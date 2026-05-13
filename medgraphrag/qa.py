import asyncio
from dataclasses import dataclass, field
from typing import Any

from medgraphrag.llm.client import openai_complete_if_cache
from medgraphrag.llm.config import get_chat_model


_NO_EVIDENCE_ANSWER = (
    "当前图谱证据不足，无法基于已检索到的医学知识图谱内容回答这个问题。"
)

_ANSWER_SYSTEM_PROMPT = """
你是医学知识图谱问答助手。请只基于提供的图谱证据回答问题。

要求：
1. 使用中文回答。
2. 不要编造证据中没有的信息。
3. 如果证据不足，明确说明“当前图谱证据不足”。
4. 对禁忌、剂量、相互作用、不良反应等高风险医学问题，避免给出个体化诊疗指令。
5. 必要时提醒具体用药需由医生或药师结合患者情况判断。
"""


@dataclass
class EvidenceItem:
    node_id: str
    labels: list[str]
    gid: str
    description: str
    score: float
    relationships: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class QAResult:
    answer: str
    evidence: list[EvidenceItem]
    metadata: dict[str, Any]


def _embed_texts(texts: list[str]) -> list[list[float]]:
    from medgraphrag.embedding.local import embed

    return embed(texts)


class GraphQA:
    """Synchronous Graph RAG QA API backed by Neo4j node embeddings."""

    def __init__(self, n4j):
        self.n4j = n4j

    def answer(
        self,
        question: str,
        gids: list[str] | None = None,
        top_k: int = 8,
        neighbor_limit: int = 30,
    ) -> QAResult:
        question = question.strip()
        if not question:
            raise ValueError("question must not be empty")

        evidence = self.retrieve_evidence(
            question=question,
            gids=gids,
            top_k=top_k,
            neighbor_limit=neighbor_limit,
        )
        metadata = {
            "question": question,
            "gids": gids,
            "top_k": top_k,
            "neighbor_limit": neighbor_limit,
            "seed_count": len(evidence),
            "evidence_count": len(evidence),
            "relationship_count": sum(len(item.relationships) for item in evidence),
            "retrieval": "embedding_cosine_1hop",
            "llm_called": False,
        }

        if not evidence:
            return QAResult(
                answer=_NO_EVIDENCE_ANSWER,
                evidence=[],
                metadata=metadata,
            )

        prompt = self._build_answer_prompt(question, evidence)
        answer = self._complete_sync(prompt)
        metadata["llm_called"] = True
        return QAResult(answer=answer, evidence=evidence, metadata=metadata)

    def retrieve_evidence(
        self,
        question: str,
        gids: list[str] | None = None,
        top_k: int = 8,
        neighbor_limit: int = 30,
    ) -> list[EvidenceItem]:
        top_k = max(1, int(top_k))
        neighbor_limit = max(0, int(neighbor_limit))
        query_embedding = _embed_texts([question])[0]

        seed_rows = self.n4j.query(
            """
            MATCH (n)
            WHERE NOT n:Summary
              AND n.embedding IS NOT NULL
              AND ($gids IS NULL OR n.gid IN $gids)
            WITH n, gds.similarity.cosine(n.embedding, $query_embedding) AS score
            RETURN
              n.id AS node_id,
              labels(n) AS labels,
              n.gid AS gid,
              n.description AS description,
              score AS score
            ORDER BY score DESC
            LIMIT $top_k
            """,
            {
                "query_embedding": query_embedding,
                "top_k": top_k,
                "gids": gids,
            },
        )

        evidence = [
            EvidenceItem(
                node_id=row.get("node_id", ""),
                labels=row.get("labels") or [],
                gid=row.get("gid", ""),
                description=row.get("description") or "",
                score=float(row.get("score") or 0.0),
            )
            for row in seed_rows
        ]
        if not evidence or neighbor_limit == 0:
            return evidence

        evidence_by_key = {(item.node_id, item.gid): item for item in evidence}
        seeds = [
            {"node_id": item.node_id, "gid": item.gid, "order": idx}
            for idx, item in enumerate(evidence)
        ]
        relationship_rows = self.n4j.query(
            """
            UNWIND $seeds AS seed
            MATCH (n)
            WHERE n.id = seed.node_id
              AND n.gid = seed.gid
              AND NOT n:Summary
            MATCH (n)-[r]-(m)
            WHERE NOT m:Summary
            WITH seed, n, r, m
            ORDER BY seed.order
            LIMIT $neighbor_limit
            RETURN
              seed.node_id AS node_id,
              seed.gid AS gid,
              CASE WHEN startNode(r) = n THEN 'outgoing' ELSE 'incoming' END AS direction,
              type(r) AS relation_type,
              m.id AS neighbor_id,
              labels(m) AS neighbor_labels,
              m.gid AS neighbor_gid,
              m.description AS neighbor_description,
              r.description AS relation_description,
              r.strength AS strength
            """,
            {
                "seeds": seeds,
                "neighbor_limit": neighbor_limit,
            },
        )

        for row in relationship_rows:
            item = evidence_by_key.get((row.get("node_id", ""), row.get("gid", "")))
            if item is None:
                continue
            item.relationships.append(
                {
                    "direction": row.get("direction", ""),
                    "relation_type": row.get("relation_type", ""),
                    "neighbor_id": row.get("neighbor_id", ""),
                    "neighbor_labels": row.get("neighbor_labels") or [],
                    "neighbor_gid": row.get("neighbor_gid", ""),
                    "neighbor_description": row.get("neighbor_description") or "",
                    "relation_description": row.get("relation_description") or "",
                    "strength": row.get("strength"),
                }
            )

        return evidence

    def _complete_sync(self, prompt: str) -> str:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                openai_complete_if_cache(
                    get_chat_model(),
                    prompt,
                    system_prompt=_ANSWER_SYSTEM_PROMPT,
                    temperature=0.2,
                    max_tokens=900,
                )
            )
        raise RuntimeError(
            "GraphQA.answer() is synchronous and cannot run inside an active "
            "event loop."
        )

    def _build_answer_prompt(
        self, question: str, evidence: list[EvidenceItem]
    ) -> str:
        evidence_text = self._format_evidence(evidence)
        return f"""
请基于下面的医学知识图谱证据回答问题。

问题：
{question}

图谱证据：
{evidence_text}

请输出：
1. 直接回答问题。
2. 简要说明依据来自哪些图谱证据。
3. 如果证据不足，请明确说明不足之处。
"""

    def _format_evidence(self, evidence: list[EvidenceItem]) -> str:
        sections = []
        for idx, item in enumerate(evidence, 1):
            labels = "/".join(item.labels) if item.labels else "Unknown"
            lines = [
                f"[{idx}] 节点: {item.node_id}",
                f"类型: {labels}",
                f"GID: {item.gid}",
                f"相似度: {item.score:.4f}",
                f"描述: {item.description or '无'}",
            ]
            if item.relationships:
                lines.append("一跳关系:")
                for rel in item.relationships:
                    triple = self._format_relationship(item.node_id, rel)
                    relation_desc = rel.get("relation_description") or "无"
                    neighbor_desc = rel.get("neighbor_description") or "无"
                    lines.append(
                        f"- {triple}; 关系描述: {relation_desc}; "
                        f"邻居描述: {neighbor_desc}"
                    )
            sections.append("\n".join(lines))
        return "\n\n".join(sections)

    def _format_relationship(self, node_id: str, rel: dict[str, Any]) -> str:
        relation_type = rel.get("relation_type") or "相关"
        neighbor_id = rel.get("neighbor_id") or "未知节点"
        if rel.get("direction") == "incoming":
            return f"{neighbor_id} -[{relation_type}]-> {node_id}"
        return f"{node_id} -[{relation_type}]-> {neighbor_id}"
