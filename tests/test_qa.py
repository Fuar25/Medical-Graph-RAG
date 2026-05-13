import unittest
from unittest.mock import AsyncMock, patch

from medgraphrag.qa import GraphQA, QAResult


class _FakeGraph:
    def __init__(self, seed_rows=None, relationship_rows=None):
        self.seed_rows = seed_rows or []
        self.relationship_rows = relationship_rows or []
        self.calls = []

    def query(self, query, params=None):
        self.calls.append((query, params or {}))
        if "gds.similarity.cosine" in query:
            return self.seed_rows
        if "UNWIND $seeds AS seed" in query:
            return self.relationship_rows
        return []


class GraphQATests(unittest.TestCase):
    def test_answer_returns_structured_result_with_evidence(self):
        graph = _FakeGraph(
            seed_rows=[
                {
                    "node_id": "二甲双胍",
                    "labels": ["药品"],
                    "gid": "gid-1",
                    "description": "用于治疗2型糖尿病的药物。",
                    "score": 0.91,
                }
            ],
            relationship_rows=[
                {
                    "node_id": "二甲双胍",
                    "gid": "gid-1",
                    "direction": "outgoing",
                    "relation_type": "禁用于",
                    "neighbor_id": "严重肾功能损害",
                    "neighbor_labels": ["禁忌"],
                    "neighbor_gid": "gid-1",
                    "neighbor_description": "严重肾功能损害患者禁用。",
                    "relation_description": "二甲双胍禁用于严重肾功能损害。",
                    "strength": "9",
                }
            ],
        )

        with (
            patch("medgraphrag.qa._embed_texts", return_value=[[0.1, 0.2]]),
            patch("medgraphrag.qa.get_chat_model", return_value="test-model"),
            patch(
                "medgraphrag.qa.openai_complete_if_cache",
                new=AsyncMock(return_value="基于证据，二甲双胍禁用于严重肾功能损害。"),
            ) as complete,
        ):
            result = GraphQA(graph).answer(
                "二甲双胍在严重肾功能损害患者中是否禁用？",
                gids=["gid-1"],
                top_k=3,
                neighbor_limit=5,
            )

        self.assertIsInstance(result, QAResult)
        self.assertEqual(result.answer, "基于证据，二甲双胍禁用于严重肾功能损害。")
        self.assertEqual(result.evidence[0].node_id, "二甲双胍")
        self.assertEqual(result.evidence[0].relationships[0]["relation_type"], "禁用于")
        self.assertTrue(result.metadata["llm_called"])
        complete.assert_called_once()

    def test_retrieval_query_passes_embedding_top_k_and_gids(self):
        graph = _FakeGraph()

        with patch("medgraphrag.qa._embed_texts", return_value=[[0.3, 0.4]]):
            GraphQA(graph).retrieve_evidence(
                "阿司匹林有什么出血风险？",
                gids=["gid-a", "gid-b"],
                top_k=4,
                neighbor_limit=0,
            )

        seed_params = graph.calls[0][1]
        self.assertEqual(seed_params["query_embedding"], [0.3, 0.4])
        self.assertEqual(seed_params["top_k"], 4)
        self.assertEqual(seed_params["gids"], ["gid-a", "gid-b"])
        self.assertIn("gds.similarity.cosine", graph.calls[0][0])

    def test_empty_evidence_returns_insufficient_answer_without_llm(self):
        graph = _FakeGraph(seed_rows=[])

        with (
            patch("medgraphrag.qa._embed_texts", return_value=[[0.1, 0.2]]),
            patch(
                "medgraphrag.qa.openai_complete_if_cache",
                new=AsyncMock(return_value="should not be used"),
            ) as complete,
        ):
            result = GraphQA(graph).answer("不存在的医学问题")

        self.assertIn("当前图谱证据不足", result.answer)
        self.assertEqual(result.evidence, [])
        self.assertFalse(result.metadata["llm_called"])
        complete.assert_not_called()


if __name__ == "__main__":
    unittest.main()
