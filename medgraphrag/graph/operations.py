import uuid

from medgraphrag.llm.summarizer import process_chunks
from medgraphrag.embedding.local import embed


def get_embedding(text: str) -> list[float]:
    return embed([text])[0]


def add_sum(n4j, content: str, gid: str):
    summary = process_chunks(content)
    if not summary:
        print("[Summary] 摘要为空，跳过Summary节点创建")
        return []

    create_summary_query = """
        CREATE (s:Summary {content: $summary, gid: $gid})
        RETURN s
        """
    s = n4j.query(create_summary_query, {"summary": summary, "gid": gid})

    link_sum_query = """
        MATCH (s:Summary {gid: $gid}), (n)
        WHERE n.gid = s.gid AND NOT n:Summary
        CREATE (s)-[:SUMMARIZES]->(n)
        RETURN s, n
        """
    n4j.query(link_sum_query, {"gid": gid})
    return s


def merge_similar_nodes(n4j, gid: str | None = None):
    if gid:
        merge_query = """
            WITH 0.5 AS threshold
            MATCH (n), (m)
            WHERE NOT n:Summary AND NOT m:Summary AND n.gid = m.gid AND n.gid = $gid AND n<>m AND apoc.coll.sort(labels(n)) = apoc.coll.sort(labels(m))
            WITH n, m,
                gds.similarity.cosine(n.embedding, m.embedding) AS similarity
            WHERE similarity > threshold
            WITH head(collect([n,m])) as nodes
            CALL apoc.refactor.mergeNodes(nodes, {properties: 'overwrite', mergeRels: true})
            YIELD node
            RETURN count(*)
        """
        return n4j.query(merge_query, {"gid": gid})
    else:
        merge_query = """
            WITH 0.5 AS threshold
            MATCH (n), (m)
            WHERE NOT n:Summary AND NOT m:Summary AND n<>m AND apoc.coll.sort(labels(n)) = apoc.coll.sort(labels(m))
            WITH n, m,
                gds.similarity.cosine(n.embedding, m.embedding) AS similarity
            WHERE similarity > threshold
            WITH head(collect([n,m])) as nodes
            CALL apoc.refactor.mergeNodes(nodes, {properties: 'overwrite', mergeRels: true})
            YIELD node
            RETURN count(*)
        """
        return n4j.query(merge_query)


def ref_link(n4j, gid1: str, gid2: str):
    trinity_query = """
        MATCH (a)
        WHERE a.gid = $gid1 AND NOT a:Summary
        WITH collect(a) AS GraphA

        MATCH (b)
        WHERE b.gid = $gid2 AND NOT b:Summary
        WITH GraphA, collect(b) AS GraphB

        UNWIND GraphA AS n
        UNWIND GraphB AS m

        WITH n, m, 0.6 AS threshold
        WHERE apoc.coll.sort(labels(n)) = apoc.coll.sort(labels(m)) AND n <> m
        WITH n, m, threshold,
            gds.similarity.cosine(n.embedding, m.embedding) AS similarity
        WHERE similarity > threshold

        MERGE (m)-[:REFERENCE]->(n)
        RETURN n, m
    """
    return n4j.query(trinity_query, {"gid1": gid1, "gid2": gid2})


def str_uuid() -> str:
    return str(uuid.uuid4())
