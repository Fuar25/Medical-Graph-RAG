import re
import asyncio
from typing import List, Dict

from medgraphrag.llm.client import openai_complete_if_cache
from medgraphrag.llm.config import get_chat_model
from medgraphrag.embedding.local import embed
from medgraphrag.graph.operations import add_sum, merge_similar_nodes
from medgraphrag.ingestion.chunking import split_text_chunks, run_chunk


DEFAULT_ENTITY_TYPES = [
    "药品",
    "活性成分",
    "禁忌",
    "用法用量",
    "给药途径",
    "不良反应",
    "警告注意事项",
    "药物相互作用",
    "特殊人群",
    "检查监测",
    "解剖部位",
    "疾病",
    "症状",
    "剂量规格",
]

ALLOWED_RELATION_TYPES = [
    "含有成分",
    "适用于",
    "禁用于",
    "用法用量",
    "给药途径",
    "不良反应",
    "警告注意",
    "药物相互作用",
    "需要监测",
    "特殊人群注意",
    "作用部位",
    "表现症状",
]

def _clean_str(input_str: str) -> str:
    if not input_str:
        return ""
    return input_str.strip().strip('"').strip("'")


def _extract_parenthesized_content(record: str) -> str:
    record = record.strip()
    if record.startswith("(") and record.endswith(")"):
        return record[1:-1]
    match = re.search(r'\((.*)\)', record)
    return match.group(1) if match else ""


def parse_extraction_response(
    response: str,
    allowed_entity_types: List[str] = None,
    tuple_delimiter: str = "<|>",
    record_delimiter: str = "##",
    completion_delimiter: str = "<|COMPLETE|>",
) -> tuple:
    entities = []
    relationships = []
    stats = {
        "invalid_records": 0,
        "invalid_entity_records": 0,
        "invalid_entity_types": 0,
        "invalid_relationship_records": 0,
        "invalid_relation_types": 0,
    }
    allowed_entity_types = allowed_entity_types or DEFAULT_ENTITY_TYPES

    if not response:
        return entities, relationships, stats

    for record in response.split(record_delimiter):
        record = record.replace(completion_delimiter, "").strip()
        if not record:
            continue

        record_content = _extract_parenthesized_content(record)
        if not record_content:
            stats["invalid_records"] += 1
            continue

        attributes = record_content.split(tuple_delimiter)
        if len(attributes) < 2:
            stats["invalid_records"] += 1
            continue

        record_type = _clean_str(attributes[0])

        if record_type == "entity":
            if len(attributes) < 4:
                stats["invalid_entity_records"] += 1
                continue
            entity = {
                'entity_name': _clean_str(attributes[1]),
                'entity_type': _clean_str(attributes[2]),
                'description': _clean_str(attributes[3]),
            }
            if entity['entity_type'] not in allowed_entity_types:
                stats["invalid_entity_types"] += 1
                continue
            if entity['entity_name']:
                entities.append(entity)

        elif record_type == "relationship":
            if len(attributes) < 6:
                stats["invalid_relationship_records"] += 1
                continue
            relation_type = _clean_str(attributes[2])
            if relation_type not in ALLOWED_RELATION_TYPES:
                stats["invalid_relation_types"] += 1
                continue
            relationship = {
                'src': _clean_str(attributes[1]),
                'relation_type': relation_type,
                'tgt': _clean_str(attributes[3]),
                'description': _clean_str(attributes[4]),
                'strength': _clean_str(attributes[5]),
            }
            if relationship['src'] and relationship['tgt']:
                relationships.append(relationship)
        else:
            stats["invalid_records"] += 1

    return entities, relationships, stats


async def _extract_entities_from_chunk(content: str, entity_types: List[str] = None) -> tuple:
    entity_types = entity_types or DEFAULT_ENTITY_TYPES
    entity_types_str = ", ".join(entity_types)
    relation_types_str = ", ".join(ALLOWED_RELATION_TYPES)
    tuple_delimiter = "<|>"
    record_delimiter = "##"
    completion_delimiter = "<|COMPLETE|>"

    prompt = f"""
你是医学知识图谱构建助手。请从药品说明书或医学文本中抽取实体和关系，并使用中文输出。

要求：
1. 实体名称、实体类型、实体描述都必须使用中文。英文药品名可保留原文，但要优先附带中文名称。
2. 实体类型只能从以下列表选择：{entity_types_str}
3. 关系类型只能从以下列表选择：{relation_types_str}
4. 药品说明书场景下，优先以"药品"或"活性成分"作为关系源实体，连接到疾病、禁忌、用法用量、给药途径、不良反应、警告注意事项、特殊人群、检查监测等实体。
5. "适应症"不要作为实体类型；把具体适应症建成"疾病"实体，并用"适用于"关系从药品或活性成分指向疾病。
6. 不要抽取泛化的"患者""医生""治疗""疾病""药物""症状"等无具体指代的节点；除非文本讨论的是具体特殊人群或具体疾病/症状。
7. 如果两个实体之间的关系类型无法确定，跳过该关系，不要输出兜底关系。
8. 关系描述必须使用中文，说明两个实体之间的具体医学含义。
9. 输出必须严格使用下面格式，不要添加解释文字。

实体格式：
("entity"{tuple_delimiter}实体名称{tuple_delimiter}实体类型{tuple_delimiter}中文描述)

关系格式：
("relationship"{tuple_delimiter}源实体名称{tuple_delimiter}关系类型{tuple_delimiter}目标实体名称{tuple_delimiter}中文关系描述{tuple_delimiter}关系强度数字1到10)

记录之间使用：{record_delimiter}
结束时输出：{completion_delimiter}

待抽取文本：
{content}
"""

    print("  [Entity Extraction] 正在提取实体和关系...")
    response = await openai_complete_if_cache(
        model=get_chat_model("gpt-4o-mini"),
        prompt=prompt,
        system_prompt="You are a helpful assistant that extracts entities and relationships from medical texts.",
    )

    entities, relationships, stats = parse_extraction_response(
        response,
        allowed_entity_types=entity_types,
        tuple_delimiter=tuple_delimiter,
        record_delimiter=record_delimiter,
        completion_delimiter=completion_delimiter,
    )

    skipped = sum(stats.values())
    if skipped:
        print(
            f"  ⚠️  跳过无效记录: 总计 {skipped} 条, "
            f"非法实体类型 {stats['invalid_entity_types']} 条, "
            f"非法关系类型 {stats['invalid_relation_types']} 条, "
            f"缺字段关系 {stats['invalid_relationship_records']} 条"
        )

    print(f"  ✅ 提取了 {len(entities)} 个实体，{len(relationships)} 个关系")
    return entities, relationships


async def _extract_all_chunks(content_chunks: List[str]) -> tuple:
    all_entities, all_relationships = [], []
    for idx, chunk in enumerate(content_chunks, 1):
        print(f"\n[块 {idx}/{len(content_chunks)}] 处理中...")
        try:
            entities, relationships = await _extract_entities_from_chunk(chunk)
            all_entities.extend(entities)
            all_relationships.extend(relationships)
        except Exception as e:
            print(f"  ⚠️  块 {idx} 提取失败，跳过: {e}")
    return all_entities, all_relationships


def _write_to_neo4j(n4j, entities: List[Dict], relationships: List[Dict], gid: str):
    print(f"  [Neo4j] 开始写入 {len(entities)} 个实体...")

    for entity in entities:
        entity_name = entity['entity_name']
        entity_type = entity['entity_type']
        description = entity['description']

        embedding_text = f"{entity_name}: {description}" if description else entity_name
        embedding = embed([embedding_text])[0]

        create_node_query = """
        MERGE (n:`%s` {id: $id, gid: $gid})
        ON CREATE SET
            n.description = $description,
            n.embedding = $embedding,
            n.source = 'medgraphrag'
        ON MATCH SET
            n.description = CASE WHEN n.description IS NULL OR n.description = ''
                                 THEN $description
                                 ELSE n.description END,
            n.embedding = CASE WHEN n.embedding IS NULL
                               THEN $embedding
                               ELSE n.embedding END
        RETURN n
        """ % entity_type

        try:
            n4j.query(create_node_query, {
                'id': entity_name,
                'gid': gid,
                'description': description,
                'embedding': embedding,
            })
        except Exception as e:
            print(f"  ⚠️  创建节点失败: {entity_name} - {e}")

    print("  ✅ 实体节点创建完成")
    print(f"  [Neo4j] 开始创建 {len(relationships)} 个关系...")

    for rel in relationships:
        src = rel['src']
        tgt = rel['tgt']
        rel_type = rel.get('relation_type', '')
        if rel_type not in ALLOWED_RELATION_TYPES:
            print(f"  ⚠️  跳过非法关系类型: {src} -[{rel_type}]-> {tgt}")
            continue

        create_rel_query = """
        MATCH (a {id: $src, gid: $gid})
        MATCH (b {id: $tgt, gid: $gid})
        MERGE (a)-[r:`%s`]->(b)
        ON CREATE SET r.description = $description, r.strength = $strength
        RETURN r
        """ % rel_type

        try:
            n4j.query(create_rel_query, {
                'src': src,
                'tgt': tgt,
                'gid': gid,
                'description': rel.get('description', ''),
                'strength': rel.get('strength', ''),
            })
        except Exception as e:
            print(f"  ⚠️  创建关系失败: {src} -> {tgt} - {e}")

    print("  ✅ 关系创建完成")


def create_metagraph_with_description(args, content: str, gid: str, n4j):
    print(f"\n{'='*60}")
    print(f"[图构建] 开始构建知识图谱 (GID: {gid[:8]}...)")
    print(f"{'='*60}")

    if args.grained_chunk:
        print("[分块] 使用细粒度分块...")
        content_chunks = run_chunk(content)
    else:
        print("[分块] 使用固定长度分块...")
        content_chunks = split_text_chunks(content)

    all_entities, all_relationships = asyncio.run(_extract_all_chunks(content_chunks))

    print(f"\n[汇总] 总共提取:")
    print(f"  - 实体: {len(all_entities)} 个")
    print(f"  - 关系: {len(all_relationships)} 个")

    print("\n[合并] 合并重复实体...")
    entity_dict = {}
    for entity in all_entities:
        name = entity['entity_name']
        if name in entity_dict:
            existing_desc = entity_dict[name]['description']
            new_desc = entity['description']
            if new_desc and new_desc not in existing_desc:
                entity_dict[name]['description'] = f"{existing_desc}; {new_desc}"
        else:
            entity_dict[name] = entity

    merged_entities = list(entity_dict.values())
    print(f"  ✅ 合并后: {len(merged_entities)} 个实体")

    print("\n[写入Neo4j] 开始...")
    _write_to_neo4j(n4j, merged_entities, all_relationships, gid)

    if args.ingraphmerge:
        print("\n[图内合并] 合并相似节点...")
        merge_similar_nodes(n4j, gid)

    if getattr(args, "skip_summary", False):
        print("\n[Summary] 已跳过摘要节点生成")
    else:
        print("\n[Summary] 创建摘要节点...")
        add_sum(n4j, content, gid)

    print(f"\n{'='*60}")
    print(f"[图构建] 完成! (GID: {gid[:8]}...)")
    print(f"{'='*60}\n")

    return n4j
