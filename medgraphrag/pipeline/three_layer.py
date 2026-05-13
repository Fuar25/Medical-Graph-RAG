from pathlib import Path

from medgraphrag.graph.store import Neo4jGraph
from medgraphrag.graph.operations import str_uuid, ref_link
from medgraphrag.ingestion.loader import load_high
from medgraphrag.extraction.entity_extractor import create_metagraph_with_description


class ThreeLayerImporter:
    """三层架构知识图谱导入器"""

    def __init__(self, neo4j_url: str, neo4j_username: str, neo4j_password: str):
        print("\n" + "="*80)
        print("三层架构知识图谱导入器")
        print("="*80)

        print("\n[连接Neo4j]...")
        self.n4j = Neo4jGraph(url=neo4j_url, username=neo4j_username, password=neo4j_password)
        print("✅ Neo4j连接成功")

        self.layer_gids: dict[str, list[str]] = {'bottom': [], 'middle': [], 'top': []}

    def clear_database(self):
        print("\n[清空数据库]...")
        result = self.n4j.query("MATCH (n) RETURN count(n) as count")
        count = result[0]['count'] if result else 0
        print(f"当前节点数: {count}")
        if count > 0:
            print("自动清空数据库...")
            self.n4j.query("MATCH (n) DETACH DELETE n")
            print("✅ 数据库已清空")
        else:
            print("数据库已经是空的")

    def import_layer(self, layer_name: str, data_path: str, args):
        print("\n" + "="*80)
        print(f"[{layer_name.upper()}层] 开始导入")
        print(f"数据路径: {data_path}")
        print("="*80)

        data_path = Path(data_path)
        if data_path.is_file():
            files = [data_path]
        else:
            files = []
            for pattern in ("*.txt", "*.pdf"):
                files.extend(data_path.glob(pattern))
                files.extend(data_path.rglob(f"*/{pattern}"))

        print(f"\n找到 {len(files)} 个文件")

        for idx, file_path in enumerate(files, 1):
            print(f"\n{'─'*80}")
            print(f"[文件 {idx}/{len(files)}] {file_path.name}")
            print(f"{'─'*80}")

            try:
                content = load_high(str(file_path))
                if not content or len(content.strip()) < 50:
                    print("⚠️  跳过: 内容太短")
                    continue

                gid = str_uuid()
                self.layer_gids[layer_name].append(gid)
                self.n4j = create_metagraph_with_description(args, content, gid, self.n4j)
                print(f"✅ 完成: {file_path.name} (GID: {gid[:8]}...)")

            except Exception as e:
                print(f"❌ 错误: {file_path.name} - {e}")
                continue

        print(f"\n{'='*80}")
        print(f"[{layer_name.upper()}层] 导入完成")
        print(f"共导入 {len(self.layer_gids[layer_name])} 个子图")
        print(f"{'='*80}")

    def create_trinity_links(self):
        print("\n" + "="*80)
        print("[Trinity链接] 开始创建跨层关系")
        print("="*80)

        total_links = 0

        print("\n[链接] Bottom → Middle")
        for bottom_gid in self.layer_gids['bottom']:
            for middle_gid in self.layer_gids['middle']:
                try:
                    result = ref_link(self.n4j, bottom_gid, middle_gid)
                    if result:
                        count = len(result)
                        total_links += count
                        if count > 0:
                            print(f"  ✅ {bottom_gid[:8]}... → {middle_gid[:8]}...: {count} 条")
                except Exception as e:
                    print(f"  ⚠️  错误: {e}")

        print("\n[链接] Middle → Top")
        for middle_gid in self.layer_gids['middle']:
            for top_gid in self.layer_gids['top']:
                try:
                    result = ref_link(self.n4j, middle_gid, top_gid)
                    if result:
                        count = len(result)
                        total_links += count
                        if count > 0:
                            print(f"  ✅ {middle_gid[:8]}... → {top_gid[:8]}...: {count} 条")
                except Exception as e:
                    print(f"  ⚠️  错误: {e}")

        print(f"\n{'='*80}")
        print("[Trinity链接] 完成")
        print(f"共创建 {total_links} 条 REFERENCE 关系")
        print(f"{'='*80}")

    def print_statistics(self):
        print("\n" + "="*80)
        print("[统计信息]")
        print("="*80)

        node_count = (self.n4j.query("MATCH (n) WHERE NOT n:Summary RETURN count(n) as count") or [{}])[0].get('count', 0)
        summary_count = (self.n4j.query("MATCH (s:Summary) RETURN count(s) as count") or [{}])[0].get('count', 0)
        rel_count = (self.n4j.query("MATCH ()-[r]->() RETURN count(r) as count") or [{}])[0].get('count', 0)
        ref_count = (self.n4j.query("MATCH ()-[r:REFERENCE]->() RETURN count(r) as count") or [{}])[0].get('count', 0)
        type_result = self.n4j.query("""
            MATCH (n) WHERE NOT n:Summary
            RETURN labels(n)[0] as type, count(n) as count
            ORDER BY count DESC LIMIT 10
        """)

        print(f"\n总体统计:")
        print(f"  - 实体节点: {node_count}")
        print(f"  - Summary节点: {summary_count}")
        print(f"  - 总关系: {rel_count}")
        print(f"  - REFERENCE关系: {ref_count}")

        print(f"\n层级统计:")
        print(f"  - Bottom层: {len(self.layer_gids['bottom'])} 个子图")
        print(f"  - Middle层: {len(self.layer_gids['middle'])} 个子图")
        print(f"  - Top层: {len(self.layer_gids['top'])} 个子图")

        print(f"\n实体类型 (前10):")
        for item in type_result:
            print(f"  - {item['type']}: {item['count']}")

        print(f"\n{'='*80}")
