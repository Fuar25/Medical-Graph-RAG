import os
import sys
import argparse

from dotenv import load_dotenv

from medgraphrag.pipeline.three_layer import ThreeLayerImporter

load_dotenv(override=True)


def main():
    parser = argparse.ArgumentParser(description='三层架构知识图谱导入')

    parser.add_argument('--bottom', type=str, help='Bottom层数据路径（医学词典）')
    parser.add_argument('--middle', type=str, help='Middle层数据路径（诊疗指南）')
    parser.add_argument('--top', type=str, help='Top层数据路径（病例）')

    parser.add_argument('--clear', action='store_true', help='清空数据库')
    parser.add_argument('--trinity', action='store_true', help='创建Trinity关系')
    parser.add_argument('--grained_chunk', action='store_true', help='使用细粒度分块')
    parser.add_argument('--ingraphmerge', action='store_true', help='图内合并相似节点')
    parser.add_argument('--skip_summary', action='store_true', help='跳过Summary节点生成')

    parser.add_argument('--neo4j-url', type=str, default=os.getenv('NEO4J_URI', 'bolt://localhost:7687'))
    parser.add_argument('--neo4j-username', type=str, default=os.getenv('NEO4J_USERNAME', 'neo4j'))
    parser.add_argument('--neo4j-password', type=str, default=os.getenv('NEO4J_PASSWORD'))

    args = parser.parse_args()

    if not args.neo4j_password:
        print("❌ 错误: 未提供 Neo4j 密码")
        print("请设置环境变量 NEO4J_PASSWORD 或使用 --neo4j-password 参数")
        sys.exit(1)

    importer = ThreeLayerImporter(args.neo4j_url, args.neo4j_username, args.neo4j_password)

    if args.clear:
        importer.clear_database()

    if args.bottom:
        importer.import_layer('bottom', args.bottom, args)
    if args.middle:
        importer.import_layer('middle', args.middle, args)
    if args.top:
        importer.import_layer('top', args.top, args)

    if args.trinity:
        importer.create_trinity_links()

    importer.print_statistics()
    print("\n🎉 所有任务完成！")


if __name__ == '__main__':
    main()
