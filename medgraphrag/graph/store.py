from neo4j import GraphDatabase


class Neo4jGraph:
    """Minimal Neo4j wrapper compatible with the query calls used in this repo."""

    def __init__(self, url: str, username: str, password: str):
        self.driver = GraphDatabase.driver(url, auth=(username, password))

    def query(self, query: str, params: dict | None = None):
        with self.driver.session() as session:
            result = session.run(query, params or {})
            return [record.data() for record in result]

    def close(self):
        self.driver.close()
