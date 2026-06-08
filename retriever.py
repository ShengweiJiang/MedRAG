"""
Retrieval module using LanceDB for vector search
"""

import os
import lancedb
from sentence_transformers import SentenceTransformer

# use absolute path for database
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB_PATH = os.path.join(BASE_DIR, "lancedb")

# module-level embedding model instance to avoid reloading
model = SentenceTransformer('all-MiniLM-L6-v2')


class Retriever:
    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db = lancedb.connect(db_path)
        self.table = self.db.open_table("medical_docs")

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        query_embedding = model.encode(query).tolist()
        results = self.table.search(query_embedding).limit(top_k).to_list()

        retrieved_docs = []
        for r in results:
            retrieved_docs.append({
                "content": r["text"],
                "score": r["_distance"]
            })
        return retrieved_docs

    def get_context(self, query: str, top_k: int = 3) -> str:
        docs = self.retrieve(query, top_k)
        context_parts = []
        for i, doc in enumerate(docs, 1):
            context_parts.append(f"[Document {i}]\n{doc['content']}")
        return "\n\n".join(context_parts)


if __name__ == "__main__":
    retriever = Retriever()
    results = retriever.retrieve("What is diabetes?")
    for i, doc in enumerate(results, 1):
        print(f"--- Document {i} ---")
        print(doc['content'][:200])
        print()