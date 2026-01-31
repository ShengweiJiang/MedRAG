"""
检索模块
"""

import lancedb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

class Retriever:
    def __init__(self, db_path: str = "./lancedb"):
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
            context_parts.append(f"[文档 {i}]\n{doc['content']}")
        return "\n\n".join(context_parts)

if __name__ == "__main__":
    retriever = Retriever()
    results = retriever.retrieve("What is diabetes?")
    for i, doc in enumerate(results, 1):
        print(f"--- 文档 {i} ---")
        print(doc['content'][:200])
        print()

