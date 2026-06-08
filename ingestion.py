"""
Document processing and vectorization module
"""

import json
import os
import lancedb
import numpy as np
from sentence_transformers import SentenceTransformer

# init embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

def load_sample_data(data_path: str) -> list[dict]:
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def create_vector_store(data: list[dict], db_path: str = "./lancedb"):
    # prepare data
    documents = []
    for item in data:
        text = f"Question: {item['question']}\n\nAnswer: {item['answer']}"
        embedding = model.encode(text).tolist()
        documents.append({
            "text": text,
            "vector": embedding
        })
    
    # create lancedb
    db = lancedb.connect(db_path)
    
    # if table exists, drop it first
    if "medical_docs" in db.table_names():
        db.drop_table("medical_docs")
    
    # create table and insert data
    table = db.create_table("medical_docs", documents)
    print(f"✓ already inserted {len(documents)} documents")
    return table

def ingest_data(data_path: str, db_path: str = "./lancedb"):
    print("processing data...")
    data = load_sample_data(data_path)
    print(f"✓ loaded {len(data)} documents")
    
    print("creating vector database...")
    table = create_vector_store(data, db_path)
    return table

if __name__ == "__main__":
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "sample_medical_data.json")
    ingest_data(data_path)

