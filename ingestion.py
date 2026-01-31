"""
文档处理和向量化模块
"""

import json
import os
import lancedb
import numpy as np
from sentence_transformers import SentenceTransformer

# 初始化 embedding 模型
model = SentenceTransformer('all-MiniLM-L6-v2')

def load_sample_data(data_path: str) -> list[dict]:
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def create_vector_store(data: list[dict], db_path: str = "./lancedb"):
    # 准备数据
    documents = []
    for item in data:
        text = f"Question: {item['question']}\n\nAnswer: {item['answer']}"
        embedding = model.encode(text).tolist()
        documents.append({
            "text": text,
            "vector": embedding
        })
    
    # 创建 LanceDB
    db = lancedb.connect(db_path)
    
    # 如果表存在就删除
    if "medical_docs" in db.table_names():
        db.drop_table("medical_docs")
    
    # 创建新表
    table = db.create_table("medical_docs", documents)
    print(f"✓ 已创建向量数据库，共 {len(documents)} 条文档")
    return table

def ingest_data(data_path: str, db_path: str = "./lancedb"):
    print("正在加载数据...")
    data = load_sample_data(data_path)
    print(f"✓ 加载了 {len(data)} 条文档")
    
    print("正在创建向量数据库...")
    table = create_vector_store(data, db_path)
    return table

if __name__ == "__main__":
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "sample_medical_data.json")
    ingest_data(data_path)

