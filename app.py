"""
Streamlit 界面
"""

import streamlit as st
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from generator import Generator, check_ollama_status
from ingestion import ingest_data

st.set_page_config(page_title="MedRAG", page_icon="🏥", layout="wide")

st.title("🏥 MedRAG - 医疗问答助手")
st.caption("基于 RAG 的医学知识问答系统")

with st.sidebar:
    st.header("⚙️ 系统状态")
    
    if check_ollama_status():
        st.success("✓ Ollama 已连接")
    else:
        st.error("✗ Ollama 未运行")
        st.info("请先运行: ollama run llama3.2")
    
    st.divider()
    st.header("📚 知识库")
    
    db_path = "./lancedb"
    data_path = "./data/sample_medical_data.json"
    
    if os.path.exists(db_path):
        st.success("✓ 知识库已初始化")
    else:
        st.warning("知识库未初始化")
        if st.button("初始化知识库"):
            with st.spinner("正在处理..."):
                ingest_data(data_path, db_path)
            st.success("✓ 完成！")
            st.rerun()
    
    st.divider()
    top_k = st.slider("检索文档数量", 1, 5, 3)
    
    st.divider()
    st.warning("⚠️ 免责声明：本系统仅供参考，不能替代专业医疗建议。")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "generator" not in st.session_state and check_ollama_status() and os.path.exists(db_path):
    st.session_state.generator = Generator()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "docs" in message:
            with st.expander("📄 参考文档"):
                for i, doc in enumerate(message["docs"], 1):
                    st.markdown(f"**文档 {i}**")
                    st.text(doc['content'])
                    st.divider()

if prompt := st.chat_input("请输入您的健康问题..."):
    if not check_ollama_status():
        st.error("请先启动 Ollama！")
    elif not os.path.exists(db_path):
        st.error("请先初始化知识库！")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("正在思考..."):
                result = st.session_state.generator.generate(prompt, top_k=top_k)
            st.markdown(result['answer'])
            with st.expander("📄 参考文档"):
                for i, doc in enumerate(result['retrieved_docs'], 1):
                    st.markdown(f"**文档 {i}**")
                    st.text(doc['content'])
                    st.divider()
        
        st.session_state.messages.append({
            "role": "assistant",
            "content": result['answer'],
            "docs": result['retrieved_docs']
        })

if len(st.session_state.messages) == 0:
    st.markdown("### 💡 试试这些问题：")
    cols = st.columns(2)
    examples = ["What is diabetes?", "What are symptoms of a heart attack?", "What causes pneumonia?", "What is hypertension?"]
    for i, q in enumerate(examples):
        if cols[i % 2].button(q, key=f"ex_{i}"):
            st.session_state.messages.append({"role": "user", "content": q})
            st.rerun()

