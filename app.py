"""
Streamlit app for MedRAG - a medical question-answering system based on Retrieval-Augmented Generation (RAG).
"""
import sys
import os
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Put current directory in sys.path for imports
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

# Custom modules (ensure these files exist in your directory)
from generator import Generator, check_api_status
from ingestion import ingest_data

st.set_page_config(page_title="MedRAG", page_icon="🏥", layout="wide")

st.title("🏥 MedRAG- Medical Question Answering System")
st.caption("based on Retrieval-Augmented Generation (RAG) with Anthropic's API and Lancedb")

# Use absolute paths for database and data file
db_path = os.path.join(BASE_DIR, "lancedb")
data_path = os.path.join(BASE_DIR, "data", "sample_medical_data.json")

# --- SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.header("⚙️ System Status")

    if check_api_status():
        st.success("✓ Anthropic API is configured")
    else:
        st.error("✗ ANTHROPIC_API_KEY is not set")
        st.info("Please set the ANTHROPIC_API_KEY environment variable")

    st.divider()
    st.header("📚 Knowledge Base")

    if os.path.exists(db_path):
        st.success("✓ Knowledge Base is initialized")
    else:
        st.warning("Knowledge Base is not initialized")
        if st.button("Initialize Knowledge Base"):
            with st.spinner("Processing..."):
                ingest_data(data_path, db_path)
            st.success("✓ Completion!")
            st.rerun()

    st.divider()
    top_k = st.slider("Retrieve Document Count", 1, 5, 3)

    st.divider()
    st.warning("⚠️ Disclaimer: This system is for reference only and cannot replace professional medical advice.")

# --- SESSION STATE INITIALIZATION ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- RENDER CHAT HISTORY ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "docs" in message and message["docs"]:
            with st.expander("📄 Reference Documents"):
                for i, doc in enumerate(message["docs"], 1):
                    st.markdown(f"**Document {i}**")
                    st.text(doc['content'])
                    st.divider()

# --- HANDLE EXAMPLES AND INPUT ---
prompt = st.chat_input("Please enter your health question...")

# Fix #3: only show examples when chat is empty AND the user hasn't just submitted a prompt,
# otherwise the examples block flashes above the first answer for one rerun.
if len(st.session_state.messages) == 0 and not prompt:
    st.markdown("### 💡 Try these questions:")
    cols = st.columns(2)
    examples = [
        "What is diabetes?",
        "What are symptoms of a heart attack?",
        "What causes pneumonia?",
        "What is hypertension?"
    ]
    for i, q in enumerate(examples):
        # If an example is clicked, assign it to the prompt variable
        if cols[i % 2].button(q, key=f"ex_{i}"):
            prompt = q

# --- GENERATION LOGIC ---
if prompt:
    if not check_api_status():
        st.error("Please configure the ANTHROPIC_API_KEY first!")
    elif not os.path.exists(db_path):
        st.error("Please initialize the knowledge base first!")
    else:
        # Fix #2: lazy-init the generator here, where API + DB checks have already passed.
        # Single source of truth, no AttributeError risk.
        if "generator" not in st.session_state:
            st.session_state.generator = Generator()

        # 1. Show user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. Generate and show assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Fix #1: guard the API call so a failure shows a clean error
                # instead of crashing the app with a traceback.
                try:
                    result = st.session_state.generator.generate(prompt, top_k=top_k)
                except Exception as e:
                    st.error(f"Generation failed: {e}")
                    st.stop()

            st.markdown(result['answer'])

            if result.get('retrieved_docs'):
                with st.expander("📄 Reference Documents"):
                    for i, doc in enumerate(result['retrieved_docs'], 1):
                        st.markdown(f"**Document {i}**")
                        st.text(doc['content'])
                        st.divider()

        # 3. Save assistant response to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": result['answer'],
            "docs": result.get('retrieved_docs', [])
        })