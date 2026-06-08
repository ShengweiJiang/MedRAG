"""
genrate module using anthropic API 
"""

import os
from anthropic import Anthropic
from retriever import Retriever


class Generator:
    def __init__(self, model_name: str = "claude-haiku-4-5"):
        self.model_name = model_name
        self.retriever = Retriever()
        self.client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

        self.system_prompt = """You are a professional healthcare assistant. Please answer the user's questions based on the provided reference materials.

Important Rules:
1. Only answer based on the information in the reference materials, do not fabricate information
2. If the reference materials do not contain relevant information, please honestly say "Based on the available information, I cannot answer this question"
3. Answers should be concise, accurate, and easy to understand
4. Please indicate the source document at the end of each answer

Disclaimer: This system is for reference only and cannot replace professional medical advice. If you have health concerns, please consult a doctor."""

    def generate(self, query: str, top_k: int = 3) -> dict:
        context = self.retriever.get_context(query, top_k)
        retrieved_docs = self.retriever.retrieve(query, top_k)

        user_prompt = f"""reference materials:
{context}

user question: {query}

please answer the question based on the reference materials: """

        try:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=1024,
                system=self.system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )
            answer = response.content[0].text
        except Exception as e:
            answer = f"Error generating answer: {str(e)}\n\nPlease check if ANTHROPIC_API_KEY is correctly configured."

        return {
            "answer": answer,
            "retrieved_docs": retrieved_docs,
            "query": query
        }


def check_api_status() -> bool:
    """Check if the API key is configured"""
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


if __name__ == "__main__":
    if check_api_status():
        gen = Generator()
        result = gen.generate("What is diabetes?")
        print(result['answer'])
    else:
        print("ANTHROPIC_API_KEY is not set")