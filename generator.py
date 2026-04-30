"""
生成模块 - 使用 Anthropic API
"""

import os
from anthropic import Anthropic
from retriever import Retriever


class Generator:
    def __init__(self, model_name: str = "claude-haiku-4-5"):
        self.model_name = model_name
        self.retriever = Retriever()
        self.client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

        self.system_prompt = """你是一个专业的医疗健康助手。请根据提供的参考资料回答用户的问题。

重要规则：
1. 只根据参考资料中的信息回答，不要编造信息
2. 如果参考资料中没有相关信息，请诚实地说"根据现有资料，我无法回答这个问题"
3. 回答要简洁、准确、易懂
4. 在回答末尾注明信息来源于哪个文档

免责声明：本系统仅供参考，不能替代专业医疗建议。如有健康问题，请咨询医生。"""

    def generate(self, query: str, top_k: int = 3) -> dict:
        context = self.retriever.get_context(query, top_k)
        retrieved_docs = self.retriever.retrieve(query, top_k)

        user_prompt = f"""参考资料：
{context}

用户问题：{query}

请根据参考资料回答问题："""

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
            answer = f"生成回答时出错: {str(e)}\n\n请检查 ANTHROPIC_API_KEY 是否正确配置。"

        return {
            "answer": answer,
            "retrieved_docs": retrieved_docs,
            "query": query
        }


def check_api_status() -> bool:
    """检查 API key 是否配置"""
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


if __name__ == "__main__":
    if check_api_status():
        gen = Generator()
        result = gen.generate("What is diabetes?")
        print(result['answer'])
    else:
        print("ANTHROPIC_API_KEY 未设置")