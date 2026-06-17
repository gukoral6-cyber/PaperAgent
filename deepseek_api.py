"""
DeepSeek API 封装模块
处理 DeepSeek Chat API 的调用
"""
from openai import OpenAI


def get_deepseek_client(api_key):
    """用 OpenAI 兼容格式创建 DeepSeek API 客户端"""
    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")


def ask_deepseek(q, ctx_docs, client, conv_context="", model="deepseek-chat"):
    """调用 DeepSeek 生成回答，支持对话上下文"""
    ctx = "\n\n---\n\n".join(d.page_content for d in ctx_docs)
    conv_section = ""
    if conv_context:
        conv_section = f"\n\n## 对话历史\n{conv_context}"
    sp = (
        "你是 PaperAgent，基于论文文档的智能问答助手。\n\n"
        "规则：\n"
        "1. 只根据检索内容回答，如果内容不足就明确告知无法回答\n"
        "2. 参考对话历史理解上下文，但以检索内容为准\n"
        "3. 引用原文时标注来源页码\n\n"
        "检索到的内容：\n---\n" + ctx + "\n---"
    )
    if conv_context:
        sp += "\n\n" + conv_section
    try:
        r = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": sp},
                {"role": "user", "content": q}
            ],
            temperature=0.3, stream=False
        )
        ans = r.choices[0].message.content
        u = {"prompt_tokens": r.usage.prompt_tokens,
             "completion_tokens": r.usage.completion_tokens,
             "total_tokens": r.usage.total_tokens}
        return ans, u
    except Exception as e:
        return f"❌ API 出错：{e}", None