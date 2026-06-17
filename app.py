"""
PaperAgent - Streamlit UI 入口
上传 PDF、对话问答（含历史记忆）、调试面板
"""
import os, time
import streamlit as st
from dotenv import load_dotenv
from paper_agent import (load_embedding_model, process_pdf,
    extract_abstract_complete, hybrid_retrieve)
from deepseek_api import get_deepseek_client, ask_deepseek
from langchain_core.documents import Document

load_dotenv()
st.set_page_config(page_title="PaperAgent", page_icon="\U0001f4c4", layout="wide")

def add_log(msg):
    ts = time.strftime('%H:%M:%S')
    st.session_state.logs.append(f"[{ts}] {msg}")

def build_conv_context(messages):
    """从对话历史构建上下文（取最近3轮问答）"""
    recent = messages[-6:]
    parts = []
    for msg in recent:
        role = "\u7528\u6237" if msg["role"] == "user" else "\u52a9\u624b"
        content = msg["content"][:300]
        parts.append(f"{role}\uff1a{content}")
    return "\n\n".join(parts)

# ---- 初始化 Session State ----
for k in ("vector_store","chunks","file_processed","last_file_name",
          "logs","pdf_total_pages","pdf_total_chars","pdf_chunk_count",
          "raw_page_texts","chat_messages"):
    if k not in st.session_state:
        if k == "logs": st.session_state[k] = []
        elif k == "chat_messages": st.session_state[k] = []
        elif k == "file_processed": st.session_state[k] = False
        elif k in ("pdf_total_pages","pdf_total_chars","pdf_chunk_count"): st.session_state[k] = 0
        elif k == "raw_page_texts": st.session_state[k] = {}
        else: st.session_state[k] = None

st.title("\U0001f4c4 PaperAgent - 论文智能问答")
st.markdown("上传 PDF \u8bba\u6587\uff0c\u57fa\u4e8e\u5185\u5bb9\u63d0\u95ee\u3002\u652f\u6301\u6458\u8981\u63d0\u53d6\u3001\u6df7\u5408\u68c0\u7d22\u3001\u5bf9\u8bdd\u8bb0\u5fc6\u3002")
st.divider()

with st.sidebar:
    st.header("\u2699\ufe0f \u914d\u7f6e")
    api_key = st.text_input(
        "DeepSeek API Key", type="password",
        value=os.getenv("DEEPSEEK_API_KEY",""),
        help="https://platform.deepseek.com"
    )
    if not api_key: st.warning("\u26a0\ufe0f \u8bf7\u586b\u5199 API Key")
    st.divider()
    st.markdown("### \U0001f4e4 \u4e0a\u4f20 PDF")
    uploaded_file = st.file_uploader("\u9009\u62e9 PDF", type=["pdf"], label_visibility="collapsed")

    if st.session_state.vector_store is not None:
        st.divider()
        st.markdown("### \U0001f4ca \u6587\u6863\u7edf\u8ba1")
        st.markdown(f"- **\u6587\u4ef6\u540d\uff1a** {st.session_state.last_file_name}")
        st.markdown(f"- **\u603b\u9875\u6570\uff1a** {st.session_state.pdf_total_pages}")
        st.markdown(f"- **\u603b\u5b57\u7b26\uff1a** {st.session_state.pdf_total_chars:,}")
        st.markdown(f"- **\u6587\u672c\u5757\uff1a** {st.session_state.pdf_chunk_count}")
    if st.session_state.logs:
        st.divider()
        st.markdown("### \U0001f4cb \u64cd\u4f5c\u65e5\u5fd7")
        with st.container(height=250):
            for e in st.session_state.logs: st.text(e)
        if st.button("\U0001f5d1\ufe0f \u6e05\u7a7a\u65e5\u5fd7"):
            st.session_state.logs = []; st.rerun()
    if st.session_state.chat_messages:
        st.divider()
        if st.button("\U0001f5d1\ufe0f \u6e05\u9664\u5bf9\u8bdd\u5386\u53f2"):
            st.session_state.chat_messages = []
            st.rerun()


# ---- 问答界面（含对话历史） ----
if st.session_state.vector_store is not None:
    
    # 显示对话历史
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    # 对话输入
    question = st.chat_input("请输入你的问题...")
    
    if question:
        # 添加用户消息
        st.session_state.chat_messages.append({"role": "user", "content": question})
        
        if not api_key:
            st.error("\u274c \u8bf7\u5148\u8f93\u5165 API Key")
        else:
            af = False; at = None; abs_debug = {}; ctx_docs = []; results = []
            
            # 摘要检测
            is_abs = any(k in question.lower().replace(" ","").replace("?","")
                       for k in ["abstract","summary","\u6458\u8981","\u6982\u8981","\u6982\u8ff0"])
            if is_abs:
                af, at, abs_debug = extract_abstract_complete(
                    st.session_state.raw_page_texts,
                    st.session_state.chunks, question)
            
            # 混合检索
            with st.spinner("\U0001f50d \u68c0\u7d22..."):
                results = hybrid_retrieve(st.session_state.vector_store,
                    st.session_state.chunks, question, top_k=8)
                ctx_docs = [d for d,_,_,_ in results]
            
            hp = [p+1 for _,_,_,p in results]
            add_log(f"\U0001f50d \u68c0\u7d22\u5b8c\u6210\uff0c\u547d\u4e2d\u9875\u7801\uff1a{hp}")
            
            # 摘要插入上下文
            if af and at:
                add_log(f"\u2705 \u63d0\u53d6\u6458\u8981\uff08{len(at)} \u5b57\u7b26\uff09")
                ctx_docs.insert(0, Document(
                    page_content=f"[\u6458\u8981\u63d0\u53d6\u7ed3\u679c]\n{at}",
                    metadata={"page":0,"source":"abstract_extraction"}
                ))
            
            # 构建对话上下文
            conv_ctx = ""
            if len(st.session_state.chat_messages) > 1:
                conv_ctx = build_conv_context(st.session_state.chat_messages)
            
            # 调用 DeepSeek
            with st.spinner("\U0001f916 \u751f\u6210\u56de\u7b54..."):
                client = get_deepseek_client(api_key)
                answer, usage = ask_deepseek(question, ctx_docs, client, conv_ctx)
                add_log(f"\U0001f4ac \u95ee\u9898\uff1a{question[:50]}...")
                if usage:
                    add_log(f"\U0001f4ca Token\uff1a{usage['prompt_tokens']} + {usage['completion_tokens']} = {usage['total_tokens']}")
            
            # 添加到对话历史
            cost_line = ""
            if usage:
                cst = usage["prompt_tokens"]*0.0005/1000 + usage["completion_tokens"]*0.002/1000
                cost_line = f"\n\n---\n\U0001f4ca Token\uff1a\u8f93\u5165 {usage['prompt_tokens']} | \u8f93\u51fa {usage['completion_tokens']} | \u603b\u8ba1 {usage['total_tokens']} token\uff08\u7ea6 \u00a5{cst:.4f}\uff09"
            st.session_state.chat_messages.append({
                "role": "assistant",
                "content": answer + cost_line
            })
            
            # ---- 检索详情（折叠） ----
            with st.expander("\U0001f4dd \u68c0\u7d22\u539f\u6587\u7247\u6bb5"):
                for i,(doc,dist,score,pg) in enumerate(results):
                    sim = 1.0/(1.0+dist)
                    prev = doc.page_content[:200]
                    if len(doc.page_content)>200: prev += "..."
                    with st.expander(f"\u7247\u6bb5 {i+1}\uff08\u7b2c {pg+1} \u9875 \u00b7 \u76f8\u4f3c\u5ea6 {sim:.1%} \u00b7 \u7efc\u5408\u5206 {score:.2f}\uff09"):
                        st.markdown("**\u524d200\u5b57\uff1a**")
                        st.markdown(f"```\n{prev}\n```")
                        st.markdown("---")
                        st.markdown("**\u5b8c\u6574\u5185\u5bb9\uff1a**")
                        st.markdown(doc.page_content)
            
            # ---- 调试面板 ----
            with st.expander("\U0001f6e1\ufe0f \u68c0\u7d22\u8c03\u8bd5\u4fe1\u606f"):
                st.markdown("**\u68c0\u7d22\u53c2\u6570\uff1a**")
                st.markdown("- Top-K: 8\uff08\u53d6 24 \u5019\u9009\u91cd\u6392\u5e8f\uff09")
                st.markdown("- \u5173\u952e\u8bcd\u5339\u914d: +0.12/\u4e2a")
                st.markdown("- \u7b2c1-2\u9875\u52a0\u6743: +0.2")
                st.markdown("- \u6807\u9898\u5339\u914d\u52a0\u6743: +0.3")
                st.markdown("---")
                st.markdown(f"**\u603b\u9875\u6570\uff1a** {st.session_state.pdf_total_pages} | **\u603b\u5b57\u7b26\uff1a** {st.session_state.pdf_total_chars:,} | **\u6587\u672c\u5757\uff1a** {st.session_state.pdf_chunk_count}")
                st.markdown("---")
                st.markdown("**\u547d\u4e2d\u8be6\u60c5\uff1a**")
                for i,(doc,dist,score,pg) in enumerate(results):
                    sim = 1.0/(1.0+dist)
                    cl = doc.page_content.lower()
                    tf = [t for t in ["abstract","introduction","\u6458\u8981","\u5f15\u8a00","conclusion","method"] if t in cl]
                    ts = f"\uff08\u542b\uff1a{', '.join(tf)}\uff09" if tf else ""
                    st.text(f"  #{i+1} \u7b2c{pg+1}\u9875 | \u76f8\u4f3c\u5ea6 {sim:.2%} | \u7efc\u5408\u5206 {score:.2f} | {len(doc.page_content)} \u5b57 {ts}")
                
                # 摘要提取详情
                if af and abs_debug.get("triggered"):
                    st.markdown("---")
                    st.markdown("**\u6458\u8981\u63d0\u53d6\u8be6\u60c5\uff1a**")
                    st.markdown(f"- \u89e6\u53d1\u72b6\u6001\uff1a{'\u2705 \u5df2\u89e6\u53d1' if abs_debug['triggered'] else '\u274c \u672a\u89e6\u53d1'}")
                    st.markdown(f"- \u6765\u6e90\u9875\u7801\uff1a{abs_debug['source_pages']}")
                    st.markdown(f"- \u7ed3\u675f\u6807\u9898\uff1a{abs_debug['end_marker']}")
                    st.markdown(f"- \u6458\u8981\u5b57\u7b26\u6570\uff1a{abs_debug['char_count']}")
                    st.markdown(f"- \u63d0\u53d6\u72b6\u6001\uff1a{abs_debug['status']}")
                    if abs_debug["reason"]:
                        st.markdown(f"- \u539f\u56e0\u8bf4\u660e\uff1a{abs_debug['reason']}")
                    if abs_debug["preview_300"]:
                        st.markdown("- \u524d300\u5b57\u9884\u89c8\uff1a")
                        st.markdown(f"  ```\n  {abs_debug['preview_300']}\n  ```")
            
        st.rerun()

else:
    st.info("\U0001f448 \u5de6\u4fa7\u8fb9\u680f\u4e0a\u4f20 PDF \u6587\u4ef6")
    st.markdown("""
    ### \U0001f680 \u5feb\u901f\u5f00\u59cb
    1. **\u586b\u5199 API Key** \u2014 \u5de6\u4fa7\u8fb9\u680f\u8f93\u5165 DeepSeek API Key
    2. **\u4e0a\u4f20 PDF** \u2014 \u70b9\u51fb\u4e0a\u4f20\u6309\u94ae\u9009\u62e9\u8bba\u6587
    3. **\u7b49\u5f85\u5904\u7406** \u2014 \u81ea\u52a8\u5b8c\u6210\uff1a\u8bfb\u53d6 \u2192 \u6309\u9875\u5207\u5206 \u2192 \u5411\u91cf\u5316
    4. **\u5f00\u59cb\u63d0\u95ee** \u2014 \u8f93\u5165\u95ee\u9898\uff0cAI \u57fa\u4e8e\u8bba\u6587\u5185\u5bb9\u56de\u7b54
    ### \u2728 \u4eae\u70b9\u529f\u80fd
    - \u2705 **\u6697\u8981\u4f18\u5148\u63d0\u53d6** \u2014 \u95ee Abstract/\u6458\u8981\u65f6\uff0c\u81ea\u52a8\u4ece\u7b2c1-3\u9875\u5b8c\u6574\u63d0\u53d6
    - \u2705 **\u6df7\u5408\u68c0\u7d22** \u2014 \u5411\u91cf + \u5173\u952e\u8bcd + \u9875\u9762\u52a0\u6743
    - \u2705 **\u5bf9\u8bdd\u8bb0\u5fc6** \u2014 \u53ef\u4ee5\u8ffd\u95ee\u524d\u4e00\u8bdd\u7684\u5185\u5bb9
    - \u2705 **\u8c03\u8bd5\u4fe1\u606f** \u2014 \u67e5\u770b\u68c0\u7d22\u53c2\u6570\u3001\u547d\u4e2d\u8be6\u60c5\u3001\u6458\u8981\u63d0\u53d6\u72b6\u6001
    """)