"""
PaperAgent - 核心业务逻辑
PDF 处理、向量化、混合检索、摘要提取
"""
import os, re
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# ============================================================
@st.cache_resource(show_spinner="\u23f3 加载 Embedding 模型（首次约 80MB）...")
def load_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

# ============================================================
# 2. DeepSeek 客户端（缓存）
# ============================================================
@st.cache_resource

def process_pdf(file_path):
    """
    读取 PDF -> 按页切分 -> 构建 FAISS 向量索引
    
    改进点：
    - 每页独立切分，不同页的内容不会混入同一个文本块
    - 每个文本块携带 page 元数据
    - 统计总页数和总字符数
    """
    pdf_name = os.path.basename(file_path)

    with st.spinner("\U0001f4d6 读取 PDF..."):
        docs = PyPDFLoader(file_path).load()
        total_pages = len(docs)
        st.success(f"\u2705 读取完成，共 {total_pages} 页")

    with st.spinner("\u2702\ufe0f 按页切分..."):
        all_chunks = []
        total_chars = 0
        for doc in docs:
            pg = doc.metadata.get("page", 0)
            txt = doc.page_content
            total_chars += len(txt)
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=500, chunk_overlap=100,
                separators=["\n\n", "\n", "\u3002", ".", " ", ""],
                length_function=len
            )
            for ct in splitter.split_text(txt):
                all_chunks.append(Document(
                    page_content=ct, metadata={"page": pg, "source": pdf_name}
                ))
        st.success(f"\u2705 切分完成，{len(all_chunks)} 个文本块")

    with st.spinner("\U0001f527 构建向量索引..."):
        vs = FAISS.from_documents(all_chunks, load_embedding_model())
        st.success("\u2705 索引完成")

    st.session_state.pdf_total_pages = total_pages
    st.session_state.pdf_total_chars = total_chars
    st.session_state.pdf_chunk_count = len(all_chunks)
    # 保存原始页面文本（用于摘要完整提取）
    st.session_state.raw_page_texts = {}
    for doc in docs:
        pg = doc.metadata.get("page", 0)
        st.session_state.raw_page_texts[pg] = doc.page_content

    return vs, all_chunks

# ============================================================
# 4. 摘要优先提取（改进：问 Abstract/摘要时从第1-2页提取）
# ============================================================

def extract_abstract_complete(raw_pages, chunks, question):
    """
    增强版摘要完整提取：
    1. 从原始页面文本提取（非切分过的块）
    2. 正则匹配 Abstract 标题到 Keywords/Introduction 结束
    3. 跨页合并 + 文本清洗
    4. 返回 (found, text, debug_info)
    """
    debug = {
        "triggered": False,
        "source_pages": [],
        "end_marker": "",
        "char_count": 0,
        "preview_300": "",
        "status": "failed",
        "reason": ""
    }

    # ---- 关键词检测 ----
    akw = ["abstract", "summary", "summarize", "overview",
           "摘要", "概要", "概述", "概述内容"]
    lq = question.lower().replace(" ", "").replace("?", "").replace("？", "")
    if not any(k in lq for k in akw):
        return False, None, debug

    debug["triggered"] = True
    add_log("U0001f50d 触发摘要专用提取逻辑")

    # ---- 获取第1-2页原始文本（非切分块） ----
    pages_text = []
    for pg in [0, 1, 2]:
        if pg in raw_pages and raw_pages[pg].strip():
            pages_text.append(raw_pages[pg])
            debug["source_pages"].append(pg + 1)
            add_log(f"  \u2714\ufe0f 获取第{pg+1}页原始文本（{len(raw_pages[pg])} 字符）")
        else:
            # 回退：从 chunks 重建该页
            pt = "".join(c.page_content + "\n" for c in chunks if c.metadata.get("page", 99) == pg)
            if pt.strip():
                pages_text.append(pt)
                debug["source_pages"].append(pg + 1)
                add_log(f"  \u26a0\ufe0f 第{pg+1}页用 chunks 重建（{len(pt)} 字符）")

    if not pages_text:
        add_log(f"  \u26a0\ufe0f 第1-3页均无有效文本")
        debug["reason"] = "PDF 未包含第1-2页内容"
        debug["status"] = "failed"
        return False, None, debug

    full_text = "\n".join(pages_text)
    add_log(f"  \U0001f4d1 合并第1-2页文本，共 {len(full_text)} 字符")

    # ---- 定义 Abstract 标题正则模式 ----
    header_pats = [
        r'Abstract[\s:.]+',
        r'ABSTRACT[\s:.]+',
        r'摘要[\s:.]+',
    ]
    abstract_start = -1
    matched_header = ""
    for pat in header_pats:
        m = re.search(pat, full_text)
        if m:
            abstract_start = m.end()
            matched_header = m.group().strip()
            add_log(f"  \u2705 匹配到摘要标题：{repr(matched_header)}")
            break

    if abstract_start == -1:
        add_log(f"  \u26a0\ufe0f 正则未匹配到 Abstract 标题")
        debug["reason"] = "未找到 Abstract / 摘要 标题"
        debug["status"] = "failed"
        return False, None, debug

    # ---- 定义结束标题模式（按优先顺序） ----
    end_pats = [
        (r'Keywords\s*[:\n\r]', 'Keywords'),
        (r'Index Terms\s*[:\n\r]', 'Index Terms'),
        (r'索引词\s*[:\n\r]', '索引词'),
        (r'Nomenclature\s*[:\n\r]', 'Nomenclature'),
        (r'(?<!\w)Introduction\s*[\n\r]', 'Introduction'),
        (r'1\s*\.?\s*Introduction', '1. Introduction'),
        (r'I\.?\s*Introduction', 'I. Introduction'),
        (r'1\s+引言', '1. 引言'),
        (r'引言\s*[\n\r]', '引言'),
        (r'关键词\s*[:\n\r]', '关键词'),
        (r'Key words\s*[:\n\r]', 'Key words'),
    ]
    abstract_end = -1
    matched_end = ""
    remaining = full_text[abstract_start:]
    for pat, label in end_pats:
        m = re.search(pat, remaining)
        if m:
            abstract_end = abstract_start + m.start()
            matched_end = label
            add_log(f"  \u2705 匹配到结束标题：{label}（偏移 {m.start()}）")
            break

    if abstract_end == -1:
        debug["reason"] = "未找到结束标题（如 Keywords / Introduction），截取 4000 字符"
        debug["status"] = "partial"
        abstract_end = min(abstract_start + 4000, len(full_text))
        matched_end = "(未找到具体结束标题)"
    else:
        debug["status"] = "ok"
    debug["end_marker"] = matched_end

    # ---- 提取摘要并清洗 ----
    raw_text = full_text[abstract_start:abstract_end].strip()
    if not raw_text:
        debug["reason"] = "摘要标题后无文本内容"
        debug["status"] = "failed"
        return False, None, debug
    add_log(f"  \U0001f4dd 提取原始文本 {len(raw_text)} 字符")

    # ---- 文本清洗 ----
    # a. 合并被 PDF 换行断开的单词
    clean = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', raw_text)
    # b. 单个换行转空格（保留段落双换行）
    clean = re.sub(r'(?<!\n)\n(?!\n)', ' ', clean)
    # c. 折叠多余空格
    clean = re.sub(r' {2,}', ' ', clean)
    # d. 去除头尾空白
    clean = clean.strip()
    # e. 去除末尾杂散标点
    clean = re.sub(r'[\s,;:+]+$', '', clean)

    if len(clean) < 20:
        debug["reason"] = "摘要内容过少（少于20字符）"
        debug["status"] = "failed"
        return False, None, debug

    debug["char_count"] = len(clean)
    debug["preview_300"] = clean[:300] + "..." if len(clean) > 300 else clean
    add_log(f"  \u2705 摘要提取完成：{len(clean)} 字符，结束于 {matched_end}")
    # 前100字符预览
    add_log(f"  \U0001f4dd 摘要开头：{clean[:100].strip()}...")

    return True, clean, debug

# ============================================================
# 5. 混合检索（改进：向量 + 关键词 + 页面加权）
# ============================================================

def hybrid_retrieve(vs, chunks, question, top_k=8):
    """
    1. 向量检索取 top_k*3 候选
    2. 重打分：向量相似度 + 关键词(+0.12/个) + 第1-2页(+0.2) + 标题匹配(+0.3)
    3. 按总分降序取 top_k
    """
    cand = vs.similarity_search_with_score(question, k=top_k * 3)

    stops = {"the","is","a","an","in","of","to","and","for","on","with",
             "what","how","why","does","do","are","can","this","that","it",
             "be","has","have","we","our","their","its","not","or","by","as","at"}
    words = question.lower().replace("?","").replace("\uff1f","").split()
    kws = [w for w in words if w not in stops and len(w) > 1]

    tags = ["abstract","introduction","conclusion","result","method",
            "summary","overview",
            "\u6458\u8981","\u5f15\u8a00","\u7ed3\u8bba","\u65b9\u6cd5",
            "\u7ed3\u679c","\u8ba8\u8bba","\u5de5\u4f5c"]

    scored = []
    for doc, dist in cand:
        base = 1.0 / (1.0 + dist)
        page = doc.metadata.get("page", 0)
        cl = doc.page_content.lower()
        bonus = 0.0
        for kw in kws:
            if kw in cl:
                bonus += 0.12
        if page <= 1:
            bonus += 0.2
        if any(t in cl for t in tags):
            bonus += 0.3
        scored.append((doc, dist, base + bonus, page))

    scored.sort(key=lambda x: x[2], reverse=True)
    return scored[:top_k]

# ============================================================
# 6. 调用 DeepSeek API 生成回答
# ============================================================
