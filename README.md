# PaperAgent - PDF 论文智能问答系统

基于 DeepSeek API + 本地 Embedding + FAISS 的 PDF 智能问答系统，支持摘要提取、混合检索和对话记忆。

## 项目架构

```
PaperAgent/
├── app.py              # Streamlit UI 入口（界面、对话管理）
├── paper_agent.py      # 核心逻辑（PDF 处理、检索、摘要提取）
├── deepseek_api.py     # DeepSeek API 封装
├── requirements.txt
└── .env               # API Key 配置
```

## 数据流

```
上传 PDF
  │
  ▼
PyPDFLoader → 按页读取
  │
  ▼
RecursiveCharacterTextSplitter → 按页切分（不混页）
  │
  ▼
sentence-transformers → 本地向量化
  │
  ▼
FAISS → 本地向量索引
  │
  ▼
用户提问
  │
  ├─ 检测是否问摘要 → 第1-3页正则提取
  │
  ├─ 混合检索（向量 + 关键词 + 页面加权）
  │
  ▼
DeepSeek Chat API → 生成回答
  │
  ▼
显示回答 + 对话历史 + 调试面板
```

## 技术选型

| 组件 | 选择 | 理由 |
|---|---|---|
| Embedding | sentence-transformers（本地） | 数据不出本机，零成本 |
| 向量库 | FAISS（CPU） | 本地运行，无需 GPU |
| LLM | DeepSeek Chat API | 价格低，兼容 OpenAI 格式 |  
| UI 框架 | Streamlit | 快速构建，社区生态好 |
| PDF 解析 | PyPDFLoader | LangChain 生态，自带页码元数据 |

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置 API Key
#   复制 .env.example 为 .env，填入 DeepSeek API Key

# 3. 启动
streamlit run app.py
```

## 功能清单

- **按页切分** — 每页独立切分，不同页不混入同一文本块
- **摘要优先提取** — 问 Abstract/摘要时，从第1-3页原始文本提取
- **混合检索** — 向量 + 关键词匹配 + 第1-2页加权 + 章节标题加权
- **对话记忆** — 可追问前一轮对话的内容
- **调试面板** — 查看检索参数、命中详情、摘要提取状态

## 功能演示

本项目支持上传论文 PDF，并基于文档内容进行结构化问答与检索调试。

### 示例流程

1. 上传一篇论文 PDF，系统会自动解析文档并展示基础统计信息：
   - 页数
   - 字符数
   - 文本块数量

2. 输入问题：

   > 这篇论文的 Abstract 是什么？

   系统会从论文中定位并提取摘要内容。

3. 继续追问：

   > 主要方法是什么？

   系统会结合当前对话上下文与检索结果，回答论文的核心方法。

4. 打开调试面板，可以查看：
   - 向量检索结果
   - 关键词检索结果
   - Hybrid Search 加权得分
   - 最终召回的文本块


## 费用估算

DeepSeek API 定价：输入 ¥0.5/1M token，输出 ¥2/1M token
每次问答约 ¥0.001-0.005，极低成本。

## 挑战与优化记录

1. **摘要提取不完整** — 从 chunk 检索改为原始页面文本 + 正则匹配
2. **f-string 兼容性问题** — Python 3.11 不支持 f-string 表达式中使用相同引号
3. **正则转义错误** — JS 字符串中双反斜杠在 Python raw string 中的层级问题
4. **代码模块化** — 从单文件 500+ 行拆分为三层架构
