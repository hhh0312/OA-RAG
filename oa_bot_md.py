#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于 Markdown 知识库的 OA 智能客服
====================================

使用 Markdown + 表格提取 (不使用 VLM)
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

from dotenv import load_dotenv
load_dotenv()

from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from markdown_loader import MarkdownLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# 导入智能切分模块
try:
    from smart_splitter import smart_split_documents
    print("✅ 智能切分模块导入成功")
except ImportError:
    print("⚠️  智能切分模块导入失败，将使用默认切分")
    smart_split_documents = None

# 导入 Rerank 模块
try:
    from rerank_model import get_rerank_retriever
    print("✅ Rerank 模块导入成功")
except ImportError:
    print("⚠️  Rerank 模块导入失败")
    get_rerank_retriever = None

# ============================================================================
# 配置
# ============================================================================

# Markdown 知识库路径
MARKDOWN_DIR = "./OA知识库_md"

# ChromaDB 路径
CHROMA_DB_DIR = "./chroma_db_markdown"

# API 配置
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_API_BASE = os.environ.get("OPENAI_API_BASE")

if not OPENAI_API_KEY:
    print("❌ 未设置 OPENAI_API_KEY")
    sys.exit(1)

if OPENAI_API_BASE:
    print(f"✅ 使用自定义 API: {OPENAI_API_BASE}")

# ============================================================================
# 文档加载
# ============================================================================

def load_markdown_documents():
    """加载 Markdown 文档"""
    print("\n📚 加载 Markdown 知识库...")
    print(f"   知识库路径: {MARKDOWN_DIR}")
    
    # 创建 Markdown 加载器（不使用 VLM）
    loader = MarkdownLoader(MARKDOWN_DIR, use_vlm=False)
    
    # 加载所有文档
    documents = loader.load_documents()
    
    print(f"\n✅ 文档加载完成")
    print(f"   - 总文档数: {len(documents)}")
    
    # 统计
    type_counts = {}
    for doc in documents:
        doc_type = doc.metadata.get("type", "unknown")
        type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
    
    print("\n📊 文档类型统计:")
    for doc_type, count in sorted(type_counts.items()):
        print(f"   - {doc_type}: {count} 个")
    
    return documents

# ============================================================================
# 向量库创建/加载
# ============================================================================

def create_or_load_vectorstore(documents):
    """创建或加载向量库"""
    print("\n🗄️  向量数据库初始化...")
    
    # 选择 Embedding 模型
    # 使用硅基流动支持的BGE模型
    embedding_model = "BAAI/bge-large-zh-v1.5"
    print(f"📊 使用 Embedding 模型: {embedding_model}")
    embeddings = OpenAIEmbeddings(
        model=embedding_model,
        openai_api_base=OPENAI_API_BASE if OPENAI_API_BASE else None
    )
    
    # 检查是否存在向量库
    if os.path.exists(CHROMA_DB_DIR):
        print("✅ 检测到现有向量库，正在加载...")
        vectorstore = Chroma(
            persist_directory=CHROMA_DB_DIR,
            embedding_function=embeddings
        )
        print(f"✅ 向量库加载完成，包含 {vectorstore._collection.count()} 个文档片段")
    else:
        print("📦 未检测到现有向量库，开始创建新库...")
        
        # 智能切分 - 适配BGE模型512 token限制
        if smart_split_documents:
            print("\n🔄 正在进行智能文本切分...")
            split_docs = smart_split_documents(
                documents, 
                chunk_size=200,  # 减小以适应512 token限制
                chunk_overlap=30
            )
            print(f"✅ 智能切分完成，共 {len(split_docs)} 个片段")
        else:
            split_docs = documents
            
        # 二次检查：强制切分超长文档
        print("🔍 检查超长文档片段...")
        final_docs = []
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=200,  # 减小以适应512 token限制
            chunk_overlap=30,
            length_function=len,
        )
        
        for doc in split_docs:
            if len(doc.page_content) > 250:  # 降低阈值
                sub_docs = text_splitter.split_documents([doc])
                final_docs.extend(sub_docs)
            else:
                final_docs.append(doc)
        
        split_docs = final_docs
        print(f"✅ 最终文档片段数: {len(split_docs)}")
        
        # 调试：打印前几个文档的长度
        print("🔍 调试：前 5 个文档的长度:")
        for i, doc in enumerate(split_docs[:5]):
            print(f"   Doc {i}: {len(doc.page_content)} chars")
        
        # 创建向量库
        print("\n⏳ 正在创建向量索引...")
        
        # 分批处理（避免批处理大小限制）
        batch_size = 4  # 进一步减小批处理大小
        if len(split_docs) > batch_size:
            print(f"💡 文档数量较多，将分批处理（每批 {batch_size} 个文档）")
            print(f"   总共 {len(split_docs)} 个片段，需要 {(len(split_docs) + batch_size - 1) // batch_size} 批\n")
            
            # 第一批创建向量库
            first_batch = split_docs[:batch_size]
            print(f"📊 处理第 1 批（{len(first_batch)} 个文档）...")
            vectorstore = Chroma.from_documents(
                documents=first_batch,
                embedding=embeddings,
                persist_directory=CHROMA_DB_DIR
            )
            print("   ✓ 第 1 批完成")
            
            # 后续批次添加到向量库
            for i in range(batch_size, len(split_docs), batch_size):
                batch_num = (i // batch_size) + 1
                batch = split_docs[i:i+batch_size]
                print(f"📊 处理第 {batch_num} 批（{len(batch)} 个文档）...")
                vectorstore.add_documents(batch)
                print(f"   ✓ 第 {batch_num} 批完成")
            
            print(f"✅ 所有 {len(split_docs)} 个文档已分批处理完成")
        else:
            vectorstore = Chroma.from_documents(
                documents=split_docs,
                embedding=embeddings,
                persist_directory=CHROMA_DB_DIR
            )
        
        print(f"✅ 向量库创建并保存至: {CHROMA_DB_DIR}")
    
    return vectorstore, documents

# ============================================================================
# QA 链创建
# ============================================================================

def create_qa_chain(vectorstore, all_documents):
    """创建问答链（支持多轮对话）"""
    print("\n🤖 正在构建问答链（多轮对话模式）...")
    
    # 选择 LLM 模型
    if OPENAI_API_BASE and "moonshot" in OPENAI_API_BASE.lower():
        model_name = "moonshot-v1-32k"
        print(f"🌙 使用 Moonshot AI 官方模型: {model_name}")
    elif OPENAI_API_BASE and "siliconflow" in OPENAI_API_BASE.lower():
        # 双模型配置（硅基流动）
        model_name = "Qwen/Qwen2.5-7B-Instruct"
        metadata_model = "Moonshot/moonshot-v1-32k"
        print(f"🤖 主LLM模型: {model_name}")
        print(f"🌙 元数据过滤模型: {metadata_model}")
    else:
        model_name = "gpt-3.5-turbo"
    
    # 创建主LLM（启用流式输出）
    llm = ChatOpenAI(
        model=model_name,
        temperature=0.7,
        max_tokens=4096,
        openai_api_base=OPENAI_API_BASE if OPENAI_API_BASE else None,
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()]
    )
    
    # 创建元数据过滤LLM（如果使用硅基流动）
    if OPENAI_API_BASE and "siliconflow" in OPENAI_API_BASE.lower():
        metadata_llm = ChatOpenAI(
            model=metadata_model,
            temperature=0.3,
            max_tokens=1024,
            openai_api_base=OPENAI_API_BASE,
        )
        print(f"✅ 双模型配置完成")
    
    # 创建检索器 - 混合检索（Vector + BM25）
    print("\n🔍 配置检索策略...")
    
    # 基础向量检索器（增加检索数量以提高召回率）
    vector_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 8,          # 增加召回数量（从 5 增至 8）
            "fetch_k": 30,   # 提前获取更多候选供 Rerank 使用（从 20 增至 30）
            "lambda_mult": 0.7
        }
    )
    print("  ✅ 向量检索器配置完成 (MMR, k=8, fetch_k=30)")
    
    # 构建BM25检索器（配置中文分词）
    if all_documents and len(all_documents) > 0:
        try:
            # 导入 jieba 分词器
            import jieba
            
            def jieba_tokenizer(text):
                """使用 jieba 对中文文本进行分词"""
                return jieba.lcut(text)
            
            print(f"  🔄 正在构建BM25关键词检索器（文档数: {len(all_documents)}，使用 jieba 分词）...")
            bm25_retriever = BM25Retriever.from_documents(
                all_documents,
                preprocess_func=jieba_tokenizer  # 关键：使用中文分词器
            )
            bm25_retriever.k = 3
            
            # 创建混合检索器
            ensemble_retriever = EnsembleRetriever(
                retrievers=[vector_retriever, bm25_retriever],
                weights=[0.4, 0.6]
            )
            retriever = ensemble_retriever
            print("  ✅ 混合检索器构建完成 (Vector 40% + BM25 60%，已启用中文分词)")
        except Exception as e:
            print(f"  ⚠️  BM25构建失败: {e}")
            print("  ⚠️  降级使用纯向量检索")
            retriever = vector_retriever
    else:
        retriever = vector_retriever
    
    # 添加 Rerank 重排序（调整阈值避免过度过滤）
    if get_rerank_retriever:
        retriever = get_rerank_retriever(
            base_retriever=retriever,
            top_n=5,
            model_name="BAAI/bge-reranker-v2-m3",
            score_threshold=0.1  # 降低阈值，从 0.3 降至 0.1，避免过度过滤
        )
    else:
        print("  ⚠️  未启用 Rerank (模块未导入)")
    
    # 自定义 Prompt Template - 注入系统背景信息（多轮对话版本，标准化输出格式）
    system_prompt = """你是一个专业、严谨且乐于助人的企业知识库助手，服务于集团自主研发的 OA 系统。

【系统背景】
• 产品名称：集团自主研发 OA 系统
• 研发历程：2018 年启动研发，历经一年精心打磨，于 2019 年推出 2.0 版本
• 核心价值：完全由集团自主研发，深度融入核心业务流程和管理理念
• 设计目标：简化工作流程、提升团队协作效率，为员工提供更智能、更便捷的办公体验

【你的角色定位】
• 专业：精通 OA 系统的功能模块、流程配置和常见使用场景
• 严谨：所有回答必须基于知识库中的【上下文信息】，严禁主观臆测
• 友好：用清晰、结构化的方式帮助用户快速找到操作路径和核心要点

【回答原则】
1. **基于事实**：
   - 只能根据下面提供的【参考文档内容】回答问题。
   - 严禁引用外部常识或编造知识库中不存在的规则、入口、字段名称等。
   - 当文档中出现 "必须"、"不得"、"不可以" 等强约束用语时，回答不得与这些约束相反，也不得弱化为仅仅是"建议"。

2. **结构化输出**：
   - 禁止输出一整段没有结构的大段文字。
   - 必须使用小标题、加粗和列表来组织内容，便于快速浏览。
   - 尤其是流程 / 操作类问题，必须拆成多个步骤或环节。

3. **关键要素提取（如上下文中有）**：
   在描述流程或操作时，优先提取并明确说明以下要素：
   - **操作入口**：在 OA 系统中的具体进入路径（例如：OA 主菜单 / 审批 / 申请审批 / 合同初审）。
   - **主要任务/动作**：用户在该环节需要完成的关键操作（例如：填写字段、上传附件、选择已有合同等）。
   - **注意事项/前提条件**：例如是否需要先完成某个前置步骤、是否有时间限制、必填字段要求等。

4. **输出格式要求**：
   - 若问题是**流程 / 操作类**，优先使用如下结构（可按实际内容增减小节）：

     ### [核心结论/步骤名称]
     *用 1–2 句话概括本次回答的核心结论，例如：线下合同审批流程主要包含三个步骤。*

     **[步骤一/具体环节名称]**
     * **操作入口**：...[如果上下文有则写明，否则省略]
     * **主要任务**：...
     * **说明/注意**：...

     **[步骤二/具体环节名称]**
     * ...（如有更多步骤，按此格式继续）

     ---
     **总结**：用简短的箭头流程或一句话总结整体流程，例如：初审 → 定稿 → 用印。

   - 若问题是**定义 / 说明类**（例如“什么是 TDS 凭证？”），可以简化为：
     - **核心结论**：用 1–3 句给出定义或结论。
     - **适用场景 / 限制条件**：简要说明在什么情况下适用，有何限制。

5. **信息来源标注（必填）**：
   - 在回答的最后必须单独一段标注【信息来源】，列出本次回答使用到的文档。
   - 建议使用如下格式，与文档元数据保持一致：
     - 8 合同.md（text）
     - 2 审批.md（table）
   - 如果同时参考了多个文档，请全部列出。

6. **处理未知 / 信息不足**：
   - 如果【参考文档内容】中没有足够信息回答用户问题，必须遵循：
     1) 以“抱歉，根据当前提供的知识库上下文，无法确定...”开头。
     2) 简要说明知识库中有哪些相关信息、缺少了什么关键信息（例如：文档只提到流程名称，但未列出具体审批人岗位）。
     3) 提供 2–3 条合理建议，例如：
        - 建议联系直属主管或对应职能部门（如人事、财务、法务）。
        - 建议在 OA 系统中尝试发起相关流程，查看系统实际提示。
        - 建议查阅最新的公司制度或公告通知。
   - 严禁在信息不足时根据常识或外部经验自行补全具体规则、数值或岗位名称。

7. **对话与表达风格**：
   - 在多轮对话中，应正确理解指代词（如“它”、“这一步”、“刚才说的流程”等），但仍需严格以文档内容为准。
   - 回答时**不要逐字重复或改写用户的问题**，直接给出结论和结构化说明即可。
   - 可以用简短的礼貌用语，但避免冗长客套，重点放在清晰、可执行的操作说明上。

【参考文档内容】
{context}

【用户问题】
{question}

【你的回答】"""

    PROMPT = PromptTemplate(
        template=system_prompt,
        input_variables=["context", "question"]
    )
    
    # 创建对话记忆
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )
    
    # 自定义问题改写 Prompt（改进多轮对话的上下文保留）
    condense_template = """基于以下对话历史和后续问题，将后续问题改写为一个独立的问题。

重要规则：
1. 必须保留对话历史中的主题词（如"招聘"、"合同"、"审批"等核心名词）
2. 如果后续问题使用了指代词（"它"、"这个"、"那"、"这些"），必须用具体名词替换
3. 如果后续问题询问"第一步"、"字段"等通用概念，必须加上主题限定（如"招聘申请的第一步"）
4. 改写后的问题应该是完整的、可以独立理解的

对话历史：
{chat_history}

后续问题: {question}

改写后的独立问题:"""
    
    condense_question_prompt = PromptTemplate(
        template=condense_template,
        input_variables=["chat_history", "question"]
    )
    
    # 构建对话式 QA 链（添加自定义的 condense_question_prompt）
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": PROMPT},
        condense_question_prompt=condense_question_prompt,
        verbose=False  # 开启调试模式，可以看到问题改写过程
    )
    
    print("✅ 多轮对话问答链构建完成")
    return qa_chain

# ============================================================================
# 主程序
# ============================================================================

def main():
    """主程序"""
    print("\n" + "="*80)
    print("🎉 欢迎使用集团 OA 系统智能客服 (Markdown 增强版)!")
    print("="*80)
    
    # 1. 加载文档
    documents = load_markdown_documents()
    
    # 2. 创建向量库
    vectorstore, all_documents = create_or_load_vectorstore(documents)
    
    # 3. 创建问答链
    qa_chain = create_qa_chain(vectorstore, all_documents)
    
    print("\n" + "="*80)
    print("💡 提示:")
    print("  • 输入您的问题，按 Enter 键提交")
    print("  • 输入 'quit', 'exit', '退出' 可结束对话")
    print("  • 输入 'test' 运行测试用例")
    print("  • 输入 'clear' 清空对话历史")
    print('  • 💬 支持多轮对话：可以追问"它"、"这个"等指代词')
    print('  • ⚡ 流式输出：回答会实时逐字显示')
    print("="*80)
    
    # 4. 交互循环
    while True:
        print("\n👤 您的问题: ", end="")
        user_input = input().strip()
        
        if not user_input:
            print("⚠️  请输入问题")
            continue
        
        if user_input.lower() in ['quit', 'exit', '退出']:
            print("\n👋 感谢使用，再见!")
            break
        
        if user_input.lower() == 'clear':
            qa_chain.memory.clear()
            print("\n🗑️  对话历史已清空")
            continue
        
        if user_input.lower() == 'test':
            test_questions = [
                "招聘申请时，招聘部门和Base字段有什么要求？",
                "合同初审需要填写哪些字段？",
                "这个系统有哪些功能模块？"
            ]
            print("\n🧪 运行测试用例...")
            qa_chain.memory.clear()  # 清空历史避免干扰
            for q in test_questions:
                print(f"\n📝 测试问题: {q}")
                print("🤖 回答: ")
                
                # 流式输出由callback自动处理
                qa_chain.invoke({"question": q})
                
                print("\n" + "-" * 80)
            qa_chain.memory.clear()  # 测试完成后清空
            continue
        
        # 处理问题
        try:
            print("\n⏳ 正在思考...")
            print(f"\n🤖 客服回答:")
            print("-" * 80)
            
            # 调用QA链（流式输出由callback自动处理）
            result = qa_chain.invoke({"question": user_input})
            
            print("\n" + "-" * 80)
            
            # 显示来源
            if result.get('source_documents'):
                print(f"\n📚 参考来源:")
                for idx, doc in enumerate(result['source_documents'][:3], 1):
                    source = doc.metadata.get('file_name', 'Unknown')
                    doc_type = doc.metadata.get('type', 'unknown')
                    print(f"  {idx}. {source} ({doc_type})")
        
        except Exception as e:
            print(f"\n❌ 处理失败: {e}")

if __name__ == "__main__":
    main()
