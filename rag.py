"""
RAG 知识库模块 —— 基于 LangChain + FAISS + sentence-transformers
支持上传 PDF、Word（.docx/.doc）、Markdown（.md）、纯文本（.txt）
"""
import os
import sqlite3
import shutil
from pathlib import Path
from typing import List, Tuple


ALLOWED_EXTENSIONS = {'.pdf', '.docx', '.doc', '.md', '.markdown', '.txt'}
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
# 使用支持中文的多语言小模型，首次运行会自动下载（约 120 MB）
EMBEDDING_MODEL = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
# 余弦距离（向量归一化后 L2 即等价余弦）阈值：< 1.0 认为相关
RELEVANCE_THRESHOLD = 1.0


def get_recursive_text_splitter():
    """兼容 LangChain 新旧版本的文本分块器导入路径。"""
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except Exception:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    return RecursiveCharacterTextSplitter


class RAGKnowledgeBase:
    """管理文档向量库的生命周期：上传、检索、删除、持久化。"""

    def __init__(self, db_path: str = 'hai_learn.db', store_dir: str = 'rag_store'):
        self.db_path = db_path
        self.store_dir = store_dir
        self.files_dir = os.path.join(store_dir, 'files')
        self.index_path = os.path.join(store_dir, 'faiss_index')
        self.vectorstore = None
        self.embeddings = None
        self.is_available = False

        os.makedirs(self.files_dir, exist_ok=True)
        self._init_db()
        self._init_components()

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------

    def _init_db(self):
        """建立存储文档元信息的 rag_documents 表。"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS rag_documents (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                filename    TEXT NOT NULL,
                original_name TEXT NOT NULL,
                doc_type    TEXT NOT NULL,
                chunk_count INTEGER DEFAULT 0,
                file_size   INTEGER DEFAULT 0,
                uploaded_by INTEGER,
                created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        conn.close()

    def _init_components(self):
        """加载 HuggingFace 嵌入模型并尝试恢复已有向量库。"""
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            self.embeddings = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True},
            )
            self._load_vectorstore()
            self.is_available = True
            print('[RAG] 向量检索组件初始化成功')
        except Exception as e:
            print(f'[RAG] 初始化失败，RAG 功能不可用: {e}')
            self.is_available = False

    def _load_vectorstore(self):
        if os.path.exists(self.index_path):
            try:
                from langchain_community.vectorstores import FAISS
                self.vectorstore = FAISS.load_local(
                    self.index_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True,
                )
                print('[RAG] FAISS 向量库加载成功')
            except Exception as e:
                print(f'[RAG] 加载 FAISS 向量库失败，将重建: {e}')
                self.vectorstore = None

    def _save_vectorstore(self):
        if self.vectorstore is not None:
            self.vectorstore.save_local(self.index_path)

    # ------------------------------------------------------------------
    # 文档上传
    # ------------------------------------------------------------------

    def add_document(self, file_path: str, original_name: str,
                     user_id: int) -> Tuple[bool, str, int]:
        """
        解析并索引一份文档。
        返回 (success, message, doc_id)
        """
        if not self.is_available:
            return False, 'RAG 组件未就绪，请确认依赖已安装', 0

        ext = Path(original_name).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            return False, f'不支持的文件格式 {ext}，请上传 PDF / Word / Markdown / TXT', 0

        try:
            RecursiveCharacterTextSplitter = get_recursive_text_splitter()
            from langchain_community.vectorstores import FAISS

            docs = self._load_document(file_path, original_name)
            if not docs:
                return False, '文档内容为空或解析失败，请检查文件是否损坏', 0

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                separators=['\n\n', '\n', '。', '！', '？', ' ', ''],
            )
            chunks = splitter.split_documents(docs)
            if not chunks:
                return False, '文档分割失败', 0

            # 先写 DB 获取 doc_id
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            file_size = os.path.getsize(file_path)
            c.execute(
                '''INSERT INTO rag_documents
                   (filename, original_name, doc_type, chunk_count, file_size, uploaded_by)
                   VALUES (?, ?, ?, ?, ?, ?)''',
                (os.path.basename(file_path), original_name,
                 ext.lstrip('.'), len(chunks), file_size, user_id),
            )
            doc_id = c.lastrowid
            conn.commit()
            conn.close()

            # 给每个 chunk 打标签
            for chunk in chunks:
                chunk.metadata['doc_id'] = doc_id
                chunk.metadata['source_name'] = original_name

            # 更新向量库
            if self.vectorstore is None:
                self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
            else:
                self.vectorstore.add_documents(chunks)
            self._save_vectorstore()

            return True, f'《{original_name}》添加成功，共 {len(chunks)} 个片段', doc_id

        except Exception as e:
            return False, f'处理文档失败: {e}', 0

    def _load_document(self, file_path: str, original_name: str) -> list:
        """按扩展名选择合适的 LangChain Loader。"""
        ext = Path(original_name).suffix.lower()
        try:
            if ext == '.pdf':
                from langchain_community.document_loaders import PyPDFLoader
                return PyPDFLoader(file_path).load()
            elif ext in ('.docx', '.doc'):
                from langchain_community.document_loaders import Docx2txtLoader
                return Docx2txtLoader(file_path).load()
            elif ext in ('.md', '.markdown', '.txt'):
                from langchain_community.document_loaders import TextLoader
                return TextLoader(file_path, encoding='utf-8').load()
        except Exception as e:
            print(f'[RAG] 加载文档失败: {e}')
        return []

    # ------------------------------------------------------------------
    # 检索
    # ------------------------------------------------------------------

    def query(self, question: str, top_k: int = 4) -> Tuple[list, list]:
        """
        对问题进行语义检索。
        返回 (相关 Document 列表, 去重后的来源文件名列表)
        若知识库为空或无相关内容，两者均为空列表。
        """
        if not self.is_available or self.vectorstore is None:
            return [], []
        try:
            results = self.vectorstore.similarity_search_with_score(question, k=top_k)
            # 过滤低相关度结果（归一化向量的 L2 距离 < 1.0 表示余弦相似度 > 0.5）
            relevant = [(doc, score) for doc, score in results if score < RELEVANCE_THRESHOLD]
            if not relevant:
                return [], []

            docs = [doc for doc, _ in relevant]
            seen: set = set()
            sources: list = []
            for doc in docs:
                name = doc.metadata.get('source_name', '未知文件')
                if name not in seen:
                    seen.add(name)
                    sources.append(name)
            return docs, sources
        except Exception as e:
            print(f'[RAG] 检索失败: {e}')
            return [], []

    # ------------------------------------------------------------------
    # 文档管理
    # ------------------------------------------------------------------

    def list_documents(self) -> List[dict]:
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''
            SELECT id, original_name, doc_type, chunk_count, file_size, created_at
            FROM rag_documents ORDER BY created_at DESC
        ''')
        rows = c.fetchall()
        conn.close()
        return [
            {'id': r[0], 'name': r[1], 'type': r[2],
             'chunks': r[3], 'size': r[4], 'created_at': r[5]}
            for r in rows
        ]

    def delete_document(self, doc_id: int) -> Tuple[bool, str]:
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('SELECT filename, original_name FROM rag_documents WHERE id = ?', (doc_id,))
        row = c.fetchone()
        if not row:
            conn.close()
            return False, '文档不存在'
        filename, original_name = row
        c.execute('DELETE FROM rag_documents WHERE id = ?', (doc_id,))
        conn.commit()
        conn.close()

        # 删除物理文件
        file_path = os.path.join(self.files_dir, filename)
        if os.path.exists(file_path):
            os.remove(file_path)

        # 重建向量库（从剩余文件重索引）
        self._rebuild_vectorstore()
        return True, f'《{original_name}》已从知识库删除'

    def _rebuild_vectorstore(self):
        """删除文档后重建整个 FAISS 索引。"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('SELECT id, filename, original_name, doc_type FROM rag_documents')
        all_docs = c.fetchall()
        conn.close()

        if not all_docs:
            self.vectorstore = None
            if os.path.exists(self.index_path):
                shutil.rmtree(self.index_path, ignore_errors=True)
            return

        RecursiveCharacterTextSplitter = get_recursive_text_splitter()
        from langchain_community.vectorstores import FAISS

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=['\n\n', '\n', '。', '！', '？', ' ', ''],
        )
        all_chunks = []
        for doc_id, filename, original_name, _ in all_docs:
            file_path = os.path.join(self.files_dir, filename)
            if not os.path.exists(file_path):
                continue
            docs = self._load_document(file_path, original_name)
            chunks = splitter.split_documents(docs)
            for chunk in chunks:
                chunk.metadata['doc_id'] = doc_id
                chunk.metadata['source_name'] = original_name
            all_chunks.extend(chunks)

        if all_chunks:
            self.vectorstore = FAISS.from_documents(all_chunks, self.embeddings)
            self._save_vectorstore()
        else:
            self.vectorstore = None
            if os.path.exists(self.index_path):
                shutil.rmtree(self.index_path, ignore_errors=True)
