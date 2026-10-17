"""
=============================================================
PART 1: Basic RAG System (ระบบ RAG พื้นฐาน)
=============================================================

ตามแนวทางของสไลด์หน้า 22-25 (Generic RAG)
องค์ประกอบหลัก:
  1. Document Loading & Fixed-size Chunking
  2. Embedding ด้วย SentenceTransformer
  3. Vector Store แบบ In-memory (Numpy)
  4. Dense Retrieval ด้วย Cosine Similarity
  5. Answer Generation ด้วย LLM (Google Gemini / OpenAI-compatible)

ไม่มี: Contextual Embedding, BM25, Hybrid Search, Reranking
=============================================================
"""

import os
import re
import numpy as np
from typing import List, Tuple
from sentence_transformers import SentenceTransformer

# นำเข้าเอกสารตัวอย่าง
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from sample_documents import SAMPLE_DOCUMENTS


# ============================================================
# STEP 1: CHUNKING - แบ่งเอกสารเป็น chunk ขนาดคงที่
# ============================================================

def fixed_size_chunking(text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
    """
    แบ่งข้อความออกเป็น chunk ขนาดคงที่ (Fixed-size Chunking)
    
    วิธีนี้ตรงกับ Generic RAG ในสไลด์:
    - แบ่งตามจำนวนตัวอักษร (character-based)
    - มี overlap เล็กน้อยเพื่อไม่ให้ข้อมูลขาดหาย
    
    ข้อจำกัด:
    - อาจตัดกลางประโยคหรือกลางบริบท
    - แต่ละ chunk ไม่รู้ว่าตัวเองมาจากส่วนไหนของเอกสาร
    """
    text = text.strip()
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start = end - overlap  # เลื่อน start โดยหักลบ overlap
    return chunks


def load_and_chunk_documents(documents: List[str], chunk_size: int = 300, overlap: int = 50) -> List[dict]:
    """โหลดเอกสารและแบ่ง chunk พร้อมเก็บ metadata เบื้องต้น"""
    all_chunks = []
    for doc_id, doc in enumerate(documents):
        chunks = fixed_size_chunking(doc, chunk_size, overlap)
        for chunk_id, chunk_text in enumerate(chunks):
            all_chunks.append({
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "text": chunk_text,
                # ไม่มี context เพิ่มเติม - นี่คือข้อจำกัดของ Basic RAG
            })
    return all_chunks


# ============================================================
# STEP 2: EMBEDDING - แปลง chunk เป็น vector
# ============================================================

def build_vector_index(chunks: List[dict], model: SentenceTransformer) -> np.ndarray:
    """
    สร้าง dense vector สำหรับแต่ละ chunk
    
    ตรงกับสไลด์ "Embedding model ทำหน้าที่แปลงข้อความใน document/chunk ให้เป็น vector"
    ใช้เฉพาะ text ของ chunk โดยตรง (ไม่มี context นำหน้า)
    """
    texts = [chunk["text"] for chunk in chunks]
    print(f"  กำลัง Embed {len(texts)} chunks...")
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    return embeddings  # shape: (num_chunks, embedding_dim)


# ============================================================
# STEP 3: RETRIEVAL - ค้นหา chunk ที่เกี่ยวข้อง
# ============================================================

def cosine_similarity_search(
    query: str,
    chunks: List[dict],
    embeddings: np.ndarray,
    model: SentenceTransformer,
    top_k: int = 3
) -> List[Tuple[dict, float]]:
    """
    ค้นหาด้วย Cosine Similarity (Dense Retrieval เท่านั้น)
    
    ตรงกับสไลด์: "Retriever เปรียบเทียบความใกล้กัน vector representation 
    ของ query กับ vector representation ของ document/chunk ที่เก็บไว้ 
    โดยใช้ similarity metrics เช่น cosine similarity"
    
    ไม่มี: BM25, Hybrid Search
    """
    # Embed query
    query_vec = model.encode([query], normalize_embeddings=True)[0]  # shape: (dim,)
    
    # คำนวณ cosine similarity ระหว่าง query กับทุก chunk
    # เนื่องจาก normalize แล้ว cosine similarity = dot product
    scores = np.dot(embeddings, query_vec)  # shape: (num_chunks,)
    
    # เรียงลำดับตาม score สูงสุด
    top_indices = np.argsort(scores)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        results.append((chunks[idx], float(scores[idx])))
    
    return results


# ============================================================
# STEP 4: ANSWER GENERATION - สร้างคำตอบด้วย LLM
# ============================================================

def build_prompt(query: str, retrieved_chunks: List[Tuple[dict, float]]) -> str:
    """สร้าง prompt สำหรับส่งให้ LLM"""
    context_parts = []
    for i, (chunk, score) in enumerate(retrieved_chunks):
        context_parts.append(
            f"[เอกสาร {i+1} | doc_id={chunk['doc_id']} | score={score:.4f}]\n{chunk['text']}"
        )
    context = "\n\n".join(context_parts)
    
    prompt = f"""คุณเป็นผู้ช่วยตอบคำถามจากเอกสารภายในองค์กร กรุณาตอบคำถามโดยอิงจากเนื้อหาที่ให้มาเท่านั้น

เนื้อหาที่ค้นหาได้:
{context}

คำถาม: {query}

คำตอบ (ตอบเป็นภาษาไทย อ้างอิงจากเอกสารที่ให้มา หากไม่มีข้อมูลให้ตอบว่าไม่พบข้อมูลในเอกสาร):"""
    return prompt


def generate_answer_with_llm(prompt: str) -> str:
    """
    ส่ง prompt ให้ LLM สร้างคำตอบ
    
    รองรับ Google Gemini API (ตัวอย่างนี้ใช้ google-generativeai)
    หากไม่มี API key จะใช้ mock response
    """
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    
    if api_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            # ใช้ LLM ตัวเดียวกับ Part 2 (gemini-1.5-flash ถูกปิดให้บริการแล้ว)
            model = genai.GenerativeModel(os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"))
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"[LLM Error: {e}]\n\nPrompt ที่จะส่ง:\n{prompt}"
    else:
        # Mock: แสดง prompt ที่จะส่ง (สำหรับกรณีไม่มี API key)
        return f"""[Mock Response - ไม่มี GOOGLE_API_KEY]

Prompt ที่จะส่งให้ LLM:
{'='*60}
{prompt}
{'='*60}

หากมี API Key จะได้คำตอบจาก LLM จริงๆ"""


# ============================================================
# STEP 5: MAIN RAG PIPELINE
# ============================================================

class BasicRAG:
    """
    ระบบ RAG พื้นฐาน (Generic RAG)
    ตามแนวทางสไลด์หน้า 22-25
    
    Pipeline:
    Document → Fixed-size Chunking → Dense Embedding → 
    Cosine Similarity Search → LLM Generation
    """
    
    def __init__(self, chunk_size: int = 300, overlap: int = 50, top_k: int = 3):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.top_k = top_k
        self.chunks: List[dict] = []
        self.embeddings: np.ndarray = None
        
        print("กำลังโหลด Embedding Model...")
        # ใช้ multilingual model รองรับภาษาไทย
        self.embed_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        print(f"โหลด Model สำเร็จ: {self.embed_model.get_sentence_embedding_dimension()} dimensions")
    
    def index_documents(self, documents: List[str]):
        """
        Indexing Phase (สไลด์: Area สีม่วง - การนำเข้าเอกสาร)
        1. โหลดและแบ่ง chunk
        2. สร้าง embedding vector
        """
        print("\n" + "="*60)
        print("INDEXING PHASE")
        print("="*60)
        
        # Step 1: Chunking
        print(f"\n[1] Fixed-size Chunking (chunk_size={self.chunk_size}, overlap={self.overlap})")
        self.chunks = load_and_chunk_documents(documents, self.chunk_size, self.overlap)
        print(f"  จำนวนเอกสาร: {len(documents)}")
        print(f"  จำนวน chunk ทั้งหมด: {len(self.chunks)}")
        
        # Step 2: Embedding
        print(f"\n[2] Dense Embedding (No Contextual Enhancement)")
        self.embeddings = build_vector_index(self.chunks, self.embed_model)
        print(f"  Embedding shape: {self.embeddings.shape}")
        
        print("\nIndexing เสร็จสิ้น!")
    
    def retrieve(self, question: str) -> List[Tuple[dict, float]]:
        """Retrieval อย่างเดียว (ไม่เรียก LLM) ใช้ใน evaluate.py"""
        return cosine_similarity_search(
            question, self.chunks, self.embeddings, self.embed_model, self.top_k
        )

    def query(self, question: str, verbose: bool = True) -> str:
        """
        Query Phase (สไลด์: Area สีเขียว - การค้นเอกสาร วิเคราะห์ และสรุปคำตอบ)
        1. Dense Retrieval ด้วย Cosine Similarity
        2. สร้างคำตอบด้วย LLM
        """
        if self.embeddings is None:
            raise ValueError("กรุณา index เอกสารก่อน ด้วย index_documents()")
        
        if verbose:
            print("\n" + "="*60)
            print(f"QUERY: {question}")
            print("="*60)
        
        # Step 3: Retrieval - Dense Search only
        if verbose:
            print(f"\n[3] Dense Retrieval - Cosine Similarity Search (top_k={self.top_k})")
        
        retrieved = cosine_similarity_search(
            question, self.chunks, self.embeddings, self.embed_model, self.top_k
        )
        
        if verbose:
            print(f"  ผลลัพธ์ที่ค้นหาได้:")
            for i, (chunk, score) in enumerate(retrieved):
                print(f"  [{i+1}] score={score:.4f} | doc_id={chunk['doc_id']}")
                print(f"       text: {chunk['text'][:100]}...")
        
        # ไม่มี Reranking ใน Basic RAG
        
        # Step 4: Build Prompt & Generate
        if verbose:
            print(f"\n[4] Answer Generation (No Reranking)")
        
        prompt = build_prompt(question, retrieved)
        answer = generate_answer_with_llm(prompt)
        
        return answer


# ============================================================
# DEMO
# ============================================================

if __name__ == "__main__":
    print("="*60)
    print("PART 1: Basic RAG System (Generic RAG)")
    print("ตามแนวทางสไลด์หน้า 22-25")
    print("="*60)
    
    # สร้างระบบ
    rag = BasicRAG(chunk_size=300, overlap=50, top_k=3)
    
    # Index เอกสาร
    rag.index_documents(SAMPLE_DOCUMENTS)
    
    # ทดสอบด้วยคำถามตัวอย่าง
    test_questions = [
        "การเดินทางไปต่างประเทศ เบิกค่าใช้จ่ายรวมแล้วไม่เกินกี่บาท",
        "พนักงานมีสิทธิ์ลาป่วยได้กี่วันต่อปี",
        "ค่าที่พักต่างประเทศเบิกได้เท่าไหร่",
    ]
    
    for q in test_questions:
        print("\n" + "~"*60)
        answer = rag.query(q, verbose=True)
        print(f"\n[คำตอบจาก LLM]:\n{answer}")
        print("~"*60)
