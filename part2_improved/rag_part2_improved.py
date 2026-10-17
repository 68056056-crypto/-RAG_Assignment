"""
=============================================================
PART 2: Improved RAG System (ระบบ RAG ที่ปรับปรุงแล้ว)
=============================================================

ปรับปรุงจาก Part 1 (Basic RAG) โดย "ไม่เปลี่ยน LLM" ที่ใช้ generate คำตอบ
(ทั้งสองส่วนเรียก generate_answer_with_llm() ตัวเดียวกัน)

ส่วนที่ปรับปรุง:
1. CHUNKING  : Structure-aware Chunking
               - แบ่งตามโครงสร้างเอกสาร (หัวข้อ "ข้อ N" / "N." + bullet "-")
               - ไม่ตัดกลางคำ/ประโยค/รายการ ไม่มี chunk ขยะ
               - หัวข้อย่อยที่ยาวเกินถูกแบ่งตามบรรทัด และทุกชิ้นยังรู้หัวข้อของตัวเอง

2. EMBEDDING : Contextual Embedding (แนวคิด Contextual Retrieval ของ Anthropic)
               - embed("เอกสาร: <ชื่อเอกสาร>" + "<หัวข้อย่อย>" + chunk)
               - แทนการ embed chunk ล้วนๆ ที่ขาดบริบท

3. SEARCH    : Hybrid Search = Dense (Cosine) + Sparse (BM25)
               - BM25 ใช้ตัวตัดคำภาษาไทย (PyThaiNLP newmm) + normalize ตัวเลข
               - รวมผลด้วย Weighted Reciprocal Rank Fusion (RRF)

4. RERANKING : Cross-Encoder Reranking แบบ Multilingual (รองรับภาษาไทย)
               - อ่าน (query, context+chunk) พร้อมกัน แล้วเรียงลำดับใหม่
               - ดึงมาเยอะ (retrieve_k=10) แล้วคัดเหลือน้อย (final_k=3)
=============================================================
"""

import os
import re
import math
import numpy as np
from typing import List, Tuple, Dict
from collections import defaultdict, Counter
from sentence_transformers import SentenceTransformer, CrossEncoder

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from sample_documents import SAMPLE_DOCUMENTS

# ใช้ embedding model ตัวเดียวกับ Part 1 เป็นค่าเริ่มต้น เพื่อให้การเปรียบเทียบยุติธรรม
# (ผลต่างที่เห็นมาจาก chunking / context / hybrid / rerank ไม่ใช่จากการเปลี่ยนโมเดล)
# สามารถลองโมเดลที่แรงกว่าได้ เช่น EMBED_MODEL=BAAI/bge-m3
EMBED_MODEL = os.environ.get("EMBED_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")

# Cross-encoder ที่ฝึกบน mMARCO (MS MARCO แปล 14 ภาษา) → รองรับหลายภาษา
# (โมเดล ms-marco-MiniLM-L-6-v2 เดิมเป็นภาษาอังกฤษล้วน ให้คะแนนภาษาไทยได้ไม่ดี)
RERANK_MODEL = os.environ.get("RERANK_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")


# ============================================================
# IMPROVEMENT 1: STRUCTURE-AWARE CHUNKING
# ============================================================

# บรรทัดที่ขึ้นหัวข้อใหม่ เช่น "ข้อ 1 ...", "1. ค่าพาหนะ", "2. ลาป่วย (Sick Leave)"
SECTION_HEADER = re.compile(r"^(ข้อ\s*\d+|\d+\.)\s*")


def get_document_title(text: str) -> str:
    """ดึงชื่อหัวข้อเอกสาร (บรรทัดแรกที่ไม่ว่าง)"""
    for line in text.strip().split("\n"):
        if line.strip():
            return line.strip()
    return "เอกสารไม่มีชื่อ"


def split_into_sections(text: str) -> Tuple[str, List[Tuple[str, List[str]]]]:
    """
    แยกเอกสารเป็น (ชื่อเอกสาร, [(หัวข้อ, [บรรทัด...]), ...])

    - บรรทัดแรก = ชื่อเอกสาร
    - บรรทัดที่ขึ้นต้นด้วย "ข้อ N" หรือ "N." = เริ่ม section ใหม่
    - bullet "-" และบรรทัดต่อเนื่อง = อยู่ใน section ปัจจุบัน
    - บรรทัดก่อนหัวข้อแรก (คำนำ) = section "บทนำ"
    """
    lines = [ln.strip() for ln in text.strip().split("\n")]
    lines = [ln for ln in lines if ln]
    title = lines[0] if lines else "เอกสารไม่มีชื่อ"

    sections: List[Tuple[str, List[str]]] = []
    current_header, current_lines = "บทนำ", []
    for ln in lines[1:]:
        if SECTION_HEADER.match(ln):
            if current_lines:
                sections.append((current_header, current_lines))
            # หัวข้อ = บรรทัดหัวข้อ (ตัดให้สั้นพอใช้เป็น context)
            current_header = ln if len(ln) <= 80 else ln[:80] + "..."
            current_lines = [ln]
        else:
            current_lines.append(ln)
    if current_lines:
        sections.append((current_header, current_lines))
    return title, sections


def structure_aware_chunking(text: str, max_chars: int = 400) -> List[Dict[str, str]]:
    """
    Structure-aware Chunking

    ปัญหาของ Fixed-size Chunking (Part 1):
      - ตัดทุก 300 ตัวอักษร → ตัดกลางคำ เช่น "...ประเภทต่างๆ ดั" | "งนี้"
      - เกิด chunk ขยะ เช่น "หาร" (เศษท้ายเอกสาร)
      - ข้อมูลเรื่องเดียวกัน (เช่นรายการค่าใช้จ่าย) ถูกแยกไปคนละ chunk

    วิธีของ Part 2:
      - 1 chunk = 1 หัวข้อย่อย (section) ของเอกสาร → ความหมายครบในตัว
      - หาก section ยาวเกิน max_chars แบ่งตามบรรทัด (ไม่ตัดกลางบรรทัด)
        และใส่บรรทัดหัวข้อนำหน้าทุกชิ้น เพื่อไม่ให้ bullet หลุดบริบท
      - section ที่สั้นมาก (< 100 ตัวอักษร เช่นคำนำ "...ดังนี้") ถูกรวมกับ section ถัดไป
    """
    title, sections = split_into_sections(text)

    # รวม section ที่สั้นมากเข้ากับ section ถัดไป
    merged: List[Tuple[str, List[str]]] = []
    carry_header, carry_lines = None, []
    for header, lines in sections:
        if carry_lines:
            header = carry_header
            lines = carry_lines + lines
            carry_header, carry_lines = None, []
        if len(" ".join(lines)) < 100:
            carry_header, carry_lines = header, lines
            continue
        merged.append((header, lines))
    if carry_lines:
        if merged:
            h, l = merged[-1]
            merged[-1] = (h, l + carry_lines)
        else:
            merged.append((carry_header, carry_lines))

    chunks: List[Dict[str, str]] = []
    for header, lines in merged:
        body = "\n".join(lines)
        if len(body) <= max_chars:
            chunks.append({"section": header, "text": body})
            continue
        # section ยาวเกิน → แบ่งตามบรรทัด โดยทุกชิ้นขึ้นต้นด้วยบรรทัดหัวข้อ
        head_line, rest = lines[0], lines[1:]
        buf: List[str] = []
        for ln in rest:
            candidate = "\n".join([head_line] + buf + [ln])
            if len(candidate) > max_chars and buf:
                chunks.append({"section": header, "text": "\n".join([head_line] + buf)})
                buf = []
            buf.append(ln)
        if buf:
            chunks.append({"section": header, "text": "\n".join([head_line] + buf)})
    return chunks


def create_contextual_chunks(documents: List[str], max_chars: int = 400) -> List[dict]:
    """
    สร้าง Contextual Chunks ตามแนวทาง Anthropic Contextual Retrieval

    แนวคิด: "ก่อน embedding ให้สร้าง context กำกับแต่ละ chunk
             แล้วจึง embed ทั้ง context + chunk"

    ที่นี่ context = ชื่อเอกสาร (rule-based, ไม่ต้องเรียก LLM) ส่วนหัวข้อย่อย
    อยู่ในบรรทัดแรกของ chunk เสมอจาก structure-aware chunking
    เช่น chunk "ค่าที่พัก: เบิกได้ไม่เกินคืนละ 1,200 บาท" จะได้ context
    "เอกสาร: ระเบียบการเบิกค่าใช้จ่ายในการเดินทางภายในประเทศ" ทำให้แยกออกจาก
    ค่าที่พักต่างประเทศ (150 USD) ได้ชัดเจน
    (ใน production อาจให้ LLM เขียน context สรุปสั้นๆ แทนได้)
    """
    all_chunks = []
    for doc_id, doc_text in enumerate(documents):
        doc_title = get_document_title(doc_text)
        pieces = structure_aware_chunking(doc_text, max_chars)
        for chunk_id, piece in enumerate(pieces):
            section = piece["section"]
            # ทุก chunk ขึ้นต้นด้วยบรรทัดหัวข้อของตัวเองอยู่แล้ว จึงเติมเฉพาะชื่อเอกสาร
            context = f"เอกสาร: {doc_title}"
            all_chunks.append({
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "doc_title": doc_title,
                "section": section,
                "text": piece["text"],                          # text ดิบของ chunk
                "context": context,                             # context ที่สร้างขึ้น
                "contextual_text": f"{context}\n{piece['text']}",  # สิ่งที่ใช้ embed / BM25 / rerank
            })
    return all_chunks


# ============================================================
# IMPROVEMENT 2: CONTEXTUAL EMBEDDING
# ============================================================

def build_contextual_vector_index(chunks: List[dict], model: SentenceTransformer) -> np.ndarray:
    """
    Contextual Embedding: embed(context + chunk) แทน embed(chunk)

    - Part 1: embed ด้วย chunk['text'] ล้วนๆ
    - Part 2: embed ด้วย chunk['contextual_text'] = ชื่อเอกสาร/หัวข้อ + chunk
    → vector สะท้อน "ความหมายในบริบทของเอกสาร" ไม่ใช่แค่ข้อความดิบ
    """
    texts = [chunk["contextual_text"] for chunk in chunks]
    print(f"  กำลัง Contextual Embed {len(texts)} chunks...")
    return model.encode(texts, show_progress_bar=False, normalize_embeddings=True)


# ============================================================
# IMPROVEMENT 3: BM25 INDEX (Sparse Retrieval) + Thai tokenizer
# ============================================================

try:
    from pythainlp.tokenize import word_tokenize as _thai_word_tokenize
    _HAS_PYTHAINLP = True
except ImportError:  # fallback: character bigram
    _HAS_PYTHAINLP = False

# คำที่พบบ่อยมากและไม่ช่วยแยกเอกสาร
THAI_STOPWORDS = {
    "ได้", "ที่", "และ", "การ", "ของ", "ใน", "ให้", "กับ", "เป็น", "มี", "ไม่", "ต่อ",
    "แล้ว", "กี่", "เท่าไหร่", "เท่าไร", "อะไร", "ไหม", "หรือ", "โดย", "ดังนี้", "ต้อง",
}


class BM25Index:
    """
    BM25 (Okapi) - Sparse/Lexical Retrieval

    score(q, d) = Σ IDF(t) · TF(t,d)·(k1+1) / (TF(t,d) + k1·(1 - b + b·|d|/avgdl))

    ดีสำหรับคำเฉพาะที่ dense embedding มักจับได้ไม่แม่น เช่น ตัวเลข (150,000),
    หน่วย (USD), คำเฉพาะ (ลาคลอด, ใบรับรองแพทย์)

    การตัดคำภาษาไทย: ภาษาไทยไม่มีช่องว่างระหว่างคำ จึงใช้ PyThaiNLP (newmm)
    ตัดเป็นคำ หากไม่ได้ติดตั้งจะ fallback เป็น character bigram
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_tf: List[Counter] = []
        self.doc_len: List[int] = []
        self.idf: Dict[str, float] = {}
        self.avg_dl: float = 0.0
        self.doc_count: int = 0

    @staticmethod
    def tokenize(text: str) -> List[str]:
        text = text.lower()
        text = re.sub(r"(?<=\d),(?=\d{3})", "", text)  # 150,000 → 150000
        if _HAS_PYTHAINLP:
            raw = _thai_word_tokenize(text, engine="newmm", keep_whitespace=False)
            tokens = []
            for tok in raw:
                tok = tok.strip()
                if not tok or tok in THAI_STOPWORDS:
                    continue
                if re.fullmatch(r"[฀-๿]+|[a-z0-9.]+", tok):
                    tokens.append(tok.strip("."))
            return [t for t in tokens if t]
        # fallback: อังกฤษ/ตัวเลขเป็นคำ + ภาษาไทยเป็น bigram
        tokens = re.findall(r"[a-z0-9]+", text)
        for run in re.findall(r"[฀-๿]+", text):
            tokens.extend(run[i:i + 2] for i in range(len(run) - 1))
        return tokens

    def build(self, chunks: List[dict]):
        corpus = [self.tokenize(chunk["contextual_text"]) for chunk in chunks]
        self.corpus_tf = [Counter(doc) for doc in corpus]
        self.doc_len = [len(doc) for doc in corpus]
        self.doc_count = len(corpus)
        self.avg_dl = (sum(self.doc_len) / self.doc_count) if self.doc_count else 1.0

        df = defaultdict(int)
        for tf in self.corpus_tf:
            for term in tf:
                df[term] += 1
        self.idf = {
            term: math.log((self.doc_count - freq + 0.5) / (freq + 0.5) + 1)
            for term, freq in df.items()
        }
        tok_name = "PyThaiNLP newmm" if _HAS_PYTHAINLP else "char-bigram (fallback)"
        print(f"  BM25 Index: {self.doc_count} chunks, vocab={len(self.idf)}, tokenizer={tok_name}")

    def score(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        scores = np.zeros(self.doc_count)
        for term in set(self.tokenize(query)):
            idf_val = self.idf.get(term)
            if idf_val is None:
                continue
            for i, tf_counter in enumerate(self.corpus_tf):
                tf = tf_counter.get(term, 0)
                if tf == 0:
                    continue
                norm = tf * (self.k1 + 1) / (
                    tf + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avg_dl)
                )
                scores[i] += idf_val * norm
        top = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in top if scores[i] > 0]


# ============================================================
# IMPROVEMENT 3 (cont.): HYBRID SEARCH + RRF FUSION
# ============================================================

def reciprocal_rank_fusion(
    dense_results: List[Tuple[int, float]],
    bm25_results: List[Tuple[int, float]],
    k: int = 60,
    dense_weight: float = 0.5,
    bm25_weight: float = 0.5,
) -> List[Tuple[int, float]]:
    """
    Weighted Reciprocal Rank Fusion: RRF(d) = Σ w_i / (k + rank_i(d))

    - ไม่ต้องปรับ scale ของ score (cosine อยู่ช่วง -1..1 ส่วน BM25 ไม่มีขอบบน)
    - chunk ที่ติดอันดับดีทั้งสองวิธีจะได้คะแนนรวมสูงสุด
    """
    fused: Dict[int, float] = defaultdict(float)
    for rank, (idx, _) in enumerate(dense_results):
        fused[idx] += dense_weight / (k + rank + 1)
    for rank, (idx, _) in enumerate(bm25_results):
        fused[idx] += bm25_weight / (k + rank + 1)
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)


def hybrid_search(
    query: str,
    chunks: List[dict],
    embeddings: np.ndarray,
    embed_model: SentenceTransformer,
    bm25_index: BM25Index,
    top_k: int = 10,
) -> List[Tuple[dict, float]]:
    """Hybrid Search: Dense + BM25 → RRF"""
    query_vec = embed_model.encode([query], normalize_embeddings=True)[0]
    dense_scores = np.dot(embeddings, query_vec)
    n = min(top_k * 2, len(chunks))
    dense_results = [(int(i), float(dense_scores[i])) for i in np.argsort(dense_scores)[::-1][:n]]
    bm25_results = bm25_index.score(query, top_k=n)
    fused = reciprocal_rank_fusion(dense_results, bm25_results)
    return [(chunks[i], s) for i, s in fused[:top_k]]


# ============================================================
# IMPROVEMENT 4: CROSS-ENCODER RERANKING (Multilingual)
# ============================================================

class CrossEncoderReranker:
    """
    Cross-Encoder Reranking

    - Bi-encoder (dense search): encode query และ chunk แยกกัน → เร็ว แต่หยาบ
    - Cross-encoder: อ่าน (query, chunk) พร้อมกันใน transformer เดียว
      → เห็นความสัมพันธ์ระดับคำต่อคำ แม่นกว่า แต่ช้ากว่า จึงใช้กับ top-N เท่านั้น

    Two-stage: Retrieval (เร็ว, recall สูง, top-10) → Rerank (แม่น, precision สูง, top-3)
    """

    def __init__(self, model_name: str = RERANK_MODEL):
        print(f"  กำลังโหลด Cross-Encoder: {model_name}")
        try:
            self.model = CrossEncoder(model_name, max_length=512)
            self.available = True
        except Exception as e:
            print(f"  โหลด Cross-Encoder ไม่ได้ ({e}) → ใช้ลำดับจาก Hybrid Search แทน")
            self.model, self.available = None, False

    def rerank(self, query: str, candidates: List[Tuple[dict, float]], top_k: int = 3) -> List[Tuple[dict, float]]:
        if not self.available or not candidates:
            return candidates[:top_k]
        # ส่ง context + chunk ให้ reranker เพื่อให้รู้ว่า chunk มาจากเอกสารไหน
        pairs = [(query, chunk["contextual_text"]) for chunk, _ in candidates]
        scores = self.model.predict(pairs)
        reranked = sorted(
            ((chunk, float(s)) for (chunk, _), s in zip(candidates, scores)),
            key=lambda x: x[1], reverse=True,
        )
        return reranked[:top_k]


# ============================================================
# ANSWER GENERATION (LLM ตัวเดียวกับ Part 1 — ไม่เปลี่ยน)
# ============================================================

def build_improved_prompt(query: str, reranked_chunks: List[Tuple[dict, float]]) -> str:
    """Prompt ที่ระบุชื่อเอกสาร/หัวข้อของแต่ละ chunk เพื่อให้ LLM อ้างอิงแหล่งที่มาได้"""
    parts = []
    for i, (chunk, score) in enumerate(reranked_chunks):
        parts.append(f"[เอกสาร {i+1}: {chunk['context']} | relevance={score:.4f}]\n{chunk['text']}")
    context = "\n\n".join(parts)
    return f"""คุณเป็นผู้ช่วยตอบคำถามจากเอกสารภายในองค์กร กรุณาตอบคำถามโดยอิงจากเนื้อหาที่ให้มาเท่านั้น

เนื้อหาที่เกี่ยวข้อง (เรียงจากความเกี่ยวข้องสูงสุด):
{context}

คำถาม: {query}

คำตอบ (ตอบเป็นภาษาไทย อ้างอิงชื่อเอกสาร หากไม่มีข้อมูลให้ตอบว่าไม่พบข้อมูลในเอกสาร):"""


def generate_answer_with_llm(prompt: str) -> str:
    """ส่ง prompt ให้ LLM (Google Gemini) — ตัวเดียวกับ Part 1 ทุกประการ"""
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        return f"[Mock Response - ไม่มี GOOGLE_API_KEY]\n\nPrompt ที่จะส่งให้ LLM:\n{'='*60}\n{prompt}\n{'='*60}"
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"))
        return model.generate_content(prompt).text
    except Exception as e:
        return f"[LLM Error: {e}]\n\nPrompt:\n{prompt}"


# ============================================================
# MAIN IMPROVED RAG PIPELINE
# ============================================================

class ImprovedRAG:
    """
    Pipeline:
    Document → Structure-aware Chunking + Context Header →
    Contextual Embedding + BM25 Index → Hybrid Search (Dense + BM25 + RRF) →
    Multilingual Cross-Encoder Reranking → LLM Generation
    """

    def __init__(self, max_chunk_chars: int = 400, retrieve_k: int = 10, final_k: int = 3,
                 use_reranker: bool = True):
        self.max_chunk_chars = max_chunk_chars
        self.retrieve_k = retrieve_k  # จำนวน chunk ที่ดึงมาก่อน rerank
        self.final_k = final_k        # จำนวน chunk ที่ส่งให้ LLM
        self.chunks: List[dict] = []
        self.embeddings: np.ndarray = None
        self.bm25_index: BM25Index = None

        print("กำลังโหลด Models...")
        self.embed_model = SentenceTransformer(EMBED_MODEL)
        print(f"  Bi-encoder: {EMBED_MODEL} ({self.embed_model.get_sentence_embedding_dimension()} dims)")
        self.reranker = CrossEncoderReranker() if use_reranker else None

    def index_documents(self, documents: List[str]):
        print("\n" + "=" * 60 + "\nINDEXING PHASE (Improved)\n" + "=" * 60)
        print(f"\n[1] Structure-aware Chunking + Contextual Header (max_chars={self.max_chunk_chars})")
        self.chunks = create_contextual_chunks(documents, self.max_chunk_chars)
        print(f"  จำนวนเอกสาร: {len(documents)} | จำนวน chunk: {len(self.chunks)}")
        ex = self.chunks[1] if len(self.chunks) > 1 else self.chunks[0]
        print(f"  ตัวอย่าง: {ex['contextual_text'][:150]!r}")

        print("\n[2] Contextual Embedding")
        self.embeddings = build_contextual_vector_index(self.chunks, self.embed_model)
        print(f"  Embedding shape: {self.embeddings.shape}")

        print("\n[3] BM25 Index")
        self.bm25_index = BM25Index()
        self.bm25_index.build(self.chunks)
        print("\nIndexing เสร็จสิ้น!")

    def retrieve(self, question: str, verbose: bool = False) -> List[Tuple[dict, float]]:
        """Retrieval อย่างเดียว (Hybrid + Rerank) ใช้ใน evaluate.py"""
        candidates = hybrid_search(question, self.chunks, self.embeddings,
                                   self.embed_model, self.bm25_index, self.retrieve_k)
        if verbose:
            print(f"\n[1] Hybrid Search (Dense + BM25 → RRF), top {len(candidates)}:")
            for i, (c, s) in enumerate(candidates[:5]):
                print(f"  [{i+1}] RRF={s:.5f} | {c['context']} | {c['text'][:60]!r}")
        if self.reranker is None:
            return candidates[:self.final_k]
        reranked = self.reranker.rerank(question, candidates, top_k=self.final_k)
        if verbose:
            print(f"\n[2] Cross-Encoder Reranking → top {self.final_k}:")
            for i, (c, s) in enumerate(reranked):
                print(f"  [{i+1}] CE={s:.4f} | {c['context']} | {c['text'][:60]!r}")
        return reranked

    def query(self, question: str, verbose: bool = True) -> str:
        if self.embeddings is None:
            raise ValueError("กรุณา index เอกสารก่อน ด้วย index_documents()")
        if verbose:
            print("\n" + "=" * 60 + f"\nQUERY: {question}\n" + "=" * 60)
        reranked = self.retrieve(question, verbose=verbose)
        if verbose:
            print("\n[3] Answer Generation with LLM")
        return generate_answer_with_llm(build_improved_prompt(question, reranked))


# ============================================================
# DEMO
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("PART 2: Improved RAG (Structure-aware + Contextual + Hybrid + Rerank)")
    print("=" * 60)

    rag = ImprovedRAG(max_chunk_chars=400, retrieve_k=10, final_k=3)
    rag.index_documents(SAMPLE_DOCUMENTS)

    test_questions = [
        "การเดินทางไปต่างประเทศ เบิกค่าใช้จ่ายรวมแล้วไม่เกินกี่บาท",
        "พนักงานมีสิทธิ์ลาป่วยได้กี่วันต่อปี",
        "ค่าที่พักต่างประเทศเบิกได้เท่าไหร่",
    ]
    for q in test_questions:
        print("\n" + "~" * 60)
        print(f"\n[คำตอบจาก LLM]:\n{rag.query(q, verbose=True)}")
        print("~" * 60)
