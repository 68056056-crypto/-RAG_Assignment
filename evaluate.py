"""
=============================================================
evaluate.py — เปรียบเทียบ Retrieval ของ Part 1 vs Part 2 แบบมีตัวเลข
=============================================================

วัดเฉพาะขั้น Retrieval (ไม่เรียก LLM) เพราะ LLM ของทั้งสองระบบเป็นตัวเดียวกัน
ถ้า retrieval ส่งเนื้อหาที่มีคำตอบให้ LLM ได้ คำตอบสุดท้ายก็จะถูกต้องตาม

ชุดคำถามทดสอบ 16 ข้อ แต่ละข้อมี
  - gold_doc   : เอกสารที่ถูกต้อง
  - gold_answer: ข้อความคำตอบที่ต้องปรากฏใน chunk (เช่น "150 USD")

Metrics (ทั้งสองระบบส่ง 3 chunks ให้ LLM เท่ากัน):
  - Answer@1   : chunk อันดับ 1 มีคำตอบครบหรือไม่
  - Answer@3   : มีคำตอบครบอยู่ใน 3 chunks ที่ส่งให้ LLM หรือไม่  ← สำคัญที่สุด
  - MRR        : 1 / อันดับของ chunk แรกที่มีคำตอบ (เฉลี่ย)
  - Doc@1      : chunk อันดับ 1 มาจากเอกสารที่ถูกต้องหรือไม่

มี Ablation ของ Part 2 เพื่อดูว่าแต่ละเทคนิคช่วยเท่าไร:
  A) Part 1 Basic                         (fixed-size + dense)
  B) + Structure-aware + Contextual chunk (dense only)
  C) + BM25 Hybrid (RRF)
  D) + Multilingual Cross-Encoder Rerank  (= Part 2 เต็มรูปแบบ)

วิธีรัน:  python evaluate.py      → พิมพ์ตาราง + บันทึก results/eval_results.md
=============================================================
"""

import os
import re
import sys
import importlib.util
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, os.path.dirname(path))
    spec.loader.exec_module(mod)
    sys.path.pop(0)
    return mod


p1 = load_module("rag_part1_basic", os.path.join(ROOT, "part1_basic", "rag_part1_basic.py"))
p2 = load_module("rag_part2_improved", os.path.join(ROOT, "part2_improved", "rag_part2_improved.py"))
DOCS = p1.SAMPLE_DOCUMENTS

# (คำถาม, gold_doc, gold_answer)
# มีทั้งคำถามตรงตัว คำถามที่ใช้คำต่างจากเอกสาร (paraphrase) และคำถามที่ตัวเลข/คำคล้ายกันข้ามเอกสาร
TEST_SET = [
    ("การเดินทางไปต่างประเทศ เบิกค่าใช้จ่ายรวมแล้วไม่เกินกี่บาท", 0, "150,000 บาท"),
    ("พนักงานมีสิทธิ์ลาป่วยได้กี่วันต่อปี", 2, "30 วันต่อปี"),
    ("ค่าที่พักต่างประเทศเบิกได้เท่าไหร่", 0, "150 USD"),
    ("ต้องยื่นขออนุมัติไปต่างประเทศล่วงหน้ากี่วัน", 0, "7 วันทำการ"),
    ("ไปดูงานต่างประเทศ เบิกค่าอาหารได้วันละเท่าไหร่", 0, "75 USD"),
    ("ไปทำงานต่างจังหวัด ค่าโรงแรมเบิกได้คืนละเท่าไร", 1, "1,200 บาท"),
    ("ขับรถตัวเองไปติดต่องานในประเทศ เบิกได้กิโลเมตรละกี่บาท", 1, "4 บาทต่อกิโลเมตร"),
    ("ค่าเบี้ยเลี้ยงเดินทางในประเทศได้วันละเท่าไร", 1, "240 บาท"),
    ("ลาคลอดได้กี่วัน", 2, "98 วัน"),
    ("ทำงานมาครบ 3 ปี ลาพักร้อนได้กี่วัน", 2, "15 วันทำการ"),
    ("ผู้จัดการได้งบซื้อโน้ตบุ๊กเท่าไหร่", 3, "60,000 บาท"),
    ("พนักงานทั่วไปเบิกค่าโทรศัพท์ได้เดือนละเท่าไร", 3, "500 บาทต่อเดือน"),
    ("คู่สมรสของพนักงานมีวงเงินประกันสุขภาพเท่าไร", 4, "150,000 บาทต่อปีต่อคน"),
    ("ตัดแว่นสายตาใหม่ บริษัทช่วยออกให้เท่าไหร่", 4, "2,000 บาทต่อปี"),
    ("งบอบรมของพนักงานแต่ละคนต่อปีมีเท่าไร", 5, "15,000 บาทต่อปี"),
    ("ได้ทุนเรียนปริญญาโทสูงสุดเท่าไร", 5, "100,000 บาท"),
]


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s)


def evaluate(name, retrieve_fn):
    a1 = a3 = d1 = 0
    rr, ctx_len, rows = [], [], []
    for q, gold_doc, gold_ans in TEST_SET:
        results = retrieve_fn(q)[:3]
        hits = [gold_ans in norm(c["text"]) for c, _ in results]
        a1 += hits[0] if hits else 0
        a3 += any(hits)
        d1 += bool(results) and results[0][0]["doc_id"] == gold_doc
        rank = next((i + 1 for i, h in enumerate(hits) if h), None)
        rr.append(1 / rank if rank else 0.0)
        ctx_len.append(sum(len(c["text"]) for c, _ in results))
        rows.append((q, rank))
    n = len(TEST_SET)
    return {
        "name": name,
        "Answer@1": a1 / n, "Answer@3": a3 / n, "MRR": float(np.mean(rr)), "Doc@1": d1 / n,
        "ctx_chars": float(np.mean(ctx_len)), "rows": rows,
    }


def main():
    print("โหลดระบบ Part 1 ...")
    basic = p1.BasicRAG(chunk_size=300, overlap=50, top_k=3)
    basic.index_documents(DOCS)

    print("\nโหลดระบบ Part 2 ...")
    improved = p2.ImprovedRAG(max_chunk_chars=400, retrieve_k=10, final_k=3)
    improved.index_documents(DOCS)

    def dense_only(q):  # B) structure-aware + contextual embedding, dense เท่านั้น
        v = improved.embed_model.encode([q], normalize_embeddings=True)[0]
        s = improved.embeddings @ v
        return [(improved.chunks[i], float(s[i])) for i in np.argsort(s)[::-1][:3]]

    def hybrid_only(q):  # C) + BM25 + RRF (ยังไม่ rerank)
        return p2.hybrid_search(q, improved.chunks, improved.embeddings,
                                improved.embed_model, improved.bm25_index, 3)

    systems = [
        ("A) Part 1: Fixed-size + Dense", basic.retrieve),
        ("B) + Structure-aware + Contextual (Dense)", dense_only),
        ("C) + BM25 Hybrid (RRF)", hybrid_only),
        ("D) + Cross-Encoder Rerank (Part 2 เต็ม)", improved.retrieve),
    ]
    results = [evaluate(name, fn) for name, fn in systems]

    lines = [
        "# Retrieval Evaluation: Part 1 vs Part 2",
        "",
        f"ชุดทดสอบ {len(TEST_SET)} คำถาม | ทุกระบบส่ง 3 chunks ให้ LLM | "
        f"embedding: `{p2.EMBED_MODEL}` | reranker: `{p2.RERANK_MODEL}`",
        "",
        "| ระบบ | Answer@1 | Answer@3 | MRR | Doc@1 | ความยาว context เฉลี่ย (ตัวอักษร) |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(f"| {r['name']} | {r['Answer@1']:.0%} | {r['Answer@3']:.0%} | "
                     f"{r['MRR']:.3f} | {r['Doc@1']:.0%} | {r['ctx_chars']:.0f} |")
    lines += ["", "## อันดับของ chunk แรกที่มีคำตอบ (— = ไม่อยู่ใน top-3)", "",
              "| คำถาม | A | B | C | D |", "|---|---|---|---|---|"]
    for i, (q, _, ans) in enumerate(TEST_SET):
        ranks = [r["rows"][i][1] for r in results]
        lines.append(f"| {q} (`{ans}`) | " + " | ".join(str(x) if x else "—" for x in ranks) + " |")

    report = "\n".join(lines)
    print("\n" + report)
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "eval_results.md"), "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print("\nบันทึกผลที่ results/eval_results.md")


if __name__ == "__main__":
    main()
