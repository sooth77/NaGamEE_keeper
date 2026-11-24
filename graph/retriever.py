"""검색 연결부 (그래프 ⑤ 노드가 부른다).

두 가지 검색기를 같은 모양으로 쓴다.
  - "kure": A의 RulebookRetriever (KURE-v3 + KURE-Reranker-nano). 공식 평가용. 모델 다운로드 필요.
  - "bm25": Kiwi 형태소 + BM25. 인터넷 없이 바로 되는 임시 검색 (화면 개발·시험용).

결과 모양 (A의 검색기와 같음):
  {"rank", "version", "chunk_id", "page", "chapter", "topic", "articles", "tables", "score", "text"}

판본 주의: 검색기의 version 은 '문서 판본'(V2/V3)이다. 사건에 적용할 판본(V1/V2/V3)이 아니다.
V2 합본 안에 V1 값도 들어 있으므로 V1 사건도 V2 문서에서 찾는다.
"""
import json
import os
import re
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"


def chunk_file(doc_release):
    return DATA / f"rulebook_{doc_release}.jsonl"


class BM25Retriever:
    name = "bm25 (임시)"

    def __init__(self, doc_release="V2"):
        from kiwipiepy import Kiwi
        from rank_bm25 import BM25Okapi
        self.doc_release = doc_release
        self.chunks = [json.loads(l) for l in open(chunk_file(doc_release), encoding="utf-8") if l.strip()]
        self.kiwi = Kiwi()
        self.bm25 = BM25Okapi([self._tok(c["text"]) for c in self.chunks])

    def _tok(self, text):
        keep = ("NNG", "NNP", "NNB", "NR", "SN", "SL", "VV", "VA", "XR", "MAG")
        return [t.form for t in self.kiwi.tokenize(text) if t.tag.startswith(keep)]

    def search(self, query, top_k=5, **_):
        scores = self.bm25.get_scores(self._tok(query))
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
        out = []
        for r, i in enumerate(order, 1):
            c = self.chunks[i]
            out.append({"rank": r, "version": self.doc_release, "chunk_id": c["chunk_id"], "page": c["page"],
                        "chapter": c.get("chapter", ""), "topic": c.get("topic", ""),
                        "articles": c.get("articles", []), "tables": c.get("tables", []),
                        "score": round(float(scores[i]), 3), "text": c["text"]})
        return out


class KureRetriever:
    """A의 검색기를 평가용 설정으로 감싼다 (폴더 감시 끔, 항상 top_k개)."""
    name = "KURE (A의 검색기)"

    def __init__(self, doc_release="V2", db_path=None):
        import tempfile
        from graph.a_rag_retriever import RulebookRetriever
        self.doc_release = doc_release
        watch = Path(tempfile.mkdtemp(prefix="nagami_chunks_"))
        (watch / f"rulebook_{doc_release}.jsonl").write_bytes(chunk_file(doc_release).read_bytes())
        self.r = RulebookRetriever(db_path=str(db_path or DATA / f"qdrant_{doc_release}"), watch_dir=str(watch),
                                   auto_sync_on_search=False, device=os.environ.get("NAGAMI_DEVICE", "auto"))

    def search(self, query, top_k=5, candidate_k=8, **_):
        res = self.r.search(query, top_k=top_k, candidate_k=candidate_k, version=self.doc_release,
                            clean_query=True, min_rerank_score=None)
        for x in res:
            x["score"] = x.pop("rerank_score", None)
        return res


class HybridRetriever:
    """하이브리드: BM25(단어 일치) 후보 + KURE(의미 검색) 후보를 합쳐서 A의 리랭커로 다시 순서를 매긴다.

    단어로만 걸리는 쪽(조항 번호·표 이름·금액)과 뜻으로만 걸리는 쪽(바꿔 말한 질문)을 둘 다 후보에 넣는다.
    최종 순서는 리랭커 점수 하나로 정한다.
    """
    name = "하이브리드 (BM25 + KURE 후보 → KURE 리랭커)"

    def __init__(self, doc_release="V2", n_bm25=10, n_dense=15):
        self.doc_release = doc_release
        self.bm25 = BM25Retriever(doc_release)
        self.kure = KureRetriever(doc_release)
        self.n_bm25, self.n_dense = n_bm25, n_dense

    def search(self, query, top_k=5, **_):
        dense = self.kure.r.search(query, top_k=self.n_dense, candidate_k=self.n_dense, version=self.doc_release,
                                   clean_query=True, min_rerank_score=None)
        lex = self.bm25.search(query, top_k=self.n_bm25)
        pool = {}
        for src, hits in (("kure", dense), ("bm25", lex)):
            for h in hits:
                d = pool.setdefault(h["chunk_id"], {**h, "from": []})
                d["from"].append(f"{src}#{h['rank']}")
        cands = list(pool.values())
        ranked = self.kure.r.reranker.rank(query, [c["text"] for c in cands], batch_size=4)
        out = []
        for r, item in enumerate(ranked[:top_k], 1):
            c = cands[item["corpus_id"]]
            out.append({"rank": r, "version": self.doc_release, "chunk_id": c["chunk_id"], "page": c["page"],
                        "chapter": c.get("chapter", ""), "topic": c.get("topic", ""),
                        "articles": c.get("articles", []), "tables": c.get("tables", []),
                        "score": round(float(item["score"]), 4), "from": c["from"], "text": c["text"]})
        return out


def make_retriever(doc_release="V2", kind=None):
    kind = kind or os.environ.get("NAGAMI_RETRIEVER", "bm25")
    if kind == "hybrid":
        return HybridRetriever(doc_release)
    if kind == "kure":
        return KureRetriever(doc_release)
    return BM25Retriever(doc_release)


def short_article(name):
    """'제19조 사용 목적' → '제19조', '부칙 제3조 생신 경과조치' → '부칙 제3조'."""
    m = re.match(r"(부칙\s*)?제\s*\d+\s*조", name or "")
    return re.sub(r"\s+", " ", m.group(0)).replace("제 ", "제") if m else ""


def short_table(name):
    m = re.match(r"표\s*(\d+)", name or "")
    return f"표 {int(m.group(1)):02d}" if m else ""
