"""Streamlit 화면용 실제 연결: 4B + A의 public12 검색 + 개선판 LangGraph."""
import os
import shutil
from pathlib import Path

import torch

from evaluation import run_eval as E
from graph.flow import Nagami

WORK = Path(os.environ.get("NAGAMI_WORK", "/content/live_work"))
M4 = os.environ["NAGAMI_M4"]


class LiveDocs:
    name = "A의 public12 검색기"

    def __init__(self, release, a_module):
        self.A, self.release, self.question, self.row = a_module, release, "", None
        chunks = WORK / f"chunks_{release}"
        chunks.mkdir(parents=True, exist_ok=True)
        shutil.copy2(E.ROOT / "data" / "rulebook_V2.jsonl", chunks / "data_v1_v2.jsonl")
        if release == "V3":
            shutil.copy2(E.ROOT / "data" / "rulebook_V3.jsonl", chunks / "data_v3.jsonl")
        self.r = a_module.RulebookRetriever(db_path=str(WORK / f"qdrant_{release}"), watch_dir=str(chunks), use_8bit=torch.cuda.is_available())
        self.latest, versions = a_module.get_latest_version(retriever=self.r, pdf_dir="/nonexistent", chunks_dir=str(chunks),
                                                            md_dir="/nonexistent", fallback=release)
        if self.latest != release:
            raise RuntimeError(f"{release} 색인의 최신 판본이 {self.latest}입니다.")
        old = [v for v in versions if v != self.latest]
        self.hint_old = f"(예: {', '.join(old)})" if old else ""

    def search(self, query, top_k=5, **_):
        core, hits, hint, mode = E.retrieve(self.r, self.A, self.question, self.release, self.latest)
        system = self.A.SYSTEM_PROMPT_TEMPLATE.format(today=E.TODAY[self.release], old_versions_hint=self.hint_old,
                                                      latest_version=self.latest, rule_4=self.A.build_rule_4(core))
        context = self.r.format_context(hits) if hits else ""
        middle = f"\n\n{hint}" if hint else ""
        docs = [{"rank": h["rank"], "version": h["version"], "chunk_id": h["chunk_id"], "page": h["page"], "chapter": h.get("chapter", ""),
                 "topic": h.get("topic", ""), "articles": h.get("articles", []), "tables": h.get("tables", []),
                 "score": h.get("rerank_score"), "text": h["text"]} for h in hits]
        self.row = {"system": system, "user": f"[참고 근거]\n{context}{middle}\n\n[질문]\n{core}", "context": context,
                    "core_question": core, "docs": docs, "search_mode": mode}
        self.llm.row = self.row
        return docs[:top_k]

    def solo_input(self):
        return self.row["context"], self.row["core_question"]


class Live:
    """V2·V3 두 그래프가 같은 4B를 함께 쓴다."""

    def __init__(self):
        self.hf = E.HF("4b", M4, False)
        self.A = E.load_public12()
        self.graphs = {}

    def get(self, release):
        if release not in self.graphs:
            docs = LiveDocs(release, self.A)
            llm = E.NagamiLLM(self.hf)
            docs.llm = llm
            self.graphs[release] = Nagami(release, llm=llm, retriever=docs, top_k=5)
        return self.graphs[release]
