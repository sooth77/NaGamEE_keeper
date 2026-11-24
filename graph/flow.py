"""나감이 LangGraph 흐름.

  ① 질문 정리 → ② 코드 판별 (규칙·값·기준일이 모두 확실하면 4B 없이 답 확정)
  → ③ 빠진 날짜 확인 → ④ 검색 → ⑤ 계산(코드)
  → ⑥ 코드 답 또는 4B 풀이 후 답(글자 단위 출력) → ⑦ 검사(코드) → ⑧ 형식 맞추기 → 답

모드
  service: 날짜가 없으면 ③에서 멈추고 사용자에게 되묻는다 (interrupt). 답을 받으면 이어간다.
  eval   : 되묻지 않는다. 날짜가 없으면 4B가 근거를 보고 추가 확인을 안내한다. 문항마다 새 대화(thread)로 시작.

설정 (Nagami(...) 인자)
  doc_release: 'V2' / 'V3' — 문서 판본. 검색 색인·판본 표·'이번 달'이 이걸 따라 바뀐다.
"""
import contextvars
import re
import time
import uuid

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from typing_extensions import TypedDict

from graph.llm import STATUSES, make_llm
from graph.retriever import make_retriever, short_article, short_table
from rules.clean import clean_question, search_query
from rules.compute import compute
from rules.dates import find_dates
from rules.guard import extra_evidence, focus_evidence, verify_answer, version_guard
from rules.tools import REFS, date_rule, pick_event_date, refs_for, solve, split_options
from rules.versions import Registry

ON_TEXT = contextvars.ContextVar("nagami_on_text", default=None)     # 화면에 글자를 바로 넘기는 함수


class State(TypedDict, total=False):
    question: str
    mode: str
    cleaned: str
    stem: str
    solved: dict
    first_at: float
    finish: str
    search_q: str
    fmt: dict
    facts: dict
    calc: dict
    docs: list
    draft: dict
    raw: str
    usage: dict
    checks: list
    result: dict
    timings: dict


VERDICT_SENTENCE = {"가능": "가능해요.", "사전 합의 필요": "사전 합의가 먼저 필요해요.", "공동 처리 불가": "공동비로는 안 돼요.",
                    "추가 확인": "판정 전에 확인할 게 있어요.", "규정에 없음": "규약집에 정해진 규정이 없어요."}

NEXT_BY_VERDICT = {"가능": "결제 기록과 증빙을 남겨 주세요.", "사전 합의 필요": "확정 전에 두 사람이 합의해 주세요.",
                   "공동 처리 불가": "초과분이나 해당 비용은 개인 처리로 나눠 주세요.", "추가 확인": "빠진 날짜와 조건을 확인해 주세요.",
                   "규정에 없음": "두 사람이 합의해 새 기준을 정해 주세요."}

def _t(state, key, t0):
    tm = dict(state.get("timings") or {})
    tm[key] = round(time.perf_counter() - t0, 3)
    return tm


class Nagami:
    def __init__(self, doc_release="V2", llm=None, retriever=None, top_k=5):
        self.doc_release = doc_release
        self.reg = Registry(doc_release)
        self.llm = llm or make_llm()
        self.retriever = retriever or make_retriever(doc_release)
        self.top_k = top_k
        self.graph = self._build()

    # ------------------------------------------------------------ 노드
    def n_clean(self, s):
        t0 = time.perf_counter()
        c = clean_question(s["question"])
        return {"cleaned": c["cleaned"], "stem": split_options(c["cleaned"])[0], "fmt": c["format"],
                "search_q": search_query(c["cleaned"]), "checks": [], "timings": _t(s, "clean", t0)}

    def n_gate(self, s):
        t0 = time.perf_counter()
        return {"solved": solve(s["cleaned"], s["fmt"], self.reg), "facts": {}, "timings": _t(s, "gate", t0)}

    def n_need_info(self, s):
        if s.get("mode") != "service" or (s.get("solved") or {}).get("code_answer"):
            return {}
        key = date_rule(s["cleaned"])
        if not key:
            return {}
        days = set(find_dates(s["question"]))
        if len(days) == 1 or (days and pick_event_date(s["question"], key)):
            return {}
        answer = str(interrupt({"field": "event_date", "item": key,
                                "say": f"판정 전에 하나만요. {self.reg.item(key)['date_basis']}이 언제예요?"}))
        return {"facts": {"item": key, "event_date": answer},
                "solved": solve(s["cleaned"], s["fmt"], self.reg, key=key, when=answer)}

    def n_retrieve(self, s):
        t0 = time.perf_counter()
        docs = self.retriever.search(s["search_q"], top_k=self.top_k)
        return {"docs": docs, "timings": _t(s, "retrieve", t0)}

    def n_compute(self, s):
        solved = s.get("solved")
        if solved and solved.get("code_answer"):
            return {"calc": solved}
        calc = compute(s.get("facts") or {}, self.reg, s["question"])
        if calc and calc.get("decision"):
            return {"calc": calc}
        base = solved or calc
        guarded = version_guard(s["cleaned"], s.get("fmt"), self.reg)
        if guarded and (not base or guarded.get("code_answer")):
            return {"calc": guarded}
        return {"calc": base}

    def n_write(self, s):
        t0 = time.perf_counter()
        docs_text = "\n\n---\n\n".join(
            f"[{d['page']}쪽 | {d.get('chapter', '')} > {d.get('topic', '')}]\n{d['text']}" for d in s["docs"])
        calc = s.get("calc")
        calc_text = None
        if calc:
            calc_text = (f"항목: {calc['item_name']} · 기준일({calc['date_basis']}): {calc.get('event_date') or '모름'} · "
                         f"적용 판본: {calc['version']}\n{calc['text']}"
                         + (f"\n코드 판정: {calc['verdict']}" if calc.get("verdict") else "")
                         + ("\n코드 결론을 바꾸지 마세요." if calc.get("decision") else ""))
        q = s["question"] if len(s["question"]) <= 1500 else s["cleaned"]
        solo_input = getattr(self.retriever, "solo_input", None)     # 평가: 4B 단독과 같은 근거·질문 글을 쓴다
        if solo_input:
            docs_text, q = solo_input()
        extra = extra_evidence(calc, s["docs"], self.doc_release)
        focus = focus_evidence(s["cleaned"], s["docs"])
        if focus:      # 원문은 그대로 두고, 질문과 직접 맞는 줄을 질문 바로 앞에 한 번 더 보여 준다
            extra = "[질문과 직접 관련된 줄] 인용은 제N조, 표 N 형식으로 쓰세요.\n" + focus + ("\n\n---\n\n" + extra if extra else "")
        self.llm.extra_text = extra
        if extra and not hasattr(self.llm, "uses_extra"):
            docs_text += "\n\n---\n\n" + extra
        if getattr(self.llm, "fake", False):
            draft, raw, usage = self.llm.write(q, docs_text, calc_text, self.reg.defaults["this_month"],
                                               calc=calc, docs=s["docs"], fmt=s["fmt"])
        else:
            draft, raw, usage = self.llm.write(q, docs_text, calc_text, self.reg.defaults["this_month"],
                                               length=(s.get("fmt") or {}).get("length"), release=self.doc_release,
                                               on_text=ON_TEXT.get())
        usage = dict(usage or {})
        first_at = usage.pop("first_answer_at", None)
        checks = list(s.get("checks") or [])
        issue = verify_answer((draft or {}).get("answer"), raw, usage.get("finish_reason"), calc, self.reg)
        if issue and hasattr(self.llm, "rewrite"):      # 첫 글자가 나간 뒤에만 실행되고, 문제가 있을 때 한 번만 다시 쓴다
            draft, raw, again = self.llm.rewrite(issue)
            usage = {**usage, "completion_tokens": (usage.get("completion_tokens") or 0) + (again.get("completion_tokens") or 0),
                     "finish_reason": again.get("finish_reason")}
            checks.append(f"답변 검증 후 재작성: {issue}")
        return {"draft": draft or {}, "raw": raw, "usage": usage, "first_at": first_at, "checks": checks,
                "finish": usage.get("finish_reason"), "timings": _t(s, "write", t0)}

    def _after_compute(self, s):
        """코드가 답을 확정했으면 4B를 부르지 않는다."""
        calc = s.get("calc") or {}
        kind = (s.get("fmt") or {}).get("kind")
        if calc.get("code_answer"):
            return "code_answer"
        return "code_answer" if calc.get("decision") and kind not in ("number", "choice", "multi") else "write"

    def n_code_answer(self, s):
        calc = s["calc"]
        answer = calc.get("code_answer") or calc["decision"]
        draft = {"verdict": calc.get("verdict") if calc.get("code_answer") else None, "answer": answer, "reason": calc["text"],
                 "citations": [], "next_action": calc.get("next_action")}
        return {"draft": draft, "raw": answer, "usage": {}, "first_at": time.perf_counter(), "finish": "code",
                "timings": _t(s, "write", time.perf_counter())}

    def n_check(self, s):
        d = dict(s["draft"])
        checks = list(s.get("checks") or [])
        calc = s.get("calc") or {}
        # 판정은 5가지 중 하나만
        if d.get("verdict") not in STATUSES:
            d["verdict"] = None
        # 코드 계산과 판정이 다르면 코드를 따른다
        if calc.get("verdict") and d.get("verdict") != calc["verdict"]:
            checks.append(f"판정을 코드 계산에 맞춰 고침: {d.get('verdict')} → {calc['verdict']}")
            d["verdict"] = calc["verdict"]
            # 답 문장도 고친 판정과 어긋나지 않게 (숫자·객관식 답은 그대로 둔다)
            kind = (s.get("fmt") or {}).get("kind")
            if kind not in ("number", "choice", "multi"):
                yesno = {"가능": "예", "공동 처리 불가": "아니오"}
                d["answer"] = (yesno.get(d["verdict"]) if kind == "yesno" and d["verdict"] in yesno
                               else VERDICT_SENTENCE.get(d["verdict"], d["verdict"]))
                if calc.get("text"):
                    d["reason"] = calc["text"]
                checks.append("답 문장도 고친 판정에 맞춰 바꿈")
        # 개정 규칙의 코드 결론 (충족·보류·성립). 숫자·객관식·긴 안내 답은 모델 답을 둔다
        if calc.get("decision") and not calc.get("code_answer"):
            d["verdict"] = None      # 충족·보류·성립은 5가지 판정이 아니다. 모델이 쓴 판정이 결론과 어긋나지 않게 비운다
            kind = (s.get("fmt") or {}).get("kind")
            if kind == "yesno":
                d["answer"] = "예" if calc.get("passed") else "아니오"
            elif kind not in ("number", "choice", "multi"):
                d["answer"] = calc["decision"]
            if kind not in ("number", "choice", "multi"):
                d["reason"] = calc["text"]
                checks.append("개정 규칙의 답을 코드 결론에 맞춤")
        # 계산에 쓴 규칙의 조항·표는 모델이 빠뜨려도 코드가 붙인다
        source = refs_for(calc["item"], self.reg) if calc.get("item") in REFS else calc.get("source", "")   # 문서 판본에 있는 근거만
        if source:
            refs = [re.sub(r"\s+", " ", m.group(0)) for m in re.finditer(r"(부칙\s*)?제\s*\d+\s*조|표\s*\d+", source)]
            d["citations"] = list(dict.fromkeys(list(d.get("citations") or []) + refs))
        # 이유 문장에 코드 계산·질문에 없는 숫자(예: 다른 판본의 상한)가 있으면 코드 계산 문장으로 바꾼다
        if calc.get("text") and d.get("reason"):
            nums = lambda t: {float(x) for x in re.findall(r"\d+(?:\.\d+)?", str(t).replace(",", ""))}
            extra = nums(d["reason"]) - nums(calc["text"]) - nums(s["question"])
            if extra:
                checks.append(f"이유 문장의 숫자({', '.join(f'{x:g}' for x in sorted(extra))})가 코드 계산과 달라서 계산 문장으로 바꿈")
                d["reason"] = calc["text"]
        # 인용은 검색된 근거·판본 표 근거 안에서만
        allowed = set()
        for x in s["docs"]:
            allowed |= {short_article(a) for a in x.get("articles", [])} | {short_table(t) for t in x.get("tables", [])}
            allowed.add(f"{x['page']}쪽")
        allowed |= {short_article(m.group(0)) for m in re.finditer(r"(부칙\s*)?제\s*\d+\s*조", source)}
        allowed |= {short_table(m.group(0)) for m in re.finditer(r"표\s*\d+", source)}
        allowed.discard("")
        keep, dropped = [], []
        for c in d.get("citations") or []:
            c = str(c).strip()
            key = short_article(c) or short_table(c) or (c if c.endswith("쪽") else "")
            (keep if key in allowed else dropped).append(c)
        if dropped:
            checks.append(f"검색 근거에 없는 인용을 뺌: {', '.join(dropped)}")
        d["citations"] = keep
        if not str(d.get("answer") or "").strip():
            checks.append("빈 답")
        length = (s.get("fmt") or {}).get("length")
        if length and not length[0] <= len(str(d.get("answer") or "")) <= length[1]:
            checks.append(f"요청 길이 {length[0]}~{length[1]}자와 다름: {len(str(d.get('answer') or ''))}자")
        return {"draft": d, "checks": checks}

    def n_format(self, s):
        d, fmt, calc = s["draft"], s["fmt"], s.get("calc") or {}
        answer = d.get("answer")
        if isinstance(answer, dict):       # 모델이 답을 표처럼 내면 사람이 읽는 문장으로 바꾼다
            answer = ", ".join(f"{k}: {v}" for k, v in answer.items())
        elif isinstance(answer, list):
            answer = ", ".join(map(str, answer))
        final = str(answer if answer is not None else "").strip()
        if fmt.get("kind") == "yesno":
            m = re.match(r"\**\s*(예|아니오|아니요|네)", final)
            if m:
                final = "아니오" if m.group(1) in ("아니오", "아니요") else "예"
        result = {
            "final_answer": final,
            "verdict": d.get("verdict"),
            "version": calc.get("version"),
            "base_date": calc.get("event_date"),
            "calc": calc.get("text"),
            "reason": d.get("reason"),
            "citations": d.get("citations", []),
            "next_action": d.get("next_action") or calc.get("next_action") or NEXT_BY_VERDICT.get(d.get("verdict"))
            or "관련 조항과 증빙을 확인해 주세요.",
            "answered_by": calc.get("answered_by") if calc.get("code_answer") else ("code_decision" if calc.get("decision") else "4b"),
            "calc_item": calc.get("item"),
            "finish": s.get("finish"),
            "first_at": s.get("first_at"),
            "checks": s.get("checks", []),
            "retrieved": [{k: x.get(k) for k in ("rank", "version", "chunk_id", "page", "articles", "tables", "score")}
                          for x in s["docs"]],
            "raw_output": s.get("raw"),
            "facts": s.get("facts"),
            "cleaned_question": s.get("cleaned"),
            "format": fmt,
            "tokens_in": (s.get("usage") or {}).get("prompt_tokens"),
            "tokens_out": (s.get("usage") or {}).get("completion_tokens"),
            "timings": s.get("timings"),
        }
        return {"result": result}

    # ------------------------------------------------------------ 그래프
    def _build(self):
        g = StateGraph(State)
        for name in ("clean", "gate", "need_info", "retrieve", "compute", "code_answer", "write", "check", "format"):
            g.add_node(name, getattr(self, f"n_{name}"))
        g.add_edge(START, "clean")
        for a, b in [("clean", "gate"), ("gate", "need_info"), ("need_info", "retrieve"), ("retrieve", "compute"),
                     ("code_answer", "check"), ("write", "check"), ("check", "format")]:
            g.add_edge(a, b)
        g.add_conditional_edges("compute", self._after_compute, {"code_answer": "code_answer", "write": "write"})
        g.add_edge("format", END)
        return g.compile(checkpointer=InMemorySaver())

    # ------------------------------------------------------------ 바깥에서 부르는 함수
    def _wrap(self, out, thread_id, t0):
        if out.get("__interrupt__"):
            v = out["__interrupt__"][0].value
            return {"type": "ask", "thread_id": thread_id, **v}
        r = dict(out["result"])
        r["latency_s"] = round(time.perf_counter() - t0, 3)
        first_at = r.pop("first_at", None)
        r["ttft_s"] = round(first_at - t0, 3) if first_at else r["latency_s"]
        return {"type": "answer", "thread_id": thread_id, **r}

    def ask(self, question, mode="service", thread_id=None, on_text=None):
        thread_id = thread_id or str(uuid.uuid4())
        t0 = time.perf_counter()
        token = ON_TEXT.set(on_text)
        try:
            out = self.graph.invoke({"question": question, "mode": mode},
                                    {"configurable": {"thread_id": thread_id}})
        finally:
            ON_TEXT.reset(token)
        return self._wrap(out, thread_id, t0)

    def resume(self, thread_id, value, on_text=None):
        t0 = time.perf_counter()
        token = ON_TEXT.set(on_text)
        try:
            out = self.graph.invoke(Command(resume=value), {"configurable": {"thread_id": thread_id}})
        finally:
            ON_TEXT.reset(token)
        return self._wrap(out, thread_id, t0)
