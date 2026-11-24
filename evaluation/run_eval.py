"""GPU 평가: 32B / 4B / 나감이(4B + public12 검색 + LangGraph).

    python evaluation/run_eval.py prepare --out <결과 폴더> --work /content/work
    python evaluation/run_eval.py run --out <결과 폴더> --m4 <4B 폴더> --m32 <32B 폴더>

- A담당의 public12와 검색기는 고치지 않고 불러 쓴다.
- 날짜 고정, 판본 분리, 검색 시간 포함, 생각 모드 끔은 이 파일에서 맞춘다.
- 세 시스템은 같은 근거와 지시문을 쓴다. 모델에는 질문만 넣고 정답 파일은 읽지 않는다.
- 답은 문항마다 저장한다. 끊기면 같은 명령으로 이어서 실행한다.
"""
import argparse
import gc
import importlib.util
import json
import os
import re
import shutil
import statistics as st
import sys
import threading
import time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
ROOT = Path(__file__).resolve().parent.parent      # NaGamEE_keeper/
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "retriever"))
os.chdir(ROOT)
QUESTION_DIR = ROOT / "evaluation" / "questions"

import torch  # noqa: E402

TODAY = {"V2": "2026-11-01", "V3": "2026-12-01"}
QUESTIONS = {("V3", "public"): "public.jsonl", ("V2", "public"): "public.jsonl", ("V3", "strength"): "strength_V3.jsonl",
             ("V2", "strength"): "strength_V2.jsonl", ("V3", "team"): "team_V3.jsonl", ("V2", "team"): "team_V2.jsonl"}
PLAN = [("V3", "public"), ("V2", "public"), ("V3", "strength"), ("V2", "strength"), ("V3", "team"), ("V2", "team")]
SYSTEMS = ("nagami", "4b", "32b")
MAX_TOKENS = 1536
REV32 = "9216db5781bf21249d130ec9da846c4624c16137"
NO_HITS = "규약집에서 관련된 기준을 찾지 못했습니다."


def load_jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def append(f, row):
    f.write(json.dumps(row, ensure_ascii=False) + "\n")
    f.flush()
    os.fsync(f.fileno())


def done_rows(path):
    if not Path(path).exists():
        return {}
    rows = {}
    for line in open(path, encoding="utf-8"):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not row.get("error"):
            rows[row["id"]] = row
    return rows


def load_public12():
    spec = importlib.util.spec_from_file_location("public12", ROOT / "retriever" / "public12_pipeline.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------- 근거 만들기
def retrieve(r, A, q, release, latest):
    raw_ver = A.detect_requested_version(q)
    init = raw_ver if raw_ver and raw_ver != "ALL" else None
    if release == "V2" and init == "V3":      # V2 문서 시점에는 V3가 없다
        init = None
    core = r.extract_core_query(q, version=init, only_latest=init is not None)
    explicit = A.detect_requested_version(core)
    if release == "V2" and explicit == "V3":
        explicit = None
    dates = r.analyze_query_dates(core)
    if explicit is not None:
        mode = f"explicit:{explicit}"
        hits = r.search(query=core, top_k=5, candidate_k=20 if explicit == "ALL" else 15, version=explicit,
                        clean_query=False, min_rerank_score=0.0)
    elif release == "V3" and dates["has_past_date"] and dates["is_amended_topic"]:
        mode = f"date_routed:{'+'.join(dates['past_versions'])}|addendum:{','.join(dates['addendum_pages'])}"
        hits = r.search_with_date_routing(query=core, past_versions=dates["past_versions"], addendum_pages=dates["addendum_pages"],
                                          latest_version=latest, total_top_k=5, candidate_k=15, min_rerank_score=0.0)
    else:      # V2 문서의 부칙 쪽 번호는 V3와 달라 날짜 라우팅을 쓰지 않는다
        mode = f"default_latest:{latest}"
        hits = r.search(query=core, top_k=5, candidate_k=15, version=latest, clean_query=False, min_rerank_score=0.0)
        if not hits:
            hits = r.search(query=core, top_k=5, candidate_k=15, version=None, only_latest=False, clean_query=False, min_rerank_score=0.0)
    hint = A.build_date_hint_block(r, dates) if release == "V3" and explicit is None else ""
    return core, hits, hint, mode


def prepare(args):
    A = load_public12()
    out = Path(args.out) / "contexts"
    out.mkdir(parents=True, exist_ok=True)
    work = Path(args.work)
    for release in ("V3", "V2"):
        stages = [s for s in PLAN if s[0] == release]
        if all(len(done_rows(out / f"{rel}_{name}.jsonl")) >= (args.limit or len(load_jsonl(QUESTION_DIR / QUESTIONS[(rel, name)])))
               for rel, name in stages):
            continue
        chunks = work / f"chunks_{release}"
        chunks.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / "data" / "rulebook_V2.jsonl", chunks / "data_v1_v2.jsonl")
        if release == "V3":
            shutil.copy2(ROOT / "data" / "rulebook_V3.jsonl", chunks / "data_v3.jsonl")
        r = A.RulebookRetriever(db_path=str(work / f"qdrant_{release}"), watch_dir=str(chunks), use_8bit=torch.cuda.is_available())
        latest, versions = A.get_latest_version(retriever=r, pdf_dir="/nonexistent", chunks_dir=str(chunks), md_dir="/nonexistent", fallback=release)
        if latest != release:
            raise SystemExit(f"{release} 색인의 최신 판본이 {latest}입니다.")
        old = [v for v in versions if v != latest]
        hint_old = f"(예: {', '.join(old)})" if old else ""
        for rel, name in stages:
            path = out / f"{rel}_{name}.jsonl"
            have = done_rows(path)
            todo = [q for q in load_jsonl(QUESTION_DIR / QUESTIONS[(rel, name)]) if str(q["id"]) not in have]
            todo = todo[:max(0, args.limit - len(have))] if args.limit else todo
            with open(path, "a", encoding="utf-8") as f:
                for n, q in enumerate(todo, 1):
                    t0 = time.perf_counter()
                    core, hits, hint, mode = retrieve(r, A, q["question"], rel, latest)
                    elapsed = round(time.perf_counter() - t0, 3)
                    system = A.SYSTEM_PROMPT_TEMPLATE.format(today=TODAY[rel], old_versions_hint=hint_old, latest_version=latest,
                                                             rule_4=A.build_rule_4(core))
                    context = r.format_context(hits) if hits else ""
                    user = f"[참고 근거]\n{context}\n\n{hint}\n\n[질문]\n{core}" if hint else f"[참고 근거]\n{context}\n\n[질문]\n{core}"
                    docs = [{"rank": h["rank"], "version": h["version"], "chunk_id": h["chunk_id"], "page": h["page"], "chapter": h.get("chapter", ""),
                             "topic": h.get("topic", ""), "articles": h.get("articles", []), "tables": h.get("tables", []),
                             "score": h.get("rerank_score"), "text": h["text"]} for h in hits]
                    append(f, {"id": str(q["id"]), "release": rel, "set": name, "question": q["question"], "core_question": core, "system": system,
                               "user": user if hits else None, "context": context, "docs": docs, "search_mode": mode, "retrieval_s": elapsed})
                    if n % 10 == 0 or n == len(todo):
                        print(f"[근거 {rel} {name}] {n}/{len(todo)}", flush=True)
        r.close()
        del r
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print("PREPARE_DONE", flush=True)


# ---------------------------------------------------------------- 모델
class HF:
    def __init__(self, name, path, nf4):
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        self.name, self.path = name, path
        self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(path)
        if nf4:
            if self.device != "cuda":
                raise SystemExit("32B NF4 실행에는 CUDA GPU가 필요합니다.")
            quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                                       bnb_4bit_compute_dtype=torch.bfloat16)
            self.model = AutoModelForCausalLM.from_pretrained(path, quantization_config=quant, device_map={"": 0})
        elif self.device == "cuda":
            self.model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map={"": 0})
        else:
            self.model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).to(self.device)
        self.model.eval()
        ids = {self.tok.convert_tokens_to_ids("<|im_end|>"), self.tok.convert_tokens_to_ids("<|endoftext|>")}
        self.stop = [i for i in ids if isinstance(i, int) and i >= 0]

    def chat(self, system, user, prefix="", on_text=None, watch=False):
        from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer
        from rules.guard import is_looping, trim_loop
        halt = threading.Event()

        class StopOnLoop(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                return torch.full((input_ids.shape[0],), halt.is_set(), device=input_ids.device, dtype=torch.bool)

        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        prompt = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = self.tok(prompt + prefix, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        n_in = inputs["input_ids"].shape[1]
        streamer = TextIteratorStreamer(self.tok, skip_prompt=True, skip_special_tokens=True)
        result = {}

        def work():
            try:
                with torch.inference_mode():
                    result["out"] = self.model.generate(**inputs, max_new_tokens=MAX_TOKENS, do_sample=False, temperature=None, top_p=None,
                                                        top_k=None, repetition_penalty=1.0, eos_token_id=self.stop,
                                                        pad_token_id=self.stop[0], streamer=streamer,
                                                        stopping_criteria=StoppingCriteriaList([StopOnLoop()]))
            except Exception as e:
                result["error"] = e
                streamer.end()

        t0, text, first, final = time.perf_counter(), "", None, None
        thread = threading.Thread(target=work, daemon=True)
        thread.start()
        for piece in streamer:
            if piece:
                text += piece
                now = time.perf_counter()
                if first is None and text.strip():
                    first = now
                if on_text:
                    on_text(piece)
                if final is None and re.search(r"\[\s*(?:최종\s*답변|단계별)", prefix + text):
                    final = now
                if watch and not halt.is_set() and "\n" in piece and is_looping(prefix + text):
                    halt.set()
        thread.join()
        if "error" in result:
            raise result["error"]
        out_ids = result["out"][0, n_in:]
        n_out = int(out_ids.shape[0])
        ended = n_out > 0 and int(out_ids[-1]) in self.stop
        full = trim_loop(prefix + text) if halt.is_set() else prefix + text
        return re.sub(r"<think>.*?</think>", "", full, flags=re.S).strip(), {
            "prompt_tokens": n_in, "completion_tokens": n_out, "finish_reason": "repeat" if halt.is_set() else ("stop" if ended else "length"),
            "started_at": t0, "first_answer_at": first, "final_at": final or (first and time.perf_counter())}


class Models:
    def __init__(self, m4, m32):
        self.paths, self.items = {"4b": (m4, False), "32b": (m32, True)}, {}
        self.total = torch.cuda.get_device_properties(0).total_memory / 2**30 if torch.cuda.is_available() else 0

    def get(self, name):
        if name not in self.items:
            if self.total < 36:
                self.items.clear()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            before = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            model = HF(name, *self.paths[name])
            model.loaded_bytes = (torch.cuda.memory_allocated() - before) if torch.cuda.is_available() else 0
            self.items[name] = model
        return self.items[name]

    def other_bytes(self, name):
        return sum(m.loaded_bytes for k, m in self.items.items() if k != name)


def split_answer(text):
    from graph.llm import _section, lead_verdict
    t = (text or "").strip()
    final = _section(t, r"최종\s*답변") or _section(t, r"단계별")
    if not final:
        final = re.split(r"\n\s*\**\[\s*참고\s*근거", t, maxsplit=1)[0]
    final = re.sub(r"\*\*|__", "", final).strip()
    refs = [re.sub(r"\s+", " ", m.group(0)) for m in re.finditer(r"(?:부칙\s*)?제\s*\d+\s*조|표\s*\d+", t)]
    reason = re.sub(r"\*\*|__", "", _section(t, r"핵심\s*(?:계산|확인)")).strip() or None
    return {"verdict": lead_verdict(final), "answer": final, "citations": list(dict.fromkeys(refs)), "reason": reason}


class FixedDocs:
    name = "public12 고정 근거"

    def __init__(self):
        self.row = None

    def solo_input(self):
        return self.row["context"], self.row["core_question"]

    def search(self, query, top_k=5, **_):
        return self.row["docs"][:top_k]


WORK_PREFIX = "[핵심 계산·조건 확인]\n"     # 질문 요약 없이 바로 풀이부터 시작한다


class NagamiLLM:
    fake = False
    uses_extra = True
    extra_text = ""

    def __init__(self, hf):
        self.hf, self.name, self.row, self.final_at = hf, f"{hf.name} (A100)", None, None

    def write(self, question, docs_text, calc_text, this_month, length=None, release="V3", on_text=None):
        user = self.row["user"] or f"[참고 근거]\n\n[질문]\n{self.row['core_question']}"
        if self.extra_text:
            user = user.replace("\n\n[질문]\n", f"\n\n[추가 근거]\n{self.extra_text}\n\n[질문]\n", 1)
        if calc_text:
            user = user.replace("\n\n[질문]\n", f"\n\n[코드 계산]\n{calc_text}\n\n[질문]\n", 1)
        self.last_user = user
        text, usage = self.hf.chat(self.row["system"], user, prefix=WORK_PREFIX, on_text=on_text, watch=True)
        self.final_at = usage.get("final_at")
        return split_answer(text), text, usage

    def rewrite(self, issue):
        user = self.last_user.replace("\n\n[질문]\n", f"\n\n[검사 결과]\n{issue}\n\n[질문]\n", 1)
        text, usage = self.hf.chat(self.row["system"], user, prefix="[최종 답변]\n", watch=True)
        self.final_at = usage.get("final_at")
        return split_answer(text), text, usage


def public_docs(row):
    return [{k: d.get(k) for k in ("rank", "version", "chunk_id", "page", "articles", "tables", "score")} for d in row["docs"]]


def answer_solo(hf, row):
    if not row["user"]:
        return {"raw_output": NO_HITS, "final_answer": NO_HITS, "citations": [], "verdict": None, "tokens_in": 0, "tokens_out": 0,
                "finish": "no_hits", "latency_s": row["retrieval_s"], "ttft_s": row["retrieval_s"], "conclusion_s": row["retrieval_s"], "answered_by": hf.name}
    text, u = hf.chat(row["system"], row["user"])
    parsed, end = split_answer(text), time.perf_counter()
    since = lambda t: round(row["retrieval_s"] + ((t or end) - u["started_at"]), 3)
    return {"raw_output": text, "final_answer": parsed["answer"], "citations": parsed["citations"], "verdict": parsed["verdict"],
            "tokens_in": u["prompt_tokens"], "tokens_out": u["completion_tokens"], "finish": u["finish_reason"],
            "latency_s": since(end), "ttft_s": since(u["first_answer_at"]), "conclusion_s": since(u["final_at"]), "answered_by": hf.name}


def answer_nagami(graphs, models, row):
    from graph.flow import Nagami
    if row["release"] not in graphs:
        llm, docs = NagamiLLM(models.get("4b")), FixedDocs()
        graphs[row["release"]] = (Nagami(row["release"], llm=llm, retriever=docs, top_k=5), llm, docs)
    nagami, llm, docs = graphs[row["release"]]
    llm.row, docs.row, llm.final_at = row, row, None
    t0 = time.perf_counter()
    out = nagami.ask(row["question"], mode="eval")
    extra = row["retrieval_s"]
    conclusion = (llm.final_at - t0 + extra) if llm.final_at else out["ttft_s"] + extra
    return {"raw_output": out.get("raw_output"), "final_answer": out.get("final_answer"), "citations": out.get("citations"), "verdict": out.get("verdict"),
            "version": out.get("version"), "base_date": out.get("base_date"), "calc": out.get("calc"), "reason": out.get("reason"),
            "calc_item": out.get("calc_item"), "checks": out.get("checks"), "format": out.get("format"), "tokens_in": out.get("tokens_in"),
            "tokens_out": out.get("tokens_out"), "finish": out.get("finish"), "latency_s": round(out["latency_s"] + extra, 3),
            "ttft_s": round(out["ttft_s"] + extra, 3), "conclusion_s": round(conclusion, 3), "answered_by": out.get("answered_by")}


def percentile(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else None


def write_meta(path, system, release, name, rows, peak_gib, hf):
    block = lambda key: {"median": round(st.median([r[key] for r in rows]), 3), "p95": percentile([r[key] for r in rows], .95),
                         "max": max(r[key] for r in rows)} if rows else None
    model_bytes = sum(p.stat().st_size for p in Path(hf.path).glob("*.safetensors"))
    meta = {"system": system, "doc_release": release, "set": name, "answered": len(rows), "model": hf.name, "model_path": str(hf.path),
            "revision": REV32 if hf.name == "32b" else None, "model_file_bytes": model_bytes, "peak_alloc_gib": peak_gib,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else hf.device, "today": TODAY[release],
            "generation": {"max_tokens": MAX_TOKENS, "temperature": 0, "do_sample": False, "thinking": False},
            "latency_s": block("latency_s"), "ttft_s": block("ttft_s"), "conclusion_s": block("conclusion_s"),
            "answered_by": {k: sum(str(r.get("answered_by")) == k for r in rows) for k in sorted({str(r.get("answered_by")) for r in rows})},
            "cut_by_length": [r["id"] for r in rows if r.get("finish") == "length"]}
    Path(path).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: meta[k] for k in ("system", "doc_release", "set", "answered", "latency_s", "ttft_s", "peak_alloc_gib", "cut_by_length")}, ensure_ascii=False), flush=True)


def run(args):
    out = Path(args.out)
    answers = out / "answers"
    answers.mkdir(parents=True, exist_ok=True)
    models, graphs = Models(args.m4, args.m32), {}
    systems = args.systems.split(",")
    for release, name in PLAN:
        rows = load_jsonl(out / "contexts" / f"{release}_{name}.jsonl")
        rows = rows[:args.limit] if args.limit else rows
        for system in SYSTEMS:
            if system not in systems:
                continue
            path = answers / f"{system}_{release}_{name}.jsonl"
            have = done_rows(path)
            todo = [r for r in rows if r["id"] not in have]
            print(f"[{system} {release} {name}] 전체 {len(rows)}, 이어서 {len(todo)}", flush=True)
            hf = models.get("4b" if system == "nagami" else system)
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            with open(path, "a", encoding="utf-8") as f:
                for n, row in enumerate(todo, 1):
                    base = {"id": row["id"], "release": release, "set": name, "system": system, "retrieved": public_docs(row),
                            "retrieval_s": row["retrieval_s"], "search_mode": row["search_mode"], "error": None}
                    try:
                        result = answer_nagami(graphs, models, row) if system == "nagami" else answer_solo(hf, row)
                        base.update(result)
                        have[row["id"]] = base
                    except Exception as e:
                        base["error"] = f"{type(e).__name__}: {e}"[:400]
                    append(f, base)
                    if n % 10 == 0 or n == len(todo) or base["error"]:
                        print(f"[{system} {release} {name}] {n}/{len(todo)} {base.get('latency_s', 'ERR')}s {base['error'] or ''}", flush=True)
            peak = None
            if torch.cuda.is_available() and todo:
                peak = round((torch.cuda.max_memory_allocated() - models.other_bytes(hf.name)) / 2**30, 3)
            done = [have[r["id"]] for r in rows if r["id"] in have]
            meta_path = answers / f"{system}_{release}_{name}.meta.json"
            if todo or not meta_path.exists():
                write_meta(meta_path, system, release, name, done, peak, hf)
    print("A100_RUN_DONE", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["prepare", "run"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", default="/content/work")
    ap.add_argument("--m4")
    ap.add_argument("--m32")
    ap.add_argument("--systems", default="nagami,4b,32b")
    ap.add_argument("--limit", type=int)
    ns = ap.parse_args()
    if ns.command == "prepare":
        prepare(ns)
    else:
        if "4b" in ns.systems or "nagami" in ns.systems:
            assert ns.m4, "--m4가 필요합니다"
        if "32b" in ns.systems:
            assert ns.m32, "--m32가 필요합니다"
        run(ns)
