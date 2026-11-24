"""답 쓰는 모델 연결부. 코드가 못 푼 문제만 4B가 풀이를 먼저 쓰고 답한다.

  write(질문, 근거, …) → 풀이를 글자 단위로 내보내고, 최종 답·이유·근거로 나눈다

연결 방법 (환경변수)
  NAGAMI_LLM_BASE_URL   예: http://localhost:8000/v1  (vLLM 등 OpenAI 형식 서버, SSH 터널)
  NAGAMI_LLM_MODEL      예: qwen3-32b
  NAGAMI_LLM_LOCAL      예: ~/Downloads/나감이네_C담당/models/4b_awq_dequant_bf16  (맥에서 직접 실행, transformers + MPS)
  설정이 없거나 서버가 꺼져 있으면 가짜 모델(FakeLLM)을 쓴다. 화면에 '가짜 모델'이라고 표시한다.
"""
import json
import os
import re
import threading
import time
import urllib.request

STATUSES = ["가능", "사전 합의 필요", "공동 처리 불가", "추가 확인", "규정에 없음"]

WORKED_MAX_TOKENS = 1100     # 4B 단독 공개 V3의 최대 출력 1030토큰이 잘리지 않는 값

# 4B 단독 평가(run_same.py)와 같은 지시문. 풀이를 먼저 써야 4B의 정확도와 근거 인용이 유지된다
WORKED_SYSTEM = """당신은 부부/가구 생활 규약집 전문 안내 도우미입니다.
반드시 아래 [참고 근거]에 적힌 내용만 사용하여 정확하게 답변하세요.
오늘 날짜는 {today}입니다.

[답변 규칙]
1. 근거에 없는 숫자나 규정은 절대 추측하거나 지어내지 마세요.
2. 사용자가 과거 판본{old_versions_hint}을 명시하지 않았으면, 가장 최신 판본({latest_version}) 기준을 우선 적용하세요.
3. 날짜가 걸치는 사건(개정 전후, 야간근무 등)은 부칙의 '기준일(최초 확정일·시작일)'과 '경과조치'를 먼저 확인해 적용 판본을 밝히세요.
4. 질문 요약을 길게 쓰거나 같은 결론을 반복하지 말고, **[핵심 계산·조건 확인] 3~5줄 → [최종 답변] → [참고 근거(판본·조항·표·쪽)]** 순서로 간결하고 정확하게 답하세요. 계산 전 첫 단어부터 성급하게 '예/아니오'를 쓰지 마세요.
5. 덧셈·뺄셈·비교는 반드시 산식을 먼저 적고 비교하세요.
6. 판단에 필요한 핵심 값(확정일·박수·합의 등)이 빠졌으면 억지로 단정 짓지 말고 추가 확인이 필요함을 명확히 밝히세요.
7. [참고 근거]만으로 판단이 불가능하면 "규약집 근거에서 확인할 수 없습니다"라고 답하세요."""
RELEASE_PROMPT = {
    "V2": {"today": "2026-11-01", "old_versions_hint": "(예: V1)", "latest_version": "V2"},
    "V3": {"today": "2026-12-01", "old_versions_hint": "(예: V2)", "latest_version": "V3"},
}


def write_limit(length=None):
    """보통 답은 풀이까지 1100토큰. '300~600자'처럼 길이를 요구하면 그 글자 수와 근거가 들어갈 만큼만 쓴다."""
    if not length:
        return WORKED_MAX_TOKENS
    return max(300, round((length[1] * 0.714 + 100) / 10) * 10)


LONG_SYSTEM = """너는 '나감이', 나미와 감식이의 공동 생활 지킴이다. 공손한 상담사 말투로 쓴다.
반드시 [근거]에 있는 내용만 쓴다. [코드 계산]이 있으면 그 판본·기준·계산 결과를 그대로 따른다.
연도가 없으면 2026년, 금액 단위가 없으면 만 원, '이번 달'은 {this_month}이다.
공백 포함 {low}~{high}자의 일반 글로 쓴다. JSON, 코드블록, 굵은 글씨, 대괄호 제목은 쓰지 않는다.
먼저 결론을 한 문장으로 쓴다. 이어서 1. 2. 3. 4. 번호를 붙여 질문이 요구한 항목을 차례로 다룬다. 번호마다 두 문장씩 쓴다.
'첫 번째 확인' 같은 제목은 쓰지 않고 바로 내용을 쓴다.
마지막 줄에 '근거: 제23조, 표 54'처럼 실제로 쓴 조항과 표만 적는다."""


def parse_long(text, length):
    """긴 답은 일반 글로 받는다. 근거 줄을 분리하고, 요청 길이를 넘으면 문장 끝에서 자른다."""
    t = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()
    parts = re.split(r"\n\s*\**근거\**\s*[:：]", t, maxsplit=1)
    answer = re.sub(r"\*\*|^\s*\[(?:결론|답)\]\s*", "", parts[0], flags=re.M).strip()
    source = parts[1] if len(parts) > 1 else answer
    citations = [re.sub(r"\s+", " ", m.group(0)) for m in re.finditer(r"(?:부칙\s*)?제\s*\d+\s*조|표\s*\d+", source)]
    if len(answer) > length[1]:
        ends = [m.end() for m in re.finditer(r"[.!?。](?=\s|$)", answer) if m.end() <= length[1]]
        if ends:
            answer = answer[:ends[-1]].strip()
    return {"verdict": None, "answer": answer, "citations": list(dict.fromkeys(citations)), "reason": None}


def _section(text, name):
    m = re.search(r"\[\s*" + name + r"[^\]\n]*\]\**\s*[:：]?(.*?)(?=\n\s*\**\[[^\]\n]{2,40}\]|\Z)", text, re.S)
    return m.group(1).strip() if m and m.group(1).strip() else ""


def lead_verdict(final):
    first = re.split(r"(?<=[.!?。])\s|\n", str(final or "").strip(), maxsplit=1)[0]
    if len(first) > 40:
        return None
    t = first.replace("사전 합의가 필요", "사전 합의 필요")
    t = re.sub(r"공동\s*처리(?:는|가)?\s*불가", "공동 처리 불가", t)
    for status in ("공동 처리 불가", "사전 합의 필요", "추가 확인", "규정에 없음"):
        if status in t:
            return status
    return "가능" if re.search(r"(^|\s)가능(\s|$|[.,。])", t) else None


def parse_worked(text):
    """[핵심 계산] → [최종 답변] → [참고 근거] 글을 나눈다. 칸이 없으면 전체를 답으로 본다."""
    t = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()
    final = re.sub(r"\*\*|__", "", _section(t, r"최종\s*답변") or t).strip()
    source = _section(t, r"참고\s*근거") or t
    refs = [re.sub(r"\s+", " ", m.group(0)) for m in re.finditer(r"(?:부칙\s*)?제\s*\d+\s*조|표\s*\d+", source)]
    return {"verdict": lead_verdict(final), "answer": final, "citations": list(dict.fromkeys(refs)),
            "reason": re.sub(r"\*\*|__", "", _section(t, r"핵심\s*계산")).strip() or None}


def run_write(chat, question, docs_text, calc_text, this_month, release, length, on_text):
    code = f"\n\n[코드 계산]\n{calc_text}" if calc_text else ""
    if length:
        system = LONG_SYSTEM.format(this_month=this_month, low=length[0], high=length[1])
        text, usage = chat(system, f"[근거]\n{docs_text}{code}\n\n[질문]\n{question}", write_limit(length), on_text)
        return parse_long(text, length), text, usage
    system = WORKED_SYSTEM.format(**RELEASE_PROMPT.get(release, RELEASE_PROMPT["V3"]))
    text, usage = chat(system, f"[참고 근거]\n{docs_text}{code}\n\n[질문]\n{question}", WORKED_MAX_TOKENS, on_text)
    return parse_worked(text), text, usage


def _post_stream(base, payload, timeout=180, on_text=None):
    """글자 단위로 받아 첫 글자가 나온 시각을 재고, 화면에 바로 넘긴다."""
    body = {**payload, "stream": True, "stream_options": {"include_usage": True}}
    req = urllib.request.Request(base.rstrip("/") + "/chat/completions",
                                 data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    text, usage, first, finish = "", {}, None, None
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            event = json.loads(data)
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices") or []:
                piece = (choice.get("delta") or {}).get("content") or ""
                if piece:
                    text += piece
                    if first is None and text.strip():
                        first = time.perf_counter()
                    if on_text:
                        on_text(piece)
                finish = choice.get("finish_reason") or finish
    return text, {**usage, "finish_reason": finish, "first_answer_at": first}


class ServerLLM:
    """OpenAI 형식 서버 (vLLM 등). 32B·7B·3B 모두 주소만 바꿔 쓴다."""

    def __init__(self, base, model):
        self.base, self.model = base, model
        self.name = f"{model} @ {base}"
        self.fake = False

    def _chat(self, system, user, max_tokens=512, on_text=None):
        return _post_stream(self.base, {
            "model": self.model, "temperature": 0, "max_tokens": max_tokens,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "chat_template_kwargs": {"enable_thinking": False},
        }, on_text=on_text)

    def write(self, question, docs_text, calc_text, this_month, length=None, release="V3", on_text=None):
        return run_write(self._chat, question, docs_text, calc_text, this_month, release, length, on_text)


class LocalLLM:
    """이 맥에서 모델을 직접 돌린다 (transformers, Apple GPU=MPS). 서버 없이 4B 확인용.
    vLLM 서버와 같은 조건: greedy(온도 0), thinking 끔, 같은 지시문."""

    def __init__(self, path):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.path = os.path.expanduser(path)
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(self.path)
        self.m = AutoModelForCausalLM.from_pretrained(self.path, dtype=torch.bfloat16).to(dev).eval()
        self.dev, self.torch, self.lock = dev, torch, threading.Lock()
        self.name = f"{os.path.basename(self.path.rstrip('/'))} (이 맥 {dev})"
        self.fake = False
        src = os.path.join(self.path, "출처.txt")
        import hashlib
        import platform
        import subprocess
        st = os.path.join(self.path, "model.safetensors")
        chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
        self.meta = {"model_path": self.path, "engine": f"transformers (local {dev})", "precision": "bf16 (4bit AWQ를 푼 값)",
                     "checkpoint": f"model.safetensors sha {hashlib.sha256(open(st, 'rb').read()).hexdigest()[:16]}" if os.path.exists(st) else None,
                     "gpu": f"{chip or platform.machine()} ({dev})",
                     "system_prompt": WORKED_SYSTEM,
                     "source_note": open(src, encoding="utf-8").read().strip() if os.path.exists(src) else None}

    def _chat(self, system, user, max_tokens=512, on_text=None):
        from transformers import TextIteratorStreamer
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        ids = self.tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False,
                                           return_tensors="pt", return_dict=True).to(self.dev)
        n_in = ids["input_ids"].shape[1]
        streamer = TextIteratorStreamer(self.tok, skip_prompt=True, skip_special_tokens=True)
        result = {}

        def generate():
            try:
                with self.torch.inference_mode():
                    result["out"] = self.m.generate(**ids, max_new_tokens=max_tokens, do_sample=False, temperature=None,
                                                    top_p=None, top_k=None, streamer=streamer)
            except Exception as e:
                result["error"] = e
                streamer.end()

        text, first = "", None
        with self.lock:
            worker = threading.Thread(target=generate, daemon=True)
            worker.start()
            for piece in streamer:
                if piece:
                    text += piece
                    if first is None and text.strip():
                        first = time.perf_counter()
                    if on_text:
                        on_text(piece)
            worker.join()
        if "error" in result:
            raise result["error"]
        n_out = result["out"].shape[1] - n_in
        return text, {"prompt_tokens": n_in, "completion_tokens": n_out, "first_answer_at": first,
                      "finish_reason": "length" if n_out >= max_tokens else "stop"}

    def write(self, question, docs_text, calc_text, this_month, length=None, release="V3", on_text=None):
        return run_write(self._chat, question, docs_text, calc_text, this_month, release, length, on_text)


class FakeLLM:
    """서버가 없을 때 쓰는 가짜 모델. 규칙으로 값을 뽑고, 정해진 틀로 답한다 (화면·흐름 시험용)."""
    name = "가짜 모델 (32B 서버 연결 전)"
    fake = True

    ITEMS = [("birthday_cap", r"생신|생일"), ("appliance_prior_agreement", r"가전|청소기|밥솥|공기청정기|세탁기|냉장고|선풍기"),
             ("domestic_lodging_cap", r"호텔|국내\s*숙박|박당|공동숙박"), ("stock_add_buy_stage1", r"추가\s*매수|추가매수|물타기|추가 주문")]
    EXCLUDE = r"가용액|예산|잔액|한도|상한|기준|포인트|캐시백|비상금|계좌|보증금|대출|전체|총 예산"

    def write(self, question, docs_text, calc_text, this_month, calc=None, docs=None, fmt=None):
        verdict = (calc or {}).get("verdict")
        cites = []
        for d in (docs or [])[:2]:
            cites += [a.split(" ")[0] for a in d.get("articles", [])[:1]] + [f"{d['page']}쪽"]
        if verdict:
            answer = {"가능": "가능해요.", "사전 합의 필요": "사전 합의가 먼저 필요해요.",
                      "공동 처리 불가": "안 돼요.", "추가 확인": "판정 전에 확인할 게 있어요."}.get(verdict, verdict)
            reason = calc.get("text", "")
        else:
            top = (docs or [{}])[0]
            answer = f"(가짜 모델) {top.get('topic', '관련 규정')}을 확인해 보세요."
            reason = "32B 서버가 연결되면 실제 답을 써요."
        out = {"verdict": verdict, "answer": answer, "reason": reason, "citations": cites,
               "next_action": (calc or {}).get("next_action") or "관련 조항을 확인해 주세요."}
        return out, json.dumps(out, ensure_ascii=False), {}


def server_ready(base):
    try:
        urllib.request.urlopen(base.rstrip("/") + "/models", timeout=2)
        return True
    except Exception:
        return False


def make_llm():
    local = os.environ.get("NAGAMI_LLM_LOCAL")
    if local:
        return LocalLLM(local)
    base = os.environ.get("NAGAMI_LLM_BASE_URL")
    if base and server_ready(base):
        return ServerLLM(base, os.environ.get("NAGAMI_LLM_MODEL", "qwen3-32b"))
    return FakeLLM()
