"""나감이 — 나미와 감식이의 공동 생활 지킴이 (Streamlit 화면)

    cd NaGamEE_keeper
    NAGAMI_M4=<4B 모델 폴더> streamlit run app/streamlit_app.py

답은 LangGraph 흐름(graph/flow.py)이 만든다.
  - 답 쓰기 모델: NAGAMI_M4 의 4B AWQ 모델 (app/live.py)
  - 검색: A의 public12 검색기 (retriever/)
날짜가 빠지면 그래프가 멈추고(interrupt) 되묻는다. 날짜를 고르면 이어서 판정한다.
"""
import base64
import html
import json
import os
import random
import sys
import time
from datetime import date
from pathlib import Path

import streamlit as st

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR.parent))     # graph, rules, evaluation
sys.path.insert(0, str(APP_DIR))            # hearts, live
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import hearts  # noqa: E402

st.set_page_config(page_title="나감이", page_icon="💗", layout="centered")

CARD = "#EBD3A9"      # 나감이 말풍선
USER = "#F5F1EA"      # 내 말풍선
MAGENTA = "#A3168A"
CHARCOAL = "#4A4440"

st.markdown(f"""
<style>
.nk-head {{display:flex;align-items:center;gap:14px;margin:0 0 8px}}
.nk-name {{margin:0;font-size:26px;font-weight:600;color:{MAGENTA}}}
.nk-family {{display:flex;align-items:flex-end;gap:0}}
.nk-heart {{margin:0 -4px 6px;display:inline-flex}}
.nk-sub {{margin:0;font-size:14px;opacity:.75}}
.nk-row {{display:flex;gap:10px;align-items:flex-start;margin:0 0 14px}}
.nk-row.me {{justify-content:flex-end}}
.nk-bub {{background:{CARD};border-radius:14px;padding:10px 14px;font-size:15px;line-height:1.65;max-width:85%}}
.nk-row.me .nk-bub {{background:{USER};border:1px solid #DCD3C4}}
.nk-say {{margin:0 0 8px}}
.nk-chip {{display:inline-block;font-size:12.5px;padding:3px 10px;border-radius:999px;margin:0 4px 4px 0;
          background:#FFFFFFAA;color:{CHARCOAL}}}
.nk-k {{display:inline-block;min-width:66px;font-size:13px;opacity:.7}}
.nk-line {{font-size:13.5px;line-height:1.8}}
.nk-fake {{font-size:11px;opacity:.6;margin-top:6px}}
.nk-legend {{display:flex;flex-wrap:wrap;gap:8px 5%;justify-content:flex-start}}
.nk-legend div {{display:flex;flex-direction:column;align-items:center;font-size:11.5px;width:30%}}
.nk-empty {{text-align:center;opacity:.8;margin:24px 0 10px}}
.nk-quote {{margin-top:10px;padding-top:9px;border-top:1px solid #0002;font-size:13.5px;line-height:1.6}}
.nk-quote span {{display:block;font-size:11.5px;opacity:.65;margin-top:2px}}
</style>
""", unsafe_allow_html=True)

# 바탕 캐릭터: 큰 나감이를 연하게 오른쪽 아래에 깔기 (SVG를 코드에 직접 넣어 인터넷 없이 보임)
BG_SVG = None          # 넣을 캐릭터가 정해지면 SVG 문자열로 채운다 (지금은 꺼 둠)
BG_OPACITY = 0.15
if BG_SVG:
    _bg_svg = BG_SVG.replace("<svg ", f'<svg opacity="{BG_OPACITY}" ', 1)
    _bg_uri = "data:image/svg+xml;base64," + base64.b64encode(_bg_svg.encode("utf-8")).decode()
if BG_SVG: st.markdown(f"""
<style>
[data-testid="stAppViewContainer"] {{
  background-image: url("{_bg_uri}");
  background-repeat: no-repeat;
  background-position: right 40px bottom 120px;
  background-size: 260px;
  background-attachment: fixed;
}}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ 상태
ss = st.session_state
ss.setdefault("messages", [])     # {"role": "user"|"bot", "text": str, "res": dict|None}
ss.setdefault("pending", None)    # 되묻기 중이면 {"question", "field"}
ss.setdefault("release", "V3")    # 화면에서 고른 규약집 판본
QUOTES = json.loads((Path(__file__).resolve().parent / "ui_quotes.json").read_text(encoding="utf-8"))


@st.cache_resource(show_spinner="나감이가 규약집을 펼치는 중이에요…")
def get_live():
    from live import Live
    return Live()


def get_nagami():
    return get_live().get(ss.release)


def mood_now():
    for m in reversed(ss.messages):
        if m["role"] == "bot" and m.get("res"):
            r = m["res"]
            if r["type"] == "ask":
                return "ask"
            return hearts.mood_of(r["verdict"]) if r.get("verdict") else "ok"
    return "ok"


def esc(s):
    return html.escape(str(s))


def bot_html(res):
    """나감이 말풍선 한 칸."""
    if res["type"] == "ask":
        inner = f'<p class="nk-say" style="margin:0">{esc(res["say"])}</p>'
        mood = "ask"
    elif res.get("error"):
        mood = "none"
        inner = f'<p class="nk-say" style="margin:0">답을 만들다 문제가 생겼어요. {esc(res["error"])}</p>'
    else:
        v = res.get("verdict")
        chips = ""
        if v:
            fill, line = hearts.colors_of(v)
            chips += f'<span class="nk-chip" style="background:{fill};color:{line};font-weight:600">{esc(v)}</span>'
        if res.get("version"):
            chips += f'<span class="nk-chip">판본 {esc(res["version"])}</span>'
        if res.get("base_date"):
            chips += f'<span class="nk-chip">기준일 {esc(res["base_date"])}</span>'
        lines = ""
        for k, label in (("calc", "계산"), ("reason", "이유"), ("citations", "근거"), ("next_action", "다음 행동")):
            val = res.get(k)
            if isinstance(val, list):
                val = " · ".join(val)
            if (k == "reason" and val == res.get("calc")) or (k in ("calc", "reason") and val == res.get("final_answer")):
                continue
            if val:
                lines += f'<div><span class="nk-k">{label}</span>{esc(val)}</div>'
        inner = (f'<p class="nk-say">{esc(res.get("final_answer") or "-")}</p>'
                 + (f"<div>{chips}</div>" if chips else "")
                 + f'<div class="nk-line">{lines}</div>')
        if res.get("checks"):
            inner += '<div class="nk-fake">검사: ' + esc(" / ".join(res["checks"])) + "</div>"
        if res.get("fake"):
            inner += '<div class="nk-fake">가짜 모델 답이에요 (32B 서버 연결 전). 판본·계산은 코드가 했어요.</div>'
        if res.get("ttft_s") is not None:
            inner += (f'<div class="nk-fake">서버 기준 첫 토큰 {res["ttft_s"]:.2f}초 · 완료 {res["latency_s"]:.2f}초 · '
                      f'문서 {esc(res.get("doc_release", ""))} · 답변 {esc(res.get("answered_by", ""))}</div>')
        if res.get("quote"):
            inner += (f'<div class="nk-quote">{esc(res["quote"]["text"])}'
                      f'<span>{esc(res["quote"]["source"])}</span></div>')
        mood = hearts.mood_of(v) if v else "ok"
    if res["type"] == "answer" and res.get("verdict"):
        face = hearts.verdict_heart(res["verdict"], 40)
    else:
        face = hearts.nagami(mood, 40)
    return f'<div class="nk-row">{face}<div class="nk-bub">{inner}</div></div>'


def user_html(text):
    return f'<div class="nk-row me"><div class="nk-bub">{esc(text)}</div></div>'


def _call(fn, *args, **kw):
    try:
        res = fn(*args, **kw)
    except Exception as e:
        return {"type": "answer", "error": f"{type(e).__name__}: {e}"}
    res["fake"] = getattr(get_nagami().llm, "fake", False)
    res["doc_release"] = ss.release
    if res.get("type") == "answer" and not res.get("error"):
        res["quote"] = random.choice(QUOTES)      # 답이 끝난 뒤 화면에만 붙인다. 모델과 채점에는 들어가지 않는다
    return res


def live_html(text):
    return (f'<div class="nk-row">{hearts.nagami("ok", 40)}'
            f'<div class="nk-bub" style="white-space:pre-wrap">{esc(text)}</div></div>')


def _streamed(fn, *args, waiting, **kw):
    """4B의 풀이를 글자가 나오는 대로 보여 주고, 끝나면 정리된 말풍선으로 바꾼다."""
    live = st.empty()
    live.markdown(live_html(waiting), unsafe_allow_html=True)
    text, shown_at = [], [0.0]

    def on_text(piece):
        text.append(piece)
        now = time.perf_counter()
        if now - shown_at[0] >= 0.05:
            shown_at[0] = now
            live.markdown(live_html("".join(text)), unsafe_allow_html=True)

    res = _call(fn, *args, on_text=on_text, **kw)
    live.empty()
    return res


def run(question):
    """질문을 보내고 결과를 대화에 붙인다. 날짜가 빠지면 되묻기 상태가 된다."""
    ss.messages.append({"role": "user", "text": question})
    st.markdown(user_html(question), unsafe_allow_html=True)
    get_nagami().retriever.question = question
    res = _streamed(get_nagami().ask, question, mode="service", waiting="규약집 넘기는 중이에요…")
    ss.messages.append({"role": "bot", "res": res})
    ss.pending = res if res["type"] == "ask" else None


def resume(value, shown):
    """되묻기에 답하면 멈춘 그래프를 이어서 돌린다."""
    ss.messages.append({"role": "user", "text": shown})
    st.markdown(user_html(shown), unsafe_allow_html=True)
    res = _streamed(get_nagami().resume, ss.pending["thread_id"], value, waiting="판본을 고르고 계산하는 중이에요…")
    ss.messages.append({"role": "bot", "res": res})
    ss.pending = res if res["type"] == "ask" else None


# ------------------------------------------------------------------ 옆 메뉴
with st.sidebar:
    st.markdown(f'<div class="nk-head">{hearts.nagami("ok", 40)}'
                f'<p class="nk-name" style="font-size:20px">나감이</p></div>', unsafe_allow_html=True)
    picked = st.radio("규약집", ["V3", "V2"], index=0 if ss.release == "V3" else 1, horizontal=True,
                      help="V2는 오늘을 2026-11-01, V3는 2026-12-01로 봅니다.")
    if picked != ss.release:
        ss.release, ss.messages, ss.pending = picked, [], None
        st.rerun()
    nag = get_nagami()
    if getattr(nag.llm, "fake", False):
        st.warning("답 쓰기: 가짜 모델 (32B 서버 연결 전)")
    else:
        st.success(f"답 쓰기: {nag.llm.name}")
    st.caption(f"검색: {nag.retriever.name} · 문서 {nag.doc_release}")
    st.markdown("**판본**")
    _vs = nag.reg.data["versions"]
    st.markdown("\n".join(
        f"- {v['id']} · {v['start'][:10]} ~ " + (f"{v['end'][:10]} 직전" if v['end'] else "0시부터")
        for v in _vs))
    st.markdown("**판정 표정**")
    st.markdown('<div class="nk-legend">' + "".join(
        f'<div>{hearts.verdict_heart(v, 40)}{v}</div>' for v in hearts.VERDICTS) + "</div>",
        unsafe_allow_html=True)
    st.write("")
    if st.button("대화 지우기", use_container_width=True):
        ss.messages, ss.pending = [], None
        st.rerun()

# ------------------------------------------------------------------ 머리
st.markdown(f'<div class="nk-head">{hearts.nagami(mood_now(), 64)}<div>'
            f'<p class="nk-name">나감이네</p>'
            f'<p class="nk-sub">나미와 감식이의 공동 생활 지킴이</p></div></div>',
            unsafe_allow_html=True)

# ------------------------------------------------------------------ 대화
if not ss.messages:
    st.markdown('<p class="nk-empty">나미와 감식이의 공동 생활 지킴이, 나감이예요.<br>'
                '무엇을 확인할까요?</p>', unsafe_allow_html=True)
    examples = [
        "어머님 생신 식사 18만 원, 선물 6만 원 냈어. 사전 합의도 했어. 공동비로 돼?",
        "2026-10-05에 세탁기 35만 원 샀어. 공동비 돼?",
        "고양이 사료값도 공동비로 낼 수 있어?",
    ]
    cols = st.columns(3)
    for c, ex in zip(cols, examples):
        if c.button(ex, use_container_width=True):
            run(ex)
            st.rerun()

for m in ss.messages:
    st.markdown(user_html(m["text"]) if m["role"] == "user" else bot_html(m["res"]),
                unsafe_allow_html=True)

# 되묻기: 기준일 고르기 (그래프가 멈춘 상태)
if ss.pending and ss.pending.get("field") == "event_date":
    label = ss.pending.get("say", "").replace("판정 전에 하나만요. ", "").replace("이 언제예요?", "")
    with st.form("ask_date", border=False):
        c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
        d = c1.date_input(label or "기준일", value=date(2026, 10, 1),
                          min_value=date(2026, 1, 1), max_value=date(2027, 12, 31))
        if c2.form_submit_button("이 날짜예요", use_container_width=True):
            resume(d.isoformat(), f"{label or '기준일'}은 {d.isoformat()}이에요.")
            st.rerun()

if q := st.chat_input("나감이에게 물어보세요 (예: 생신 선물 20만 원 공동비 돼?)"):
    run(q)
    st.rerun()
