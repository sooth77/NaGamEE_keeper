"""V3에서 바뀐 규칙 중 금액 합계만으로 끝나지 않는 5개를 코드로 판단한다.

  월 원리금 상한 · 개별주 비중 · 추가매수 대기 · 손님 숙박 고지 · 금·토 귀가

원칙
- 값은 코드에 적지 않고 판본 표(versions_*.json)에서 읽는다. 공통 수치만 규약 근거와 함께 상수로 둔다.
- 값은 작은 모델이 아니라 질문 글에서 규칙으로 읽는다.
- 판단에 필요한 값이 하나라도 없으면 결론을 내지 않고 그 판본의 기준값만 모델에 알려준다.
- 이 규칙들의 결과는 충족·보류·성립 같은 말이라 5가지 판정(verdict)에 넣지 않고 decision으로 돌려준다.
"""
import re
import unicodedata
from datetime import date
from decimal import Decimal

from rules.tools import pick_event_date
from rules.versions import BEFORE, UNKNOWN

LTV_LIMIT = Decimal(60)            # 제49조
STOCK_ORDER_MAX = Decimal(80)      # 제61조·표 57: 종목당 80만 원
STOCK_ALL_LIMIT = Decimal(20)      # 표 33: 개별주 전체 20%
CASH_FLOOR = Decimal(15)           # 표 33: 주문 후 현금 15%
STAGE2_DROP = Decimal(20)          # 표 57
STOP_DROP = Decimal(25)            # 제60조
NUM = r"(\d[\d,]*(?:\.\d+)?)"
CLOCK = r"(\d{1,2})\s*시(?:\s*(\d{1,2})\s*분)?"


def _text(question):
    return re.sub(r"[ \t]+", " ", unicodedata.normalize("NFKC", question or ""))


def _dec(x):
    return Decimal(str(x).replace(",", ""))


def _fmt(x):
    s = format(Decimal(x).normalize(), "f")
    return s


def _money(pattern, text):
    m = re.search(pattern + r"\s*" + NUM + r"\s*만", text)
    return _dec(m.group(1)) if m else None


def _pct(part, whole):
    return (part / whole * 100).quantize(Decimal("0.01"))


def detect(question):
    """질문이 어느 개정 규칙을 묻는지 (판본 표의 항목 키). 없으면 None."""
    t = _text(question)
    if re.search(r"원리금|월\s*상환액", t) and "실수령" in t and "심사" in t:
        return "housing_monthly_cap"
    if "거래일" in t and re.search(r"추가\s*(주문|매수)", t):
        return "stock_add_buy_wait"
    if re.search(r"계좌", t) and re.search(r"종목", t) and re.search(r"만\s*원\s*(추가\s*)?(주문|매수)", t):
        return "stock_single_cap"
    if re.search(r"[1-3]\s*박", t) and re.search(r"\d+\s*시간\s*전", t) and "고지" in t and "퇴원" not in t:
        return "guest_stay_notice"
    if "귀가" in t and "연장" in t and "동의" in t and re.search(r"회식|술|음주|모임", t):
        return "drinking_return_fri_sat"
    return None


DATE_WORDS = {
    "housing_monthly_cap": ("심사",), "stock_add_buy_wait": ("제출", "주문"), "stock_single_cap": ("제출", "주문", "점검"),
    "guest_stay_notice": ("시작", "숙박"), "drinking_return_fri_sat": ("시작", "회식", "모임"),
}


def pick_date(text, key, fallback=None):
    """이 규칙의 기준일. 날짜가 여럿이면 기준 단어에 가장 가까운 날짜를 쓰고, 못 정하면 None."""
    return pick_event_date(text, key, fallback)


def _agreed(t):
    if re.search(r"미합의|합의\s*(는|가|도)?\s*(없|안\s*했|아직)|동의\s*(는|가|도)?\s*없", t):
        return False
    if re.search(r"합의\s*(는|가|도|를)?\s*(있|했|함|완료|받)|합의를 갖", t):
        return True
    return None


def _housing(t, value, ver):
    rate = _dec(value["비율"])
    income = _money(r"실수령(?:액)?", t)
    m = re.search(r"월\s*상환액[^\d\n]{0,30}" + NUM + r"\s*만", t)
    new_pay = _dec(m.group(1)) if m else None
    old_pay = _money(r"기존\s*(?:월\s*)?(?:원리금|상환액)", t)
    if old_pay is None and re.search(r"기존\s*(대출|원리금|상환액)?\s*(은|는|이|가)?\s*없", t):
        old_pay = Decimal(0)
    ref = f"{ver} 월 원리금 상한: 실수령액의 {_fmt(rate)}% (기존+신규 잔존 원리금 합계로 비교), 담보 {LTV_LIMIT}% 이내"
    if income is None or new_pay is None or old_pay is None:
        return ref, None, None
    cap, total = income * rate / 100, old_pay + new_pay
    month_ok = total <= cap
    parts, ok = [], month_ok
    price, loan = _money(r"매매가", t), _money(r"신규\s*대출", t)
    if price and loan:
        ltv = _pct(loan, price)
        ltv_ok = loan * 100 <= price * LTV_LIMIT
        ok = ok and ltv_ok
        parts.append(f"담보비율 {_fmt(ltv)}%로 {LTV_LIMIT}%{' 이내예요' if ltv_ok else '를 넘어요'}.")
    parts.append(f"월 원리금 {_fmt(old_pay)} + {_fmt(new_pay)} = {_fmt(total)}만 원은 {ver} 상한 {_fmt(cap)}만 원"
                 f"(실수령 {_fmt(income)}만 원의 {_fmt(rate)}%){' 이내예요' if month_ok else '을 넘어요'}.")
    parts.append("내부 수치 요건을 통과해요." if ok else "내부 심사 요건을 통과하지 못해요.")
    return ref, " ".join(parts), ok


def _stock(t, reg, ver, key):
    wait = reg.value("stock_add_buy_wait", ver)
    single = _dec(reg.value("stock_single_cap", ver))
    stage1 = _dec(reg.value("stock_add_buy_stage1", ver))
    order_rate = _dec(reg.value("stock_stage_order_limit", ver))
    ref = (f"{ver} 투자 기준: 1단계 {_fmt(stage1)}% 이상 하락, 직전 매수 뒤 {wait}거래일, 주문 {_fmt(order_rate)}% 이하, "
           f"한 종목 {_fmt(single)}%와 {STOCK_ORDER_MAX}만 원 중 작은 값, 개별주 전체 {STOCK_ALL_LIMIT}%, 주문 후 현금 {CASH_FLOOR}% 이상")
    account = _money(r"계좌(?:\s*총액)?(?:은|는|이)?", t)
    m = re.search(NUM + r"\s*만\s*원?\s*(?:을|의)?\s*(?:추가\s*)?(?:주문|매수)", t)
    order = _dec(m.group(1)) if m else None
    target = _money(r"(?:대상\s*)?종목(?:\s*평가액)?(?:은|는|이)?", t)
    if account is None or order is None or target is None:
        return ref, None, None

    parts, fails = [], []
    limit = min(account * single / 100, STOCK_ORDER_MAX)
    after = target + order
    (parts if after <= limit else fails).append(f"주문 후 종목 {_fmt(after)}만 원은 {ver} 한도 {_fmt(limit)}만 원 {'이내' if after <= limit else '초과'}")
    all_stock, cash = _money(r"개별주\s*전체", t), _money(r"현금", t)
    if all_stock is not None:
        a = all_stock + order
        cap = account * STOCK_ALL_LIMIT / 100
        (parts if a <= cap else fails).append(f"개별주 전체 {_fmt(a)}만 원은 {_fmt(cap)}만 원 {'이내' if a <= cap else '초과'}")
    if cash is not None:
        c = cash - order
        floor = account * CASH_FLOOR / 100
        (parts if c >= floor else fails).append(f"주문 후 현금 {_fmt(c)}만 원은 최소 {_fmt(floor)}만 원 {'이상' if c >= floor else '미달'}")

    complete = all_stock is not None and cash is not None
    if key == "stock_add_buy_wait":
        d = re.search(r"(\d+(?:\.\d+)?)\s*%\s*하락", t)
        w = re.search(r"직전\s*매수\s*(?:뒤|후|이후)?\s*(\d+)\s*거래일", t) or re.search(r"(\d+)\s*거래일", t)
        first_stage = bool(re.search(r"1단계\s*(는|가)?\s*(미사용|아직|안\s*썼)", t))
        if d is None or w is None or not first_stage:
            return ref, None, None
        drop, days = _dec(d.group(1)), int(w.group(1))
        if drop >= STOP_DROP:
            fails.append(f"{_fmt(drop)}% 하락은 {STOP_DROP}% 이상이라 추가매수 보류")
        else:
            (parts if drop >= stage1 else fails).append(f"하락률 {_fmt(drop)}%는 1단계 기준 {_fmt(stage1)}% {'이상' if drop >= stage1 else '미만'}")
        (parts if days >= wait else fails).append(f"직전 매수 뒤 {days}거래일은 {ver} 대기 {wait}거래일 {'충족' if days >= wait else '미달'}")
        cap = account * order_rate / 100
        (parts if order <= cap else fails).append(f"주문 {_fmt(order)}만 원은 {_fmt(order_rate)}% 한도 {_fmt(cap)}만 원 {'이내' if order <= cap else '초과'}")
        complete = complete and all(x in t for x in ("공시", "합의")) and bool(re.search(r"(주택\s*)?적립\s*완료", t))
    body = ", ".join(fails + parts) + "."
    if fails:
        return ref, f"{body} 따라서 이 주문은 보류해야 해요.", False
    if not complete:
        return ref, None, None
    return ref, f"{body} 숫자 요건을 모두 충족해 주문할 수 있어요.", True


def _guest(t, value, ver):
    n = int(re.search(r"([1-3])\s*박", t).group(1))
    rule = value[f"{n}박"]
    need_h = int(re.search(r"(\d+)시간", rule).group(1))
    need_agree = "합의" in rule
    ref = f"{ver} {n}박 기준: {rule}"
    hours = int(re.search(r"(\d+)\s*시간\s*전", t).group(1))
    agreed = _agreed(t)
    if need_agree and agreed is None:
        return ref, None, None
    notice_ok = hours >= need_h
    agree_ok = not need_agree or agreed is True
    ok = notice_ok and agree_ok
    parts = [f"{ver}의 {n}박은 {rule}예요.",
             f"{hours}시간 전 고지는 {'기한을 지켰어요' if notice_ok else f'{need_h - hours}시간 부족해요'}."]
    if need_agree:
        parts.append("사전 합의도 있어요." if agreed else "사전 합의가 없어요.")
    parts.append("숙박 절차를 충족해요." if ok else "숙박 절차를 충족하지 못해요.")
    return ref, " ".join(parts), ok


def _clock_minutes(hour, minute):
    h, m = int(hour), int(minute or 0)
    return (h + 24 if h < 12 else h) * 60 + m     # 12시 전은 다음 날 새벽으로 센다


def _show(mins):
    return f"{(mins // 60) % 24:02d}시 {mins % 60:02d}분"


def _drinking(t, value, ver, when):
    y, mo, d = map(int, when[:10].split("-"))
    if date(y, mo, d).weekday() not in (4, 5):      # 금·토 시작 모임만 개정 대상
        return None, None, None
    base = _clock_minutes(*value["기본 귀가"].replace("다음 날 ", "").split(":"))
    deadline = _clock_minutes(*value["연장 요청 마감"].replace("다음 날 ", "").split(":"))
    ref = f"{ver} 금·토 시작 모임: 기본 귀가 다음 날 {_show(base)}, 연장 요청 마감 다음 날 {_show(deadline)}, 원래 약속 전 동의 필요"
    req = re.search(CLOCK + r"에\s*(?:\d{1,2}\s*시(?:\s*\d{1,2}\s*분)?\s*(?:까지\s*)?귀가로\s*)?(?:귀가\s*)?연장", t)
    con = re.search(CLOCK + r"에\s*(?:상대\s*)?(?:의\s*)?동의", t)
    if not req or not con:
        return ref, None, None
    r, c = _clock_minutes(req.group(1), req.group(2)), _clock_minutes(con.group(1), con.group(2))
    req_ok, con_ok = r <= deadline, c < base
    ok = req_ok and con_ok
    parts = [f"{ver} 기본 귀가는 다음 날 {_show(base)}, 연장 요청 마감은 {_show(deadline)}이에요.",
             f"{_show(r)} 요청은 마감을 {'지켰어요' if req_ok else '넘겼어요'}.",
             f"{_show(c)} 동의는 원래 약속 {'전이에요' if con_ok else '전이 아니에요'}.",
             "귀가 시각 변경이 성립해요." if ok else "귀가 시각 변경은 성립하지 않아요."]
    if "택시" in t and re.search(r"개인\s*(회식|모임)", t):
        parts.append("개인 모임의 귀가 택시비는 개인 부담이에요.")
    return ref, " ".join(parts), ok


def check_revised(question, reg, fallback_date=None):
    """개정 규칙 판단 결과. 해당 규칙이 아니거나 기준일을 못 정하면 None."""
    key = detect(question)
    if key is None:
        return None
    t = _text(question)
    when = pick_date(t, key, fallback_date)
    if not when:
        return None
    ver = reg.pick(when)
    it = reg.item(key)
    base = {"item": key, "item_name": it["name"], "date_basis": it["date_basis"], "event_date": when,
            "version": ver, "source": it["source"], "verdict": None, "decision": None, "passed": None}
    if ver == UNKNOWN:
        return None
    if ver == BEFORE:
        return {**base, "verdict": "추가 확인", "text": "규약 시행(2026-06-01) 전 사건이에요.",
                "next_action": "당시 별도 규정이 있었는지 확인해 주세요."}
    value = reg.value(key, ver)
    if key == "housing_monthly_cap":
        ref, decision, passed = _housing(t, value, ver)
    elif key in ("stock_add_buy_wait", "stock_single_cap"):
        ref, decision, passed = _stock(t, reg, ver, key)
    elif key == "guest_stay_notice":
        ref, decision, passed = _guest(t, value, ver)
    else:
        ref, decision, passed = _drinking(t, value, ver, when)
        if ref is None:
            return None
    return {**base, "decision": decision, "passed": passed, "text": decision or ref,
            "next_action": "판단에 쓴 날짜와 기록을 남겨 주세요." if decision else "빠진 값을 확인해 주세요."}
