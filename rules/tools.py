"""확실할 때만 코드가 답하는 계산 도구 (그래프 ② 판별 노드).

원칙
- 규칙 신호, 필요한 값, 기준일이 모두 확실할 때만 답을 낸다. 하나라도 모르면 None을 돌려 4B에게 넘긴다.
- 판본별 수치는 판본 표(versions_*.json)에서 읽는다. 판본과 무관한 공통 수치만 규약 조항과 함께 상수로 둔다.
- 설명형·복수 질문은 코드가 답하지 않는다.
"""
import re
import unicodedata
from datetime import date
from decimal import ROUND_FLOOR, ROUND_HALF_UP, Decimal, getcontext

from rules.dates import find_date_spans
from rules.versions import BEFORE, UNKNOWN

getcontext().prec = 50
LTV_LIMIT = Decimal(60)            # 제49조
SINGLE_MAX = Decimal(80)           # 제61조·표 33
STOCK_ALL = Decimal(20)            # 제54조·표 33
CASH_FLOOR = Decimal(15)           # 제54조·표 33
GENERAL_AGREEMENT = Decimal(20)    # 제21조·표 54
TRIP_CAP = Decimal(100)            # 제88조
CAR_PER_KM = Decimal(200)          # 제93조
PROTECT_HOURS = 12                 # 제69조
SWAP_HOURS = 3                     # 제70조
NUM = r"(\d[\d,]*(?:\.\d+)?)"
CLOCK = r"(\d{1,2})\s*시(?:\s*(\d{1,2})\s*분)?"

# 규칙별 근거. V2 문서에 없는 부칙은 V2용 근거를 쓴다
REFS = {
    "birthday_cap": ("제23조, 제79조, 부칙 제3조, 표 54", None),
    "appliance_prior_agreement": ("제24조, 제28조, 부칙 제4조, 표 54", None),
    "housing_monthly_cap": ("제49조, 부칙 제12조, 표 53", "제49조, 표 27"),
    "stock_single_cap": ("제61조, 부칙 제11조, 표 33, 표 57", "제61조, 표 33"),
    "stock_add_buy_wait": ("제58조, 부칙 제11조, 표 31, 표 57", "제58조, 표 31"),
    "guest_stay_notice": ("제84조, 부칙 제7조, 표 44, 표 56", None),
    "drinking_return_fri_sat": ("제95조, 부칙 제13조, 표 50, 표 55", "제95조, 표 50"),
    "domestic_lodging_cap": ("제89조, 제90조, 부칙 제9조, 표 47, 표 57", None),
    "max_order": ("제54조, 제58조, 제61조, 표 31, 표 33", None),
    "max_loan": ("제49조, 제50조, 표 27", None),
    "loan_payment": ("제50조", None),
    "trip_remaining": ("제88조, 제89조, 부칙 제9조, 표 46, 표 57", None),
    "car_cost": ("제93조, 표 49", None),
    "partial_sell": ("제59조, 표 32", None),
    "protect_end": ("제69조, 표 37", None),
    "swap_deadline": ("제70조, 표 37", None),
    "confirmed_total": ("제24조, 표 14", None),
    "add_buy_stop": ("제60조, 표 31", None),
}
NAMES = {"max_order": "최대 추가 주문액", "max_loan": "최대 대출액", "loan_payment": "원리금균등 월 상환액",
         "trip_remaining": "여행 건당 남은 한도", "car_cost": "공동 차량비", "partial_sell": "일반 부분매도 수량",
         "protect_end": "야간근무 보호 종료", "swap_deadline": "일반 교대 요청 마감"}
BASIS = {
    "birthday_cap": [r"첫\s*확정\s*결제", r"처음\s*(?:으로\s*)?[^.,]{0,24}?결제", r"확정\s*결제", r"결제"],
    "appliance_prior_agreement": [r"(?:처음|최초)\s*(?:으로\s*)?확정\s*주문", r"확정\s*주문", r"주문"],
    "domestic_lodging_cap": [r"예약\s*확정", r"확정"],
    "housing_monthly_cap": [r"심사일", r"심사"],
    "stock_single_cap": [r"주문\s*제출일", r"제출일", r"제출", r"매수\s*주문", r"주문"],
    "stock_add_buy_wait": [r"주문\s*제출일", r"제출일", r"제출", r"추가\s*주문", r"주문"],
    "guest_stay_notice": [r"숙박\s*시작", r"시작", r"부터"],
    "drinking_return_fri_sat": [r"시작한", r"시작", r"회식|술자리|모임"],
    "max_loan": [r"심사일", r"심사"],
    "max_order": [r"주문\s*제출일", r"제출일", r"제출", r"기준", r"주문"],
    "trip_remaining": [r"예약\s*확정", r"확정"],
}
FORMAT_SENTENCE = re.compile(r"^(?:예\s*/\s*아니|숫자|기호|시각만|시간 수만|거래일 수만|소수|만 원 미만)")
OPEN_ASK = re.compile(r"각각|이유|설명|왜|어떻게|어떤|무엇|누가|어디|언제|며칠|정리|비교해|나열")
YES_ASK = re.compile(r"받아야|돼|되나|될까|되는지|됩니까|가능|있나|있는지|안이|이내|통과|충족|지켰|채웠|맞아|맞나|맞는지|인정|해야|필요|괜찮|문제없|인지|나요|까요")


def normalize(text):
    t = unicodedata.normalize("NFKC", text or "")
    t = re.sub(r"(\d+)\s*만\s*(\d)\s*천(?:\s*원)?", r"\1.\2만 원", t)
    t = re.sub(r"(?<![\d.만])(\d)\s*천\s*원", r"0.\1만 원", t)
    return re.sub(r"[ \t]+", " ", t).strip()


def split_options(text):
    """'… A 가 B 나 C 다 D 라. 기호만' → (보기를 뺀 질문, {'A': '가', ...})."""
    t = normalize(text)
    m = re.search(r"(?:^|\s)A[.):]?\s+(.+?)\s+B[.):]?\s+(.+?)\s+C[.):]?\s+(.+?)\s+D[.):]?\s+(.+?)(?=[.?!](?:\s|$)|\n|$)", t, re.S)
    if not m:
        return t, {}
    rest = t[m.end():].lstrip(".?! ")
    stem = (t[:m.start()].rstrip() + (" " + rest if rest else "")).strip()
    return stem, {k: m.group(i).strip() for i, k in enumerate("ABCD", 1)}


def ask_sentence(stem):
    parts = [x.strip() for x in re.split(r"(?<=[.?!])\s+|\n", stem) if x.strip()]
    parts = [x for x in parts if not FORMAT_SENTENCE.search(x)]
    return parts[-1] if parts else stem


def route(stem, options=None):
    s = stem
    option_text = " ".join((options or {}).values())
    money = re.search(r"\d\s*만\s*원", s)
    if re.search(r"원금", s) and re.search(r"금리", s) and re.search(r"\d+\s*년", s) and "월 상환액" in s:
        return "loan_payment"
    if "최대 대출액" in s and "실수령" in s:
        return "max_loan"
    if re.search(r"원리금|월\s*상환액", s) and "실수령" in s:
        return "housing_monthly_cap"
    if re.search(r"\d+\s*km", s) and "차량비" in s:
        return "car_cost"
    if re.search(r"부분\s*매도|일반\s*매도", s) and re.search(r"\d+\s*%\s*이익", s):
        return "partial_sell"
    drop = re.search(r"최초\s*(?:체결)?\s*가?\s*대비\s*(\d+(?:\.\d+)?)\s*%\s*하락", s)
    if drop and float(drop.group(1)) >= 25 and re.search(r"추가|단계|주문|매수", s + option_text):
        return "add_buy_stop"
    if "확정액" in s and len(re.findall(r"\d\s*만\s*원", s)) >= 2 and re.search(r"할인|포인트|캐시백|배송|설치", s):
        return "confirmed_total"
    if "거래일" in s and re.search(r"추가\s*(?:주문|매수)", s):
        return "stock_add_buy_wait"
    if "계좌" in s and re.search(r"최대\s*추가\s*주문", s):
        return "max_order"
    if "계좌" in s and "종목" in s and re.search(r"상한|한도", s + option_text) and re.search(r"(?:더|추가로?)\s*(?:매수|사|삽|담|주문)", s):
        return "stock_single_cap"
    if "보호" in s and "야간근무" in s and re.search(r"끝|종료", s) and re.search(r"몇\s*시|시각", s):
        return "protect_end"
    if re.search(r"교대\s*요청\s*마감", s) or ("교대" in s and re.search(r"요청\s*마감|마감\s*시각", s)):
        return "swap_deadline"
    if re.search(r"[1-3]\s*박", s) and re.search(r"\d+\s*시간\s*전", s) and re.search(r"처음|최초|고지", s):
        return "guest_stay_notice"
    if re.search(r"회식|술자리|음주|모임", s) and re.search(r"귀가|들어", s) and re.search(r"\d{1,2}\s*시", s):
        return "drinking_return_fri_sat"
    if money and "여행 건당" in s:
        return "trip_remaining"
    if money and "생신" in s and "결제" in s:
        return "birthday_cap"
    if re.search(r"수리|고장", s):
        return None
    if (re.search(r"확정\s*주문", s) and (money or "기준" in s) and re.search(r"합의|가전", s)) or ("가전" in s and "주문" in s and money):
        return "appliance_prior_agreement"
    if re.search(r"숙소|숙박|패키지", s) and re.search(r"예약\s*확정", s) and money:
        return "domestic_lodging_cap"
    return None


def pick_event_date(text, key, fallback=None):
    """규칙의 기준 단어에 가장 가까운 날짜. 여러 날짜 중 하나로 못 정하면 fallback이 질문에 있을 때만 쓴다."""
    spans = find_date_spans(text)
    days = list(dict.fromkeys(d for *_, d in spans))
    if not days:
        return fallback
    if len(days) == 1:
        return days[0]
    t = unicodedata.normalize("NFKC", text)
    for pattern in BASIS.get(key, []):
        best = None
        for m in re.finditer(pattern, t):
            for s, e, d in spans:
                gap = m.start() - e if e <= m.start() else s - m.end() if s >= m.end() else 0
                between = t[e:m.start()] if e <= m.start() else t[m.end():s]
                if gap < 0 or gap > 30 or re.search(r"[.?!]\s", between):
                    continue
                if best is None or gap < best[0]:
                    best = (gap, d)
        if best:
            return best[1]
    return fallback if fallback in days else None


def agreed(t):
    if re.search(r"미합의|합의\s*(?:는|가|도)?\s*(?:없|안\s*했|아직)|동의\s*(?:는|가|도)?\s*없", t):
        return False
    if re.search(r"합의[^.?!]{0,28}(?:갖췄|갖춤|있습니다|있어|했|함|완료|끝|받았|줬)|(?:합의|동의)해\s*줬|모두\s*합의|동의했", t):
        return True
    return None


def dec(x):
    return Decimal(str(x).replace(",", ""))


def fmt(x):
    d = Decimal(x)
    return format(d.normalize(), "f") if d == d.to_integral() or d.as_tuple().exponent >= -6 else format(d.quantize(Decimal("0.01")), "f")


def money(pattern, t):
    m = re.search(pattern + r"[^\d\n]{0,8}?" + NUM + r"\s*만", t)
    return dec(m.group(1)) if m else None


def spend_items(t, planned_ok=False):
    """쓰는 돈 목록. 미래 포인트·미확정 캐시백·가용액·한도는 빼고 즉시할인은 음수."""
    out = []
    for m in re.finditer(NUM + r"\s*만\s*원", t):
        before = re.split(r"[,.;\n]", t[max(0, m.start() - 24):m.start()])[-1]
        after = t[m.end():m.end() + 14]
        if re.search(r"포인트|캐시백|적립|가용|예산|한도|상한|기준|계좌|보증금|미확정", before):
            continue
        if re.search(r"받을|준대|준다|미정" if planned_ok else r"받을|예정|준대|준다|미정", after):
            continue
        value = dec(m.group(1))
        out.append((before.strip(), -value if re.search(r"할인|쿠폰|깎", before) else value))
    return out


def total_of(t):
    items = spend_items(t)
    if not items:
        return None
    named = [v for label, v in items if re.search(r"확정액|합계|총액", label)]
    return named[-1] if named else sum(v for _, v in items)


def clock_minutes(hour, minute):
    h, m = int(hour), int(minute or 0)
    return (h + 24 if h < 12 else h) * 60 + m


def show_clock(mins):
    return f"{(mins // 60) % 24:02d}시 {mins % 60:02d}분"


def r(passed=None, number=None, unit=None, text="", pick=None, verdict=None):
    return {"passed": passed, "number": number, "unit": unit, "text": text, "pick": pick, "verdict": verdict}


# ---------------------------------------------------------------- 규칙별 계산
def birthday(s, ask, reg, ver):
    cap, total = dec(reg.value("birthday_cap", ver)), total_of(s)
    if total is None:
        return None
    text = f"같은 행사 합계 {fmt(total)}만 원, {ver} 상한 {fmt(cap)}만 원"
    if "합계" in ask and re.search(r"몇|얼마", ask):
        return r(number=total, unit="만 원", text=text)
    if "상한" in ask and re.search(r"몇|얼마", ask):
        return r(number=cap, unit="만 원", text=text)
    if total > cap:
        return r(False, text=text + "을 넘어요.", verdict="공동 처리 불가")
    if total >= GENERAL_AGREEMENT:
        a = agreed(s)
        if a is None:
            return None
        if not a:
            return r(False, text=text + " 이내지만 사전 합의가 필요해요.", verdict="사전 합의 필요")
    return r(True, text=text + " 이내예요.", verdict="가능")


def appliance(s, ask, reg, ver):
    th, cap = dec(reg.value("appliance_prior_agreement", ver)), dec(reg.value("appliance_order_cap", ver))
    if "기준" in ask and re.search(r"몇|얼마", ask) and "확정액" not in ask:
        return r(number=th, unit="만 원", text=f"{ver} 가전 사전 합의 기준 {fmt(th)}만 원 이상")
    total = total_of(s)
    if total is None:
        return None
    text = f"확정액 {fmt(total)}만 원, {ver} 가전 합의 기준 {fmt(th)}만 원"
    if re.search(r"확정액|비교할", ask) and re.search(r"몇|얼마", ask):
        return r(number=total, unit="만 원", text=text)
    need = total >= th
    def pick(o):
        if total > cap:
            return "상한 초과" in o
        return "사전 합의 필요" in o if need else bool(re.search(r"의무 없|필요 없", o))
    return r(need, text=text + (" 이상이라 사전 합의가 필요해요." if need else " 미만이라 사전 합의 의무가 없어요."), pick=pick)


def lodging(s, ask, reg, ver):
    cap = dec(reg.value("domestic_lodging_cap", ver))
    if "상한" in ask and re.search(r"몇|얼마", ask):
        return r(number=cap, unit="만 원", text=f"{ver} 국내 숙박 박당 상한 {fmt(cap)}만 원")
    total = total_of(s)
    nights = re.search(r"([1-9])\s*박", s)
    if total is None or not nights:
        return None
    n = 1 if re.search(r"1\s*박\s*(?:에|금액|당)", s) else int(nights.group(1))
    per = total / n
    text = f"박당 {fmt(per)}만 원, {ver} 상한 {fmt(cap)}만 원"
    if "박당" in ask and re.search(r"몇|얼마", ask):
        return r(number=per, unit="만 원", text=text)
    def pick(o):
        m = re.search(r"박당\s*([\d.]+)\s*만\s*원으로\s*상한\s*(이내|초과)", o)
        return bool(m) and dec(m.group(1)) == per and (m.group(2) == "이내") == (per <= cap)
    return r(per <= cap, text=text + (" 이내예요." if per <= cap else "을 넘어요."), pick=pick,
             verdict="가능" if per <= cap else "공동 처리 불가")


def housing_values(s):
    income = money(r"실수령(?:액)?(?:은|는|이)?", s)
    m = re.search(r"월\s*상환액[^\d\n]{0,14}" + NUM + r"\s*만", s)
    new = dec(m.group(1)) if m else None
    m = re.search(r"(?:잔존|기존)[^\d.,]{0,10}(?:원리금|상환액)[^\d\n]{0,8}" + NUM + r"\s*만", s)
    old = dec(m.group(1)) if m else (Decimal(0) if re.search(r"기존\s*(?:대출|원리금|상환액)?\s*(?:은|는|이|가)?\s*없", s) else None)
    m = re.search(r"(?:매매가|집값)(?:은|는|이)?\s*" + NUM + r"\s*만", s)
    price = dec(m.group(1)) if m else None
    m = re.search(r"(?:신규\s*)?대출(?:은|액)?\s*" + NUM + r"\s*만", s)
    loan = dec(m.group(1)) if m else None
    return income, old, new, price, loan


def housing(s, ask, reg, ver):
    rate = dec(reg.value("housing_monthly_cap", ver)["비율"])
    income, old, new, price, loan = housing_values(s)
    if income is None:
        return None
    cap = income * rate / 100
    if "상한" in ask and re.search(r"몇|얼마", ask):
        return r(number=cap, unit="만 원", text=f"{ver} 월 원리금 상한 {fmt(cap)}만 원(실수령 {fmt(income)}만 원의 {fmt(rate)}%)")
    if old is None:
        return None
    if re.search(r"신규\s*대출[^.?!]{0,8}월\s*상환액[^.?!]{0,12}최대", ask):
        room = cap - old
        return r(number=max(room, Decimal(0)), unit="만 원", text=f"{ver} 상한 {fmt(cap)}만 원 - 기존 {fmt(old)}만 원 = {fmt(room)}만 원")
    if new is None:
        return None
    total = old + new
    month_ok = total <= cap
    ltv_ok = None if not (price and loan) else loan * 100 <= price * LTV_LIMIT
    text = f"월 원리금 {fmt(old)}+{fmt(new)}={fmt(total)}만 원, {ver} 상한 {fmt(cap)}만 원({fmt(rate)}%)"
    def pick(o):
        if ltv_ok is None:
            return False
        want = "둘 다 통과" if ltv_ok and month_ok else "월 원리금만" if ltv_ok else "담보비율만" if month_ok else "둘 다 초과"
        return want in o
    return r(month_ok, text=text + (" 이내예요." if month_ok else "을 넘어요."), pick=pick)


def stock_values(s):
    account = money(r"계좌(?:\s*(?:총액|전체|평가액))?(?:은|는|이)?", s)
    target = money(r"(?:해당|대상|이)?\s*(?:개별주|종목)(?:은|의)?(?:\s*지금)?(?:\s*(?:평가액|보유액))?(?:은|는|이)?", s)
    m = re.search(NUM + r"\s*만\s*원\s*(?:을|만|어치)?\s*(?:더|추가로?)\s*(?:매수|사|삽|담|주문)", s)
    return account, target, dec(m.group(1)) if m else None


def single(s, ask, reg, ver):
    account, target, order = stock_values(s)
    if account is None or target is None or order is None:
        return None
    rate = dec(reg.value("stock_single_cap", ver))
    limit = min(account * rate / 100, SINGLE_MAX)
    after = target + order
    ok = after <= limit
    text = f"주문 후 {fmt(after)}만 원, {ver} 한도 {fmt(limit)}만 원({fmt(rate)}%와 80만 원 중 작은 값)"
    def pick(o):
        m = re.search(r"한도\s*([\d.]+)\s*만\s*원,?\s*주문\s*후\s*(이내|초과)", o)
        return bool(m) and dec(m.group(1)) == limit and (m.group(2) == "이내") == ok
    return r(ok, text=text + (" 이내예요." if ok else "을 넘어요."), pick=pick)


def wait(s, ask, reg, ver):
    if "대기" not in ask:
        return None
    m = re.search(r"(\d+)\s*거래일", s)
    if not m:
        return None
    days, need = int(m.group(1)), int(reg.value("stock_add_buy_wait", ver))
    text = f"{days}거래일 경과, {ver} 대기 {need}거래일"
    if re.search(r"몇\s*거래일", ask):
        return r(number=Decimal(max(need - days, 0)), unit="거래일", text=text)
    return r(days >= need, text=text + (" 충족이에요." if days >= need else " 미달이에요."))


def guest(s, ask, reg, ver):
    nights = int(re.search(r"([1-3])\s*박", s).group(1))
    rule = reg.value("guest_stay_notice", ver)[f"{nights}박"]
    need = int(re.search(r"(\d+)시간", rule).group(1))
    hours = int(re.search(r"(\d+)\s*시간\s*전", s).group(1))
    text = f"{ver} {nights}박 기준은 {rule}, 실제 {hours}시간 전 고지"
    if re.search(r"몇\s*시간", ask):
        return r(number=Decimal(max(need - hours, 0)), unit="시간", text=text)
    need_agree = "합의" in rule
    a = agreed(s)
    if need_agree and a is None:
        return None
    notice_ok, agree_ok = hours >= need, not need_agree or a is True
    ok = notice_ok and agree_ok
    def pick(o):
        m = re.search(r"(\d+)시간 전 고지 필요,?\s*절차\s*(미충족|충족)", o)
        if m:
            return int(m.group(1)) == need and (m.group(2) == "충족") == ok
        want = "모두 충족" if ok else "둘 다 부족" if not notice_ok and not agree_ok else "고지 기한만" if not notice_ok else "합의만"
        return want in o
    return r(ok, text=text + (", 절차를 충족해요." if ok else ", 절차를 충족하지 못해요."), pick=pick)


def drinking(s, ask, reg, ver, when):
    y, m, d = map(int, when[:10].split("-"))
    if date(y, m, d).weekday() not in (4, 5):
        return None
    value = reg.value("drinking_return_fri_sat", ver)
    base = clock_minutes(*value["기본 귀가"].replace("다음 날 ", "").split(":"))
    due = clock_minutes(*value["연장 요청 마감"].replace("다음 날 ", "").split(":"))
    if re.search(r"연장\s*요청\s*없이", s):
        back = re.findall(CLOCK + r"\s*에\s*(?:귀가|들어)", s)
        if len(back) != 1:
            return None
        mins = clock_minutes(*back[0])
        ok = mins <= base
        return r(ok, text=f"{show_clock(mins)} 귀가, {ver} 금·토 기본 귀가 {show_clock(base)}" + (" 이내예요." if ok else "를 넘었어요."))
    req = re.search(CLOCK + r"\s*에\s*(?:" + CLOCK + r"\s*까지[^.?!]{0,24})?연장", s)
    con = re.search(CLOCK + r"\s*에\s*(?:상대\s*)?(?:의\s*)?동의", s)
    if not req or not con:
        return None
    a, b = clock_minutes(req.group(1), req.group(2)), clock_minutes(con.group(1), con.group(2))
    ok = a <= due and b < base
    return r(ok, text=f"{show_clock(a)} 요청, {show_clock(b)} 동의, {ver} 요청 마감 {show_clock(due)}, 원래 약속 {show_clock(base)}"
             + (". 변경이 성립해요." if ok else ". 변경이 성립하지 않아요."))


def max_order(s, ask, reg, ver):
    account = money(r"계좌(?:\s*(?:총액|전체|평가액))?(?:은|는|이)?", s)
    if account is None:
        return None
    limits = []
    if re.search(r"단계별", ask):
        limits.append(("단계 상한", account * dec(reg.value("stock_stage_order_limit", ver)) / 100))
    if re.search(r"한\s*종목|종목\s*보유", ask):
        target = money(r"(?:대상\s*)?(?:개별주|종목)(?:\s*평가액)?(?:은|는|이)?", s)
        if target is None:
            return None
        limits.append(("종목 여유", min(account * dec(reg.value("stock_single_cap", ver)) / 100, SINGLE_MAX) - target))
    if "현금" in ask:
        cash = money(r"현금(?:은|는|이)?", s)
        if cash is None:
            return None
        limits.append(("현금 여유", cash - account * CASH_FLOOR / 100))
    if "개별주 전체" in ask:
        whole = money(r"개별주\s*전체(?:\s*평가액)?(?:은|는|이)?", s)
        if whole is None:
            return None
        limits.append(("전체 여유", account * STOCK_ALL / 100 - whole))
    if len(limits) < 2:
        return None
    value = max(min(v for _, v in limits), Decimal(0))
    return r(number=value, unit="만 원", text=", ".join(f"{k} {fmt(v)}만 원" for k, v in limits) + f" → 최대 {fmt(value)}만 원")


def monthly_rate(s):
    m = re.search(r"연\s*(?:금리\s*)?" + NUM + r"\s*%", s)
    y = re.search(r"(?<!\d)(\d{1,2})\s*년", s)
    return (dec(m.group(1)) / 100 / 12, int(y.group(1)) * 12) if m and y else (None, None)


def max_loan(s, ask, reg, ver):
    rate = dec(reg.value("housing_monthly_cap", ver)["비율"])
    income, old, _, price, _ = housing_values(s)
    mr, n = monthly_rate(s)
    if income is None or old is None or price is None or not mr:
        return None
    pay = income * rate / 100 - old
    by_month = (pay * (1 - (1 + mr) ** (-n)) / mr).quantize(Decimal(1), ROUND_FLOOR)
    by_ltv = (price * LTV_LIMIT / 100).quantize(Decimal(1), ROUND_FLOOR)
    value = max(min(by_month, by_ltv), Decimal(0))
    return r(number=value, unit="만 원", text=f"담보 기준 {fmt(by_ltv)}만 원, 월 {fmt(pay)}만 원 기준 {fmt(by_month)}만 원 → 작은 값 {fmt(value)}만 원")


def loan_payment(s, ask):
    p = money(r"원금(?:은|이)?", s)
    mr, n = monthly_rate(s)
    if p is None or not mr:
        return None
    value = (p * mr / (1 - (1 + mr) ** (-n))).quantize(Decimal("0.01"), ROUND_HALF_UP)
    return r(number=value, unit="만 원", text=f"P×r÷[1-(1+r)^(-n)], n={n} → 월 {fmt(value)}만 원")


def trip_remaining(s, ask, reg, ver):
    if not re.search(r"더\s*쓸|남은", ask):
        return None
    cap = dec(reg.value("domestic_lodging_cap", ver))
    total = Decimal(0)
    for label, value in spend_items(s, planned_ok=True):
        m = re.search(r"([1-9])\s*박[^\d]{0,8}각", label)
        if m:
            if value > cap:
                return None
            value *= int(m.group(1))
        total += value
    left = TRIP_CAP - total
    return r(number=max(left, Decimal(0)), unit="만 원", text=f"{ver} 박당 상한 {fmt(cap)}만 원, 여행 상한 100만 원 - 예정 {fmt(total)}만 원 = {fmt(left)}만 원")


def car_cost(s, ask):
    whole = re.search(r"총\s*(\d[\d,]*)\s*km", s)
    away = re.search(r"(\d[\d,]*)\s*km\s*(?:는|은)?\s*개인", s)
    if not whole or not away:
        return None
    km = dec(whole.group(1)) - dec(away.group(1))
    fees = Decimal(0)
    for m in re.finditer(NUM + r"\s*원", s):
        before = re.split(r"[,.]", s[max(0, m.start() - 16):m.start()])[-1]
        if "개인" not in before and re.search(r"통행료|주차료", before):
            fees += dec(m.group(1))
    value = km * CAR_PER_KM + fees
    return r(number=value, unit="원", text=f"공용 {fmt(km)}km × 200원 + 통행·주차 {fmt(fees)}원 = {fmt(value)}원")


def partial_sell(s, ask):
    days, gain, qty = re.search(r"(\d+)\s*일이?\s*지", s), re.search(r"(\d+(?:\.\d+)?)\s*%\s*이익", s), re.search(r"(\d+)\s*주를?\s*보유", s)
    if not (days and gain and qty) or "첫" not in s or int(days.group(1)) < 30 or not 20 <= float(gain.group(1)) < 30:
        return None
    value = Decimal(int(qty.group(1)) // 3)
    return r(number=value, unit="주", text=f"30일 경과, 이익 {gain.group(1)}%, 보유 {qty.group(1)}주의 3분의 1은 소수주를 버려 {fmt(value)}주")


def protect_end(s, ask):
    m = re.search(r"야간근무[^.?!]{0,8}?(\d{1,2})\s*시(?:\s*(\d{1,2})\s*분)?에\s*(?:끝|종료)", s)
    if not m or "보호 대상" not in s or m.group(2):
        return None
    hour = (int(m.group(1)) + PROTECT_HOURS) % 24
    return r(number=Decimal(hour), unit="시", text=f"{int(m.group(1)):02d}시 종료 + {PROTECT_HOURS}시간 = {hour:02d}시")


def swap_deadline(s, ask):
    m = re.search(r"(\d{1,2})\s*시(?:\s*(\d{1,2})\s*분)?(?:에\s*예정|\s*시작)", s)
    if not m or m.group(2):
        return None
    hour = (int(m.group(1)) - SWAP_HOURS) % 24
    return r(number=Decimal(hour), unit="시", text=f"{int(m.group(1)):02d}시 - {SWAP_HOURS}시간 = {hour:02d}시")


def confirmed_total(s, ask):
    items = []
    for clause in re.split(r"[,;\n]|(?<=[가-힣])\.\s", s):      # 항목마다 앞말만 본다. 뒤 항목의 '예정'이 앞 항목을 지우지 않게 한다
        for m in re.finditer(NUM + r"\s*만\s*원", clause):
            label = clause[:m.start()]
            if re.search(r"포인트|캐시백|적립|예정|미확정|환급|가용|예산|한도|상한|기준", label):
                continue
            value = dec(m.group(1))
            items.append((label.strip(), -value if re.search(r"할인|쿠폰", label) else value))
    if len(items) < 2:
        return None
    value = sum(v for _, v in items)
    parts = " ".join(f"{'+' if v > 0 else '-'}{fmt(abs(v))}" for _, v in items).lstrip("+")
    return r(number=value, unit="만 원", text=f"확정액 = {parts} = {fmt(value)}만 원. 미래 포인트와 미확정 캐시백은 빼지 않아요.")


def add_buy_stop(s, ask):
    drop = re.search(r"최초\s*(?:체결)?\s*가?\s*대비\s*(\d+(?:\.\d+)?)\s*%\s*하락", s)
    if not drop or float(drop.group(1)) < 25:
        return None
    return r(passed=False, text=f"최초 체결가 대비 {drop.group(1)}% 하락은 25% 이상이라 추가매수를 멈추고 투자 가정을 다시 심의해요.",
             pick=lambda o: "보류" in o)


NO_DATE = {"loan_payment": loan_payment, "car_cost": car_cost, "partial_sell": partial_sell,
           "protect_end": protect_end, "swap_deadline": swap_deadline, "confirmed_total": confirmed_total, "add_buy_stop": add_buy_stop}
DATED = {"birthday_cap": birthday, "appliance_prior_agreement": appliance, "domestic_lodging_cap": lodging,
         "housing_monthly_cap": housing, "stock_single_cap": single, "stock_add_buy_wait": wait, "guest_stay_notice": guest,
         "max_order": max_order, "max_loan": max_loan, "trip_remaining": trip_remaining}


def refs_for(key, reg):
    main, v2 = REFS[key]
    return v2 if v2 and reg.doc_release == "V2" else main


def wants_number(ask, f):
    return f.get("kind") == "number" or bool(re.search(r"몇\s*(?:만\s*원|원|시간|거래일|주|시)|얼마", ask))


def yes_value(key, ask, passed):
    if passed is None or OPEN_ASK.search(ask) or not YES_ASK.search(ask):
        return None
    if key == "appliance_prior_agreement":
        return passed if re.search(r"합의", ask) and re.search(r"필요|받아야|해야", ask) else None
    return passed


def code_answer(res, key, ask, f, options):
    if options:
        if not res.get("pick"):
            return None
        hits = [k for k, o in options.items() if res["pick"](o)]
        return hits[0] if len(hits) == 1 else None
    if wants_number(ask, f):
        if res.get("number") is None:
            return None
        n = fmt(res["number"])
        return f"{n}{'' if res['unit'] in ('시', '시간', '거래일', '주', '원') else ' '}{res['unit']}"
    value = yes_value(key, ask, res.get("passed"))
    if value is None:
        return None
    word = "예" if value else "아니오"
    return word if f.get("kind") == "yesno" else f"{word}. {res['text']}"


DATE_KEYS = ("birthday_cap", "appliance_prior_agreement", "domestic_lodging_cap", "housing_monthly_cap", "stock_single_cap",
             "stock_add_buy_wait", "guest_stay_notice", "drinking_return_fri_sat")
APPLIANCE_WORDS = r"가전|세탁기|건조기|냉장고|청소기|밥솥|공기청정기|식기세척기|제습기|에어컨|티비|TV|전자레인지|오븐|커피머신"


def date_rule(text):
    """기준일이 있어야 판단할 수 있는 규칙이면 그 키. 4B를 부르지 않고 되묻기 여부를 정한다."""
    t = normalize(text)
    stem, options = split_options(t)
    key = route(stem, options)
    if key in DATE_KEYS:
        return key
    if not re.search(r"\d\s*만\s*원", t) or re.search(r"수리|고장", t):
        return None
    if "생신" in t:
        return "birthday_cap"
    if re.search(APPLIANCE_WORDS, t) and re.search(r"샀|사려|살까|구매|구입|주문|결제", t):
        return "appliance_prior_agreement"
    if re.search(r"숙소|호텔|숙박비|공동\s*숙박", t):
        return "domestic_lodging_cap"
    return None


def solve(question, f, reg, key=None, when=None):
    """코드가 맡을 수 있으면 calc 형식의 결과, 아니면 None. code_answer가 None이면 4B가 참고할 계산만 담긴다."""
    stem, options = split_options(question)
    key = key or route(stem, options)
    if key is None or f.get("length") or f.get("multi") or len(stem) > 420:
        return None
    ask = ask_sentence(stem)
    if OPEN_ASK.search(ask) and key != "add_buy_stop":
        return None
    if key in NO_DATE:
        when, ver = None, reg.doc_release
        res = NO_DATE[key](stem, ask)
    else:
        when = when or pick_event_date(stem, key)
        ver = reg.pick(when) if when else UNKNOWN
        if ver in (BEFORE, UNKNOWN):
            return None
        res = drinking(stem, ask, reg, ver, when) if key == "drinking_return_fri_sat" else DATED[key](stem, ask, reg, ver)
    if res is None:
        return None
    item = reg.items.get(key) or {}
    answer = code_answer(res, key, ask, f, options)
    return {"item": key, "item_name": item.get("name") or NAMES.get(key, key), "date_basis": item.get("date_basis") or "해당 없음",
            "event_date": when, "version": ver, "source": refs_for(key, reg), "verdict": res.get("verdict") if answer else None,
            "decision": None, "passed": res.get("passed"), "text": res["text"], "code_answer": answer,
            "answered_by": "code" if answer else "4b_with_code_hint",
            "next_action": "판단에 쓴 날짜와 금액 기록을 남겨 주세요."}
