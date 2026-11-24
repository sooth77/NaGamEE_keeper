"""계산 노드 (그래프 ⑥). 판본 표의 값으로 코드가 계산하고 판정을 제안한다.

지금 아는 항목: 부모님 생신 상한, 가전 사전 합의 기준·주문 상한, 국내 숙박 박당 상한.
모르는 항목이면 None 을 돌려주고, 답 쓰기(LLM)가 근거만 보고 판단한다.
비교 규칙: '이상'은 기준 포함(>=), '초과'는 기준 제외(>), 상한 '이내'는 포함(<=). 중간 반올림 없음.
"""
import re

from rules.revised_checks import check_revised
from rules.tools import pick_event_date, route, split_options
from rules.versions import BEFORE, UNKNOWN

GENERAL_AGREEMENT = 20   # 제21조·표 54: 20만 원 이상 사전 합의
AMOUNT_ITEMS = ("birthday_cap", "appliance_prior_agreement", "domestic_lodging_cap")   # 금액 합계로 판정하는 항목
# 4B가 고른 계산기가 질문과 맞는지 보는 최소 신호. 신호가 없으면 계산기를 켜지 않는다
ITEM_SIGNAL = {"birthday_cap": r"생신|생일", "appliance_prior_agreement": r"가전|주문|구매|구입|샀|사려|살까",
               "domestic_lodging_cap": r"숙소|호텔|박당|패키지|숙박비|공동\s*숙박|국내\s*숙박|여행"}


def _fmt(x):
    return f"{x:g}"


def _total(amounts):
    vals = [a["sign"] * a["value"] for a in amounts or [] if a.get("unit", "만원") == "만원"]
    return sum(vals), " + ".join(_fmt(a["value"]) if a["sign"] > 0 else f"(-{_fmt(a['value'])})" for a in amounts or [])


def compute(facts, reg, question=None):
    item = facts.get("item")
    if question:      # 월 원리금·투자·손님·귀가는 질문 글에서 값을 읽어 판단
        revised = check_revised(question, reg, facts.get("event_date"))
        # 개정 규칙으로 판별되면 모델이 고른 금액 계산기보다 우선한다 (월 원리금이 숙박 계산기로 가는 경우 방지)
        if revised:
            return revised
        routed = route(*split_options(question))     # 코드가 다른 규칙으로 판별하면 4B가 고른 계산기를 켜지 않는다
        if item in ITEM_SIGNAL and ((routed and routed != item) or not re.search(ITEM_SIGNAL[item], question)):
            return None
    if item not in ("birthday_cap", "appliance_prior_agreement", "domestic_lodging_cap", "stock_add_buy_stage1"):
        return None
    when = (pick_event_date(question, item, facts.get("event_date")) if question else None) or facts.get("event_date")
    ver = reg.pick(when)
    base = {"item": item, "item_name": reg.item(item)["name"], "date_basis": reg.item(item)["date_basis"],
            "event_date": when, "version": ver, "source": reg.item(item)["source"]}
    if ver == UNKNOWN:
        return {**base, "verdict": "추가 확인",
                "text": f"{base['date_basis']}을 알 수 없어 판본을 정할 수 없어요.",
                "next_action": f"{base['date_basis']}을 알려 주세요."}
    if ver == BEFORE:
        return {**base, "verdict": "추가 확인", "text": "규약 시행(2026-06-01) 전 사건이에요.",
                "next_action": "당시 별도 규정이 있었는지 확인해 주세요."}
    if not facts.get("amounts") and item != "stock_add_buy_stage1":
        return {**base, "verdict": None, "text": f"{ver} 기준값 {reg.value(item, ver)}만 원 (금액 정보 없음)"}
    total, expr = _total(facts.get("amounts"))
    agreed = facts.get("agreed")

    if item == "birthday_cap":
        cap = reg.value(item, ver)
        calc = f"{expr} = {_fmt(total)}만 원, {ver} 상한 {cap}만 원"
        if total > cap:
            return {**base, "verdict": "공동 처리 불가", "total": total, "value": cap,
                    "text": f"{calc} → {_fmt(total - cap)}만 원 초과",
                    "next_action": f"초과한 {_fmt(total - cap)}만 원은 개인 처리, 또는 예외 해당 여부 확인"}
        if total >= GENERAL_AGREEMENT and agreed is not True:
            return {**base, "verdict": "사전 합의 필요", "total": total, "value": cap,
                    "text": f"{calc} 이내지만 20만 원 이상이라 첫 확정 결제 전 합의 필요",
                    "next_action": "첫 확정 결제 전에 두 사람이 범위와 상한에 합의해 주세요."}
        return {**base, "verdict": "가능", "total": total, "value": cap, "text": f"{calc} 이내",
                "next_action": "영수증을 같은 행사로 묶어 기록해 주세요."}

    if item == "appliance_prior_agreement":
        th, cap = reg.value(item, ver), reg.value("appliance_order_cap", ver)
        calc = f"{expr} = {_fmt(total)}만 원, {ver} 합의 기준 {th}만 원 이상, 주문 상한 {cap}만 원"
        if total > cap:
            return {**base, "verdict": "공동 처리 불가", "total": total, "value": th,
                    "text": f"{calc} → 주문 상한 초과", "next_action": "주문을 나누거나 개인 부담분을 정해 주세요."}
        if total >= th and agreed is not True:
            return {**base, "verdict": "사전 합의 필요", "total": total, "value": th,
                    "text": f"{calc} → 기준 이상", "next_action": "확정 주문 전에 두 사람이 합의해 주세요."}
        return {**base, "verdict": "가능", "total": total, "value": th,
                "text": f"{calc} → {'합의 있음' if total >= th else '기준 미만'}",
                "next_action": "공용 등록과 가용 예산을 확인해 주세요."}

    if item == "domestic_lodging_cap":
        cap = reg.value(item, ver)
        n = facts.get("nights") or 1
        per = total / n
        calc = f"{expr} = {_fmt(total)}만 원 ÷ {n}박 = 박당 {_fmt(round(per, 4))}만 원, {ver} 상한 {cap}만 원"
        if per > cap:
            return {**base, "verdict": "공동 처리 불가", "total": total, "value": cap,
                    "text": f"{calc} → 초과", "next_action": "초과분은 개인 부담으로 나눠 주세요."}
        return {**base, "verdict": "가능", "total": total, "value": cap, "text": f"{calc} 이내",
                "next_action": "예약 확정일과 명세를 기록해 주세요."}

    # 추가매수: 하락률은 LLM이 근거로 판단, 코드는 기준값만 알려준다
    th = reg.value(item, ver)
    return {**base, "verdict": None, "value": th,
            "text": f"{ver} 첫 추가매수 기준: 최초 체결가 대비 {th}% 이상 하락 (기준일: 추가 주문 제출일)"}
