"""개선판 안전장치. 모델을 부르지 않으므로 첫 토큰을 늦추지 않는다.

- version_guard: 개정 항목의 날짜가 있으면 적용 판본을 4B에 알려 주고, 날짜를 모른다고 하면 코드가 추가 확인으로 답한다.
- extra_evidence: 개정 항목의 근거 조항·표가 검색에서 빠졌으면 그 쪽을 최대 2쪽 더한다.
- verify_answer: 답이 끝난 뒤 다른 판본의 값을 썼는지, 최종 답이 잘렸는지 검사한다.
"""
import json
import re
from pathlib import Path

from rules.tools import REFS, date_rule, pick_event_date, refs_for, split_options
from rules.versions import BEFORE, UNKNOWN

DATA = Path(__file__).resolve().parent.parent / "data"
UNKNOWN_DATE = re.compile(r"모르|모름|불명|기억이?\s*안|확인이?\s*안|언제인지|알아봤|나중에\s*(?:주문|결제|샀|구매)|\d+\s*월인지")
REF = re.compile(r"(?:부칙\s*)?제\s*\d+\s*조|표\s*\d+")
_BOOK = {}


def _unit(value, unit):
    return f"{value}{unit}".replace("만원", "만 원")


def _source(key, item, reg):
    return refs_for(key, reg) if key in REFS else item.get("source", "")


def _base(key, item, reg, when, version, text):
    return {"item": key, "item_name": item["name"], "date_basis": item["date_basis"], "event_date": when, "version": version,
            "source": _source(key, item, reg), "verdict": None, "decision": None, "passed": None, "text": text, "code_answer": None,
            "answered_by": "4b_with_version_hint", "next_action": f"{item['date_basis']} 기록을 남겨 주세요."}


def version_guard(question, fmt, reg):
    key = date_rule(question)
    if not key or key not in reg.items or (fmt or {}).get("length"):
        return None
    item = reg.items[key]
    values = item["values"]
    stem = split_options(question)[0]
    when = pick_event_date(stem, key)
    if when:
        version = reg.pick(when)
        if version in (BEFORE, UNKNOWN) or version not in values:
            return None
        text = f"{item['name']}: {item['date_basis']} {when} → {version} 기준 {_unit(values[version], item.get('unit', ''))}"
        return _base(key, item, reg, when, version, text)
    if not UNKNOWN_DATE.search(stem) or len({str(v) for v in values.values()}) < 2:
        return None
    listed = ", ".join(f"{v} {_unit(x, item.get('unit', ''))}" for v, x in values.items())
    text = f"{item['date_basis']}을 알아야 적용 판본을 정할 수 있어요. 판본별 기준은 {listed}예요."
    options = split_options(question)[1]
    answer = f"추가 확인이 필요해요. {text}"
    if options:
        hits = [k for k, o in options.items() if "추가 확인" in o]
        if len(hits) != 1:
            return None
        answer = hits[0]
    return {**_base(key, item, reg, None, None, text), "verdict": "추가 확인", "code_answer": answer, "answered_by": "code_guard",
            "next_action": f"{item['date_basis']}을 확인해 주세요."}


def _book(release):
    if release not in _BOOK:
        rows = [json.loads(l) for l in open(DATA / f"rulebook_{release}.jsonl", encoding="utf-8") if l.strip()]
        index = {}
        for row in rows:
            for label in list(row.get("articles") or []) + list(row.get("tables") or []):
                m = REF.search(str(label))
                if m:
                    index.setdefault(re.sub(r"\s", "", m.group(0)), []).append(row)
        _BOOK[release] = index
    return _BOOK[release]


def extra_evidence(calc, docs, release, limit=2):
    if not calc or not calc.get("source"):
        return ""
    have = {int(d["page"]) for d in docs or [] if d.get("version") in (None, release)}
    index, out, seen = _book(release), [], set()
    for ref in REF.findall(calc["source"]):
        for row in index.get(re.sub(r"\s", "", ref), [])[:1]:
            page = int(row["page"])
            if page not in have and page not in seen and len(out) < limit:
                seen.add(page)
                out.append(row["text"])
    return "\n\n---\n\n".join(out)


def verify_answer(answer, raw, finish, calc, reg):
    """고칠 이유가 있으면 한 문장, 없으면 None."""
    if finish in ("length", "repeat") and not re.search(r"\[\s*최종\s*답변", raw or ""):
        return "최종 답변까지 쓰지 못했습니다. 계산을 줄이고 최종 답변과 근거를 쓰세요."
    if not calc or not calc.get("version") or calc.get("item") not in reg.items:
        return None
    values = reg.items[calc["item"]]["values"]
    right = values.get(calc["version"])
    if not isinstance(right, (int, float)):
        return None
    nums = {float(x.replace(",", "")) for x in re.findall(r"\d[\d,]*(?:\.\d+)?", str(answer))}
    others = {float(v) for k, v in values.items() if k != calc["version"] and isinstance(v, (int, float)) and v != right}
    if float(right) not in nums and nums & others:
        return f"적용 판본은 {calc['version']}이고 기준값은 {right}입니다. 다른 판본의 값을 쓰지 말고 다시 판단하세요."
    return None


STOP = {"있는", "없는", "하는", "되는", "해서", "하고", "그리고", "에서", "으로", "까지", "부터", "이번", "다음", "질문", "규약", "기준", "확인", "경우",
        "알려", "답해", "주세요", "해줘", "이거", "좀", "대해", "어떻게", "무엇", "얼마", "가능", "필요"}


def _terms(text):
    out = set()
    for w in re.findall(r"[가-힣]{2,}|\d+(?:\.\d+)?", text):
        w = re.sub(r"(은|는|이|가|을|를|의|에|도|로|과|와|만|요|야)$", "", w)
        if len(w) >= 2 and w not in STOP:
            out.add(w)
    return out


def focus_evidence(question, docs, limit=6, max_chars=800):
    """질문의 낱말이 많이 겹치는 조항 문장과 표 행을 원문 그대로 앞에 모은다. 원문 근거는 지우지 않는다."""
    q = _terms(question)
    if not q or not docs:
        return ""
    picked = []
    for d in docs:
        title, header, article = "", "", ""
        lines = str(d.get("text") or "").splitlines()
        for i, line in enumerate(lines):
            t = line.strip()
            found = re.match(r"표\s*\d+", t)
            if found:
                title, header = re.sub(r"\s+", " ", found.group(0)), ""
            found = re.match(r"(?:부칙\s*)?제\s*\d+\s*조", t)
            if found:
                article, title = re.sub(r"\s+", " ", found.group(0)), ""
                if len(t) <= 30:
                    continue
            if not t or t.startswith("[") or re.fullmatch(r"[|\-\s]+", t):
                continue
            is_row = t.startswith("|")
            if is_row and i + 1 < len(lines) and re.fullmatch(r"[|\-\s]+", lines[i + 1].strip()):
                header = t
                continue
            hit = q & _terms(t)
            score = sum(2 if x[0].isdigit() else 1 for x in hit)
            if len(hit) >= 2:
                name = title if is_row and title else article
                label = (f"[{name}] " if name else "") + (f"{header} " if is_row and header else "")
                picked.append((score, -int(d.get("rank") or 9), label + t))
    out, size = [], 0
    for _, _, text in sorted(picked, key=lambda x: (-x[0], -x[1])):
        if text in out or size + len(text) > max_chars or len(out) >= limit:
            continue
        out.append(text)
        size += len(text)
    return "\n".join(out)


def _line_key(line):
    t = re.sub(r"^[\s\-*•]*(?:\d+[.)]\s*)?", "", line)
    return re.sub(r"[\s*]+", "", t)


def is_looping(text):
    """같은 줄이 세 번 나오면 True. 끝나지 않은 마지막 줄은 보지 않는다."""
    lines = str(text or "").split("\n")[:-1]
    if not lines:
        return False
    keys = [_line_key(x) for x in lines]
    return len(keys[-1]) >= 10 and keys.count(keys[-1]) >= 3


def trim_loop(text):
    """반복이 시작되기 전까지만 남긴다."""
    out, seen = [], {}
    for line in str(text or "").split("\n")[:-1]:
        key = _line_key(line)
        if len(key) >= 10:
            seen[key] = seen.get(key, 0) + 1
            if seen[key] >= 3:
                break
        out.append(line)
    return "\n".join(out).strip()
