"""질문에서 날짜 찾기. 연도가 없으면 2026년 (평가 키트 README)."""
import re
import unicodedata

DEFAULT_YEAR = 2026


def find_dates(text):
    t = unicodedata.normalize("NFKC", text or "")
    out = []
    for m in re.finditer(r"(20\d\d)\s*[-./년]\s*(\d{1,2})\s*[-./월]\s*(\d{1,2})\s*일?", t):
        out.append((m.start(), int(m[1]), int(m[2]), int(m[3])))
    for m in re.finditer(r"(?<![\d.\-/])(\d{1,2})\s*월\s*(\d{1,2})\s*일", t):
        if not any(s <= m.start() < s + 14 for s, *_ in out):
            out.append((m.start(), DEFAULT_YEAR, int(m[1]), int(m[2])))
    out.sort()
    return [f"{y:04d}-{mo:02d}-{d:02d}" for _, y, mo, d in out if 1 <= mo <= 12 and 1 <= d <= 31]


def find_date_spans(text):
    """[(시작 위치, 끝 위치, 'YYYY-MM-DD')]. 같은 날짜가 여러 번 나오면 각각 돌려준다."""
    t = unicodedata.normalize("NFKC", text or "")
    out = []
    for m in re.finditer(r"(20\d\d)\s*[-./년]\s*(\d{1,2})\s*[-./월]\s*(\d{1,2})\s*일?", t):
        out.append((m.start(), m.end(), int(m[1]), int(m[2]), int(m[3])))
    for m in re.finditer(r"(?<![\d.\-/])(\d{1,2})\s*월\s*(\d{1,2})\s*일", t):
        if not any(s <= m.start() < e for s, e, *_ in out):
            out.append((m.start(), m.end(), DEFAULT_YEAR, int(m[1]), int(m[2])))
    out.sort()
    return [(s, e, f"{y:04d}-{mo:02d}-{d:02d}") for s, e, y, mo, d in out if 1 <= mo <= 12 and 1 <= d <= 31]


def find_date(text):
    ds = find_dates(text)
    return ds[0] if ds else None
