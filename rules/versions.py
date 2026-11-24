"""판본 표 읽기와 판본 고르기.

판본 값은 코드에 적지 않고 data/versions_<문서판본>.json 에서 읽는다.
V3 문서가 오면 versions_v3.json 만 추가하면 된다.

    from rules.versions import Registry
    reg = Registry("V2")
    reg.pick("2026-10-01T00:00")          # 'V2'
    reg.value("birthday_cap", "V1")      # 25
"""
import json
from datetime import datetime
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"

BEFORE = "규정 범위 밖"    # 첫 판본 시행 전 (부칙 제1조, 표 53)
UNKNOWN = "추가 확인"      # 기준일을 모름 (표 53)


def _dt(x):
    if x is None or isinstance(x, datetime):
        return x
    s = str(x).strip()
    if len(s) == 10:                      # 날짜만 있으면 그날 00시
        s += "T00:00"
    return datetime.fromisoformat(s)


class Registry:
    def __init__(self, doc_release="V2", path=None):
        p = Path(path) if path else DATA / f"versions_{doc_release.lower()}.json"
        self.data = json.loads(p.read_text(encoding="utf-8"))
        self.doc_release = self.data["doc_release"]
        self.items = {it["key"]: it for it in self.data["items"]}
        self.versions = [(v["id"], _dt(v["start"]), _dt(v["end"])) for v in self.data["versions"]]

    def pick(self, when):
        """기준일(날짜 또는 날짜+시각) → 'V1' / 'V2' / ... / BEFORE / UNKNOWN.

        구간은 [start, end) 이라 경계 시각은 새 판본이 된다 (부칙 제2조).
        """
        t = _dt(when)
        if t is None:
            return UNKNOWN
        for vid, start, end in self.versions:
            if start <= t and (end is None or t < end):
                return vid
        return BEFORE

    def value(self, key, version):
        return self.items[key]["values"][version]

    def item(self, key):
        return self.items[key]

    def item_value_at(self, key, when):
        """항목 key 의 기준일 when 에서의 (판본, 값). 판본을 못 정하면 값은 None."""
        v = self.pick(when)
        vals = self.items[key]["values"]
        return v, vals.get(v)

    @property
    def defaults(self):
        return self.data["defaults"]
