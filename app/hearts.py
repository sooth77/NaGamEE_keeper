"""나감이 하트 그림 (SVG). 인터넷 없이 그려진다.

- 나감이 본인: 손그림 마젠타 하트. 표정만 바뀌고 색은 그대로.
- 판정 하트 5개: 반듯한 하트, 판정마다 색과 표시가 다르다.
"""

# 판정 이름 → (표정, 하트 안쪽 색, 선 색, 얼굴 색, 칩 바탕, 칩 글자)
VERDICTS = {
    "가능":           ("ok",    "#A9D67A", "#5E8F2E", "#2C2C2A", "#E6F4D3", "#4A7A22"),
    "사전 합의 필요":  ("agree", "#CFAA84", "#8A6340", "#2C2C2A", "#F2E6D8", "#7A5535"),
    "공동 처리 불가":  ("no",    "#8C2341", "#5E1229", "#FFFFFF", "#F2D9DF", "#7A1F3D"),
    "추가 확인":      ("ask",   "#93CFF2", "#3E8FC4", "#2C2C2A", "#DDF0FB", "#2A75A8"),
    "규정에 없음":    ("none",  "#C4C2BB", "#7A7974", "#2C2C2A", "#F1EFE8", "#5F5E5A"),
}
ME = ("#E26CC9", "#A3168A")  # 나감이: 안쪽 마젠타(조금 연하게), 선은 진한 마젠타
ME_INK = "#2C2C2A"           # 얼굴은 원래대로 진한 색

CLEAN = "M32 56 C14 44 6 33 9 22 C12 12 26 9 32 20 C38 9 52 12 55 22 C58 33 50 44 32 56 Z"
HAND = ("M32 55 C21 48 7 37 9 23 C10 13 19 8 26 11 C29 13 31 16 32 20 "
        "C33 15 37 10 44 10 C53 10 57 18 55 27 C53 38 42 47 32 55 Z")
HAND2 = "M31 52 C20 45 10 35 11 24 C12 16 18 12 25 13 M34 18 C36 13 40 12 45 12 C51 13 55 19 53 27"
INK = "#2C2C2A"


def _face(mood, line, ink=INK):
    INK = ink  # noqa: N806
    eyes = f'<circle cx="24" cy="30" r="2.6" fill="{INK}"/><circle cx="40" cy="30" r="2.6" fill="{INK}"/>'
    mouth = f'<path d="M27 38 Q32 43 37 38" stroke="{INK}" stroke-width="2" fill="none" stroke-linecap="round"/>'
    extra = ""
    if mood == "ok":
        eyes = (f'<path d="M21 31 Q24 27 27 31 M37 31 Q40 27 43 31" stroke="{INK}" '
                f'stroke-width="2" fill="none" stroke-linecap="round"/>')
    elif mood == "no":
        eyes = (f'<path d="M20 27 L27 29 M44 27 L37 29" stroke="{INK}" stroke-width="2" stroke-linecap="round"/>'
                f'<circle cx="24" cy="32" r="2.4" fill="{INK}"/><circle cx="40" cy="32" r="2.4" fill="{INK}"/>')
        mouth = f'<path d="M27 40 L37 40" stroke="{INK}" stroke-width="2" stroke-linecap="round"/>'
        extra = f'<path d="M28 58 L36 65 M36 58 L28 65" stroke="{line}" stroke-width="3" stroke-linecap="round"/>'
    elif mood == "agree":
        mouth = f'<circle cx="32" cy="40" r="2.6" fill="none" stroke="{INK}" stroke-width="2"/>'
        extra = f'<path d="M59 4 l0 10 M59 19 l0 0.5" stroke="{line}" stroke-width="3" stroke-linecap="round"/>'
    elif mood == "ask":
        mouth = f'<path d="M28 40 Q32 38 36 40" stroke="{INK}" stroke-width="2" fill="none" stroke-linecap="round"/>'
        extra = f'<text x="55" y="17" font-size="15" font-weight="500" fill="{line}">?</text>'
    elif mood == "none":
        mouth = (f'<path d="M27 40 Q30 38 32 40 Q34 42 37 40" stroke="{INK}" stroke-width="2" '
                 f'fill="none" stroke-linecap="round"/>')
        extra = f'<text x="50" y="10" font-size="12" fill="{line}">...</text>'
    blush = '<circle cx="18" cy="36" r="3" fill="#F4C0D1"/><circle cx="46" cy="36" r="3" fill="#F4C0D1"/>'
    return blush + eyes + mouth + extra


def _svg(body, face, size):
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 66 66" xmlns="http://www.w3.org/2000/svg" '
            f'aria-hidden="true">{body}{face}</svg>')


def nagami(mood="ok", size=54):
    """나감이 본인 (손그림 마젠타 하트). mood: ok / agree / no / ask / none."""
    fill, line = ME
    body = (f'<path d="{HAND}" fill="{fill}" stroke="{line}" stroke-width="2.4" '
            f'stroke-linejoin="round" stroke-linecap="round"/>'
            f'<path d="{HAND2}" fill="none" stroke="{line}" stroke-width="1.2" '
            f'stroke-linecap="round" opacity="0.55"/>')
    return _svg(body, _face(mood, line, ME_INK), size)


def verdict_heart(verdict, size=42):
    """판정 하트 (반듯한 하트, 판정 색)."""
    mood, fill, line, ink, _, _ = VERDICTS.get(verdict, VERDICTS["규정에 없음"])
    body = f'<path d="{CLEAN}" fill="{fill}" stroke="{line}" stroke-width="2.4" stroke-linejoin="round"/>'
    return _svg(body, _face(mood, line, ink), size)


def mood_of(verdict):
    return VERDICTS.get(verdict, VERDICTS["규정에 없음"])[0]


def colors_of(verdict):
    """판정 칩 색 (바탕, 글자)."""
    v = VERDICTS.get(verdict, VERDICTS["규정에 없음"])
    return v[4], v[5]
