"""질문 정리 도구 (그래프 ① 노드).

하는 일
  1. 글자 정리: 전각→반각(４２→42), 이모티콘·ㅋㅋ·ㅠㅠ·ㅇㅇ 제거, 느낌표·물음표 반복 줄이기
  2. 형식 지시 떼기: 예/아니오만, 숫자만, 300~500자, 한 문장, 객관식 보기(A~D) …
  3. 잡담 걸러내기: 사건 정보(숫자·날짜·규약 단어)나 물음이 있는 줄만 남긴다
  4. 안전장치: 원문의 숫자가 정리본에서 빠지면, 그 숫자가 있는 줄을 되살린다

원문은 그대로 두고 정리본을 따로 만든다. 답 쓰기에는 원문, 검색에는 정리본을 쓸 수 있다.
"""
import re
import unicodedata

EMOJI_RE = re.compile(r"[\U0001F000-\U0001FAFF☀-➿️‍]")
JAMO_RE = re.compile(r"[ㄱ-ㅎㅏ-ㅣ\u1100-\u11FF]{2,}|(?<![가-힣])[ㄱ-ㅎㅏ-ㅣ\u1100-\u11FF](?![가-힣])")

# 사건이나 규약과 상관없는 정형 잡담 줄
FILLER_LINES = [
    r"질문에 적은 (출력 )?형식(으로|대로) 부탁", r"이거 좀 봐주(셈|세요)", r"이거 좀 봐줘", r"나 이거 헷갈림",
    r"질문에 적은 (출력 )?형식대로 부탁", r"아래 상황을 확인해 주시겠어요", r"아래 경우가 궁금해",
    r"먼저 확인할 질문과 관련 문장을 적습니다", r"안녕하세요,? 규약 적용을 문의드립니다",
    r"정리 요청: 인용한 사례와 질문을 규약 기준으로 확인해 주세요",
    r"\[붙여 넣은 상담 메모", r"^\?+$", r"^!+$",
]
FILLER_RE = re.compile("|".join(FILLER_LINES))

# 규약·사건 단어 (이 중 하나라도 있으면 사건 정보로 본다)
RULE_WORDS = (
    "공동비 공동 합의 동의 생신 생일 가전 숙박 상한 한도 판본 결제 예약 근무 당번 교대 야간 설거지 음식물 "
    "투자 주문 매수 매도 주식 종목 대출 비상금 계좌 적립 예금 연금 전세 보증금 매매 견적 금리 "
    "취소 환불 수수료 손님 방문 고지 가사 외주 명절 시댁 처가 장모 장인 시어 부모님 수리 누수 "
    "구독 차량 택시 귀가 회식 영수증 증빙 회의록 규약 공용 청소기 밥솥 공기청정기 가용액 예산 할인 배송 "
    "나미 감석 개정 경과 부칙 기준일 확정 상여 이체 선납 순자산 유동 보험 여행 호텔 박수 1박 2박 3박 "
).split()
# 물음·요청으로 끝나는 문장 (문장 끝에서만 본다)
ASK_END_RE = re.compile(
    r"(\?|까|나|니|야|해|돼|되나|인가|는가|줘|주세요|주라|해쥬|부탁헤|부탁함|"
    r"답해|알려|구해|계산해|판단해|판정해|설명해|제안해|적어|골라)\s*[.!]?\s*$"
)
ASK_WORD_RE = re.compile(r"무엇|뭐야|어디|얼마|몇|어떻게|어느")
QUOTE_RE = re.compile(r"^(메모 인용|문의자 기록|문의자 발언|메모 낭독)\s*[:>]")
NUM_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
FORMAT_HINT_RE = re.compile(r"예\s*/\s*아니|예 또는 아니|숫자만|숫자로|단위로|자로|한 문장|24시간제|소수|골라|고르")


def normalize(text):
    t = JAMO_RE.sub(" ", text or "")             # NFKC 전에 한 번 (ㅋㅋ)
    t = unicodedata.normalize("NFKC", t)
    t = EMOJI_RE.sub(" ", t)
    t = JAMO_RE.sub(" ", t)                       # NFKC 후에 한 번 (ᄒᄒ)
    t = re.sub(r"([!?.~])\1{1,}", r"\1", t)          # !!!! → !, ????? → ?
    t = re.sub(r"[ \t]+", " ", t)
    return t


def extract_format(text):
    """질문이 요구한 답 형식."""
    t = normalize(text)
    fmt = {}
    if re.search(r"예\s*/\s*아니오|예 또는 아니오|예/아니요", t):
        fmt["kind"] = "yesno"
    opts = re.findall(r"(?:^|\s)([A-D])\s*[가-힣0-9]", t)
    if len(set(opts)) >= 3:
        fmt["kind"] = "choice"
        fmt["multi"] = bool(re.search(r"모두 고르|모두 골라|여러 개|해당하는 것을 모두", t))
    if re.search(r"숫자만|숫자로|만 원 단위로|원 단위로|금액만|시각만|시각으로 답|몇\s*(만\s*)?원|몇\s*%|몇\s*주|몇\s*시|몇\s*거래일", t):
        fmt.setdefault("kind", "number")
    m = re.search(r"(\d{2,4})\s*[~∼-]\s*(\d{2,4})\s*자", t)
    if m:
        fmt["length"] = [int(m[1]), int(m[2])]
        fmt["kind"] = "free"                  # 글자 수를 정한 답은 숫자 답이 아니라 서술형이다
    if "한 문장" in t:
        fmt["one_sentence"] = True
    if re.search(r"소수\s*둘째", t):
        fmt["decimals"] = 2
    if "24시간제" in t:
        fmt["clock24"] = True
    fmt.setdefault("kind", "free")
    return fmt


def _keep_sentence(s):
    s = s.strip()
    if not s or FILLER_RE.search(s):
        return False
    if NUM_RE.search(s) or QUOTE_RE.search(s) or re.search(r"(^|\s)[A-D]\s*[가-힣0-9]", s):
        return True
    if re.search(r"제\s*\d+\s*조|표\s*\d+|V\d|합의|동의", s):      # 합의 여부는 숫자가 없어도 판정에 필요한 사실이다
        return True
    words = sum(1 for w in RULE_WORDS if w in s)
    if words >= 2:
        return True
    if words >= 1 and (ASK_END_RE.search(s) or ASK_WORD_RE.search(s)):
        return True
    return bool(ASK_END_RE.search(s)) and len(s) <= 60 and not s.endswith(("니다.", "습니다", "다."))


def _sentences(line):
    return [x for x in re.split(r"(?<=[.!?])\s+(?=[^\s])", line) if x.strip()]


def _keep(line):
    """줄 안의 문장을 하나씩 보고, 남길 문장만 이어 붙인다. 남길 게 없으면 None."""
    keep = [x for x in _sentences(line.strip()) if _keep_sentence(x)]
    return " ".join(keep) if keep else None


def clean_question(text):
    """→ {"cleaned", "format", "dropped_lines", "restored", "orig_chars", "clean_chars"}"""
    norm = normalize(text)
    lines = [l.strip() for l in norm.split("\n")]
    kept = [k for k in (_keep(l) for l in lines) if k]
    # 안전장치: 원문 숫자가 모두 남았는지
    restored = []
    have = set(NUM_RE.findall(" ".join(kept)))
    for l in lines:
        miss = set(NUM_RE.findall(l)) - have
        if miss and l not in kept and l:
            kept.append(l)
            restored.append(l)
            have |= miss
    cleaned = "\n".join(dict.fromkeys(kept))
    if not cleaned.strip():                  # 다 지워졌으면 원문(정리만)으로
        cleaned = re.sub(r"\n{2,}", "\n", norm).strip()
    return {
        "cleaned": cleaned,
        "format": extract_format(cleaned + "\n" + "\n".join(
            l for l in lines if len(l) <= 80 and FORMAT_HINT_RE.search(l))),
        "dropped_lines": sum(1 for l in lines if l) - len(kept),
        "restored": restored,
        "orig_chars": len(text or ""),
        "clean_chars": len(cleaned),
    }


def numbers_preserved(orig, cleaned):
    a = set(NUM_RE.findall(normalize(orig)))
    b = set(NUM_RE.findall(cleaned))
    return a <= b, sorted(a - b)


_kiwi = None


def search_query(cleaned):
    """검색용: 띄어쓰기 복원 ('두금액 모두공동비' → '두 금액 모두 공동비'). 형식 지시 줄은 뺀다."""
    global _kiwi
    lines = [l for l in cleaned.split("\n")
             if not re.search(r"답해줘|알려줘|답해 주세요|형식|글자|한 문장으로|숫자만|예/아니오", l) or NUM_RE.search(l)]
    q = " ".join(lines) or cleaned
    try:
        if _kiwi is None:
            from kiwipiepy import Kiwi
            _kiwi = Kiwi()
        q = _kiwi.space(q, reset_whitespace=False)
    except Exception:
        pass
    return q
