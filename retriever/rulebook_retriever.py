"""
실시간 자동 버전 관리형 규약집 RAG 검색 모듈
- 잡담 폭탄 방어 + 날짜 기반 판본/부칙 라우팅(표 07) + chunk_id 중복 차단 + 안전 Fallback
"""

from datetime import date
import glob
import hashlib
import json
import os
import re
import threading
import unicodedata
import uuid
from typing import Any, Dict, List, Optional, Tuple

import torch
from qdrant_client import QdrantClient, models
from sentence_transformers import CrossEncoder, SentenceTransformer

JAMO_SPAM_RE = re.compile(r"[ㄱ-ㅎㅏ-ㅣ]+")
EMOJI_RE = re.compile(r"[\U00010000-\U0010ffff\u2600-\u27bf]")
MULTI_TILDE_RE = re.compile(r"(?<!\d)~+(?!\d)")

# 상담 메모/질문 래퍼 및 구어체 군더더기 패턴
WRAPPER_NOISE_RE = re.compile(
    r"(?:"
    r"\[붙여\s*넣은\s*상담\s*메모[^\]]*\]|"
    r"먼저\s*확인할\s*질문과\s*관련\s*문장을\s*적습니다\.?|"
    r"정리\s*요청\s*:[^\n]*|"
    r"(?:메모\s*(?:인용|낭독)|문의자\s*(?:기록|발언))\s*[:>：]\s*|"
    r"(?:^|\n)\s*(?:이거\s*좀\s*봐주[셈세]|나\s*이거\s*헷갈림|(?:질문에\s*적은\s*(?:출력\s*)?형식|ㅇㅇ\s*질문에\s*적은\s*형식))[^\n]*"
    r")"
)

OPENING_FLUFF_RE = re.compile(
    r"^(?:"
    r"안녕(?:하세요|하십니까|하신가요|해요)?|"
    r"반갑습니다|반가워요|반가워|하이|헬로|여보세요|"
    r"실례(?:지만|합니다)|죄송(?:하지만|한데|해요|합니다)|미안(?:한데|하지만)|"
    r"저기(?:요)?|있잖아요|혹시(?:나\s*해서)?|다름이\s*아니라|"
    r"좋은\s*(?:아침|하루|주말)(?:이에요|예요|되세요)?|"
    r"잘\s*지내셨(?:어요|나요)|바쁘(?:신데|실\s*텐데)|"
    r"(?:질문|문의)(?:\s*하나|\s*좀)?\s*(?:할게요|할게|드려요|드립니다|있어요|있습니다)|"
    r"(?:궁금한\s*게|여쭤볼\s*게)\s*(?:있는데|있어요|있어서요|있습니다)|"
    r"(?:갑자기\s*)?궁금해서(?:요)?|"
    r"하나만\s*(?:물어볼게|물어볼게요|여쭤볼게요|여쭤볼게)"
    r")(?=$|[\s,!?.])[\s,!?.]*"
)

CLOSING_FLUFF_RE = re.compile(
    r"[\s,!?.]*(?:"
    r"감사(?:합니다|해요|합니당|드려요|드립니다)|"
    r"고맙습니다|고마워요|고마워|땡큐|"
    r"수고(?:하세요|하십쇼|하셔요|해|해요)|"
    r"좋은\s*하루\s*(?:보내세요|되세요)|"
    r"(?:빠른\s*답변\s*|답변\s*|잘\s*)?부탁(?:드립니다|드려요|해요|해|합니다)|"
    r"미리\s*감사(?:드립니다|합니다)"
    r")[.!?\s]*$"
)

INLINE_FILLER_RE = re.compile(
    r"(?:^|\s)(?:아이고|어머나|세상에|헐|대박|우와|제발|혹시|어쨌든|아무튼|그러니까|근데|그런데)(?=$|[\s,!?.])"
)

# 표 01 문서 관리 기준 기본 시행 기간 (문서 내 파싱 시 자동 보강)
DEFAULT_VERSION_PERIODS: Dict[str, Tuple[str, str]] = {
    "V1": ("2026-06-01", "2026-09-30"),
    "V2": ("2026-10-01", "2026-11-30"),
    "V3": ("2026-12-01", ""),
}

# 표 07 개정 항목별 부칙 대조표 페이지 매핑 (data_v3 기준 V1/V2/V3 대조표)
ADDENDUM_TOPIC_RULES: List[Tuple[str, re.Pattern]] = [
    ("page_53", re.compile(r"담보|월\s*상환|실수령|원리금|매수\s*자금|대출\s*한도|주택\s*월")),
    ("page_54", re.compile(r"생신|가족\s*행사|장인|장모|가전|밥솥|청소기|공기청정기|선풍기|세탁기")),
    ("page_55", re.compile(r"야간\s*근무|연속\s*근무|보호\s*교대|보호\s*시간|음식물|배출|당번|음주|술자리|회식|귀가")),
    ("page_56", re.compile(r"손님|친구|당일\s*방문|외박|숙박\s*시작|1박|2박|3박|4박|퇴원\s*돌봄")),
    ("page_57", re.compile(r"여행|국내\s*숙박|균일|박별|예약\s*취소|순취소비|추가\s*매수|추가\s*주문|하락|거래일|개별주|종목")),
]

# 날짜 추출 정규식
YMD_DATE_RE = re.compile(
    r"(?<!\d)(20\d{2})\s*[년.\-/]\s*(1[0-2]|0?[1-9])\s*[월.\-/]\s*(3[01]|[12]\d|0?[1-9])\s*일?"
)
MD_KOR_DATE_RE = re.compile(
    r"(?<![0-9가-힣])(1[0-2]|0?[1-9])\s*월\s*(3[01]|[12]\d|0?[1-9])\s*일"
)
MD_SLASH_DATE_RE = re.compile(
    r"(?<![0-9./\-])(1[0-2]|0?[1-9])\s*/\s*(3[01]|[12]\d|0?[1-9])(?![0-9./\-배만억원점%])"
)
MONTH_ONLY_DATE_RE = re.compile(
    r"(?<![0-9개])(6|7|8|9|10|11|12)\s*월(?=\s*(?:초|중순|말|기준|당시|분|부터|까지|에|의|에는|에도|에서|으로|로|신청|확정|체결|결제|예약|발생|진행|적용|판본|시행|개정|이전|이후|전|후))"
)
FRACTION_CONTEXT_RE = re.compile(
    r"(?:비율|지분|분담|가중치|절반|할인|수준|가량|정도|이상|이하|초과|미만|만큼)"
)
SLASH_DATE_CONTEXT_RE = re.compile(
    r"(?:기준|당시|확정|신청|계약|체결|결제|예약|발생|시행|개정|부터|까지|에|의|날|일자|일|자|전|후|이전|이후)"
)


class RulebookRetriever:
    def __init__(
        self,
        db_path: str = "./qdrant_db",
        collection_name: str = "rulebook_versions",
        qdrant_url: Optional[str] = None,
        watch_dir: Optional[str] = "./chunks_dir",
        auto_sync_on_search: bool = True,
        embed_model_id: str = "nlpai-lab/KURE-v3",
        reranker_model_id: str = "nlpai-lab/KURE-Reranker-nano",
        device: str = "auto",
        use_8bit: bool = True,
        default_year: int = 2026,
    ):
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._watch_thread: Optional[threading.Thread] = None
        self._last_folder_snapshot: Dict[str, Tuple[int, int]] = {}
        self.default_year = default_year
        self.version_periods: Dict[str, Tuple[str, str]] = dict(DEFAULT_VERSION_PERIODS)

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        if self.device == "cuda":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            dtype = torch.float32

        self.collection_name = collection_name
        self.watch_dir = watch_dir
        self.auto_sync_on_search = auto_sync_on_search

        if self.device == "cuda" and use_8bit:
            from transformers import BitsAndBytesConfig
            bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)
            self.embedder = SentenceTransformer(
                embed_model_id,
                model_kwargs={"quantization_config": bnb_cfg, "device_map": {"": 0}},
            )
        else:
            self.embedder = SentenceTransformer(
                embed_model_id,
                device=self.device,
                model_kwargs={"torch_dtype": dtype},
            )

        self.reranker = CrossEncoder(
            reranker_model_id,
            max_length=8192,
            device=self.device,
        )
        if self.device == "cuda":
            self.reranker.model.to(dtype)

        if qdrant_url:
            self.client = QdrantClient(url=qdrant_url)
        else:
            self.client = QdrantClient(path=db_path)

        if self.watch_dir:
            os.makedirs(self.watch_dir, exist_ok=True)
            self.sync_folder()

    @staticmethod
    def clean_query_text(query: str) -> str:
        """NFKC 전각 숫자 정규화 + 줄바꿈(\n) 보존 + 래퍼/추임새 제거"""
        q = unicodedata.normalize("NFKC", query).strip()
        if not q:
            return ""

        q = JAMO_SPAM_RE.sub(" ", q)
        q = EMOJI_RE.sub(" ", q)
        q = MULTI_TILDE_RE.sub(" ", q)
        q = WRAPPER_NOISE_RE.sub("\n", q)
        q = re.sub(r"\^{2,}|;{2,}|T_T|ㅜ_ㅜ", " ", q)
        q = re.sub(r"!+|\.{2,}", " ", q)
        q = re.sub(r"\?{2,}", "?", q)

        lines = []
        for raw_line in q.splitlines():
            line = re.sub(r"[ \t]+", " ", raw_line).strip()
            if not line or re.fullmatch(r"[\W_]+", line):
                continue
            while True:
                prev = line
                line = OPENING_FLUFF_RE.sub("", line).strip()
                line = CLOSING_FLUFF_RE.sub("", line).strip()
                if line == prev:
                    break
            line = INLINE_FILLER_RE.sub(" ", line)
            line = re.sub(r"^[,.\s]+", "", line)
            line = re.sub(r"[ \t]+", " ", line).strip()
            if line:
                lines.append(line)

        return "\n".join(lines)

    def extract_core_query(
        self,
        query: str,
        version: Optional[str] = None,
        target_date: Optional[str] = None,
        only_latest: bool = True,
    ) -> str:
        """긴 잡담 폭탄(Haystack)이 들어온 경우 규약 관련 핵심 문단만 자동 추출"""
        with self._lock:
            cleaned = self.clean_query_text(query)
            if not cleaned or len(cleaned) <= 450:
                return cleaned

            seen = set()
            paras = []
            for p in cleaned.splitlines():
                p = p.strip()
                if p and p not in seen:
                    seen.add(p)
                    paras.append(p)

            if len(paras) < 6 or not self.client.collection_exists(self.collection_name):
                return "\n".join(paras)

            conditions = []
            if version is not None:
                norm_ver = self._normalize_version(version)
                if norm_ver != "ALL":
                    matched_versions = self._resolve_primary_versions(norm_ver)
                    if matched_versions:
                        conditions.append(
                            models.FieldCondition(
                                key="version", match=models.MatchAny(any=matched_versions)
                            )
                        )
            elif target_date is not None:
                matched_ver = self.resolve_version_by_date(target_date)
                if matched_ver:
                    conditions.append(
                        models.FieldCondition(
                            key="version", match=models.MatchValue(value=matched_ver)
                        )
                    )
            elif only_latest:
                conditions.append(
                    models.FieldCondition(
                        key="is_latest", match=models.MatchValue(value=True)
                    )
                )
            q_filter = models.Filter(must=conditions) if conditions else None

            if hasattr(self.embedder, "encode_query"):
                vecs = self.embedder.encode_query(paras, batch_size=16, normalize_embeddings=True)
            else:
                vecs = self.embedder.encode(paras, batch_size=16, normalize_embeddings=True)

            reqs = [
                models.QueryRequest(query=v.tolist(), filter=q_filter, limit=1, with_payload=True)
                for v in vecs
            ]
            batch_res = self.client.query_batch_points(
                collection_name=self.collection_name, requests=reqs
            )

            pairs, valid_idx = [], []
            for idx, (p, res) in enumerate(zip(paras, batch_res)):
                if res.points:
                    pairs.append((p, res.points[0].payload["text"]))
                    valid_idx.append(idx)

            scores = [-10.0] * len(paras)
            if pairs:
                preds = self.reranker.predict(pairs, batch_size=16)
                for idx, sc in zip(valid_idx, preds):
                    scores[idx] = float(sc)

            median_sc = sorted(scores)[len(scores) // 2]
            kept = []
            for p, sc in zip(paras, scores):
                has_q = "?" in p
                has_num = bool(re.search(r"\d", p))
                has_choice = bool(re.search(r"(?:^|\s)[ABCD](?=[\s\.\)]|[가-힣])", p))
                is_req_end = bool(re.search(r"(?:골라|답해|알려|설명|판단|판정|계산|제안)[가-힣\.\?!]*$", p))
                is_short_signal = len(p) <= 75 and (has_q or has_num or has_choice or is_req_end or "예/아니오" in p)

                if sc >= max(-1.5, median_sc + 1.5) or is_short_signal or ((has_q or has_num) and sc >= median_sc - 0.5):
                    kept.append(p)

            if len(kept) < 2:
                top_idxs = sorted(range(len(paras)), key=lambda i: scores[i], reverse=True)[:3]
                kept = [paras[i] for i in sorted(top_idxs)]

            return "\n".join(kept)

    @staticmethod
    def _normalize_version(version: str) -> str:
        return version.strip().upper()

    @staticmethod
    def _parse_version_num(version: str) -> int:
        nums = re.findall(r"\d+", version)
        return max(int(x) for x in nums) if nums else 0

    @staticmethod
    def _make_point_id(version: str, chunk_id: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{version}:{chunk_id}"))

    @staticmethod
    def _file_hash(file_path: str) -> str:
        with open(file_path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()[:16]

    @staticmethod
    def _prev_day_iso(iso_str: str) -> str:
        d = date.fromisoformat(iso_str)
        return date.fromordinal(d.toordinal() - 1).isoformat()

    def _update_periods_from_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """표 01 문서 관리에서 판본별 시행일(최초 시행 / 제N차 개정 시행)을 자동 파싱해 보강"""
        if not chunks:
            return
        head_text = "\n".join(c.get("text", "") for c in chunks[:2])

        init_m = re.search(
            r"최초\s*시행\s*\|\s*(20\d{2})\s*년\s*(\d{1,2})\s*월\s*(\d{1,2})\s*일",
            head_text,
        )
        revs = re.findall(
            r"제\s*(\d+)\s*차\s*개정\s*시행\s*\|\s*(20\d{2})\s*년\s*(\d{1,2})\s*월\s*(\d{1,2})\s*일",
            head_text,
        )
        if not init_m and not revs:
            return

        starts: Dict[int, str] = {}
        if init_m:
            y, m, d = int(init_m.group(1)), int(init_m.group(2)), int(init_m.group(3))
            starts[1] = f"{y:04d}-{m:02d}-{d:02d}"
            self.default_year = y
        for rev_no, y, m, d in revs:
            starts[int(rev_no) + 1] = f"{int(y):04d}-{int(m):02d}-{int(d):02d}"

        if not starts:
            return
        sorted_nums = sorted(starts.keys())
        for idx, v_num in enumerate(sorted_nums):
            v_key = f"V{v_num}"
            ef_from = starts[v_num]
            ef_to = self._prev_day_iso(starts[sorted_nums[idx + 1]]) if idx + 1 < len(sorted_nums) else ""
            self.version_periods[v_key] = (ef_from, ef_to)

    def _infer_effective_range(self, covered: List[str]) -> Tuple[str, str]:
        ranges = [self.version_periods[v] for v in covered if v in self.version_periods]
        if not ranges:
            return "", ""
        ef_from = min(r[0] for r in ranges if r[0]) if any(r[0] for r in ranges) else ""
        ef_to = "" if any(r[1] == "" for r in ranges) else max(r[1] for r in ranges)
        return ef_from, ef_to

    def _detect_version_info(self, jsonl_path: str) -> Tuple[str, List[str], int, str, str]:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            chunks = [json.loads(line) for line in f if line.strip()]
        if not chunks:
            raise ValueError(f"빈 JSONL 파일입니다: {jsonl_path}")

        self._update_periods_from_chunks(chunks)

        ef_from = chunks[0].get("effective_from", "")
        ef_to = chunks[0].get("effective_to", "")

        if chunks[0].get("version"):
            primary = self._normalize_version(str(chunks[0]["version"]))
            raw_cov = chunks[0].get("covered_versions") or [primary]
            covered = list(dict.fromkeys([self._normalize_version(str(v)) for v in raw_cov] + [primary]))
            if not ef_from and not ef_to:
                ef_from, ef_to = self._infer_effective_range(covered)
            return primary, covered, self._parse_version_num(primary), ef_from, ef_to

        stem = os.path.splitext(os.path.basename(jsonl_path))[0]
        fn_nums = re.findall(r"(?:^|[_\-\s.])v(\d{1,3})(?=$|[_\-\s.])", stem, re.IGNORECASE)
        if fn_nums:
            nums = sorted(set(int(x) for x in fn_nums))
            covered = [f"V{n}" for n in nums]
            if not ef_from and not ef_to:
                ef_from, ef_to = self._infer_effective_range(covered)
            return f"V{nums[-1]}", covered, nums[-1], ef_from, ef_to

        found_nums = set()
        for c in chunks:
            for m in re.findall(r"(?<![A-Za-z0-9])[Vv](\d{1,2})(?![A-Za-z0-9])", c.get("text", "")):
                found_nums.add(int(m))
        if found_nums:
            nums = sorted(found_nums)
            covered = [f"V{n}" for n in nums]
            if not ef_from and not ef_to:
                ef_from, ef_to = self._infer_effective_range(covered)
            return f"V{nums[-1]}", covered, nums[-1], ef_from, ef_to

        if not ef_from and not ef_to:
            ef_from, ef_to = self._infer_effective_range(["V1"])
        return "V1", ["V1"], 1, ef_from, ef_to

    def _get_current_folder_snapshot(self) -> Dict[str, Tuple[int, int]]:
        if not self.watch_dir or not os.path.isdir(self.watch_dir):
            return {}
        snapshot: Dict[str, Tuple[int, int]] = {}
        for fpath in sorted(glob.glob(os.path.join(self.watch_dir, "*.jsonl"))):
            fname = os.path.basename(fpath)
            if fname.startswith(".") or fname.endswith((".tmp", ".part")):
                continue
            try:
                st = os.stat(fpath)
                snapshot[fpath] = (st.st_mtime_ns, st.st_size)
            except OSError:
                continue
        return snapshot

    def _has_folder_changes(self) -> bool:
        return self._get_current_folder_snapshot() != self._last_folder_snapshot

    def _scroll_all(self, scroll_filter: Optional[models.Filter] = None, with_payload: Any = False) -> List[Any]:
        records, offset = [], None
        while True:
            points, next_offset = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=scroll_filter,
                limit=256,
                offset=offset,
                with_payload=with_payload,
                with_vectors=False,
            )
            records.extend(points)
            if next_offset is None:
                break
            offset = next_offset
        return records

    def _ensure_collection(self, vector_size: int) -> None:
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
            )

    def sync_folder(self, folder_path: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._lock:
            target_dir = folder_path or self.watch_dir
            if not target_dir or not os.path.isdir(target_dir):
                return self.list_versions()
            jsonl_files = [
                p for p in sorted(glob.glob(os.path.join(target_dir, "*.jsonl")))
                if not os.path.basename(p).startswith((".", "~")) and not p.endswith((".tmp", ".part"))
            ]
            if not jsonl_files:
                self._last_folder_snapshot = {}
                return self.list_versions()

            parsed_files, synced_snapshot = [], {}
            for fpath in jsonl_files:
                try:
                    st = os.stat(fpath)
                    primary, covered, ver_num, ef_from, ef_to = self._detect_version_info(fpath)
                    parsed_files.append((ver_num, fpath, primary, covered, ef_from, ef_to, st))
                except (json.JSONDecodeError, OSError, ValueError) as e:
                    print(f"[Sync 보류] {os.path.basename(fpath)} 읽기 대기: {e}")

            parsed_files.sort(key=lambda x: (x[0], x[1]))
            for ver_num, fpath, primary, covered, ef_from, ef_to, st in parsed_files:
                self.upsert_version(
                    jsonl_path=fpath, version=primary, covered_versions=covered,
                    effective_from=ef_from, effective_to=ef_to, set_as_latest=False,
                )
                synced_snapshot[fpath] = (st.st_mtime_ns, st.st_size)

            versions = self.list_versions()
            if versions:
                highest = max(versions, key=lambda v: v["version_num"])
                if not highest["is_latest"]:
                    self.set_latest_version(highest["version"])
            if target_dir == self.watch_dir:
                self._last_folder_snapshot = synced_snapshot
            return versions

    def start_watching(self, interval: float = 2.0) -> None:
        if not self.watch_dir:
            raise ValueError("watch_dir가 설정되어 있어야 실시간 감시가 가능합니다.")
        if self._watch_thread and self._watch_thread.is_alive():
            return
        self._stop_event.clear()

        def _watch_loop() -> None:
            while not self._stop_event.is_set():
                try:
                    if self._has_folder_changes():
                        print(f"\n[Watcher] '{self.watch_dir}' 변경 감지 -> 자동 업데이트 시작")
                        self.sync_folder()
                except Exception as e:
                    print(f"[Watcher 경고] 다음 주기에 재시도합니다: {e}")
                self._stop_event.wait(interval)

        self._watch_thread = threading.Thread(target=_watch_loop, daemon=True)
        self._watch_thread.start()
        print(f"[Watcher] '{self.watch_dir}' 실시간 감시 ON ({interval}초 주기)")

    def stop_watching(self) -> None:
        self._stop_event.set()
        if self._watch_thread and self._watch_thread.is_alive():
            self._watch_thread.join(timeout=5.0)
        self._watch_thread = None

    def upsert_version(
        self, jsonl_path: str, version: str, covered_versions: Optional[List[str]] = None,
        effective_from: str = "", effective_to: str = "", set_as_latest: bool = True, force_reembed: bool = False,
    ) -> int:
        with self._lock:
            norm_ver = self._normalize_version(version)
            ver_num = self._parse_version_num(norm_ver)
            norm_covered = list(dict.fromkeys([self._normalize_version(v) for v in (covered_versions or [])] + [norm_ver]))
            src_hash = self._file_hash(jsonl_path)

            if self.client.collection_exists(self.collection_name) and not force_reembed:
                existing_pts = self._scroll_all(
                    scroll_filter=models.Filter(must=[models.FieldCondition(key="version", match=models.MatchValue(value=norm_ver))]),
                    with_payload=["source_hash", "covered_versions", "effective_from", "effective_to", "is_latest"],
                )
                if existing_pts and existing_pts[0].payload.get("source_hash") == src_hash:
                    p0 = existing_pts[0].payload
                    if (
                        p0.get("covered_versions") != norm_covered
                        or p0.get("effective_from", "") != effective_from
                        or p0.get("effective_to", "") != effective_to
                    ):
                        self.client.set_payload(
                            collection_name=self.collection_name,
                            payload={
                                "covered_versions": norm_covered,
                                "effective_from": effective_from,
                                "effective_to": effective_to,
                            },
                            points=[p.id for p in existing_pts],
                        )
                    if set_as_latest and not p0.get("is_latest", False):
                        self.set_latest_version(norm_ver)
                    return len(existing_pts)

            with open(jsonl_path, "r", encoding="utf-8") as f:
                chunks = [json.loads(line) for line in f if line.strip()]
            docs = [c["text"] for c in chunks]
            if hasattr(self.embedder, "encode_document"):
                doc_embs = self.embedder.encode_document(docs, batch_size=8, normalize_embeddings=True)
            else:
                doc_embs = self.embedder.encode(docs, batch_size=8, normalize_embeddings=True)
            self._ensure_collection(int(doc_embs.shape[1]))

            old_ver_pts = self._scroll_all(
                scroll_filter=models.Filter(must=[models.FieldCondition(key="version", match=models.MatchValue(value=norm_ver))]),
                with_payload=False,
            )
            if old_ver_pts:
                self.client.delete(collection_name=self.collection_name, points_selector=models.PointIdsList(points=[p.id for p in old_ver_pts]))
            if set_as_latest:
                old_latest_pts = self._scroll_all(
                    scroll_filter=models.Filter(must=[models.FieldCondition(key="is_latest", match=models.MatchValue(value=True))]),
                    with_payload=False,
                )
                if old_latest_pts:
                    self.client.set_payload(collection_name=self.collection_name, payload={"is_latest": False}, points=[p.id for p in old_latest_pts])

            points = [
                models.PointStruct(
                    id=self._make_point_id(norm_ver, chunk["chunk_id"]),
                    vector=vec.tolist(),
                    payload={
                        "version": norm_ver, "version_num": ver_num, "covered_versions": norm_covered,
                        "effective_from": effective_from, "effective_to": effective_to, "is_latest": set_as_latest,
                        "source_hash": src_hash, "chunk_id": chunk["chunk_id"], "page": chunk["page"],
                        "chapter": chunk.get("chapter", ""), "topic": chunk.get("topic", ""),
                        "articles": chunk.get("articles", []), "tables": chunk.get("tables", []), "text": chunk["text"],
                    },
                )
                for chunk, vec in zip(chunks, doc_embs)
            ]
            self.client.upsert(collection_name=self.collection_name, points=points)
            print(f"[Qdrant] {norm_ver} (포함={norm_covered}, {effective_from}~{effective_to or '현재'}) 저장 완료 ({len(points)}개 청크)")
            return len(points)

    def set_latest_version(self, version: str) -> None:
        with self._lock:
            if not self.client.collection_exists(self.collection_name):
                raise ValueError("컬렉션이 아직 생성되지 않았습니다.")
            norm_ver = self._normalize_version(version)
            matched_primary = self._resolve_primary_versions(norm_ver)
            if not matched_primary:
                raise ValueError(f"Qdrant에서 '{norm_ver}' 판본을 찾을 수 없습니다.")
            target_ver = matched_primary[0]
            target_pts = self._scroll_all(
                scroll_filter=models.Filter(must=[models.FieldCondition(key="version", match=models.MatchValue(value=target_ver))]),
                with_payload=False,
            )
            old_latest_pts = self._scroll_all(
                scroll_filter=models.Filter(must=[models.FieldCondition(key="is_latest", match=models.MatchValue(value=True))]),
                with_payload=False,
            )
            if old_latest_pts:
                self.client.set_payload(collection_name=self.collection_name, payload={"is_latest": False}, points=[p.id for p in old_latest_pts])
            self.client.set_payload(collection_name=self.collection_name, payload={"is_latest": True}, points=[p.id for p in target_pts])
            print(f"[Qdrant] 기본 최신판을 '{target_ver}'(으)로 자동 지정했습니다.")

    def delete_version(self, version: str) -> int:
        with self._lock:
            if not self.client.collection_exists(self.collection_name):
                return 0
            norm_ver = self._normalize_version(version)
            pts = self._scroll_all(
                scroll_filter=models.Filter(must=[models.FieldCondition(key="version", match=models.MatchValue(value=norm_ver))]),
                with_payload=False,
            )
            if not pts:
                return 0
            self.client.delete(collection_name=self.collection_name, points_selector=models.PointIdsList(points=[p.id for p in pts]))
            return len(pts)

    def list_versions(self) -> List[Dict[str, Any]]:
        with self._lock:
            if not self.client.collection_exists(self.collection_name):
                return []
            all_pts = self._scroll_all(with_payload=["version", "version_num", "covered_versions", "effective_from", "effective_to", "is_latest"])
            summary: Dict[str, Dict[str, Any]] = {}
            for pt in all_pts:
                p = pt.payload
                ver = p["version"]
                if ver not in summary:
                    summary[ver] = {
                        "version": ver, "version_num": p.get("version_num", 0),
                        "covered_versions": p.get("covered_versions", [ver]),
                        "effective_from": p.get("effective_from", ""), "effective_to": p.get("effective_to", ""),
                        "is_latest": p.get("is_latest", False), "chunk_count": 0,
                    }
                summary[ver]["chunk_count"] += 1
            return sorted(summary.values(), key=lambda x: x["version_num"])

    def _resolve_primary_versions(self, requested_version: str) -> List[str]:
        norm_req = self._normalize_version(requested_version)
        return [v["version"] for v in self.list_versions() if norm_req == v["version"] or norm_req in v["covered_versions"]]

    def map_date_to_fine_version(self, target_date: str) -> Optional[str]:
        """세부 판본(V1 / V2 / V3) 단위로 날짜 매핑"""
        candidates = []
        for ver, (ef_from, ef_to) in self.version_periods.items():
            if (not ef_from or target_date >= ef_from) and (not ef_to or target_date <= ef_to):
                candidates.append((ef_from, self._parse_version_num(ver), ver))
        if candidates:
            candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
            return candidates[0][2]
        return self.resolve_version_by_date(target_date)

    def resolve_version_by_date(self, target_date: str) -> Optional[str]:
        candidates = [
            v for v in self.list_versions()
            if (not v["effective_from"] or target_date >= v["effective_from"])
            and (not v["effective_to"] or target_date <= v["effective_to"])
        ]
        if not candidates:
            return None
        candidates.sort(key=lambda x: (x["effective_from"], x["version_num"]), reverse=True)
        return candidates[0]["version"]

    @staticmethod
    def detect_addendum_pages(query: str) -> List[str]:
        """표 07 개정 항목 키워드 기반으로 필요한 부칙 대조표 페이지(page_53~57) 반환"""
        pages: List[str] = []
        for page_id, pat in ADDENDUM_TOPIC_RULES:
            if pat.search(query):
                pages.append(page_id)
        return pages

    def analyze_query_dates(self, query: str) -> Dict[str, Any]:
        """질문 내 날짜 표현 추출, 판본(V1/V2/V3) 매핑, 개정 항목 부칙 여부 판별"""
        norm_q = unicodedata.normalize("NFKC", query)
        spans: List[Tuple[int, int]] = []
        extracted: List[Dict[str, str]] = []

        def _overlaps(s: int, e: int) -> bool:
            return any(not (e <= ps or s >= pe) for ps, pe in spans)

        def _try_add(raw: str, y: int, m: int, d: int, s: int, e: int) -> None:
            if _overlaps(s, e):
                return
            try:
                iso = date(y, m, d).isoformat()
            except ValueError:
                return
            fine_ver = self.map_date_to_fine_version(iso) or "V3"
            spans.append((s, e))
            extracted.append({"raw": raw.strip(), "iso": iso, "version": fine_ver})

        for m in YMD_DATE_RE.finditer(norm_q):
            _try_add(m.group(0), int(m.group(1)), int(m.group(2)), int(m.group(3)), m.start(), m.end())

        for m in MD_KOR_DATE_RE.finditer(norm_q):
            _try_add(m.group(0), self.default_year, int(m.group(1)), int(m.group(2)), m.start(), m.end())

        for m in MD_SLASH_DATE_RE.finditer(norm_q):
            if _overlaps(m.start(), m.end()):
                continue
            mon, day = int(m.group(1)), int(m.group(2))
            ctx_window = norm_q[max(0, m.start() - 12):min(len(norm_q), m.end() + 12)]
            if FRACTION_CONTEXT_RE.search(ctx_window) and not SLASH_DATE_CONTEXT_RE.search(ctx_window):
                continue
            if mon < 6 and not SLASH_DATE_CONTEXT_RE.search(norm_q[m.end():min(len(norm_q), m.end() + 10)]):
                continue
            _try_add(m.group(0), self.default_year, mon, day, m.start(), m.end())

        for m in MONTH_ONLY_DATE_RE.finditer(norm_q):
            if _overlaps(m.start(), m.end()):
                continue
            mon = int(m.group(1))
            _try_add(m.group(0), self.default_year, mon, 15, m.start(), m.end())

        versions = list(dict.fromkeys(d["version"] for d in extracted))
        latest_ver = self.list_versions()[-1]["version"] if self.list_versions() else "V3"
        past_versions = [v for v in versions if self._parse_version_num(v) < self._parse_version_num(latest_ver)]
        addendum_pages = self.detect_addendum_pages(norm_q)

        return {
            "dates": extracted,
            "versions": versions,
            "past_versions": past_versions,
            "has_past_date": len(past_versions) > 0,
            "addendum_pages": addendum_pages,
            "is_amended_topic": len(addendum_pages) > 0,
            "latest_version": latest_ver,
        }

    def _fetch_chunks_by_ids(
        self,
        version: str,
        chunk_ids: List[str],
        score_query: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """특정 판본에서 chunk_id 리스트로 청크를 직접 조회"""
        if not chunk_ids or not self.client.collection_exists(self.collection_name):
            return []
        matched_versions = self._resolve_primary_versions(version)
        if not matched_versions:
            return []

        pts = self._scroll_all(
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(key="version", match=models.MatchAny(any=matched_versions)),
                    models.FieldCondition(key="chunk_id", match=models.MatchAny(any=chunk_ids)),
                ]
            ),
            with_payload=True,
        )
        if not pts:
            return []

        order_map = {cid: idx for idx, cid in enumerate(chunk_ids)}
        pts.sort(key=lambda p: order_map.get(p.payload.get("chunk_id", ""), 999))

        scores = [5.0] * len(pts)
        if score_query:
            pairs = [(score_query, p.payload["text"]) for p in pts]
            preds = self.reranker.predict(pairs, batch_size=4)
            scores = [float(s) for s in preds]

        recs: List[Dict[str, Any]] = []
        for idx, (pt, sc) in enumerate(zip(pts, scores), start=1):
            payload = pt.payload
            recs.append({
                "rank": idx,
                "version": payload["version"],
                "covered_versions": payload.get("covered_versions", [payload["version"]]),
                "is_latest": payload.get("is_latest", False),
                "effective_from": payload.get("effective_from", ""),
                "effective_to": payload.get("effective_to", ""),
                "chunk_id": payload["chunk_id"],
                "page": payload["page"],
                "chapter": payload.get("chapter", ""),
                "topic": payload.get("topic", ""),
                "articles": payload.get("articles", []),
                "tables": payload.get("tables", []),
                "rerank_score": round(sc, 4),
                "dense_score": 1.0,
                "text": payload["text"],
            })
        return recs

    def search(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int = 15,
        version: Optional[str] = None,
        target_date: Optional[str] = None,
        only_latest: bool = True,
        chapter: Optional[str] = None,
        page: Optional[int] = None,
        clean_query: bool = True,
        min_rerank_score: Optional[float] = 0.0,
    ) -> List[Dict[str, Any]]:
        with self._lock:
            if self.watch_dir and self.auto_sync_on_search and self._has_folder_changes():
                self.sync_folder()
            if not self.client.collection_exists(self.collection_name):
                return []

            if clean_query:
                search_q = self.extract_core_query(
                    query, version=version, target_date=target_date, only_latest=only_latest
                )
            else:
                search_q = query.strip()

            if not search_q:
                return []

            conditions = []
            if version is not None:
                norm_ver = self._normalize_version(version)
                if norm_ver != "ALL":
                    matched_versions = self._resolve_primary_versions(norm_ver)
                    if not matched_versions:
                        return []
                    conditions.append(models.FieldCondition(key="version", match=models.MatchAny(any=matched_versions)))
            elif target_date is not None:
                matched_ver = self.resolve_version_by_date(target_date)
                if matched_ver is None:
                    return []
                conditions.append(models.FieldCondition(key="version", match=models.MatchValue(value=matched_ver)))
            elif only_latest:
                conditions.append(models.FieldCondition(key="is_latest", match=models.MatchValue(value=True)))

            if chapter:
                conditions.append(models.FieldCondition(key="chapter", match=models.MatchValue(value=chapter)))
            if page is not None:
                conditions.append(models.FieldCondition(key="page", match=models.MatchValue(value=page)))

            query_filter = models.Filter(must=conditions) if conditions else None

            if hasattr(self.embedder, "encode_query"):
                q_vec = self.embedder.encode_query([search_q], normalize_embeddings=True)[0].tolist()
            else:
                q_vec = self.embedder.encode([search_q], normalize_embeddings=True)[0].tolist()

            hits = self.client.query_points(
                collection_name=self.collection_name,
                query=q_vec,
                query_filter=query_filter,
                limit=max(top_k, candidate_k),
                with_payload=True,
            ).points

            if not hits:
                return []

            cand_docs = [h.payload["text"] for h in hits]
            ranked = self.reranker.rank(search_q, cand_docs, batch_size=4)

            results: List[Dict[str, Any]] = []
            fallback_results: List[Dict[str, Any]] = []

            for item in ranked:
                score = float(item["score"])
                hit = hits[item["corpus_id"]]
                payload = hit.payload
                rec = {
                    "rank": len(results) + 1,
                    "version": payload["version"],
                    "covered_versions": payload.get("covered_versions", [payload["version"]]),
                    "is_latest": payload.get("is_latest", False),
                    "effective_from": payload.get("effective_from", ""),
                    "effective_to": payload.get("effective_to", ""),
                    "chunk_id": payload["chunk_id"],
                    "page": payload["page"],
                    "chapter": payload.get("chapter", ""),
                    "topic": payload.get("topic", ""),
                    "articles": payload.get("articles", []),
                    "tables": payload.get("tables", []),
                    "rerank_score": round(score, 4),
                    "dense_score": round(float(hit.score), 4),
                    "text": payload["text"],
                }
                if len(fallback_results) < min(3, top_k) and score >= -1.8:
                    fallback_results.append(rec)

                if min_rerank_score is not None and score < min_rerank_score:
                    continue
                if len(results) >= top_k:
                    break
                results.append(rec)

            if not results and fallback_results:
                for idx, r in enumerate(fallback_results, start=1):
                    r["rank"] = idx
                return fallback_results

            return results

    def search_with_date_routing(
        self,
        query: str,
        past_versions: List[str],
        addendum_pages: List[str],
        latest_version: str = "V3",
        total_top_k: int = 5,
        candidate_k: int = 15,
        min_rerank_score: Optional[float] = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        개정 항목 + 과거 날짜(V1/V2) 질의용 정밀 라우팅 (chunk_id 중복 제거):
        1) data_v3(latest_version)에서 주제에 맞는 부칙 대조표(page_53~57)를 근거 1순위로 확보
        2) 본문 청크는 data.jsonl(V1/V2)에서 검색하여 V3 본문 수치(7거래일·35만 등) 혼입 차단
        3) 부족분만 latest_version에서 중복 없는 chunk_id로 보충
        """
        with self._lock:
            combined: List[Dict[str, Any]] = []
            seen_chunk_ids = set()

            # 1. data_v3(통합 합본)에서 표 07이 가리키는 해당 부칙 페이지(page_53~57) 최상단 고정
            target_addendums = addendum_pages[:2]
            if target_addendums:
                add_chunks = self._fetch_chunks_by_ids(
                    version=latest_version,
                    chunk_ids=target_addendums,
                    score_query=query,
                )
                for h in add_chunks:
                    cid = h["chunk_id"]
                    if cid not in seen_chunk_ids:
                        seen_chunk_ids.add(cid)
                        combined.append(h)

            # 2. 과거 본문(data.jsonl: V1/V2) 검색 (부칙/중복 chunk_id는 제외)
            past_ver_for_body = past_versions[0] if past_versions else "V2"
            remain = max(1, total_top_k - len(combined))
            body_hits = self.search(
                query=query,
                top_k=total_top_k,
                candidate_k=candidate_k,
                version=past_ver_for_body,
                clean_query=False,
                min_rerank_score=min_rerank_score,
            )
            for h in body_hits:
                cid = h["chunk_id"]
                if cid in seen_chunk_ids:
                    continue
                seen_chunk_ids.add(cid)
                combined.append(h)
                if len(combined) >= total_top_k:
                    break

            # 3. 혹시 V1/V2에 본문 히트가 부족해 total_top_k 미만이면 V3에서 중복 없는 청크만 보충
            if len(combined) < total_top_k:
                extra_hits = self.search(
                    query=query,
                    top_k=total_top_k,
                    candidate_k=candidate_k,
                    version=latest_version,
                    clean_query=False,
                    min_rerank_score=min_rerank_score,
                )
                for h in extra_hits:
                    cid = h["chunk_id"]
                    if cid in seen_chunk_ids:
                        continue
                    seen_chunk_ids.add(cid)
                    combined.append(h)
                    if len(combined) >= total_top_k:
                        break

            for idx, r in enumerate(combined, start=1):
                r["rank"] = idx
            return combined

    @staticmethod
    def format_context(results: List[Dict[str, Any]]) -> str:
        if not results:
            return ""
        blocks = []
        for r in results:
            covered = r.get("covered_versions", [r["version"]])
            ver_label = "/".join(covered) if len(covered) > 1 else r["version"]
            ver_tag = f"{ver_label} (최신)" if r["is_latest"] else f"{ver_label} (과거판)"
            period = f"시행 {r['effective_from']}~{r['effective_to']}" if r["effective_from"] else "시행일 전체"
            header = f"[근거 {r['rank']} | 판본: {ver_tag} | {period} | {r['chunk_id']} | {r['chapter']} > {r['topic']}]"
            blocks.append(f"{header}\n{r['text']}")
        return "\n\n---\n\n".join(blocks)

    def close(self) -> None:
        self.stop_watching()
        self.client.close()