import os
import re
import glob
import json
import time
from datetime import date
from typing import Any, List, Optional, Set, Tuple
from tqdm import tqdm
from openai import OpenAI

try:
    from pdf_processor_final import sync_pdf_folder
except ImportError:
    def sync_pdf_folder(*args, **kwargs):
        print("[알림] sync_pdf_folder 없음 -> 기존 ./chunks_dir JSONL 파일 사용")
        return []

from rulebook_retriever import RulebookRetriever

# NFD/NFC 한글 파일명 자동 탐색. 평가 파일이 없어도 모듈은 불러올 수 있게 None으로 둔다.
Q_PATH = next(iter(sorted(glob.glob("./NewDataSet_V3/*01*.jsonl"))), None)
GT_PATH = next(iter(sorted(glob.glob("./NewDataSet_V3/*02*.jsonl"))), None)
OUT_PATH = "./eval_100_results_v12.jsonl"

VERSION_PATTERN = re.compile(r"(?<![A-Za-z0-9])[Vv](\d{1,2})(?![A-Za-z0-9])")
VERSION_COMPARE_RE = re.compile(
    r"(?:판본|버전|규약|개정|이전\s*기준|V\d).{0,12}(?:비교|달라|변경|차이|대조|바뀌)"
    r"|(?:비교|대조)(?:해|하|를|가)"
)
LONG_FORM_RE = re.compile(
    r"(\d{3}\s*[~\-∼]\s*\d{3}\s*자|\d{3}\s*자\s*(?:이상|내외|이내|정도))"
)


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _extract_version_from_item(item: Any) -> Optional[str]:
    if not isinstance(item, dict):
        return None
    for key, val in item.items():
        k_low = str(key).lower()
        if k_low in ("version", "ver") or k_low.endswith("_version"):
            if isinstance(val, str):
                m = VERSION_PATTERN.search(val)
                if m:
                    return f"V{int(m.group(1))}"
    for nested_key in ("metadata", "meta", "payload"):
        nested = item.get(nested_key)
        if isinstance(nested, dict):
            res = _extract_version_from_item(nested)
            if res:
                return res
    return None


def _extract_versions_from_json_data(data: Any) -> Set[int]:
    nums: Set[int] = set()
    if isinstance(data, dict):
        ver = _extract_version_from_item(data)
        if ver:
            nums.add(int(ver[1:]))
        for list_key in ("chunks", "items", "documents", "docs", "data", "records"):
            sub = data.get(list_key)
            if isinstance(sub, list):
                for item in sub:
                    sub_ver = _extract_version_from_item(item)
                    if sub_ver:
                        nums.add(int(sub_ver[1:]))
    elif isinstance(data, list):
        for item in data:
            ver = _extract_version_from_item(item)
            if ver:
                nums.add(int(ver[1:]))
    return nums


def _extract_versions_from_retriever(retriever: Any) -> Set[int]:
    nums: Set[int] = set()

    if hasattr(retriever, "list_versions") and callable(retriever.list_versions):
        try:
            for info in retriever.list_versions():
                if isinstance(info, dict):
                    m = VERSION_PATTERN.search(str(info.get("version", "")))
                    if m:
                        nums.add(int(m.group(1)))
                    for cov in info.get("covered_versions", []) or []:
                        m = VERSION_PATTERN.search(str(cov))
                        if m:
                            nums.add(int(m.group(1)))
        except Exception:
            pass
    if nums:
        return nums

    for attr in ("available_versions", "versions", "get_versions"):
        val = getattr(retriever, attr, None)
        if callable(val):
            try:
                val = val()
            except Exception:
                val = None
        if isinstance(val, (list, tuple, set)):
            for item in val:
                m = VERSION_PATTERN.search(str(item))
                if m:
                    nums.add(int(m.group(1)))
    if nums:
        return nums

    for attr in ("chunks", "docs", "documents", "metadata", "records"):
        data = getattr(retriever, attr, None)
        if isinstance(data, dict):
            data = list(data.values())
        if isinstance(data, list):
            for item in data:
                ver = _extract_version_from_item(item)
                if ver:
                    nums.add(int(ver[1:]))
    return nums


def _scan_dir_filenames_for_versions(dir_path: str, exts: Tuple[str, ...]) -> Set[int]:
    nums: Set[int] = set()
    if not dir_path or not os.path.isdir(dir_path):
        return nums
    for root, _, files in os.walk(dir_path):
        for fname in files:
            if exts and not fname.lower().endswith(exts):
                continue
            rel_path = os.path.relpath(os.path.join(root, fname), dir_path)
            for m in VERSION_PATTERN.finditer(rel_path):
                nums.add(int(m.group(1)))
    return nums


def detect_available_versions(
    retriever: Any = None,
    pdf_dir: str = "./pdf_dir",
    chunks_dir: str = "./chunks_dir",
    md_dir: str = "./cleaned_md",
) -> List[str]:
    nums: Set[int] = set()

    if retriever is not None:
        nums |= _extract_versions_from_retriever(retriever)

    if not nums and chunks_dir and os.path.isdir(chunks_dir):
        for root, _, files in os.walk(chunks_dir):
            for fname in files:
                fpath = os.path.join(root, fname)
                low = fname.lower()
                try:
                    if low.endswith(".jsonl"):
                        with open(fpath, "r", encoding="utf-8") as f:
                            for line in f:
                                line = line.strip()
                                if line:
                                    nums |= _extract_versions_from_json_data(json.loads(line))
                    elif low.endswith(".json"):
                        with open(fpath, "r", encoding="utf-8") as f:
                            nums |= _extract_versions_from_json_data(json.load(f))
                except Exception:
                    continue

    if not nums:
        nums |= _scan_dir_filenames_for_versions(pdf_dir, (".pdf",))
        nums |= _scan_dir_filenames_for_versions(chunks_dir, (".json", ".jsonl", ".md", ".txt"))
        nums |= _scan_dir_filenames_for_versions(md_dir, (".md", ".txt"))

    return [f"V{n}" for n in sorted(nums)]


def get_latest_version(
    retriever: Any = None,
    pdf_dir: str = "./pdf_dir",
    chunks_dir: str = "./chunks_dir",
    md_dir: str = "./cleaned_md",
    fallback: str = "V3",
) -> Tuple[str, List[str]]:
    versions = detect_available_versions(
        retriever=retriever,
        pdf_dir=pdf_dir,
        chunks_dir=chunks_dir,
        md_dir=md_dir,
    )
    if not versions:
        return fallback, [fallback]
    return versions[-1], versions


def detect_requested_version(query: str) -> Optional[str]:
    if VERSION_COMPARE_RE.search(query):
        return "ALL"
    target_match = re.search(
        r"(?<![A-Za-z0-9])[Vv](\d{1,2})\s*(?:기준|으로|로|에\s*따라|에서|만)",
        query,
    )
    if target_match:
        return f"V{int(target_match.group(1))}"
    vers = sorted(set(int(v) for v in VERSION_PATTERN.findall(query)))
    if not vers:
        return None
    if len(vers) >= 2:
        return "ALL"
    return f"V{vers[0]}"


def build_date_hint_block(retriever: RulebookRetriever, date_info: dict) -> str:
    """개정 항목 + 과거 날짜가 감지된 질문에 한해 기준일 및 부칙 확인 힌트 생성"""
    if not (date_info.get("has_past_date") and date_info.get("is_amended_topic")):
        return ""
    lines = ["[질문 날짜·판본 참고]"]
    for d in date_info.get("dates", []):
        ver = d["version"]
        ef_from, ef_to = retriever.version_periods.get(ver, ("", ""))
        period_str = f"{ef_from}~{ef_to or '현재'}" if ef_from else ""
        lines.append(f"- 감지된 날짜: '{d['raw']}' → {d['iso']} ({ver} 시행기간: {period_str})")

    lines.append(
        "- 반드시 [근거 1]의 부칙 경과조치와 표 07 기준일(최초 확정일·근무 시작일·예약 확정일·추가 주문 제출일)을 확인해 적용 판본(V1/V2/V3)을 결정하세요. 과거 기준일 사안에 V3 수치를 소급 적용하지 마세요."
    )
    return "\n".join(lines)


DEFAULT_RULE_4 = (
    "4. 질문 요약을 길게 쓰거나 같은 결론을 반복하지 말고, "
    "**[핵심 계산·조건 확인] 3~5줄 → [최종 답변] → [참고 근거(판본·조항·표·쪽)]** 순서로 "
    "간결하고 정확하게 답하세요. 계산 전 첫 단어부터 성급하게 '예/아니오'를 쓰지 마세요."
)

SYSTEM_PROMPT_TEMPLATE = """당신은 부부/가구 생활 규약집 전문 안내 도우미입니다.
반드시 아래 [참고 근거]에 적힌 내용만 사용하여 정확하게 답변하세요.
오늘 날짜는 {today}입니다.

[답변 규칙]
1. 근거에 없는 숫자나 규정은 절대 추측하거나 지어내지 마세요.
2. 사용자가 과거 판본{old_versions_hint}을 명시하지 않았으면, 가장 최신 판본({latest_version}) 기준을 우선 적용하세요.
3. 날짜가 걸치는 사건(개정 전후, 야간근무 등)은 부칙의 '기준일(최초 확정일·시작일)'과 '경과조치'를 먼저 확인해 적용 판본을 밝히세요.
{rule_4}
5. 덧셈·뺄셈·비교는 반드시 산식을 먼저 적고 비교하세요.
6. 판단에 필요한 핵심 값(확정일·박수·합의 등)이 빠졌으면 억지로 단정 짓지 말고 추가 확인이 필요함을 명확히 밝히세요.
7. [참고 근거]만으로 판단이 불가능하면 "규약집 근거에서 확인할 수 없습니다"라고 답하세요."""


def build_rule_4(core_q: str) -> str:
    m = LONG_FORM_RE.search(core_q)
    if not m:
        return DEFAULT_RULE_4
    req_len = m.group(1).strip()
    return (
        f"4. 이 질문은 상세 분량({req_len})을 요구합니다. '3~5줄 간결' 제약을 적용하지 말고, "
        f"**[핵심 확인 항목] → [단계별 실행·판정 제안(요구 분량 {req_len}에 맞춰 공백 포함 350~550자 내외로 구체 서술)] → [참고 근거(판본·조항·표·쪽)]** 순서로 답하세요. "
        "질문에 포함된 모든 대상·예외·합산·합의·기준일·가용액·누락 시 추가 확인 사항을 하나도 빠뜨리지 말고 구체적으로 적으세요."
    )


def main():
    pdf_dir = "./pdf_dir"
    chunks_dir = "./chunks_dir"
    md_dir = "./cleaned_md"

    print(f"질문 파일: {Q_PATH}")
    print(f"정답 파일: {GT_PATH}")
    print("[1/3] RAG Retriever 초기화 중...")
    sync_pdf_folder(pdf_dir=pdf_dir, chunks_dir=chunks_dir, md_dir=md_dir)
    retriever = RulebookRetriever(db_path="./qdrant_db", watch_dir=chunks_dir, use_8bit=True)

    latest_version, available_versions = get_latest_version(
        retriever=retriever,
        pdf_dir=pdf_dir,
        chunks_dir=chunks_dir,
        md_dir=md_dir,
        fallback="V3",
    )
    old_versions = [v for v in available_versions if v != latest_version]
    old_versions_hint = f"(예: {', '.join(old_versions)})" if old_versions else ""
    print(f"      탐지된 규약집 판본: {available_versions} -> 기본 최신 판본: {latest_version}")
    print(f"      시행 기간 테이블: {retriever.version_periods}")

    llm_client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

    questions = load_jsonl(Q_PATH)
    gts = load_jsonl(GT_PATH)
    gt_map = {str(g["id"]): g for g in gts}

    print(f"[2/3] 총 {len(questions)}문항 평가 시작...")
    results, latencies = [], []

    for item in tqdm(questions, desc="100문항 평가"):
        qid = str(item["id"])
        q_text = item["question"]

        # 1. 명시적 판본(V1/V2/V3)이 있을 때만 추출 필터 적용, 없으면 전체에서 잡담만 제거
        q_ver_raw = detect_requested_version(q_text)
        init_ver = q_ver_raw if (q_ver_raw and q_ver_raw != "ALL") else None
        core_q = retriever.extract_core_query(
            q_text,
            version=init_ver,
            only_latest=(init_ver is not None),
        )

        # 핵심 질문(core_q)에서만 명시적 판본 여부 판정 (잡담 속 '달라/변경' 오탐 방지)
        explicit_ver = detect_requested_version(core_q)
        date_info = retriever.analyze_query_dates(core_q)

        if explicit_ver is not None:
            search_mode = f"explicit:{explicit_ver}"
            cand_k = 20 if explicit_ver == "ALL" else 15
            hits = retriever.search(
                query=core_q,
                top_k=5,
                candidate_k=cand_k,
                version=explicit_ver,
                clean_query=False,
                min_rerank_score=0.0,
            )
        elif date_info["has_past_date"] and date_info["is_amended_topic"]:
            search_mode = (
                f"date_routed:{'+'.join(date_info['past_versions'])}"
                f"|addendum:{','.join(date_info['addendum_pages'])}"
            )
            hits = retriever.search_with_date_routing(
                query=core_q,
                past_versions=date_info["past_versions"],
                addendum_pages=date_info["addendum_pages"],
                latest_version=latest_version,
                total_top_k=5,
                candidate_k=15,
                min_rerank_score=0.0,
            )
        else:
            search_mode = f"default_latest:{latest_version}"
            hits = retriever.search(
                query=core_q,
                top_k=5,
                candidate_k=15,
                version=latest_version,
                clean_query=False,
                min_rerank_score=0.0,
            )
            if not hits:
                hits = retriever.search(
                    query=core_q,
                    top_k=5,
                    candidate_k=15,
                    version=None,
                    only_latest=False,
                    clean_query=False,
                    min_rerank_score=0.0,
                )

        ctx = retriever.format_context(hits) if hits else ""
        date_hint = build_date_hint_block(retriever, date_info) if explicit_ver is None else ""
        rule_4 = build_rule_4(core_q)

        t0 = time.perf_counter()
        if not hits:
            pred = "규약집에서 관련된 기준을 찾지 못했습니다."
        else:
            sys_p = SYSTEM_PROMPT_TEMPLATE.format(
                today=date.today().isoformat(),
                old_versions_hint=old_versions_hint,
                latest_version=latest_version,
                rule_4=rule_4,
            )
            if date_hint:
                usr_p = f"[참고 근거]\n{ctx}\n\n{date_hint}\n\n[질문]\n{core_q}"
            else:
                usr_p = f"[참고 근거]\n{ctx}\n\n[질문]\n{core_q}"

            resp = llm_client.chat.completions.create(
                model="verdict-4b",
                messages=[
                    {"role": "system", "content": sys_p},
                    {"role": "user", "content": usr_p},
                ],
                temperature=0.0,
                frequency_penalty=0.0,
                max_tokens=1536,
            )
            pred = resp.choices[0].message.content.strip()
        lat = round(time.perf_counter() - t0, 2)
        latencies.append(lat)

        results.append({
            "id": qid,
            "question": q_text,
            "pred": pred,
            "context": ctx,
            "gt": gt_map.get(qid, {}),
            "search_mode": search_mode,
            "detected_dates": date_info["dates"],
            "is_long_form": bool(LONG_FORM_RE.search(core_q)),
            "sources": [
                {"chunk_id": h["chunk_id"], "version": h["version"], "score": round(h["rerank_score"], 3)}
                for h in hits
            ],
            "latency_sec": lat,
        })

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n[3/3] 100문항 완료! 평균 응답시간: {sum(latencies)/len(latencies):.2f}초 -> {OUT_PATH}")


if __name__ == "__main__":
    main()