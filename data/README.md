# data

| 파일 | 내용 | 포함 |
|---|---|---|
| `versions_v2.json` | V2 문서 기준 판본표 (V1·V2) | 포함 |
| `versions_v3.json` | V3 문서 기준 판본표 (V1·V2·V3), 항목별 기준일과 판본별 값 | 포함 |
| `rulebook_V2.jsonl` | V2 규약집 쪽 단위 조각 60개 | **제외** |
| `rulebook_V3.jsonl` | V3 규약집 쪽 단위 조각 60개 | **제외** |

규약집 원문은 과제 운영진이 제공한 자료라 저장소에 넣지 않았어요.
실행하려면 두 `rulebook_*.jsonl` 파일을 이 폴더에 넣어 주세요. 한 줄의 형식은 이래요.

```json
{"chunk_id": "page_06", "page": "6", "chapter": "제1장 총칙", "topic": "고지 합의와 답변 상태",
 "articles": ["제7조 고지와 사전 합의", "제8조 상태 판정"], "tables": ["표 06 결정 상태"], "text": "..."}
```

공개 평가를 다시 돌리려면 운영진 질문 파일을 `evaluation/questions/public.jsonl`(`id`, `question`)로 넣어 주세요.
판본표는 사람이 전체를 검수하지 않은 상태(`reviewed_by_human: false`)예요.
