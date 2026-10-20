import os
import sys
import json

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from app import app

client = app.test_client()

print("=" * 60)
print("  AI Data Insight Builder 전체 흐름(E2E) 종합 검증")
print("=" * 60)

# 1. 서버 헬스체크
print("\n[Step 1] GET /health")
r_health = client.get("/health")
assert r_health.status_code == 200, "Health check failed"
d_health = r_health.get_json()
print(f"  ✓ 상태: {d_health.get('status')}")
print(f"  ✓ API 키 설정됨: {d_health.get('gemini_api_key_set')}")
print(f"  ✓ 모델: {d_health.get('gemini_model')}")

# 2. 메인 화면
print("\n[Step 2] GET / (메인 UI)")
r_index = client.get("/")
assert r_index.status_code == 200, "Index load failed"
assert "AI Data Insight Builder" in r_index.text, "Title missing"
print("  ✓ 메인 HTML 및 Chart.js CDN 로드 성공")

# 3. CSV 업로드 및 진단
print("\n[Step 3] POST /inspect (World Bank WDI 인터넷 보급률)")
csv_file = "data/API_IT.NET.USER.ZS_DS2_en_csv_v2_468465/API_IT.NET.USER.ZS_DS2_en_csv_v2_468465.csv"
with open(csv_file, "rb") as f:
    r_inspect = client.post("/inspect", data={"file": (f, "API_IT.NET.USER.ZS_DS2_en_csv_v2_468465.csv")})

assert r_inspect.status_code == 200, "Inspect failed"
d_inspect = r_inspect.get_json()
print(f"  ✓ 감지된 인코딩: {d_inspect['encoding']}")
print(f"  ✓ 건너뛴 헤더 줄 수: {d_inspect['skipped_header_rows']}줄")
print(f"  ✓ 데이터 크기: {d_inspect['rows']}행 x {d_inspect['columns']}열")
print(f"  ✓ 권장 기준연도: {d_inspect['recommended_reference_year']}년")
print(f"  ✓ 집계 행 감지: {d_inspect['aggregate_rows_count']}개")
print(f"  ✓ 경고 목록: {len(d_inspect['warnings'])}개 생성")

# 4. AI 분석 제안 요청
print("\n[Step 4] POST /suggest (Gemini API)")
r_suggest = client.post("/suggest")
assert r_suggest.status_code == 200, "Suggest failed"
d_suggest = r_suggest.get_json()
suggestions = d_suggest.get("suggestions", [])
print(f"  ✓ 제안된 분석 수: {len(suggestions)}개")
for idx, s in enumerate(suggestions, 1):
    print(f"    ({idx}) 도구: {s['tool']} ({s.get('tool_name')}) [기준: {s.get('reference_period')}]")

# 5. 제안된 도구 실행 (Pandas 계산)
print("\n[Step 5] POST /run (도구 실행 - top_bottom_n)")
r_run = client.post("/run", json={"tool": "top_bottom_n", "params": {"reference_period": "2024", "n": 5}})
assert r_run.status_code == 200, "Run failed"
d_run = r_run.get_json()
print(f"  ✓ 실행 완료: {d_run.get('tool')}")
print(f"  ✓ 사용한 행 수: {d_run.get('used_rows')}개")
print(f"  ✓ 제외한 행 수: {d_run.get('excluded_rows')}개")
print(f"  ✓ 제외 사유: {d_run.get('excluded_reasons')}")
print(f"  ✓ 차트 타입: {d_run.get('chart_data', {}).get('type')}")
top1 = d_run["result"]["top_n"][0]
print(f"  ✓ 1위 국가: {top1['label']} ({top1['value']}%)")

# 6. AI 정식 해석 보고서 생성 (모드 B)
print("\n[Step 6] POST /explain (모드 B 정식 보고서)")
r_explain = client.post("/explain", json={"mode": "B"})
assert r_explain.status_code == 200, "Explain failed"
d_explain = r_explain.get_json()
report = d_explain.get("explanation", "")
print(f"  ✓ 보고서 작성 완료 (총 {len(report)}자)")

required_headers = [
    "## 무엇을 봤는가",
    "## 데이터 출처와 정의",
    "## 기준 시점과 제외된 항목",
    "## 계산 결과",
    "## 읽어낼 수 있는 것",
    "## 이 데이터로는 알 수 없는 것",
    "## 다음에 확인해볼 것"
]
print("  ✓ 7대 필수 섹션 출력 검증:")
for h in required_headers:
    has_h = h in report
    print(f"     {h}: {'통과 (OK)' if has_h else '누락 (FAIL)'}")
    assert has_h, f"누락된 헤더: {h}"

# 7. 데이터 결과에 대한 추가 질문 (POST /ask)
print("\n[Step 7] POST /ask (추가 질문 질의응답)")
question = "상위 1위 국가와 2위 국가의 차이는 얼마인가요?"
r_ask = client.post("/ask", json={"question": question})
assert r_ask.status_code == 200, f"Ask failed: {r_ask.text}"
d_ask = r_ask.get_json()
print(f"  ✓ 질문: {d_ask.get('question')}")
print(f"  ✓ 답변 (길이 {len(d_ask.get('answer'))}자): {d_ask.get('answer')[:120]}...")

# 8. 자연어 질의 기반 맞춤 분석 & 자동 시각화 (POST /query)
print("\n[Step 8] POST /query (자연어로 질문하고 맞춤 시각화·답변 생성)")
nl_query = "한국과 미국의 2015년부터 2024년까지 인터넷 보급률 추이 비교해줘"
r_query = client.post("/query", json={"query": nl_query})
assert r_query.status_code == 200, f"Query failed: {r_query.text}"
d_query = r_query.get_json()
print(f"  ✓ 자연어 질문: \"{d_query.get('query')}\"")
print(f"  ✓ 매핑된 도구: {d_query.get('tool_used')} ({d_query.get('tool_name')})")
print(f"  ✓ 분석 의도: {d_query.get('intent')}")
print(f"  ✓ 차트 타입: {d_query.get('run_result', {}).get('chart_data', {}).get('type')}")
print(f"  ✓ 답변 브리핑 (길이 {len(d_query.get('answer'))}자): {d_query.get('answer')[:120]}...")

print("\n" + "=" * 60)
print("  🎉 전체 엔드투엔드 파이프라인 (자연어 질의 & 시각화 포함) 100% 통과 완료!")
print("=" * 60)
