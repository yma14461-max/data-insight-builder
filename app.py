"""
AI Data Insight Builder — app.py
Flask 웹 서버 메인 파일
"""

import os
import io
import json
import logging
import pandas as pd
from flask import Flask, jsonify, render_template, request
from dotenv import load_dotenv
from google import genai
from google.genai import types
import tools

# ── 환경변수 로드 ──────────────────────────────────────────
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL   = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
FLASK_DEBUG    = os.getenv("FLASK_DEBUG", "false").lower() == "true"
MAX_UPLOAD_MB  = int(os.getenv("MAX_UPLOAD_MB", "10"))

# ── 로깅 설정 ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG if FLASK_DEBUG else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Flask 앱 생성 ─────────────────────────────────────────
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

# ── 메모리에 보관할 데이터셋 (1개만) ──────────────────────
current_dataset = {
    "df": None,           # pandas DataFrame
    "filename": None,     # 원본 파일명
    "diagnosis": None,    # /inspect 진단 결과
    "last_run": None,     # /run 마지막 실행 결과
}


# ═══════════════════════════════════════════════════════════
#  GET /health — 서버 상태 확인
# ═══════════════════════════════════════════════════════════
@app.route("/health", methods=["GET"])
def health():
    """서버 상태와 API Key 설정 여부만 반환 (키 값은 반환 금지)"""
    logger.info("[REQUEST]  GET /health")

    status = {
        "status": "ok",
        "gemini_api_key_set": bool(GEMINI_API_KEY),
        "gemini_model": GEMINI_MODEL,
        "max_upload_mb": MAX_UPLOAD_MB,
        "dataset_loaded": current_dataset["df"] is not None,
    }

    logger.info("[RESPONSE] status=%s api_key_set=%s model=%s",
                status["status"],
                status["gemini_api_key_set"],
                status["gemini_model"])
    return jsonify(status)


# ═══════════════════════════════════════════════════════════
#  GET / — 메인 페이지
# ═══════════════════════════════════════════════════════════
@app.route("/")
def index():
    """index.html 반환"""
    return render_template("index.html")


# ═══════════════════════════════════════════════════════════
#  POST /inspect — CSV 업로드 및 진단 (1차: 인코딩·헤더 감지)
# ═══════════════════════════════════════════════════════════

# 인코딩 시도 순서
ENCODING_ORDER = ["utf-8-sig", "utf-8", "cp949", "euc-kr"]


def _try_read_csv(raw_bytes):
    """인코딩을 순서대로 시도하여 DataFrame으로 읽는다.
    성공하면 (df, encoding, skipped_rows) 반환.
    모두 실패하면 (None, None, None) 반환.
    """
    for enc in ENCODING_ORDER:
        try:
            text = raw_bytes.decode(enc)
            logger.info("[ENCODING] %s 시도 → 디코딩 성공", enc)
        except (UnicodeDecodeError, LookupError):
            logger.info("[ENCODING] %s 시도 → 실패", enc)
            continue

        # ── 헤더 줄 자동 탐지 ──────────────────────────────
        lines = text.splitlines()
        header_row = 0
        max_fields = 0

        # 앞쪽 20줄을 검사하여 비어있지 않은 필드 수가 가장 많은 줄을 헤더로 추정
        # 빈 줄은 건너뛴다. 동률이면 먼저 나온 줄이 헤더(데이터 행보다 위에 있으므로)
        check_limit = min(20, len(lines))
        for i in range(check_limit):
            line = lines[i].strip()
            if not line:
                continue  # 빈 줄 무시
            # 쉼표로 나눈 뒤, 실제 내용이 있는 필드 수를 센다
            parts = line.split(",")
            non_empty = sum(1 for p in parts if p.strip().strip('"') != "")
            if non_empty > max_fields:
                max_fields = non_empty
                header_row = i

        skipped_rows = header_row
        logger.info("[HEADER]   헤더 줄 감지 결과: %d번째 줄 (건너뛴 줄 수: %d)",
                    header_row + 1, skipped_rows)

        try:
            df = pd.read_csv(
                io.StringIO(text),
                header=header_row,
                skip_blank_lines=False,
                on_bad_lines="warn",
            )
            logger.info("[CSV]      parsing 완료 rows=%d columns=%d",
                        len(df), len(df.columns))
            return df, enc, skipped_rows
        except Exception as e:
            logger.warning("[CSV]      %s 인코딩으로 파싱 실패: %s", enc, e)
            continue

    return None, None, None


def _detect_empty_unnamed_columns(df):
    """이름 없는 열, Unnamed 열, 전체가 빈 열을 감지한다."""
    warnings = []
    unnamed_cols = []
    empty_cols = []

    for col in df.columns:
        col_str = str(col)
        # Unnamed 열 감지
        if col_str.startswith("Unnamed") or col_str.strip() == "":
            unnamed_cols.append(col_str)
        # 전체가 빈 열 감지
        if df[col].isna().all():
            empty_cols.append(col_str)

    if unnamed_cols:
        warnings.append(f"이름이 없는 열(Unnamed)이 {len(unnamed_cols)}개 있습니다: {unnamed_cols}")
    if empty_cols:
        warnings.append(f"값이 전혀 없는 빈 열이 {len(empty_cols)}개 있습니다: {empty_cols}")

    return unnamed_cols, empty_cols, warnings


@app.route("/inspect", methods=["POST"])
def inspect_csv():
    """CSV 업로드 → 인코딩·헤더 감지 → 진단 결과 JSON 반환"""
    logger.info("[REQUEST]  POST /inspect")

    # ── 파일 유효성 검사 ───────────────────────────────────
    if "file" not in request.files:
        logger.warning("[ERROR]    파일이 전송되지 않음")
        return jsonify({"error": "파일을 선택해주세요."}), 400

    file = request.files["file"]
    if file.filename == "":
        logger.warning("[ERROR]    파일명이 비어있음")
        return jsonify({"error": "파일을 선택해주세요."}), 400

    if not file.filename.lower().endswith(".csv"):
        logger.warning("[ERROR]    CSV가 아닌 파일: %s", file.filename)
        return jsonify({"error": "CSV 파일만 업로드할 수 있습니다."}), 400

    raw_bytes = file.read()
    file_size = len(raw_bytes)
    logger.info("[FILE]     name=%s size=%d bytes", file.filename, file_size)

    if file_size == 0:
        logger.warning("[ERROR]    빈 파일")
        return jsonify({"error": "빈 파일입니다. 내용이 있는 CSV를 업로드해주세요."}), 400

    # ── 인코딩 순차 시도 + 헤더 감지 + 파싱 ────────────────
    df, detected_encoding, skipped_rows = _try_read_csv(raw_bytes)

    if df is None:
        logger.error("[ERROR]    모든 인코딩 시도 실패")
        return jsonify({
            "error": "파일 인코딩을 인식할 수 없습니다. "
                     "UTF-8 또는 CP949(한글 Windows) 형식의 CSV를 사용해주세요."
        }), 400

    if len(df) == 0:
        logger.warning("[ERROR]    파싱 후 행이 0개")
        return jsonify({"error": "파일을 읽었지만 데이터가 없습니다. 헤더만 있는 파일인지 확인해주세요."}), 400

    # ── 빈 열·Unnamed 열 감지 ──────────────────────────────
    unnamed_cols, empty_cols, col_warnings = _detect_empty_unnamed_columns(df)

    # 빈 열/Unnamed 열 제거 (분석 품질을 위해)
    cols_to_drop = list(set(unnamed_cols + empty_cols))
    if cols_to_drop:
        existing = [c for c in cols_to_drop if c in df.columns]
        if existing:
            df = df.drop(columns=existing)
            logger.info("[CSV]      빈/Unnamed 열 %d개 제거: %s", len(existing), existing)

    # ── 열 분류 ────────────────────────────────────────────
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    string_cols = df.select_dtypes(include=["object"]).columns.tolist()

    # 연도 후보 열 감지: 열 이름이 4자리 숫자인 열
    year_candidate_cols = [c for c in df.columns if str(c).strip().isdigit() and len(str(c).strip()) == 4]

    dtypes_dict = {col: str(dtype) for col, dtype in df.dtypes.items()}

    # ── 연도별 데이터 충실도 & 권장 기준연도 (Step 13) ───────
    year_coverage = {}
    recommended_year = None
    if year_candidate_cols:
        # 연도 이름 숫자 기준 정렬
        sorted_year_cols = sorted(year_candidate_cols, key=lambda x: int(str(x).strip()))
        for y_col in sorted_year_cols:
            valid_cnt = int(df[y_col].notna().sum())
            ratio = round(valid_cnt / len(df) * 100, 1)
            year_coverage[str(y_col)] = {
                "count": valid_cnt,
                "ratio_percent": ratio
            }

        # 권장 기준연도: 데이터 충실도 70% 이상인 최근 연도 (없으면 최근 10년 중 최다 값 연도)
        qualifying_years = [
            y for y in sorted_year_cols
            if year_coverage[str(y)]["ratio_percent"] >= 70.0
        ]
        if qualifying_years:
            recommended_year = str(qualifying_years[-1])
        else:
            # 70% 이상이 없으면 값이 가장 많은 연도 중 가장 최근 연도
            max_cnt = max(v["count"] for v in year_coverage.values()) if year_coverage else 0
            best_candidates = [
                y for y in sorted_year_cols
                if year_coverage[str(y)]["count"] == max_cnt
            ]
            recommended_year = str(best_candidates[-1]) if best_candidates else None

    # ── 집계 행(Aggregate Rows) 감지 (Step 13) ───────────────
    aggregate_keywords = [
        "world", "oecd members", "euro area", "european union",
        "high income", "low income", "middle income", "upper middle", "lower middle",
        "income", "area", "total", "sub-saharan", "latin america",
        "east asia", "south asia", "north america", "arab world",
        "ida & ibrd", "dividend", "countries"
    ]
    aggregate_rows_found = []

    # 문자열 열(특히 Country Name 같은 식별자 열)에서 검색
    id_col = string_cols[0] if string_cols else None
    if id_col:
        for idx, val in df[id_col].items():
            val_lower = str(val).lower()
            matched_kws = [kw for kw in aggregate_keywords if kw in val_lower]
            if matched_kws:
                aggregate_rows_found.append({
                    "index": int(idx),
                    "name": str(val),
                    "matched_keywords": matched_kws
                })

    # ── 결측치 및 중복치 계산 ──────────────────────────────
    missing_count = df.isnull().sum().to_dict()
    missing_ratio = {col: round(float(v) / len(df) * 100, 1) for col, v in missing_count.items()}
    duplicate_rows = int(df.duplicated().sum())

    # ── 진단 결과 구성 ─────────────────────────────────────
    warnings_list = list(col_warnings)  # 빈 열/Unnamed 경고
    if duplicate_rows > 0:
        warnings_list.append(f"중복된 행이 {duplicate_rows}개 있습니다.")
    if skipped_rows > 0:
        warnings_list.append(f"헤더 앞에 메타정보 줄이 {skipped_rows}개 있어 건너뛰었습니다.")
    if aggregate_rows_found:
        warnings_list.append(
            f"지역·소득그룹 등 집계 행 후보가 {len(aggregate_rows_found)}개 감지되었습니다. "
            f"순위/분포 분석 시 '집계 행 제외'를 권장합니다."
        )
    if recommended_year:
        cov_info = year_coverage.get(recommended_year, {})
        warnings_list.append(
            f"권장 기준연도는 {recommended_year}년입니다 "
            f"(충실도: {cov_info.get('ratio_percent', 0)}%, {cov_info.get('count', 0)}/{len(df)}개 값 채워짐)."
        )

    diagnosis = {
        "filename": file.filename,
        "file_size_bytes": file_size,
        "encoding": detected_encoding,
        "skipped_header_rows": skipped_rows,
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": df.columns.tolist(),
        "dtypes": dtypes_dict,
        "numeric_columns": numeric_cols,
        "string_columns": string_cols,
        "year_candidate_columns": year_candidate_cols,
        "year_coverage": year_coverage,
        "recommended_reference_year": recommended_year,
        "aggregate_rows": aggregate_rows_found,
        "aggregate_rows_count": len(aggregate_rows_found),
        "missing_count": missing_count,
        "missing_ratio_percent": missing_ratio,
        "duplicate_rows": duplicate_rows,
        "unnamed_columns": unnamed_cols,
        "empty_columns": empty_cols,
        "preview": df.head(10).fillna("").to_dict(orient="records"),
        "warnings": warnings_list,
    }

    # ── 메모리에 저장 ──────────────────────────────────────
    current_dataset["df"] = df
    current_dataset["filename"] = file.filename
    current_dataset["diagnosis"] = diagnosis

    logger.info("[DIAGNOSE] 진단 완료 rows=%d cols=%d aggregates=%d rec_year=%s warnings=%d",
                len(df), len(df.columns), len(aggregate_rows_found), recommended_year, len(warnings_list))
    logger.info("[RESPONSE] status=ok")

    return jsonify(diagnosis)



# ═══════════════════════════════════════════════════════════
#  POST /suggest — Gemini 기반 분석 제안 (카탈로그 제한, JSON 강제)
# ═══════════════════════════════════════════════════════════
@app.route("/suggest", methods=["POST"])
def suggest_analyses():
    """진단 결과를 Gemini에 전달하고, 도구 카탈로그 내에서 3~5개 분석 제안 수신 (JSON 강제)"""
    logger.info("[REQUEST]  POST /suggest")

    if current_dataset["diagnosis"] is None or current_dataset["df"] is None:
        logger.warning("[ERROR]    데이터셋 미업로드 상태에서 suggest 요청")
        return jsonify({"error": "먼저 CSV 파일을 업로드하여 진단을 완료해주세요."}), 400

    if not GEMINI_API_KEY:
        logger.error("[ERROR]    GEMINI_API_KEY 미설정")
        return jsonify({"error": "GEMINI_API_KEY가 설정되지 않았습니다. .env 파일을 확인해주세요."}), 500

    diag = current_dataset["diagnosis"]
    df = current_dataset["df"]

    # ── 도구 카탈로그 요약 ─────────────────────────────────
    catalog_summary = {}
    for k, v in tools.TOOL_CATALOG.items():
        catalog_summary[k] = {
            "name": v["name"],
            "description": v["description"],
            "requires_reference_period": v["requires_reference_period"]
        }

    # ── 진단 결과 요약 ─────────────────────────────────────
    diag_summary = {
        "filename": diag["filename"],
        "rows": diag["rows"],
        "columns": diag["columns"],
        "string_columns": diag["string_columns"],
        "numeric_columns_sample": diag["numeric_columns"][:10],
        "year_candidate_columns": diag["year_candidate_columns"],
        "recommended_reference_year": diag["recommended_reference_year"],
        "aggregate_rows_count": diag["aggregate_rows_count"],
        "warnings": diag["warnings"]
    }

    # ── Gemini 프롬프트 구성 ──────────────────────────────
    system_instruction = (
        "당신은 데이터 분석 전문가 AI입니다.\n"
        "제공된 데이터셋 진단 결과 요약과 '도구 카탈로그'를 바탕으로, "
        "사용자가 이 데이터셋에서 실행할 수 있는 가장 가치 있고 의미 있는 분석 3~5개를 추천하십시오.\n\n"
        "반드시 지켜야 할 절대 규칙:\n"
        "1. 반드시 아래 명시된 '도구 카탈로그'에 존재하는 도구(tool 키)만 제안해야 합니다. 없는 도구를 지어내지 마십시오.\n"
        "2. 반드시 데이터셋에 실제 존재하는 열(column)만 파라미터(params)에 사용해야 합니다.\n"
        "3. 기준시점(reference_period)이 필요한 도구는 가급적 '권장 기준연도'(recommended_reference_year)를 사용하십시오.\n"
        "4. 오직 아래 JSON 형식으로만 응답해야 합니다. 마크다운 코드블록이나 부연 설명 없이 순수 JSON 객체만 반환하십시오.\n\n"
        "응답 JSON 스키마:\n"
        "{\n"
        '  "suggestions": [\n'
        "    {\n"
        '      "tool": "도구ID (예: top_bottom_n)",\n'
        '      "params": {\n'
        '        "reference_period": "2024",\n'
        '        "label_column": "Country Name",\n'
        '        "n": 10,\n'
        '        "exclude_aggregates": true\n'
        "      },\n"
        '      "reference_period": "2024",\n'
        '      "why": "이 데이터셋에서 이 분석이 의미 있는 명확한 이유",\n'
        '      "caution": "결과를 해석할 때 주의해야 할 점 (예: 결측치, 이상치, 집계단위 혼동 등)"\n'
        "    }\n"
        "  ]\n"
        "}"
    )

    user_content = (
        f"### 데이터 진단 요약:\n{json.dumps(diag_summary, ensure_ascii=False, indent=2)}\n\n"
        f"### 사용 가능한 도구 카탈로그:\n{json.dumps(catalog_summary, ensure_ascii=False, indent=2)}\n\n"
        f"위 정보에 기반하여 3~5개의 최적 분석을 JSON으로 제안해주세요."
    )

    logger.info("[AI]       suggest 요청 시작 (모델: %s)", GEMINI_MODEL)
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=f"{system_instruction}\n\n{user_content}",
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.2
            )
        )
        logger.info("[AI]       suggest 응답 수신 완료")

        raw_text = response.text.strip()
        if raw_text.startswith("```"):
            raw_text = raw_text.split("\n", 1)[1]
            if raw_text.endswith("```"):
                raw_text = raw_text.rsplit("\n", 1)[0]
            raw_text = raw_text.strip()

        parsed = json.loads(raw_text)
        raw_suggestions = parsed.get("suggestions", [])
        logger.info("[AI]       JSON 파싱 결과: %d개 원시 제안", len(raw_suggestions))

    except Exception as e:
        logger.error("[ERROR]    Gemini suggest 호출/파싱 실패: %s", e, exc_info=True)
        # 안전한 폴백: 카탈로그 기반 기본 제안 3개 자동 구성
        rec_yr = diag.get("recommended_reference_year", "2024")
        lbl_col = diag.get("string_columns", ["Country Name"])[0] if diag.get("string_columns") else "Country Name"
        raw_suggestions = [
            {
                "tool": "top_bottom_n",
                "params": {"reference_period": rec_yr, "label_column": lbl_col, "n": 10, "exclude_aggregates": True},
                "reference_period": rec_yr,
                "why": f"{rec_yr}년 기준 보급률 상위 10개국과 하위 10개국을 비교하여 국가 간 디지털 격차를 파악합니다.",
                "caution": "순위 왜곡을 막기 위해 지역/소득그룹 집계 행을 배제했는지 확인해야 합니다."
            },
            {
                "tool": "trend_line",
                "params": {"entities": ["Korea, Rep.", "United States", "World"], "label_column": lbl_col},
                "reference_period": f"1990-{rec_yr}",
                "why": "대한민국, 미국 등 주요국과 전 세계 평균의 연도별 보급 추이를 시계열로 비교합니다.",
                "caution": "초기 연도에는 결측치가 다수 존재할 수 있습니다."
            },
            {
                "tool": "change_rate",
                "params": {"start_period": "2014", "reference_period": rec_yr, "n": 10, "label_column": lbl_col, "exclude_aggregates": True},
                "reference_period": f"2014-{rec_yr}",
                "why": f"최근 10년간({rec_yr}년 대비 10년 전) 가장 급격한 성장을 이룬 국가들을 확인합니다.",
                "caution": "초기값이 극히 낮았던 국가는 소폭의 상승으로도 높은 백분율 변화율이 나타날 수 있습니다."
            }
        ]

    # ── 도구 유효성 검증 및 카탈로그 제한 ────────────────────
    validated_suggestions = []
    for s in raw_suggestions:
        t_name = s.get("tool")
        if t_name in tools.TOOL_CATALOG:
            validated_suggestions.append({
                "tool": t_name,
                "tool_name": tools.TOOL_CATALOG[t_name]["name"],
                "params": s.get("params", {}),
                "reference_period": s.get("reference_period") or diag.get("recommended_reference_year", "2024"),
                "why": s.get("why", ""),
                "caution": s.get("caution", "")
            })

    # 최소 3개 보장
    if len(validated_suggestions) < 3:
        rec_yr = diag.get("recommended_reference_year", "2024")
        lbl_col = diag.get("string_columns", ["Country Name"])[0] if diag.get("string_columns") else "Country Name"
        fallback_candidates = [
            ("top_bottom_n", {"reference_period": rec_yr, "label_column": lbl_col, "n": 10, "exclude_aggregates": True}, f"{rec_yr}년 상위/하위 순위 비교"),
            ("trend_line", {"entities": ["Korea, Rep.", "United States"], "label_column": lbl_col}, "주요 국가 시계열 추이 분석"),
            ("compare_one_vs_all", {"reference_period": rec_yr, "target_entity": "Korea, Rep.", "label_column": lbl_col, "exclude_aggregates": True}, "대한민국 vs 전체 국가 평균 대비")
        ]
        existing_tools = {s["tool"] for s in validated_suggestions}
        for t_key, t_params, t_why in fallback_candidates:
            if t_key not in existing_tools and len(validated_suggestions) < 3:
                validated_suggestions.append({
                    "tool": t_key,
                    "tool_name": tools.TOOL_CATALOG[t_key]["name"],
                    "params": t_params,
                    "reference_period": rec_yr,
                    "why": t_why,
                    "caution": "해석 시 집계 단위 제외 여부와 결측치를 확인하십시오."
                })

    logger.info("[RESPONSE] status=ok suggestions_count=%d", len(validated_suggestions))
    return jsonify({"suggestions": validated_suggestions})



# ═══════════════════════════════════════════════════════════
#  POST /run — 도구 실행 (Pandas 계산, 기준시점 필수, 제외 행 수 반환)
# ═══════════════════════════════════════════════════════════
@app.route("/run", methods=["POST"])
def run_tool():
    """선택된 분석 도구를 Pandas로 직접 실행하고 결과 및 차트 데이터 반환"""
    logger.info("[REQUEST]  POST /run")

    if current_dataset["df"] is None:
        logger.warning("[ERROR]    데이터셋이 로드되지 않음")
        return jsonify({"error": "분석할 데이터셋이 없습니다. 먼저 CSV 파일을 업로드해주세요."}), 400

    payload = request.get_json(silent=True) or {}
    tool_name = payload.get("tool")
    params = payload.get("params", {})

    if not tool_name:
        logger.warning("[ERROR]    tool 파라미터 누락")
        return jsonify({"error": "실행할 도구(tool)를 지정해주세요."}), 400

    if tool_name not in tools.TOOL_CATALOG:
        logger.warning("[ERROR]    카탈로그에 없는 도구 요청: %s", tool_name)
        return jsonify({"error": f"도구 카탈로그에 등록되지 않은 도구입니다: {tool_name}"}), 400

    tool_info = tools.TOOL_CATALOG[tool_name]
    df = current_dataset["df"]

    # ── 기준시점 필수 검증 (도구 7~12번 및 requires_reference_period가 True인 도구) ──
    if tool_info.get("requires_reference_period", False):
        ref_period = params.get("reference_period") or payload.get("reference_period")
        if not ref_period:
            logger.warning("[ERROR]    도구 %s 실행 시 필수 기준시점 누락", tool_name)
            return jsonify({"error": f"도구 '{tool_name}'은(는) 기준 시점(reference_period)이 필수입니다."}), 400
        else:
            params["reference_period"] = str(ref_period)

    # ── 집계 행 제외 기본값 보장 (기본: 제외) ────────────────
    if "exclude_aggregates" not in params:
        params["exclude_aggregates"] = True

    # ── 도구 실행 (순수 Pandas 연산) ────────────────────────
    try:
        func = tool_info["func"]
        result = func(df, **params)
    except Exception as e:
        logger.error("[ERROR]    도구 %s 실행 중 예외 발생: %s", tool_name, e, exc_info=True)
        return jsonify({"error": f"도구 실행 중 오류가 발생했습니다: {str(e)}"}), 500

    if isinstance(result, dict) and "error" in result:
        logger.warning("[ERROR]    도구 %s 실행 실패: %s", tool_name, result["error"])
        return jsonify(result), 400

    # ── 메모리에 마지막 실행 결과 보관 (/explain용) ─────────
    current_dataset["last_run"] = result

    # ── Backend Log 준수 ───────────────────────────────────
    # [TOOL] 도구명, 파라미터, 사용 행 수, 제외 행 수
    logger.info("[TOOL]     name=%s params=%s used_rows=%s excluded_rows=%s",
                tool_name, params, result.get("used_rows"), result.get("excluded_rows"))
    logger.info("[RESPONSE] status=ok")

    return jsonify(result)



# ═══════════════════════════════════════════════════════════
#  POST /explain — Gemini 기반 결과 해석 (모드 A/B)
# ═══════════════════════════════════════════════════════════
@app.route("/explain", methods=["POST"])
def explain_analysis():
    """Pandas 계산 결과를 Gemini에 전달하여 해석 보고서 생성 (모드 A: 비교용 허술한 모드 / 모드 B: 정식 검증 모드)"""
    logger.info("[REQUEST]  POST /explain")

    if not GEMINI_API_KEY:
        logger.error("[ERROR]    GEMINI_API_KEY 미설정")
        return jsonify({"error": "GEMINI_API_KEY가 설정되지 않았습니다."}), 500

    payload = request.get_json(silent=True) or {}
    mode = str(payload.get("mode", "B")).upper()

    if mode not in ["A", "B"]:
        mode = "B"

    diag = current_dataset.get("diagnosis")
    run_res = payload.get("run_result") or current_dataset.get("last_run")

    client = genai.Client(api_key=GEMINI_API_KEY)

    # ── 모드 A (비교용, 의도적으로 허술하게): 계산 결과 없이 진단 요약만 전달 ────
    if mode == "A":
        logger.info("[AI]       explain 요청 시작 (모드: A - 비교용)")
        diag_brief = {
            "filename": diag["filename"] if diag else "uploaded_data.csv",
            "rows": diag["rows"] if diag else "알 수 없음",
            "columns": diag["columns"] if diag else "알 수 없음",
            "sample_columns": diag["column_names"][:10] if diag else []
        }
        prompt = (
            f"다음 데이터 파일이 업로드되었습니다:\n"
            f"{json.dumps(diag_brief, ensure_ascii=False, indent=2)}\n\n"
            f"이 데이터를 분석해서 종합 보고서를 써줘."
        )

        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.7)
            )
            explanation_text = response.text.strip()
            logger.info("[AI]       explain 응답 수신 완료 (모드: A)")
        except Exception as e:
            logger.error("[ERROR]    모드 A 해석 생성 실패: %s", e)
            return jsonify({"error": f"해석 생성 실패: {str(e)}"}), 500

        logger.info("[RESPONSE] status=ok mode=A")
        return jsonify({
            "mode": "A",
            "mode_name": "모드 A (비교용 - 계산 결과 없는 허술한 보고서)",
            "explanation": explanation_text
        })

    # ── 모드 B (정식): Pandas 계산 결과 기반 8대 규칙 및 7대 섹션 강제 ────────
    if run_res is None:
        logger.warning("[ERROR]    /run 결과 없이 모드 B explain 요청")
        return jsonify({"error": "해석할 분석 결과가 없습니다. 먼저 분석 도구를 실행(/run)해주세요."}), 400

    logger.info("[AI]       explain 요청 시작 (모드: B - 정식)")

    # 지표 및 데이터 메타정보
    meta_info = {
        "dataset_name": diag.get("filename", "World Bank WDI Data") if diag else "World Bank WDI Data",
        "indicator_code": "IT.NET.USER.ZS",
        "indicator_name": "Individuals using the Internet (% of population)",
        "source": "World Bank / International Telecommunication Union (ITU)",
        "definition": "최근 3개월 이내에 어떤 기기(컴퓨터, 스마트폰, 디지털TV 등)로든 인터넷을 사용한 인구의 비율(%)",
        "unit": run_res.get("unit", "%")
    }

    # 계산 결과 요약 (차트 객체는 제외하고 순수 수치 표와 메타정보만 전달)
    clean_calc_result = {
        "tool": run_res.get("tool"),
        "reference_period": run_res.get("reference_period"),
        "used_rows": run_res.get("used_rows"),
        "excluded_rows": run_res.get("excluded_rows"),
        "excluded_reasons": run_res.get("excluded_reasons"),
        "used_columns": run_res.get("used_columns"),
        "unit": run_res.get("unit"),
        "caution": run_res.get("caution"),
        "result_numbers": run_res.get("result")
    }

    system_instruction = (
        "당신은 공공 데이터 전문 수석 데이터 사이언티스트입니다.\n"
        "제공된 'Pandas 계산 결과'만을 근거로 아래 8대 절대 규칙과 7대 필수 섹션 형식을 엄격히 지켜 해석 보고서를 작성하십시오.\n\n"
        "### 8대 절대 규칙 (위반 시 보고서 실격):\n"
        "1. 반드시 제공된 '계산 결과'에 명시된 숫자만 사용하십시오. 스스로 재계산하거나 임의의 숫자를 추정하지 마십시오.\n"
        "2. 직접 계산하거나 추측하지 마십시오.\n"
        "3. 계산 결과에 등장하지 않는 국가, 연도, 열 이름을 언급하지 마십시오.\n"
        "4. '기준 시점'과 '제외된 행 수 및 사유'를 본문에 반드시 명확히 기술하십시오.\n"
        "5. 상관관계를 절대 인과관계(원인과 결과)로 표현하지 마십시오.\n"
        "6. 지표의 공식 정의와 출처를 보고서 앞부분(데이터 출처와 정의)에 정확히 표기하십시오.\n"
        "7. 데이터에서 직접 확인할 수 없는 한계점은 반드시 '이 데이터로는 확인할 수 없다'라고 단정하여 표현하십시오.\n"
        "8. 중학생도 쉽게 이해할 수 있는 명료한 한국어로 서술하되, 데이터 과학적 정확성을 절대 희생하지 마십시오.\n\n"
        "### 필수 출력 마크다운 형식 (아래 7개 헤더를 그대로 사용하십시오):\n"
        "## 무엇을 봤는가\n"
        "## 데이터 출처와 정의\n"
        "## 기준 시점과 제외된 항목\n"
        "## 계산 결과\n"
        "## 읽어낼 수 있는 것\n"
        "## 이 데이터로는 알 수 없는 것\n"
        "## 다음에 확인해볼 것\n"
    )

    user_content = (
        f"### 지표 메타데이터:\n{json.dumps(meta_info, ensure_ascii=False, indent=2)}\n\n"
        f"### Pandas 계산 결과 (이 숫자들만 사용):\n{json.dumps(clean_calc_result, ensure_ascii=False, indent=2)}\n\n"
        f"위 계산 결과를 바탕으로 7대 섹션 형식에 맞춰 정식 해석 보고서를 작성해주세요."
    )

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=f"{system_instruction}\n\n{user_content}",
            config=types.GenerateContentConfig(temperature=0.2)
        )
        explanation_text = response.text.strip()
        logger.info("[AI]       explain 응답 수신 완료 (모드: B)")
    except Exception as e:
        logger.error("[ERROR]    모드 B 해석 생성 실패: %s", e, exc_info=True)
        return jsonify({"error": f"해석 생성 실패: {str(e)}"}), 500

    logger.info("[RESPONSE] status=ok mode=B")
    return jsonify({
        "mode": "B",
        "mode_name": "모드 B (정식 - Pandas 검증 기반 7대 섹션 보고서)",
        "explanation": explanation_text
    })



# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    logger.info("=" * 50)
    logger.info("AI Data Insight Builder 서버 시작")
    logger.info("  GEMINI_MODEL   = %s", GEMINI_MODEL)
    logger.info("  API_KEY 설정됨 = %s", bool(GEMINI_API_KEY))
    logger.info("  MAX_UPLOAD_MB  = %d", MAX_UPLOAD_MB)
    logger.info("  FLASK_DEBUG    = %s", FLASK_DEBUG)
    logger.info("=" * 50)
    app.run(host="127.0.0.1", port=5000, debug=FLASK_DEBUG)
