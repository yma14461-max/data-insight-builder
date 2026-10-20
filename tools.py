"""
AI Data Insight Builder — tools.py
Pandas 기반 데이터 계산 및 분석 도구 모듈

규칙:
1. 숫자 계산은 반드시 Pandas가 한다. Gemini에게 계산을 시키지 않는다.
2. 모든 도구는 계산 결과와 함께:
   - 기준시점 (reference_period)
   - 사용한 행 수 (used_rows)
   - 제외한 행 수와 이유 (excluded_rows, excluded_reasons)
   - 사용한 열 (used_columns)
   - 단위 (unit)
   - 주의사항 (caution)
   을 반드시 반환한다.
3. NaN 및 Infinity는 JSON 직렬화가 가능하도록 안전하게 None 또는 유효 숫자로 처리한다.
"""

import math
import numpy as np
import pandas as pd

# 집계 행(Aggregate Rows) 식별용 기본 키워드 목록
DEFAULT_AGGREGATE_KEYWORDS = [
    "world", "oecd members", "euro area", "european union",
    "high income", "low income", "middle income", "upper middle", "lower middle",
    "income", "area", "total", "sub-saharan", "latin america",
    "east asia", "south asia", "north america", "arab world",
    "ida & ibrd", "dividend", "countries", "fragile and conflict"
]


def sanitize_val(val):
    """NaN, Inf, numpy 타입을 JSON 직렬화 가능한 순수 파이썬 타입으로 변환"""
    if pd.isna(val) or val is None:
        return None
    if isinstance(val, (float, np.floating)):
        if math.isnan(val) or math.isinf(val):
            return None
        return round(float(val), 4)
    if isinstance(val, (int, np.integer)):
        return int(val)
    return val


def sanitize_records(records):
    """딕셔너리 리스트 내 모든 값을 JSON 안전 형태로 변환"""
    cleaned = []
    for row in records:
        cleaned.append({k: sanitize_val(v) for k, v in row.items()})
    return cleaned


def find_year_columns(df):
    """4자리 숫자로 된 연도 열 목록을 찾아 정렬하여 반환"""
    year_cols = [c for c in df.columns if str(c).strip().isdigit() and len(str(c).strip()) == 4]
    return sorted(year_cols, key=lambda x: int(str(x).strip()))


def find_label_column(df, label_column=None):
    """항목명/국가명으로 사용할 대표 문자열 열 반환"""
    if label_column and label_column in df.columns:
        return label_column
    str_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if str_cols:
        return str_cols[0]
    return df.columns[0]


def is_aggregate_row(val, keywords=None):
    """주어진 문자열이 집계 행(World, Region 등) 키워드를 포함하는지 판별"""
    if not isinstance(val, str):
        return False
    kws = keywords or DEFAULT_AGGREGATE_KEYWORDS
    val_lower = val.lower()
    return any(kw in val_lower for kw in kws)


# ═══════════════════════════════════════════════════════════════════
#  도구 1: reshape_wide_to_long — 연도 열을 행으로 접기
# ═══════════════════════════════════════════════════════════════════
def reshape_wide_to_long(df, id_vars=None, year_columns=None, var_name="Year", value_name="Value", unit="%"):
    """
    가로로 펼쳐진 연도 열들을 '연도', '값' 행으로 변환 (Wide → Long)
    """
    total_original_rows = len(df)
    year_cols = year_columns or find_year_columns(df)

    if not year_cols:
        return {
            "error": "변환할 연도 열(4자리 숫자 열)을 찾을 수 없습니다.",
            "tool": "reshape_wide_to_long"
        }

    if id_vars is None:
        id_vars = [c for c in df.columns if c not in year_cols]

    # melt 수행
    melted = pd.melt(
        df,
        id_vars=id_vars,
        value_vars=year_cols,
        var_name=var_name,
        value_name=value_name
    )

    # 결측치 제외
    initial_long_rows = len(melted)
    valid_melted = melted.dropna(subset=[value_name]).copy()
    excluded_missing_count = initial_long_rows - len(valid_melted)

    # 숫자 변환 시도
    valid_melted[value_name] = pd.to_numeric(valid_melted[value_name], errors="coerce")
    valid_melted = valid_melted.dropna(subset=[value_name])

    sample_preview = sanitize_records(valid_melted.head(20).to_dict(orient="records"))

    return {
        "tool": "reshape_wide_to_long",
        "result": {
            "original_shape": [total_original_rows, len(df.columns)],
            "long_shape": [len(valid_melted), len(valid_melted.columns)],
            "id_variables": id_vars,
            "year_range": f"{year_cols[0]} ~ {year_cols[-1]}",
            "year_columns_count": len(year_cols),
            "preview": sample_preview
        },
        "chart_data": None,
        "reference_period": f"{year_cols[0]}-{year_cols[-1]}",
        "used_rows": len(valid_melted),
        "excluded_rows": excluded_missing_count,
        "excluded_reasons": f"연도별 결측치(NaN) 행 {excluded_missing_count}개 제외",
        "used_columns": id_vars + [var_name, value_name],
        "unit": unit,
        "caution": "Wide 형태를 Long 형태로 변환하여 시계열 추이 분석 및 그룹화 연산에 최적화된 테이블을 생성했습니다."
    }


# ═══════════════════════════════════════════════════════════════════
#  도구 2: quality_report — 결측·중복·타입·이상한 열 이름 점검
# ═══════════════════════════════════════════════════════════════════
def quality_report(df, unit="건"):
    """
    데이터셋 전체의 품질 종합 진단 (결측치, 중복행, 데이터타입, 이상값 범위 검사)
    """
    total_rows = len(df)
    total_cols = len(df.columns)

    # 1. 중복 행
    duplicate_count = int(df.duplicated().sum())

    # 2. 열별 결측치
    missing_by_col = {}
    for col in df.columns:
        m_cnt = int(df[col].isna().sum())
        missing_by_col[str(col)] = {
            "missing_count": m_cnt,
            "missing_ratio_percent": round(m_cnt / total_rows * 100, 1) if total_rows > 0 else 0
        }

    # 3. 이상한 열 이름 (Unnamed, 빈칸, 공백 포함 등)
    suspicious_cols = []
    for col in df.columns:
        c_str = str(col)
        if c_str.startswith("Unnamed") or c_str.strip() == "" or " " in c_str.strip():
            suspicious_cols.append(c_str)

    # 4. 숫자형 열 값 범위 (0~100% 지표인 경우 100 초과 또는 음수 감지)
    outlier_warnings = []
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    for col in numeric_cols:
        col_series = df[col].dropna()
        if len(col_series) > 0:
            c_min = float(col_series.min())
            c_max = float(col_series.max())
            if c_max > 100.0:
                outlier_warnings.append(f"열 '{col}'에 100을 초과하는 값({c_max})이 존재합니다.")
            if c_min < 0.0:
                outlier_warnings.append(f"열 '{col}'에 음수 값({c_min})이 존재합니다.")

    quality_summary = {
        "total_rows": total_rows,
        "total_columns": total_cols,
        "duplicate_rows": duplicate_count,
        "numeric_columns_count": len(numeric_cols),
        "string_columns_count": len(df.select_dtypes(include=["object"]).columns),
        "suspicious_columns": suspicious_cols,
        "outlier_warnings": outlier_warnings,
        "column_missing_details": missing_by_col
    }

    return {
        "tool": "quality_report",
        "result": quality_summary,
        "chart_data": {
            "type": "bar",
            "labels": list(missing_by_col.keys())[:20],
            "datasets": [{
                "label": "결측률 (%)",
                "data": [v["missing_ratio_percent"] for v in list(missing_by_col.values())[:20]]
            }]
        },
        "reference_period": "전체 데이터 기간",
        "used_rows": total_rows,
        "excluded_rows": 0,
        "excluded_reasons": "전체 행을 대상으로 무결성을 진단함 (제외 행 없음)",
        "used_columns": df.columns.tolist(),
        "unit": unit,
        "caution": "결측률이 50%를 넘는 열은 통계적 대표성이 떨어지므로 단독 기준 시점으로 사용하지 마십시오."
    }


# ═══════════════════════════════════════════════════════════════════
#  도구 3: year_coverage — 연도별 값 개수와 권장 기준연도
# ═══════════════════════════════════════════════════════════════════
def year_coverage(df, year_columns=None, unit="%"):
    """
    각 연도별 데이터 수집 건수 및 충실도 계산, 최적의 권장 기준연도 산출
    """
    total_rows = len(df)
    year_cols = year_columns or find_year_columns(df)

    if not year_cols:
        return {
            "error": "연도 열(4자리 숫자)을 찾을 수 없습니다.",
            "tool": "year_coverage"
        }

    coverage_list = []
    best_year = None
    best_cnt = -1

    for y in year_cols:
        valid_cnt = int(df[y].notna().sum())
        ratio = round(valid_cnt / total_rows * 100, 1) if total_rows > 0 else 0
        coverage_list.append({
            "year": str(y),
            "valid_count": valid_cnt,
            "missing_count": total_rows - valid_cnt,
            "coverage_ratio_percent": ratio
        })

    # 권장 기준연도: 70% 이상 충실도를 가진 가장 최근 연도
    qualifying = [c for c in coverage_list if c["coverage_ratio_percent"] >= 70.0]
    if qualifying:
        best_year = qualifying[-1]["year"]
        rec_info = qualifying[-1]
    else:
        # 70% 이상이 없으면 최다 값 보유 연도
        sorted_by_cnt = sorted(coverage_list, key=lambda x: (x["valid_count"], int(x["year"])))
        rec_info = sorted_by_cnt[-1]
        best_year = rec_info["year"]

    return {
        "tool": "year_coverage",
        "result": {
            "recommended_reference_year": best_year,
            "recommended_year_coverage": rec_info,
            "total_entities": total_rows,
            "evaluated_years_count": len(year_cols),
            "coverage_table": coverage_list
        },
        "chart_data": {
            "type": "line",
            "labels": [c["year"] for c in coverage_list],
            "datasets": [{
                "label": "데이터 충실도 (%)",
                "data": [c["coverage_ratio_percent"] for c in coverage_list]
            }]
        },
        "reference_period": f"{year_cols[0]} ~ {year_cols[-1]} (권장: {best_year})",
        "used_rows": total_rows,
        "excluded_rows": 0,
        "excluded_reasons": "모든 행의 연도별 데이터 유무를 집계함 (제외 행 없음)",
        "used_columns": year_cols,
        "unit": unit,
        "caution": f"최신 연도라 하더라도 집계 미완료로 결측률이 높을 수 있습니다. 충실도가 높은 {best_year}년을 우선 활용하십시오."
    }


# ═══════════════════════════════════════════════════════════════════
#  도구 4: detect_aggregates — 집계 행 후보 탐지
# ═══════════════════════════════════════════════════════════════════
def detect_aggregates(df, label_column=None, keywords=None, unit="개"):
    """
    개별 개체(국가)가 아닌 대륙, 소득그룹, 전세계 등 집계 행을 탐지하고 분리
    """
    total_rows = len(df)
    label_col = find_label_column(df, label_column)
    kws = keywords or DEFAULT_AGGREGATE_KEYWORDS

    aggregate_records = []
    normal_indices = []

    for idx, row in df.iterrows():
        val = str(row[label_col])
        val_lower = val.lower()
        matched = [k for k in kws if k in val_lower]
        if matched:
            aggregate_records.append({
                "index": int(idx),
                "name": val,
                "matched_keywords": matched
            })
        else:
            normal_indices.append(int(idx))

    agg_count = len(aggregate_records)
    pure_count = len(normal_indices)

    return {
        "tool": "detect_aggregates",
        "result": {
            "total_rows": total_rows,
            "aggregate_rows_count": agg_count,
            "individual_rows_count": pure_count,
            "label_column": label_col,
            "aggregate_samples": aggregate_records[:15]
        },
        "chart_data": {
            "type": "pie",
            "labels": ["개별 국가/개체", "집계 단위(대륙·소득그룹)"],
            "datasets": [{
                "data": [pure_count, agg_count]
            }]
        },
        "reference_period": "전체 데이터",
        "used_rows": total_rows,
        "excluded_rows": 0,
        "excluded_reasons": "전체 행 대상 집계 행 판별 (제외 행 없음)",
        "used_columns": [label_col],
        "unit": unit,
        "caution": f"총 {agg_count}개의 집계 행이 포함되어 있습니다. 순위 산출(Top/Bottom) 시에는 이 집계 행들을 반드시 제외해야 순위 왜곡이 없습니다."
    }


# ═══════════════════════════════════════════════════════════════════
#  도구 5: describe_numeric — 개수·평균·중앙값·최소·최대·표준편차
# ═══════════════════════════════════════════════════════════════════
def describe_numeric(df, value_column=None, reference_period=None, exclude_aggregates=True, label_column=None, unit="%"):
    """
    선택한 연도/수치 열에 대한 7대 기초 통계량(개수, 평균, 중앙값, 표준편차, 사분위수, 최소, 최대) 계산
    """
    total_original_rows = len(df)
    target_col = str(reference_period) if reference_period and str(reference_period) in df.columns else value_column

    if not target_col or target_col not in df.columns:
        # 기본값으로 가장 최근 연도 열 선택
        year_cols = find_year_columns(df)
        target_col = year_cols[-1] if year_cols else None

    if not target_col or target_col not in df.columns:
        return {
            "error": "분석할 수치 열 또는 기준시점을 찾을 수 없습니다.",
            "tool": "describe_numeric"
        }

    label_col = find_label_column(df, label_column)
    work_df = df.copy()
    excluded_reasons = []
    excluded_count = 0

    # 1. 집계 행 제외 (기본값: 제외)
    if exclude_aggregates and label_col in work_df.columns:
        is_agg = work_df[label_col].apply(lambda x: is_aggregate_row(str(x)))
        agg_cnt = int(is_agg.sum())
        if agg_cnt > 0:
            work_df = work_df[~is_agg]
            excluded_count += agg_cnt
            excluded_reasons.append(f"집계 행(대륙·소득그룹 등) {agg_cnt}개 제외")

    # 2. 결측치 제외
    series = pd.to_numeric(work_df[target_col], errors="coerce")
    missing_cnt = int(series.isna().sum())
    if missing_cnt > 0:
        series = series.dropna()
        excluded_count += missing_cnt
        excluded_reasons.append(f"기준시점 결측치(NaN) {missing_cnt}개 제외")

    used_rows = len(series)
    if used_rows == 0:
        return {
            "error": f"열 '{target_col}'에 계산 가능한 유효 데이터가 없습니다.",
            "tool": "describe_numeric"
        }

    # 기초 통계량 계산
    mean_val = float(series.mean())
    median_val = float(series.median())
    std_val = float(series.std()) if used_rows > 1 else 0.0
    min_val = float(series.min())
    q25_val = float(series.quantile(0.25))
    q75_val = float(series.quantile(0.75))
    max_val = float(series.max())
    iqr_val = q75_val - q25_val

    stats_summary = {
        "column": target_col,
        "count": used_rows,
        "mean": sanitize_val(mean_val),
        "median": sanitize_val(median_val),
        "std": sanitize_val(std_val),
        "min": sanitize_val(min_val),
        "q25": sanitize_val(q25_val),
        "q75": sanitize_val(q75_val),
        "max": sanitize_val(max_val),
        "iqr": sanitize_val(iqr_val)
    }

    return {
        "tool": "describe_numeric",
        "result": stats_summary,
        "chart_data": {
            "type": "bar",
            "labels": ["최소", "1사분위(25%)", "중앙값(50%)", "평균", "3사분위(75%)", "최대"],
            "datasets": [{
                "label": f"{target_col} 요약 통계 ({unit})",
                "data": [
                    sanitize_val(min_val),
                    sanitize_val(q25_val),
                    sanitize_val(median_val),
                    sanitize_val(mean_val),
                    sanitize_val(q75_val),
                    sanitize_val(max_val)
                ]
            }]
        },
        "reference_period": str(target_col),
        "used_rows": used_rows,
        "excluded_rows": excluded_count,
        "excluded_reasons": ", ".join(excluded_reasons) if excluded_reasons else "제외 없음",
        "used_columns": [target_col],
        "unit": unit,
        "caution": f"평균({sanitize_val(mean_val)}{unit})과 중앙값({sanitize_val(median_val)}{unit})의 차이가 클 경우, 극단적인 이상치에 의해 평균이 왜곡되었을 수 있습니다."
    }


# ═══════════════════════════════════════════════════════════════════
#  도구 6: trend_line — 선택 항목들의 연도별 추이 (선 차트)
# ═══════════════════════════════════════════════════════════════════
def trend_line(df, entities=None, label_column=None, year_columns=None, unit="%"):
    """
    선택된 주요 국가/개체들의 연도별 시계열 변화 추이를 선 차트 데이터로 생성
    """
    total_original_rows = len(df)
    label_col = find_label_column(df, label_column)
    year_cols = year_columns or find_year_columns(df)

    if not year_cols:
        return {
            "error": "시계열을 생성할 연도 열을 찾을 수 없습니다.",
            "tool": "trend_line"
        }

    # 대상 항목 선정 (지정되지 않은 경우: World 및 데이터가 풍부한 대표 국가 자동 선택)
    if not entities:
        available_entities = df[label_col].dropna().astype(str).tolist()
        candidates = ["World", "Korea, Rep.", "United States", "Japan", "Germany", "China"]
        selected = [c for c in candidates if c in available_entities]
        if not selected:
            selected = available_entities[:5]
        entities = selected

    datasets = []
    trend_table = []
    matched_rows = 0

    # 색상 팔레트
    colors = ["#2563eb", "#dc2626", "#16a34a", "#d97706", "#9333ea", "#0891b2", "#4b5563"]

    for i, entity in enumerate(entities):
        row_match = df[df[label_col].astype(str) == str(entity)]
        if row_match.empty:
            continue
        matched_rows += 1
        row_series = row_match.iloc[0]

        values = []
        for y in year_cols:
            v = row_series.get(y)
            num_v = sanitize_val(pd.to_numeric(v, errors="coerce"))
            values.append(num_v)

        color = colors[i % len(colors)]
        datasets.append({
            "label": str(entity),
            "data": values,
            "borderColor": color,
            "backgroundColor": color,
            "tension": 0.2,
            "fill": False
        })
        trend_table.append({
            "entity": str(entity),
            "start_value": values[0] if values else None,
            "latest_value": next((v for v in reversed(values) if v is not None), None)
        })

    excluded_rows = total_original_rows - matched_rows

    return {
        "tool": "trend_line",
        "result": {
            "selected_entities": entities,
            "years_range": f"{year_cols[0]} ~ {year_cols[-1]}",
            "trend_summary": trend_table
        },
        "chart_data": {
            "type": "line",
            "labels": [str(y) for y in year_cols],
            "datasets": datasets
        },
        "reference_period": f"{year_cols[0]}-{year_cols[-1]}",
        "used_rows": matched_rows,
        "excluded_rows": excluded_rows,
        "excluded_reasons": f"선택된 {matched_rows}개 개체 외 나머지 {excluded_rows}개 행 제외",
        "used_columns": [label_col] + year_cols,
        "unit": unit,
        "caution": "연도별 결측치가 존재하는 구간에서는 선이 끊기거나 보간되어 보일 수 있습니다."
    }


# ═══════════════════════════════════════════════════════════════════
#  도구 7: top_bottom_n — 기준연도 기준 상위·하위 N (수평 막대)
# ═══════════════════════════════════════════════════════════════════
def top_bottom_n(df, reference_period=None, value_column=None, label_column=None, n=10, top_n=None, bottom_n=None, mode="both", exclude_aggregates=True, unit="%", **kwargs):
    """
    기준연도(reference_period 필수) 기준 상위 N개 및 하위 N개 항목 추출 및 수평 막대 차트 생성
    """
    # [절대 규칙] 기준시점 필수 검증
    target_period = str(reference_period) if reference_period else (str(value_column) if value_column else None)
    if not target_period or target_period not in df.columns:
        return {
            "error": "기준 시점(reference_period)은 필수 파라미터입니다. 유효한 연도 열을 지정해야 합니다.",
            "tool": "top_bottom_n"
        }

    total_original_rows = len(df)
    label_col = find_label_column(df, label_column)
    work_df = df.copy()
    excluded_reasons = []
    excluded_count = 0

    # 1. 집계 행 제외 (기본값: 제외)
    if exclude_aggregates and label_col in work_df.columns:
        is_agg = work_df[label_col].apply(lambda x: is_aggregate_row(str(x)))
        agg_cnt = int(is_agg.sum())
        if agg_cnt > 0:
            work_df = work_df[~is_agg]
            excluded_count += agg_cnt
            excluded_reasons.append(f"집계 행(대륙·지역그룹 등) {agg_cnt}개 제외")

    # 2. 기준 시점 수치 변환 및 결측치 제외
    work_df["_numeric_val"] = pd.to_numeric(work_df[target_period], errors="coerce")
    missing_cnt = int(work_df["_numeric_val"].isna().sum())
    if missing_cnt > 0:
        work_df = work_df.dropna(subset=["_numeric_val"])
        excluded_count += missing_cnt
        excluded_reasons.append(f"기준시점({target_period}) 결측치(NaN) {missing_cnt}개 제외")

    used_rows = len(work_df)
    if used_rows == 0:
        return {
            "error": f"기준시점 '{target_period}'에 유효한 데이터가 전혀 없습니다.",
            "tool": "top_bottom_n"
        }

    # 정렬
    sorted_df = work_df.sort_values(by="_numeric_val", ascending=False)
    
    # 상위/하위 개수 산출 (개별 지정 지원: 예: top_n=10, bottom_n=30)
    actual_top_n = min(int(top_n if top_n is not None else n), used_rows)
    actual_bottom_n = min(int(bottom_n if bottom_n is not None else n), used_rows)

    top_df = sorted_df.head(actual_top_n)
    bottom_df = sorted_df.tail(actual_bottom_n).sort_values(by="_numeric_val", ascending=True)

    top_list = [
        {"rank": i + 1, "label": str(r[label_col]), "value": sanitize_val(r["_numeric_val"])}
        for i, (_, r) in enumerate(top_df.iterrows())
    ]
    bottom_list = [
        {"rank": used_rows - actual_bottom_n + i + 1, "label": str(r[label_col]), "value": sanitize_val(r["_numeric_val"])}
        for i, (_, r) in enumerate(bottom_df.sort_values(by="_numeric_val", ascending=False).iterrows())
    ]

    # 모드 판별: mode 파라미터 또는 top_n/bottom_n 존재 여부
    is_both = (mode == "both") or ("bottom_n" in kwargs and "top_n" in kwargs) or ("하위" in str(kwargs.get("intent", "")))

    if is_both:
        # 상위와 하위를 한 차트에 합치되 색상과 라벨로 명확히 분리
        combined_labels = []
        combined_values = []
        combined_colors = []

        for item in top_list:
            combined_labels.append(f"[{item['rank']}위] {item['label']}")
            combined_values.append(item["value"])
            combined_colors.append("#2563eb")  # 상위는 신뢰감 있는 파란색

        for item in reversed(bottom_list):  # 최하위부터 위로
            combined_labels.append(f"[{item['rank']}위] {item['label']}")
            combined_values.append(item["value"])
            combined_colors.append("#ef4444")  # 하위는 대비되는 빨간색

        chart_data = {
            "type": "bar",
            "indexAxis": "y",
            "labels": combined_labels,
            "datasets": [{
                "label": f"{target_period}년 상위 {actual_top_n}개(파랑) & 하위 {actual_bottom_n}개(빨강) ({unit})",
                "data": combined_values,
                "backgroundColor": combined_colors,
                "borderWidth": 1
            }]
        }
    elif mode == "bottom":
        chart_data = {
            "type": "bar",
            "indexAxis": "y",
            "labels": [f"[{item['rank']}위] {item['label']}" for item in bottom_list],
            "datasets": [{
                "label": f"{target_period}년 하위 {actual_bottom_n} ({unit})",
                "data": [item["value"] for item in bottom_list],
                "backgroundColor": "#ef4444"
            }]
        }
    else:
        chart_data = {
            "type": "bar",
            "indexAxis": "y",
            "labels": [f"[{item['rank']}위] {item['label']}" for item in top_list],
            "datasets": [{
                "label": f"{target_period}년 상위 {actual_top_n} ({unit})",
                "data": [item["value"] for item in top_list],
                "backgroundColor": "#2563eb"
            }]
        }

    return {
        "tool": "top_bottom_n",
        "result": {
            "reference_period": target_period,
            "top_n_count": actual_top_n,
            "bottom_n_count": actual_bottom_n,
            "top_n": top_list,
            "bottom_n": bottom_list,
            "total_ranked_entities": used_rows
        },
        "chart_data": chart_data,
        "reference_period": target_period,
        "used_rows": used_rows,
        "excluded_rows": excluded_count,
        "excluded_reasons": ", ".join(excluded_reasons) if excluded_reasons else "제외 없음",
        "used_columns": [label_col, target_period],
        "unit": unit,
        "caution": f"순위 왜곡을 방지하기 위해 집계 단위 행을 제외하고 기준 시점({target_period}년)에 유효한 값을 가진 {used_rows}개 개체만으로 순위를 산출했습니다."
    }


# ═══════════════════════════════════════════════════════════════════
#  도구 8: group_summary — 그룹별 평균·합계·개수 (막대)
# ═══════════════════════════════════════════════════════════════════
def group_summary(df, reference_period=None, group_column=None, value_column=None, label_column=None, exclude_aggregates=True, unit="%"):
    """
    범주형 열(소득그룹, 지역 등) 또는 분위수 구간별로 데이터 요약 통계(평균, 중앙값, 개수) 산출
    """
    total_original_rows = len(df)
    target_period = str(reference_period) if reference_period and str(reference_period) in df.columns else (str(value_column) if value_column else None)

    if not target_period or target_period not in df.columns:
        year_cols = find_year_columns(df)
        target_period = year_cols[-1] if year_cols else None

    if not target_period or target_period not in df.columns:
        return {
            "error": "그룹 통계를 계산할 수치 열 또는 기준시점을 찾을 수 없습니다.",
            "tool": "group_summary"
        }

    label_col = find_label_column(df, label_column)
    work_df = df.copy()
    excluded_reasons = []
    excluded_count = 0

    # 1. 집계 행 제외 (기본값: 제외)
    if exclude_aggregates and label_col in work_df.columns:
        is_agg = work_df[label_col].apply(lambda x: is_aggregate_row(str(x)))
        agg_cnt = int(is_agg.sum())
        if agg_cnt > 0:
            work_df = work_df[~is_agg]
            excluded_count += agg_cnt
            excluded_reasons.append(f"집계 행 {agg_cnt}개 제외")

    # 2. 결측치 제외
    work_df["_numeric_val"] = pd.to_numeric(work_df[target_period], errors="coerce")
    missing_cnt = int(work_df["_numeric_val"].isna().sum())
    if missing_cnt > 0:
        work_df = work_df.dropna(subset=["_numeric_val"])
        excluded_count += missing_cnt
        excluded_reasons.append(f"기준시점({target_period}) 결측치 {missing_cnt}개 제외")

    used_rows = len(work_df)
    if used_rows == 0:
        return {
            "error": "유효한 데이터가 없습니다.",
            "tool": "group_summary"
        }

    # 3. 그룹 열 결정: 명시된 열이 없으면 사분위수(4개 분위수 그룹)로 자동 구간 분할
    used_group_col = group_column
    if not used_group_col or used_group_col not in work_df.columns or work_df[used_group_col].nunique() < 2:
        # 사분위수 기반 4개 분위 그룹 자동 생성
        work_df["_auto_group"] = pd.qcut(
            work_df["_numeric_val"],
            q=4,
            labels=["하위 25% (Q1)", "중하위 25% (Q2)", "중상위 25% (Q3)", "상위 25% (Q4)"],
            duplicates="drop"
        )
        used_group_col = "_auto_group"
        group_display_name = "수치 분위수 그룹 (사분위)"
    else:
        group_display_name = used_group_col

    # groupby 연산 (Pandas)
    grouped = work_df.groupby(used_group_col, observed=False)["_numeric_val"].agg(
        count="count",
        mean="mean",
        median="median",
        min="min",
        max="max"
    ).reset_index()

    summary_records = []
    for _, r in grouped.iterrows():
        summary_records.append({
            "group": str(r[used_group_col]),
            "count": int(r["count"]),
            "mean": sanitize_val(r["mean"]),
            "median": sanitize_val(r["median"]),
            "min": sanitize_val(r["min"]),
            "max": sanitize_val(r["max"])
        })

    chart_labels = [r["group"] for r in summary_records]
    chart_means = [r["mean"] for r in summary_records]

    return {
        "tool": "group_summary",
        "result": {
            "group_column": group_display_name,
            "reference_period": target_period,
            "groups_count": len(summary_records),
            "summary_table": summary_records
        },
        "chart_data": {
            "type": "bar",
            "labels": chart_labels,
            "datasets": [{
                "label": f"그룹별 평균 ({unit})",
                "data": chart_means,
                "backgroundColor": "#3b82f6"
            }]
        },
        "reference_period": target_period,
        "used_rows": used_rows,
        "excluded_rows": excluded_count,
        "excluded_reasons": ", ".join(excluded_reasons) if excluded_reasons else "제외 없음",
        "used_columns": [used_group_col, target_period],
        "unit": unit,
        "caution": f"그룹별 표본 수(count) 차이가 클 경우 소수 그룹의 평균이 전체 비교를 왜곡할 수 있으니 개수도 함께 확인하십시오."
    }


# ═══════════════════════════════════════════════════════════════════
#  도구 9: change_rate — 두 시점 사이 변화량·변화율 (막대)
# ═══════════════════════════════════════════════════════════════════
def change_rate(df, reference_period=None, start_period=None, end_period=None, n=10, label_column=None, exclude_aggregates=True, unit="%"):
    """
    [기준시점 필수] 두 시점(시작연도 vs 종료/기준연도) 사이의 절대 변화량(%p) 및 변화율(%) 계산
    """
    target_end = str(reference_period) if reference_period else (str(end_period) if end_period else None)
    if not target_end or target_end not in df.columns:
        return {
            "error": "기준 시점(종료 시점 reference_period)은 필수 파라미터입니다.",
            "tool": "change_rate"
        }

    # 시작 시점이 지정되지 않은 경우: 종료 시점 기준 10년 전 (또는 가능한 가장 이른 연도)
    target_start = str(start_period) if start_period else None
    if not target_start or target_start not in df.columns:
        year_cols = find_year_columns(df)
        if target_end in year_cols:
            idx = year_cols.index(target_end)
            target_start = year_cols[max(0, idx - 10)]
        else:
            target_start = year_cols[0] if year_cols else None

    if not target_start or target_start not in df.columns or target_start == target_end:
        return {
            "error": f"비교할 유효한 시작 시점(start_period)을 찾을 수 없습니다. (종료시점: {target_end})",
            "tool": "change_rate"
        }

    total_original_rows = len(df)
    label_col = find_label_column(df, label_column)
    work_df = df.copy()
    excluded_reasons = []
    excluded_count = 0

    # 1. 집계 행 제외
    if exclude_aggregates and label_col in work_df.columns:
        is_agg = work_df[label_col].apply(lambda x: is_aggregate_row(str(x)))
        agg_cnt = int(is_agg.sum())
        if agg_cnt > 0:
            work_df = work_df[~is_agg]
            excluded_count += agg_cnt
            excluded_reasons.append(f"집계 행 {agg_cnt}개 제외")

    # 2. 수치 변환 및 두 시점 모두 유효한 행만 선택
    work_df["_start_val"] = pd.to_numeric(work_df[target_start], errors="coerce")
    work_df["_end_val"] = pd.to_numeric(work_df[target_end], errors="coerce")

    valid_mask = work_df["_start_val"].notna() & work_df["_end_val"].notna()
    missing_cnt = int((~valid_mask).sum())
    if missing_cnt > 0:
        work_df = work_df[valid_mask].copy()
        excluded_count += missing_cnt
        excluded_reasons.append(f"시작({target_start}) 또는 종료({target_end}) 결측 행 {missing_cnt}개 제외")

    used_rows = len(work_df)
    if used_rows == 0:
        return {
            "error": f"두 시점({target_start} ~ {target_end}) 모두 데이터가 존재하는 개체가 없습니다.",
            "tool": "change_rate"
        }

    # 변화량(절대 차이) 및 변화율 계산
    work_df["_diff"] = work_df["_end_val"] - work_df["_start_val"]
    # 시작값이 0 초과일 때만 백분율 변화율 계산
    work_df["_pct_change"] = np.where(
        work_df["_start_val"] > 0,
        (work_df["_diff"] / work_df["_start_val"]) * 100.0,
        np.nan
    )

    # 변화량 기준 정렬
    sorted_df = work_df.sort_values(by="_diff", ascending=False)
    actual_n = min(int(n), used_rows)

    top_increase = [
        {
            "rank": i + 1,
            "label": str(r[label_col]),
            "start_val": sanitize_val(r["_start_val"]),
            "end_val": sanitize_val(r["_end_val"]),
            "diff": sanitize_val(r["_diff"]),
            "pct_change": sanitize_val(r["_pct_change"])
        }
        for i, (_, r) in enumerate(sorted_df.head(actual_n).iterrows())
    ]

    top_decrease = [
        {
            "rank": i + 1,
            "label": str(r[label_col]),
            "start_val": sanitize_val(r["_start_val"]),
            "end_val": sanitize_val(r["_end_val"]),
            "diff": sanitize_val(r["_diff"]),
            "pct_change": sanitize_val(r["_pct_change"])
        }
        for i, (_, r) in enumerate(sorted_df.tail(actual_n).sort_values(by="_diff", ascending=True).iterrows())
    ]

    chart_labels = [item["label"] for item in top_increase]
    chart_diffs = [item["diff"] for item in top_increase]

    return {
        "tool": "change_rate",
        "result": {
            "start_period": target_start,
            "end_period": target_end,
            "top_increases": top_increase,
            "top_decreases": top_decrease,
            "total_compared_entities": used_rows
        },
        "chart_data": {
            "type": "bar",
            "labels": chart_labels,
            "datasets": [{
                "label": f"{target_start}→{target_end} 증가폭 상위 ({unit}p)",
                "data": chart_diffs,
                "backgroundColor": "#10b981"
            }]
        },
        "reference_period": f"{target_start}-{target_end} (기준시점: {target_end})",
        "used_rows": used_rows,
        "excluded_rows": excluded_count,
        "excluded_reasons": ", ".join(excluded_reasons) if excluded_reasons else "제외 없음",
        "used_columns": [label_col, target_start, target_end],
        "unit": f"{unit}p (포인트 변화)",
        "caution": f"초기값({target_start}년)이 매우 낮았던 국가는 작은 절대치 상승으로도 높은 변화율(%)을 보일 수 있으므로 절대 증가량(diff)과 함께 해석하십시오."
    }


# ═══════════════════════════════════════════════════════════════════
#  도구 10: distribution — 구간별 분포 (히스토그램)
# ═══════════════════════════════════════════════════════════════════
def distribution(df, reference_period=None, value_column=None, bins=10, label_column=None, exclude_aggregates=True, unit="%"):
    """
    [기준시점 필수] 기준 시점 데이터의 수치 구간별 빈도수 및 밀도를 계산하여 히스토그램 차트 데이터 생성
    """
    target_period = str(reference_period) if reference_period else (str(value_column) if value_column else None)
    if not target_period or target_period not in df.columns:
        return {
            "error": "기준 시점(reference_period)은 필수 파라미터입니다.",
            "tool": "distribution"
        }

    total_original_rows = len(df)
    label_col = find_label_column(df, label_column)
    work_df = df.copy()
    excluded_reasons = []
    excluded_count = 0

    # 1. 집계 행 제외
    if exclude_aggregates and label_col in work_df.columns:
        is_agg = work_df[label_col].apply(lambda x: is_aggregate_row(str(x)))
        agg_cnt = int(is_agg.sum())
        if agg_cnt > 0:
            work_df = work_df[~is_agg]
            excluded_count += agg_cnt
            excluded_reasons.append(f"집계 행 {agg_cnt}개 제외")

    # 2. 결측치 제외
    series = pd.to_numeric(work_df[target_period], errors="coerce").dropna()
    missing_cnt = len(work_df) - len(series)
    if missing_cnt > 0:
        excluded_count += missing_cnt
        excluded_reasons.append(f"기준시점({target_period}) 결측치 {missing_cnt}개 제외")

    used_rows = len(series)
    if used_rows == 0:
        return {
            "error": f"기준시점 '{target_period}'에 유효한 데이터가 없습니다.",
            "tool": "distribution"
        }

    # 히스토그램 구간 계산 (numpy)
    bin_count = max(3, min(int(bins), 20))
    counts, bin_edges = np.histogram(series, bins=bin_count)

    bin_table = []
    labels = []
    for i in range(len(counts)):
        b_start = round(float(bin_edges[i]), 1)
        b_end = round(float(bin_edges[i + 1]), 1)
        cnt = int(counts[i])
        ratio = round(cnt / used_rows * 100, 1) if used_rows > 0 else 0
        lbl = f"{b_start}~{b_end}{unit}"
        labels.append(lbl)
        bin_table.append({
            "range": lbl,
            "min": b_start,
            "max": b_end,
            "count": cnt,
            "ratio_percent": ratio
        })

    return {
        "tool": "distribution",
        "result": {
            "reference_period": target_period,
            "bins_count": bin_count,
            "total_samples": used_rows,
            "distribution_table": bin_table
        },
        "chart_data": {
            "type": "bar",
            "labels": labels,
            "datasets": [{
                "label": f"{target_period}년 빈도수 (개체 수)",
                "data": [int(c) for c in counts],
                "backgroundColor": "#6366f1"
            }]
        },
        "reference_period": target_period,
        "used_rows": used_rows,
        "excluded_rows": excluded_count,
        "excluded_reasons": ", ".join(excluded_reasons) if excluded_reasons else "제외 없음",
        "used_columns": [target_period],
        "unit": "개체 수",
        "caution": "특정 구간에 빈도가 과도하게 집중되어 있는 경우, 비정규 분포(양극화 또는 좌우 비대칭) 특성을 감안하여 평균보다는 중앙값 위주로 데이터를 평가해야 합니다."
    }


# ═══════════════════════════════════════════════════════════════════
#  도구 11: correlation_scatter — 두 숫자 열 상관계수 (산점도)
# ═══════════════════════════════════════════════════════════════════
def correlation_scatter(df, reference_period=None, x_column=None, y_column=None, label_column=None, exclude_aggregates=True, unit=""):
    """
    [기준시점 필수] 두 수치 열 간의 피어슨 상관계수(r) 계산 및 산점도 데이터 생성
    [주의사항 절대 규칙] '상관관계는 인과관계가 아니다'를 반드시 포함한다.
    """
    # x열과 y열 결정: reference_period가 있으면 x열 또는 y열로 사용
    col_x = str(x_column) if x_column and str(x_column) in df.columns else (str(reference_period) if reference_period and str(reference_period) in df.columns else None)
    col_y = str(y_column) if y_column and str(y_column) in df.columns else None

    # 만약 y열이 지정되지 않은 경우: x열 이전 10년 전 연도 열을 y열로 자동 매칭
    year_cols = find_year_columns(df)
    if not col_x and year_cols:
        col_x = year_cols[-1]
    if not col_y and year_cols and len(year_cols) >= 2:
        idx = year_cols.index(col_x) if col_x in year_cols else len(year_cols) - 1
        col_y = year_cols[max(0, idx - 5)]

    if not col_x or not col_y or col_x not in df.columns or col_y not in df.columns or col_x == col_y:
        return {
            "error": "상관분석을 수행할 두 개의 유효한 수치 열(또는 기준시점)을 지정해야 합니다.",
            "tool": "correlation_scatter"
        }

    total_original_rows = len(df)
    label_col = find_label_column(df, label_column)
    work_df = df.copy()
    excluded_reasons = []
    excluded_count = 0

    # 1. 집계 행 제외
    if exclude_aggregates and label_col in work_df.columns:
        is_agg = work_df[label_col].apply(lambda x: is_aggregate_row(str(x)))
        agg_cnt = int(is_agg.sum())
        if agg_cnt > 0:
            work_df = work_df[~is_agg]
            excluded_count += agg_cnt
            excluded_reasons.append(f"집계 행 {agg_cnt}개 제외")

    # 2. 결측치 제외
    work_df["_x"] = pd.to_numeric(work_df[col_x], errors="coerce")
    work_df["_y"] = pd.to_numeric(work_df[col_y], errors="coerce")

    valid_mask = work_df["_x"].notna() & work_df["_y"].notna()
    missing_cnt = int((~valid_mask).sum())
    if missing_cnt > 0:
        work_df = work_df[valid_mask].copy()
        excluded_count += missing_cnt
        excluded_reasons.append(f"두 변수 중 결측치 포함 행 {missing_cnt}개 제외")

    used_rows = len(work_df)
    if used_rows < 3:
        return {
            "error": "상관계수를 계산하기 위한 유효 행 수가 부족합니다 (최소 3개 필요).",
            "tool": "correlation_scatter"
        }

    # 피어슨 상관계수 (Pandas .corr)
    corr_val = float(work_df["_x"].corr(work_df["_y"]))

    # 상관 강도 해석
    abs_corr = abs(corr_val)
    if abs_corr >= 0.7:
        strength = "매우 강한 상관관계"
    elif abs_corr >= 0.4:
        strength = "뚜렷한 상관관계"
    elif abs_corr >= 0.2:
        strength = "약한 상관관계"
    else:
        strength = "거의 상관관계 없음"
    direction = "양(+)" if corr_val >= 0 else "음(-)"

    # 산점도 차트 데이터
    scatter_points = [
        {"x": sanitize_val(r["_x"]), "y": sanitize_val(r["_y"]), "label": str(r[label_col])}
        for _, r in work_df.iterrows()
    ]

    return {
        "tool": "correlation_scatter",
        "result": {
            "x_column": col_x,
            "y_column": col_y,
            "correlation_coefficient_r": sanitize_val(corr_val),
            "correlation_strength": f"{direction}의 {strength} (r={round(corr_val, 4)})",
            "sample_size": used_rows,
            "points_count": len(scatter_points)
        },
        "chart_data": {
            "type": "scatter",
            "datasets": [{
                "label": f"{col_x} vs {col_y} 산점도",
                "data": [{"x": p["x"], "y": p["y"]} for p in scatter_points],
                "backgroundColor": "#ec4899"
            }]
        },
        "reference_period": f"{col_x} & {col_y}",
        "used_rows": used_rows,
        "excluded_rows": excluded_count,
        "excluded_reasons": ", ".join(excluded_reasons) if excluded_reasons else "제외 없음",
        "used_columns": [label_col, col_x, col_y],
        "unit": unit,
        "caution": "⚠️ [중요 원칙] 상관관계는 인과관계가 아니다. 두 변수가 같은 방향으로 움직인다고 해서 한쪽이 다른 쪽의 직접적 원인이라고 단정할 수 없습니다."
    }


# ═══════════════════════════════════════════════════════════════════
#  도구 12: compare_one_vs_all — 특정 항목 vs 전체 평균·중앙값
# ═══════════════════════════════════════════════════════════════════
def compare_one_vs_all(df, reference_period=None, target_entity=None, value_column=None, label_column=None, exclude_aggregates=True, unit="%"):
    """
    [기준시점 필수] 관심 있는 특정 국가/항목의 값을 전체 집단의 평균 및 중앙값과 직관적으로 비교 (강조 막대)
    """
    target_period = str(reference_period) if reference_period else (str(value_column) if value_column else None)
    if not target_period or target_period not in df.columns:
        return {
            "error": "기준 시점(reference_period)은 필수 파라미터입니다.",
            "tool": "compare_one_vs_all"
        }

    total_original_rows = len(df)
    label_col = find_label_column(df, label_column)
    work_df = df.copy()
    excluded_reasons = []
    excluded_count = 0

    # 대상 항목 결정 (지정되지 않은 경우 대한민국 또는 첫 번째 유효 국가)
    target_name = str(target_entity) if target_entity else None
    if not target_name:
        all_labels = work_df[label_col].dropna().astype(str).tolist()
        if "Korea, Rep." in all_labels:
            target_name = "Korea, Rep."
        else:
            target_name = all_labels[0] if all_labels else None

    # 1. 집계 행 제외 (단, target_entity가 집계 행인 경우 본인은 유지)
    if exclude_aggregates and label_col in work_df.columns:
        is_agg = work_df[label_col].apply(lambda x: is_aggregate_row(str(x)) and str(x) != target_name)
        agg_cnt = int(is_agg.sum())
        if agg_cnt > 0:
            work_df = work_df[~is_agg]
            excluded_count += agg_cnt
            excluded_reasons.append(f"집계 행 {agg_cnt}개 제외")

    # 2. 수치 변환 및 결측치 제외
    work_df["_numeric_val"] = pd.to_numeric(work_df[target_period], errors="coerce")
    missing_cnt = int(work_df["_numeric_val"].isna().sum())
    if missing_cnt > 0:
        work_df = work_df.dropna(subset=["_numeric_val"])
        excluded_count += missing_cnt
        excluded_reasons.append(f"기준시점({target_period}) 결측치 {missing_cnt}개 제외")

    used_rows = len(work_df)
    if used_rows == 0:
        return {
            "error": f"기준시점 '{target_period}'에 유효한 데이터가 없습니다.",
            "tool": "compare_one_vs_all"
        }

    # 대상 항목의 수치 추출
    target_row = work_df[work_df[label_col].astype(str) == target_name]
    if target_row.empty:
        return {
            "error": f"개체 '{target_name}'의 유효한 {target_period}년 데이터를 찾을 수 없습니다.",
            "tool": "compare_one_vs_all"
        }

    target_val = float(target_row["_numeric_val"].iloc[0])
    all_mean = float(work_df["_numeric_val"].mean())
    all_median = float(work_df["_numeric_val"].median())

    # 순위 산출
    work_df["_rank"] = work_df["_numeric_val"].rank(ascending=False, method="min")
    target_rank = int(work_df[work_df[label_col].astype(str) == target_name]["_rank"].iloc[0])

    diff_from_mean = target_val - all_mean
    diff_from_median = target_val - all_median

    return {
        "tool": "compare_one_vs_all",
        "result": {
            "target_entity": target_name,
            "reference_period": target_period,
            "target_value": sanitize_val(target_val),
            "group_mean": sanitize_val(all_mean),
            "group_median": sanitize_val(all_median),
            "diff_from_mean": sanitize_val(diff_from_mean),
            "diff_from_median": sanitize_val(diff_from_median),
            "target_rank": target_rank,
            "total_entities": used_rows,
            "percentile": round((used_rows - target_rank + 1) / used_rows * 100, 1)
        },
        "chart_data": {
            "type": "bar",
            "labels": [target_name, f"전체 평균 ({used_rows}개국)", f"전체 중앙값 ({used_rows}개국)"],
            "datasets": [{
                "label": f"{target_period}년 비교 ({unit})",
                "data": [sanitize_val(target_val), sanitize_val(all_mean), sanitize_val(all_median)],
                "backgroundColor": ["#ef4444", "#94a3b8", "#64748b"]
            }]
        },
        "reference_period": target_period,
        "used_rows": used_rows,
        "excluded_rows": excluded_count,
        "excluded_reasons": ", ".join(excluded_reasons) if excluded_reasons else "제외 없음",
        "used_columns": [label_col, target_period],
        "unit": unit,
        "caution": f"{target_name}은(는) 전체 {used_rows}개 비교 대상 중 {target_rank}위를 차지하며, 중앙값 대비 {round(abs(diff_from_median), 1)}{unit} {'높습니다' if diff_from_median >= 0 else '낮습니다'}."
    }


# ═══════════════════════════════════════════════════════════════════
#  도구 카탈로그 레지스트리 (12개 전체 등록)
# ═══════════════════════════════════════════════════════════════════
TOOL_CATALOG = {
    "reshape_wide_to_long": {
        "func": reshape_wide_to_long,
        "name": "연도 열을 행으로 변환 (Wide to Long)",
        "description": "연도별로 가로로 나열된 데이터를 시계열 분석에 적합한 세로형(행) 테이블로 재구성합니다.",
        "requires_reference_period": False,
        "chart_type": None
    },
    "quality_report": {
        "func": quality_report,
        "name": "데이터 무결성 및 품질 진단",
        "description": "결측률, 중복 행, 이상치(100% 초과 등)를 종합 검사하여 데이터 품질 리포트를 작성합니다.",
        "requires_reference_period": False,
        "chart_type": "bar"
    },
    "year_coverage": {
        "func": year_coverage,
        "name": "연도별 데이터 충실도 및 권장 기준연도 산출",
        "description": "연도별 실제 수집된 데이터 비율을 계산하고, 가장 신뢰성 높은 권장 기준연도를 찾아냅니다.",
        "requires_reference_period": False,
        "chart_type": "line"
    },
    "detect_aggregates": {
        "func": detect_aggregates,
        "name": "집계 단위 행 후보 탐지 및 분리",
        "description": "개별 국가가 아닌 대륙, 소득그룹, 전세계 등 집계 행을 탐지하여 순위 왜곡을 방지합니다.",
        "requires_reference_period": False,
        "chart_type": "pie"
    },
    "describe_numeric": {
        "func": describe_numeric,
        "name": "7대 기초 통계량 요약",
        "description": "기준 시점에 대한 평균, 중앙값, 표준편차, 사분위수(IQR), 최소/최대 분포를 계산합니다.",
        "requires_reference_period": False,
        "chart_type": "bar"
    },
    "trend_line": {
        "func": trend_line,
        "name": "주요 개체 연도별 시계열 추이 (선 차트)",
        "description": "선택한 주요 국가들의 연도별 수치 변화 추세를 다중 선 그래프로 시각화합니다.",
        "requires_reference_period": False,
        "chart_type": "line"
    },
    "top_bottom_n": {
        "func": top_bottom_n,
        "name": "기준연도 상위·하위 순위 랭킹 (수평 막대)",
        "description": "지정한 기준 시점 기준으로 가장 높은 상위 N개국과 가장 낮은 하위 N개국을 추출합니다.",
        "requires_reference_period": True,
        "chart_type": "bar"
    },
    "group_summary": {
        "func": group_summary,
        "name": "그룹별 요약 통계 (막대 차트)",
        "description": "사분위수 구간 또는 범주형 그룹별 평균, 중앙값, 표본 수를 산출합니다.",
        "requires_reference_period": False,
        "chart_type": "bar"
    },
    "change_rate": {
        "func": change_rate,
        "name": "두 시점 간 변화량 및 성장률 분석 (막대)",
        "description": "시작 시점과 종료 시점 사이의 절대 증가폭(%p)과 백분율 변화율(%)을 계산합니다.",
        "requires_reference_period": True,
        "chart_type": "bar"
    },
    "distribution": {
        "func": distribution,
        "name": "구간별 빈도 분포 (히스토그램)",
        "description": "기준 시점의 수치를 일정한 구간(Bins)으로 나누어 분포의 치우침을 분석합니다.",
        "requires_reference_period": True,
        "chart_type": "bar"
    },
    "correlation_scatter": {
        "func": correlation_scatter,
        "name": "두 변수 간 상관계수 분석 (산점도)",
        "description": "두 시점 또는 두 변수 간 피어슨 상관계수(r)를 구하고 산점도를 제공합니다.",
        "requires_reference_period": True,
        "chart_type": "scatter"
    },
    "compare_one_vs_all": {
        "func": compare_one_vs_all,
        "name": "특정 항목 vs 전체 평균·중앙값 비교 (강조 막대)",
        "description": "관심 있는 특정 국가(예: 한국)의 수치를 전체 집단 평균 및 중앙값과 직관적으로 대비합니다.",
        "requires_reference_period": True,
        "chart_type": "bar"
    }
}

