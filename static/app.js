/**
 * AI Data Insight Builder — app.js
 * 바닐라 자바스크립트 + Chart.js
 * 
 * Frontend Console Log 규격:
 * [PAGE]    페이지 로드 완료
 * [UPLOAD]  파일 선택 (파일명, 크기)
 * [INSPECT] 진단 요청 시작 / 응답 수신 (행 수, 열 수, 경고 수)
 * [SUGGEST] 제안 요청 시작 / 응답 수신 (제안 개수)
 * [RUN]     도구 실행 시작 (도구명, 기준시점) / 결과 수신
 * [CHART]   차트 렌더 완료 (차트 종류)
 * [EXPLAIN] 해석 요청 시작 (모드) / 응답 수신
 * [ERROR]   오류 발생
 */

// ── 전역 상태 ───────────────────────────────────────────────
const state = {
    selectedFile: null,
    diagnosis: null,
    lastRunResult: null,
    chartInstance: null
};

// 12개 도구 목록 (수동 선택용)
const TOOL_OPTIONS = [
    { id: "top_bottom_n", name: "기준연도 상위·하위 순위 랭킹 (수평 막대)" },
    { id: "trend_line", name: "주요 개체 연도별 시계열 추이 (선 차트)" },
    { id: "change_rate", name: "두 시점 간 변화량 및 성장률 (막대)" },
    { id: "compare_one_vs_all", name: "특정 항목 vs 전체 평균·중앙값 (강조 막대)" },
    { id: "describe_numeric", name: "7대 기초 통계량 요약" },
    { id: "distribution", name: "구간별 빈도 분포 (히스토그램)" },
    { id: "correlation_scatter", name: "두 변수 간 상관계수 (산점도)" },
    { id: "group_summary", name: "그룹별 요약 통계" },
    { id: "quality_report", name: "데이터 무결성 및 품질 진단" },
    { id: "year_coverage", name: "연도별 데이터 충실도 분석" },
    { id: "detect_aggregates", name: "집계 단위 행 후보 탐지" },
    { id: "reshape_wide_to_long", name: "연도 열을 행으로 변환 (Wide to Long)" }
];

// ── 1. 페이지 로드 초기화 ──────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
    console.log("[PAGE]    페이지 로드 완료");
    initHealthCheck();
    initDropZone();
    initManualToolOptions();
    bindEvents();
});

// 서버 상태 확인
async function initHealthCheck() {
    try {
        const res = await fetch("/health");
        const data = await res.json();
        const textElem = document.getElementById("server-status-text");
        const dotElem = document.querySelector(".status-dot");
        if (data.status === "ok") {
            textElem.textContent = `서버 연결됨 (모델: ${data.gemini_model || "Gemini"})`;
            dotElem.className = "status-dot online";
        } else {
            textElem.textContent = "서버 상태 불안정";
            dotElem.className = "status-dot offline";
        }
    } catch (e) {
        console.error("[ERROR]   서버 상태 확인 실패:", e);
        document.getElementById("server-status-text").textContent = "서버 연결 불가";
        document.querySelector(".status-dot").className = "status-dot offline";
    }
}

// 수동 도구 셀렉트 옵션 채우기
function initManualToolOptions() {
    const sel = document.getElementById("manual-tool-select");
    sel.innerHTML = TOOL_OPTIONS.map(t => `<option value="${t.id}">${t.name}</option>`).join("");
}

// ── 2. 파일 드래그 앤 드롭 & 업로드 ────────────────────────
function initDropZone() {
    const dropZone = document.getElementById("drop-zone");
    const fileInput = document.getElementById("file-input");

    dropZone.addEventListener("click", () => fileInput.click());

    ["dragenter", "dragover"].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.add("dragover");
        });
    });

    ["dragleave", "drop"].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.remove("dragover");
        });
    });

    dropZone.addEventListener("drop", (e) => {
        const files = e.dataTransfer.files;
        if (files && files.length > 0) {
            handleFileSelection(files[0]);
        }
    });

    fileInput.addEventListener("change", (e) => {
        if (e.target.files && e.target.files.length > 0) {
            handleFileSelection(e.target.files[0]);
        }
    });
}

function handleFileSelection(file) {
    if (!file.name.toLowerCase().endsWith(".csv")) {
        alert("CSV 파일만 업로드할 수 있습니다.");
        console.error("[ERROR]   비 CSV 파일 선택됨:", file.name);
        return;
    }
    state.selectedFile = file;
    console.log(`[UPLOAD]  파일 선택 (파일명: ${file.name}, 크기: ${file.size} bytes)`);

    document.getElementById("selected-file-name").textContent = file.name;
    document.getElementById("selected-file-size").textContent = formatFileSize(file.size);
    document.getElementById("upload-actions").style.display = "flex";
}

function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + " B";
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
    return (bytes / (1024 * 1024)).toFixed(1) + " MB";
}

// ── 3. 이벤트 바인딩 ────────────────────────────────────────
function bindEvents() {
    // 진단 버튼
    document.getElementById("btn-inspect").addEventListener("click", runInspect);
    // 미리보기 토글 버튼
    document.getElementById("btn-toggle-preview").addEventListener("click", togglePreview);
    // AI 제안 요청 버튼
    document.getElementById("btn-suggest").addEventListener("click", runSuggest);
    // 수동 도구 실행 버튼
    document.getElementById("btn-manual-run").addEventListener("click", runManualTool);
    // AI 해석 생성 버튼
    document.getElementById("btn-explain").addEventListener("click", runExplain);
    // Q&A 폼 제출
    const qaForm = document.getElementById("qa-form");
    if (qaForm) {
        qaForm.addEventListener("submit", handleQaSubmit);
    }
    // Q&A 추천 질문 칩 클릭
    document.querySelectorAll(".qa-chip").forEach(chip => {
        chip.addEventListener("click", () => {
            const q = chip.getAttribute("data-question");
            const qaInput = document.getElementById("qa-input");
            if (qaInput) {
                qaInput.value = q;
                sendQaQuestion(q);
            }
        });
    });
}

// ── 4. 진단(/inspect) 실행 ─────────────────────────────────
async function runInspect() {
    if (!state.selectedFile) {
        alert("먼저 CSV 파일을 선택해주세요.");
        return;
    }

    const spinner = document.getElementById("inspect-loading");
    spinner.style.display = "flex";
    console.log("[INSPECT] 진단 요청 시작");

    const formData = new FormData();
    formData.append("file", state.selectedFile);

    try {
        const res = await fetch("/inspect", {
            method: "POST",
            body: formData
        });
        const data = await res.json();
        spinner.style.display = "none";

        if (!res.ok) {
            throw new Error(data.error || "진단에 실패했습니다.");
        }

        state.diagnosis = data;
        console.log(`[INSPECT] 응답 수신 (행 수: ${data.rows}, 열 수: ${data.columns}, 경고 수: ${data.warnings.length})`);

        renderDiagnosis(data);
    } catch (err) {
        spinner.style.display = "none";
        console.error(`[ERROR]   진단 오류: ${err.message}`);
        alert(`진단 오류: ${err.message}`);
    }
}

function renderDiagnosis(data) {
    document.getElementById("badge-rows-cols").textContent = `${data.rows.toLocaleString()}행 / ${data.columns}열`;
    document.getElementById("diag-encoding").textContent = data.encoding.toUpperCase();
    document.getElementById("diag-skipped-header").textContent = `${data.skipped_header_rows}개 행`;
    document.getElementById("diag-rec-year").textContent = data.recommended_reference_year ? `${data.recommended_reference_year}년` : "미감지";
    document.getElementById("diag-aggregates").textContent = `${data.aggregate_rows_count}개 감지`;

    // 경고 목록 렌더링
    const warnList = document.getElementById("warnings-list");
    warnList.innerHTML = data.warnings.map(w => `<li>${escapeHtml(w)}</li>`).join("");

    // 수동 기준연도 인풋 기본값 채우기
    if (data.recommended_reference_year) {
        document.getElementById("manual-ref-period").value = data.recommended_reference_year;
    }

    // 미리보기 테이블 렌더링
    renderPreviewTable(data.preview);

    // 진단 섹션 표시 및 스크롤
    const diagSection = document.getElementById("diagnosis-section");
    diagSection.style.display = "block";
    diagSection.scrollIntoView({ behavior: "smooth" });
}

function renderPreviewTable(records) {
    if (!records || records.length === 0) return;
    const thead = document.getElementById("preview-thead");
    const tbody = document.getElementById("preview-tbody");

    const headers = Object.keys(records[0]);
    thead.innerHTML = `<tr>${headers.map(h => `<th>${escapeHtml(h)}</th>`).join("")}</tr>`;
    tbody.innerHTML = records.map(r => `
        <tr>${headers.map(h => `<td>${escapeHtml(String(r[h] !== null && r[h] !== undefined ? r[h] : ""))}</td>`).join("")}</tr>
    `).join("");
}

function togglePreview() {
    const wrapper = document.getElementById("preview-table-wrapper");
    const btn = document.getElementById("btn-toggle-preview");
    if (wrapper.style.display === "none") {
        wrapper.style.display = "block";
        btn.textContent = "미리보기 (상위 10행) 접기 ▲";
    } else {
        wrapper.style.display = "none";
        btn.textContent = "미리보기 (상위 10행) 펼치기 ▼";
    }
}

// ── 5. AI 제안(/suggest) 실행 ──────────────────────────────
async function runSuggest() {
    const spinner = document.getElementById("suggest-loading");
    const suggestSection = document.getElementById("suggest-section");
    suggestSection.style.display = "block";
    spinner.style.display = "flex";
    suggestSection.scrollIntoView({ behavior: "smooth" });

    console.log("[SUGGEST] 제안 요청 시작");

    try {
        const res = await fetch("/suggest", { method: "POST" });
        const data = await res.json();
        spinner.style.display = "none";

        if (!res.ok) {
            throw new Error(data.error || "분석 제안을 가져오지 못했습니다.");
        }

        const suggestions = data.suggestions || [];
        console.log(`[SUGGEST] 응답 수신 (제안 개수: ${suggestions.length})`);
        renderSuggestions(suggestions);
    } catch (err) {
        spinner.style.display = "none";
        console.error(`[ERROR]   제안 요청 오류: ${err.message}`);
        alert(`제안 요청 오류: ${err.message}`);
    }
}

function renderSuggestions(suggestions) {
    const container = document.getElementById("suggestions-container");
    container.innerHTML = "";

    if (suggestions.length === 0) {
        container.innerHTML = "<p>제안 가능한 분석이 없습니다.</p>";
        return;
    }

    suggestions.forEach(s => {
        const card = document.createElement("div");
        card.className = "suggestion-card";

        const refPeriod = s.reference_period || s.params?.reference_period || "";

        card.innerHTML = `
            <div>
                <span class="sugg-tool-tag">${escapeHtml(s.tool)}</span>
                <h3 class="sugg-title">${escapeHtml(s.tool_name || s.tool)}</h3>
                <p class="sugg-why"><strong>💡 제안 이유:</strong> ${escapeHtml(s.why)}</p>
                <div class="sugg-caution">⚠️ <strong>주의:</strong> ${escapeHtml(s.caution)}</div>
            </div>
            <button class="btn btn-primary" style="margin-top: 12px;">이 분석 실행하기 ▶</button>
        `;

        card.querySelector("button").addEventListener("click", () => {
            executeTool(s.tool, s.params || {}, refPeriod);
        });

        container.appendChild(card);
    });
}

// 수동 도구 실행
function runManualTool() {
    const tool = document.getElementById("manual-tool-select").value;
    const refPeriod = document.getElementById("manual-ref-period").value.trim();
    const params = {};
    if (refPeriod) {
        params.reference_period = refPeriod;
    }
    executeTool(tool, params, refPeriod);
}

// ── 6. 도구 실행(/run) 및 차트 렌더링 ──────────────────────
async function executeTool(toolName, params, refPeriod) {
    console.log(`[RUN]     도구 실행 시작 (도구명: ${toolName}, 기준시점: ${refPeriod || "없음"})`);

    try {
        const res = await fetch("/run", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ tool: toolName, params: params, reference_period: refPeriod })
        });
        const data = await res.json();

        if (!res.ok) {
            throw new Error(data.error || "도구 실행에 실패했습니다.");
        }

        state.lastRunResult = data;
        console.log(`[RUN]     결과 수신 (사용한 행: ${data.used_rows}, 제외: ${data.excluded_rows})`);

        renderResult(data);
    } catch (err) {
        console.error(`[ERROR]   도구 실행 오류: ${err.message}`);
        alert(`도구 실행 오류: ${err.message}`);
    }
}

function renderResult(data) {
    document.getElementById("res-tool-name").textContent = data.tool;
    document.getElementById("res-ref-period").textContent = data.reference_period || "전체";
    document.getElementById("res-used-rows").textContent = `${data.used_rows.toLocaleString()}개`;
    document.getElementById("res-excluded-rows").textContent = `${data.excluded_rows.toLocaleString()}개`;
    document.getElementById("res-exclusion-reason").querySelector("span").textContent = data.excluded_reasons || "제외 없음";
    document.getElementById("badge-used-rows").textContent = `${data.used_rows}개 행 계산 완료`;

    // 1. 차트 렌더링
    renderChart(data.chart_data);

    // 2. 수치 테이블 렌더링
    renderResultTable(data.result);

    // 3. 결과 섹션 표시
    const resSection = document.getElementById("result-section");
    resSection.style.display = "block";
    resSection.scrollIntoView({ behavior: "smooth" });

    // 4. 결과에 대한 Q&A 섹션 표시
    const qaSection = document.getElementById("qa-section");
    if (qaSection) {
        qaSection.style.display = "block";
    }
}

// Chart.js 인터랙티브 차트 렌더링
function renderChart(chartData) {
    const canvas = document.getElementById("insight-chart");
    const ctx = canvas.getContext("2d");

    if (state.chartInstance) {
        state.chartInstance.destroy();
        state.chartInstance = null;
    }

    if (!chartData || !chartData.datasets || chartData.datasets.length === 0) {
        document.getElementById("chart-wrapper").style.display = "none";
        return;
    }

    document.getElementById("chart-wrapper").style.display = "block";
    const chartType = chartData.type || "bar";

    const config = {
        type: chartType,
        data: {
            labels: chartData.labels || [],
            datasets: chartData.datasets.map(ds => ({
                label: ds.label || "값",
                data: ds.data || [],
                backgroundColor: ds.backgroundColor || (chartType === "line" ? "rgba(37,99,235,0.1)" : "rgba(37,99,235,0.85)"),
                borderColor: ds.borderColor || "#2563eb",
                borderWidth: 1.5,
                tension: ds.tension !== undefined ? ds.tension : 0.2,
                fill: ds.fill !== undefined ? ds.fill : false
            }))
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            indexAxis: chartData.indexAxis || "x",
            plugins: {
                legend: {
                    display: true,
                    position: "top"
                },
                tooltip: {
                    padding: 10,
                    boxPadding: 4
                }
            },
            scales: chartType === "pie" ? {} : {
                x: {
                    grid: { color: "#f1f5f9" }
                },
                y: {
                    grid: { color: "#f1f5f9" },
                    beginAtZero: true
                }
            }
        }
    };

    state.chartInstance = new Chart(ctx, config);
    console.log(`[CHART]   차트 렌더 완료 (차트 종류: ${chartType})`);
}

function renderResultTable(resObj) {
    const thead = document.getElementById("result-thead");
    const tbody = document.getElementById("result-tbody");

    if (!resObj) {
        thead.innerHTML = "";
        tbody.innerHTML = "<tr><td>결과 데이터 없음</td></tr>";
        return;
    }

    // 배열 형태 결과 (예: top_n, distribution_table, summary_table)
    let records = null;
    if (resObj.top_n) records = resObj.top_n;
    else if (resObj.distribution_table) records = resObj.distribution_table;
    else if (resObj.summary_table) records = resObj.summary_table;
    else if (resObj.top_increases) records = resObj.top_increases;
    else if (resObj.coverage_table) records = resObj.coverage_table;
    else if (resObj.preview) records = resObj.preview;

    if (records && Array.isArray(records) && records.length > 0) {
        const headers = Object.keys(records[0]);
        thead.innerHTML = `<tr>${headers.map(h => `<th>${escapeHtml(h)}</th>`).join("")}</tr>`;
        tbody.innerHTML = records.map(r => `
            <tr>${headers.map(h => `<td>${escapeHtml(String(r[h] !== null && r[h] !== undefined ? r[h] : ""))}</td>`).join("")}</tr>
        `).join("");
    } else {
        // 단일 객체 (키-값)
        thead.innerHTML = `<tr><th>지표 항목</th><th>계산 값</th></tr>`;
        tbody.innerHTML = Object.entries(resObj)
            .filter(([_, v]) => typeof v !== "object" || v === null)
            .map(([k, v]) => `<tr><td><strong>${escapeHtml(k)}</strong></td><td>${escapeHtml(String(v !== null ? v : "-"))}</td></tr>`)
            .join("");
    }
}

// ── 7. AI 해석(/explain) 실행 ──────────────────────────────
async function runExplain() {
    const selectedMode = document.querySelector('input[name="explain-mode"]:checked').value;
    console.log(`[EXPLAIN] 해석 요청 시작 (모드: ${selectedMode})`);

    const spinner = document.getElementById("explain-loading");
    const explainSection = document.getElementById("explain-section");
    const reportBox = document.getElementById("report-content");

    explainSection.style.display = "block";
    spinner.style.display = "flex";
    reportBox.innerHTML = "";
    explainSection.scrollIntoView({ behavior: "smooth" });

    try {
        const res = await fetch("/explain", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ mode: selectedMode })
        });
        const data = await res.json();
        spinner.style.display = "none";

        if (!res.ok) {
            throw new Error(data.error || "해석 보고서 생성에 실패했습니다.");
        }

        console.log(`[EXPLAIN] 응답 수신 (모드: ${data.mode})`);
        document.getElementById("badge-explain-mode").textContent = data.mode_name;
        document.getElementById("badge-explain-mode").className = `badge ${data.mode === "B" ? "badge-success" : "badge-calc"}`;

        renderMarkdown(data.explanation, reportBox);
    } catch (err) {
        spinner.style.display = "none";
        console.error(`[ERROR]   해석 생성 오류: ${err.message}`);
        alert(`해석 생성 오류: ${err.message}`);
    }
}

// 순수 JS 기반 마크다운 뷰어 (7대 섹션 렌더링)
function renderMarkdown(mdText, targetElem) {
    if (!mdText) {
        targetElem.innerHTML = "<p>보고서 내용이 비어있습니다.</p>";
        return;
    }

    let html = mdText
        // 줄바꿈 통일
        .replace(/\r\n/g, "\n")
        // 헤더 2
        .replace(/^## (.*$)/gim, '<h2>$1</h2>')
        // 헤더 3
        .replace(/^### (.*$)/gim, '<h3>$1</h3>')
        // 볼드
        .replace(/\*\*(.*?)\*\*/gim, '<strong>$1</strong>')
        // 목록
        .replace(/^\- (.*$)/gim, '<li>$1</li>')
        // 인용구
        .replace(/^> (.*$)/gim, '<blockquote>$1</blockquote>');

    // li를 ul로 감싸기
    html = html.replace(/(<li>.*<\/li>)/gims, '<ul>$1</ul>');
    // 중복 ul 정리
    html = html.replace(/<\/ul>\s*<ul>/gim, '');

    // 일반 문단 처리
    const paragraphs = html.split(/\n{2,}/);
    html = paragraphs.map(p => {
        p = p.trim();
        if (p.startsWith('<h') || p.startsWith('<ul') || p.startsWith('<block')) {
            return p;
        }
        return `<p>${p.replace(/\n/g, '<br>')}</p>`;
    }).join("\n");

    targetElem.innerHTML = html;
}

// ── 유틸리티: HTML 이스케이프 ──────────────────────────────
function escapeHtml(str) {
    if (typeof str !== "string") return str;
    return str
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

// ── 7. 데이터 결과 추가 질문 Q&A 핸들러 ──────────────────────
function handleQaSubmit(e) {
    e.preventDefault();
    const qaInput = document.getElementById("qa-input");
    const question = qaInput.value.trim();
    if (!question) return;
    sendQaQuestion(question);
}

async function sendQaQuestion(question) {
    const qaInput = document.getElementById("qa-input");
    const btnSubmit = document.getElementById("btn-qa-submit");
    const chatBox = document.getElementById("qa-chat-box");
    const loadingElem = document.getElementById("qa-loading");

    // 사용자 메시지 추가
    appendChatMessage("user", question);
    qaInput.value = "";
    qaInput.disabled = true;
    btnSubmit.disabled = true;
    loadingElem.style.display = "flex";
    chatBox.scrollTop = chatBox.scrollHeight;

    console.log(`[QA]      질문 요청 시작: "${question}"`);

    try {
        const res = await fetch("/ask", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ question: question })
        });
        const data = await res.json();
        loadingElem.style.display = "none";
        qaInput.disabled = false;
        btnSubmit.disabled = false;
        qaInput.focus();

        if (!res.ok) {
            throw new Error(data.error || "답변을 가져오지 못했습니다.");
        }

        console.log(`[QA]      답변 수신 완료 (길이: ${data.answer ? data.answer.length : 0})`);
        appendChatMessage("ai", data.answer);
    } catch (err) {
        loadingElem.style.display = "none";
        qaInput.disabled = false;
        btnSubmit.disabled = false;
        console.error(`[ERROR]   QA 오류: ${err.message}`);
        appendChatMessage("ai", `⚠️ 오류가 발생했습니다: ${escapeHtml(err.message)}`);
    }

    chatBox.scrollTop = chatBox.scrollHeight;
}

function appendChatMessage(sender, text) {
    const chatBox = document.getElementById("qa-chat-box");
    const msgDiv = document.createElement("div");
    msgDiv.className = `qa-message ${sender === "user" ? "qa-user" : "qa-ai"}`;

    const avatarDiv = document.createElement("div");
    avatarDiv.className = "qa-avatar";
    avatarDiv.textContent = sender === "user" ? "👤" : "🤖";

    const bubbleDiv = document.createElement("div");
    bubbleDiv.className = "qa-bubble";

    if (sender === "user") {
        const p = document.createElement("p");
        p.textContent = text;
        bubbleDiv.appendChild(p);
    } else {
        renderMarkdown(text, bubbleDiv);
    }

    msgDiv.appendChild(avatarDiv);
    msgDiv.appendChild(bubbleDiv);
    chatBox.appendChild(msgDiv);
    chatBox.scrollTop = chatBox.scrollHeight;
}

