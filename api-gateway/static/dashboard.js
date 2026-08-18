let currentCategoryFilter = "ALL";
let cachedModelsData = [];
let cachedMetricsData = [];
let cachedDatasetsData = [];

document.addEventListener("DOMContentLoaded", () => {
    initDashboard();
});

async function initDashboard() {
    const refreshBtn = document.getElementById("refreshBtn");
    if (refreshBtn) {
        refreshBtn.addEventListener("click", fetchAllData);
    }
    
    // Category Filter Tabs Event Listener
    const filterTabs = document.querySelectorAll(".filter-tab");
    filterTabs.forEach(tab => {
        tab.addEventListener("click", (e) => {
            filterTabs.forEach(t => t.classList.remove("active"));
            e.target.classList.add("active");
            currentCategoryFilter = e.target.getAttribute("data-category") || "ALL";
            renderFilteredViews();
        });
    });

    await fetchAllData();
}

async function fetchAllData() {
    setLoadingState(true);
    try {
        await Promise.all([
            fetchSummary(),
            fetchHealth(),
            fetchModels(),
            fetchMetrics(),
            fetchDatasets(),
            fetchActivity(),
            fetchExtensionStatus()
        ]);
        updateLastUpdated();
    } catch (err) {
        console.error("Error fetching dashboard telemetry:", err);
    } finally {
        setLoadingState(false);
    }
}

function setLoadingState(isLoading) {
    const refreshBtn = document.getElementById("refreshBtn");
    const refreshIcon = document.getElementById("refreshIcon");
    if (refreshBtn && refreshIcon) {
        if (isLoading) {
            refreshBtn.disabled = true;
            refreshIcon.classList.add("spinner");
        } else {
            refreshBtn.disabled = false;
            refreshIcon.classList.remove("spinner");
        }
    }
}

function updateLastUpdated() {
    const now = new Date();
    const timeStr = now.toTimeString().split(" ")[0];
    const el = document.getElementById("lastUpdatedTime");
    if (el) el.textContent = timeStr;
}

function renderFilteredViews() {
    renderModelsTable();
    renderMetricsTable();
    renderDatasetsTable();
}

async function fetchSummary() {
    try {
        const res = await fetch("/api/developer/summary");
        if (!res.ok) return;
        const data = await res.json();

        const statusBadge = document.getElementById("overallStatusBadge");
        const statusText = document.getElementById("overallStatusText");
        
        if (statusBadge && statusText) {
            statusBadge.className = `status-badge ${data.overall_status}`;
            if (data.overall_status === "operational") {
                statusText.textContent = "● All Systems Operational";
            } else if (data.overall_status === "degraded") {
                statusText.textContent = "● Partial Degradation";
            } else {
                statusText.textContent = "● System Down";
            }
        }

        const healthyServicesVal = document.getElementById("healthyServicesVal");
        if (healthyServicesVal) healthyServicesVal.textContent = `${data.healthy_services} / ${data.total_services}`;

        const modelsLoadedVal = document.getElementById("modelsLoadedVal");
        if (modelsLoadedVal) modelsLoadedVal.textContent = data.total_models_loaded;

        const datasetRecordsVal = document.getElementById("datasetRecordsVal");
        if (datasetRecordsVal) datasetRecordsVal.textContent = data.total_dataset_records.toLocaleString();
        
    } catch (e) {
        console.error("fetchSummary error:", e);
    }
}

async function fetchHealth() {
    try {
        const res = await fetch("/api/developer/health");
        if (!res.ok) return;
        const data = await res.json();
        
        const container = document.getElementById("servicesHealthGrid");
        if (!container) return;

        container.innerHTML = data.services.map(s => {
            const latencyStr = s.latency_ms !== null ? `${s.latency_ms} ms` : "N/A";
            const statusClass = s.status;
            
            return `
                <div class="service-card">
                    <div class="service-card-top">
                        <div>
                            <div class="service-name">${escapeHtml(s.name)}</div>
                            <div class="service-role">${escapeHtml(s.type)}</div>
                        </div>
                        <span class="status-badge ${statusClass}">
                            <span class="status-dot ${statusClass}"></span>
                            ${capitalize(s.status)}
                        </span>
                    </div>
                    <div class="service-bottom">
                        <span>Port: <span class="port-tag">${s.port}</span></span>
                        <span>Latency: <strong>${latencyStr}</strong></span>
                    </div>
                </div>
            `;
        }).join("");
    } catch (e) {
        console.error("fetchHealth error:", e);
    }
}

async function fetchModels() {
    try {
        const res = await fetch("/api/developer/models");
        if (!res.ok) return;
        const data = await res.json();
        cachedModelsData = data.models || [];
        renderModelsTable();
    } catch (e) {
        console.error("fetchModels error:", e);
    }
}

function renderModelsTable() {
    const tbody = document.getElementById("modelsTableBody");
    if (!tbody) return;

    const filtered = currentCategoryFilter === "ALL" 
        ? cachedModelsData 
        : cachedModelsData.filter(m => m.category.toUpperCase() === currentCategoryFilter.toUpperCase());

    if (filtered.length === 0) {
        tbody.innerHTML = `<tr><td colspan="8" style="text-align: center; color: var(--text-secondary); padding: 20px;">No models configured for this category</td></tr>`;
        return;
    }

    tbody.innerHTML = filtered.map(m => {
        const badgeClass = `badge-${m.category.toLowerCase()}`;
        const statusClass = m.status === "loaded" ? "healthy" : "down";
        
        return `
            <tr>
                <td><span class="badge ${badgeClass}">${escapeHtml(m.category)}</span></td>
                <td><strong>${escapeHtml(m.name)}</strong></td>
                <td>${escapeHtml(m.algorithm)}</td>
                <td>
                    <span class="status-badge ${statusClass}">
                        <span class="status-dot ${statusClass}"></span>
                        ${capitalize(m.status)}
                    </span>
                </td>
                <td>${escapeHtml(m.version)}</td>
                <td class="mono">${m.file_size_mb} MB</td>
                <td class="mono">${escapeHtml(m.features_count.toString())}</td>
                <td>${escapeHtml(m.last_trained)}</td>
            </tr>
        `;
    }).join("");
}

async function fetchMetrics() {
    try {
        const res = await fetch("/api/developer/metrics");
        if (!res.ok) return;
        const data = await res.json();
        cachedMetricsData = data.metrics || [];
        renderMetricsTable();
    } catch (e) {
        console.error("fetchMetrics error:", e);
    }
}

function renderMetricsTable() {
    const tbody = document.getElementById("metricsTableBody");
    if (!tbody) return;

    const filtered = currentCategoryFilter === "ALL" 
        ? cachedMetricsData 
        : cachedMetricsData.filter(m => m.category.toUpperCase() === currentCategoryFilter.toUpperCase());

    if (filtered.length === 0) {
        tbody.innerHTML = `<tr><td colspan="6" style="text-align: center; color: var(--text-secondary); padding: 20px;">No evaluation metrics for this category</td></tr>`;
        return;
    }

    tbody.innerHTML = filtered.map(m => {
        const badgeClass = `badge-${m.category.toLowerCase()}`;
        return `
            <tr>
                <td><span class="badge ${badgeClass}">${escapeHtml(m.category)}</span></td>
                <td><strong>${escapeHtml(m.model)}</strong></td>
                <td class="text-right mono">${escapeHtml(m.mae)}</td>
                <td class="text-right mono">${escapeHtml(m.rmse)}</td>
                <td class="text-right mono"><strong>${escapeHtml(m.r2)}</strong></td>
                <td class="text-right mono">${escapeHtml(m.mape)}</td>
            </tr>
        `;
    }).join("");
}

async function fetchDatasets() {
    try {
        const res = await fetch("/api/developer/datasets");
        if (!res.ok) return;
        const data = await res.json();
        cachedDatasetsData = data.datasets || [];
        renderDatasetsTable();
    } catch (e) {
        console.error("fetchDatasets error:", e);
    }
}

function renderDatasetsTable() {
    const tbody = document.getElementById("datasetsTableBody");
    if (!tbody) return;

    const filtered = currentCategoryFilter === "ALL" 
        ? cachedDatasetsData 
        : cachedDatasetsData.filter(d => d.category.toUpperCase() === currentCategoryFilter.toUpperCase());

    if (filtered.length === 0) {
        tbody.innerHTML = `<tr><td colspan="6" style="text-align: center; color: var(--text-secondary); padding: 20px;">No dataset records for this category</td></tr>`;
        return;
    }

    tbody.innerHTML = filtered.map(d => {
        const badgeClass = `badge-${d.category.toLowerCase()}`;
        return `
            <tr>
                <td><span class="badge ${badgeClass}">${escapeHtml(d.category)}</span></td>
                <td><strong>${escapeHtml(d.name)}</strong></td>
                <td class="text-right mono"><strong>${d.records.toLocaleString()}</strong></td>
                <td class="mono">${escapeHtml(d.size)}</td>
                <td>
                    <span class="status-badge healthy">
                        <span class="status-dot healthy"></span>
                        ${escapeHtml(d.quality)}
                    </span>
                </td>
                <td>${escapeHtml(d.last_updated)}</td>
            </tr>
        `;
    }).join("");
}

async function fetchActivity() {
    try {
        const res = await fetch("/api/developer/activity");
        if (!res.ok) return;
        const data = await res.json();
        
        const container = document.getElementById("activityList");
        if (!container) return;

        container.innerHTML = data.events.map(ev => {
            return `
                <div class="activity-item">
                    <span class="activity-time">${escapeHtml(ev.time)}</span>
                    <span class="activity-tag">${escapeHtml(ev.type)}</span>
                    <span style="flex-grow: 1;">${escapeHtml(ev.description)}</span>
                </div>
            `;
        }).join("");
    } catch (e) {
        console.error("fetchActivity error:", e);
    }
}

async function fetchExtensionStatus() {
    try {
        const res = await fetch("/api/developer/extension-status");
        if (!res.ok) return;
        const data = await res.json();
        
        const extStatusBadge = document.getElementById("extStatusBadge");
        if (extStatusBadge) {
            extStatusBadge.className = "status-badge healthy";
            extStatusBadge.innerHTML = `<span class="status-dot healthy"></span> Verified Active`;
        }

        const extSupportedCats = document.getElementById("extSupportedCats");
        if (extSupportedCats) {
            extSupportedCats.textContent = data.supported_categories.map(c => c.toUpperCase()).join(", ");
        }

        const extGatewayUrl = document.getElementById("extGatewayUrl");
        if (extGatewayUrl) {
            extGatewayUrl.textContent = data.gateway_url;
        }
    } catch (e) {
        console.error("fetchExtensionStatus error:", e);
    }
}

function escapeHtml(str) {
    if (!str) return "";
    return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#039;");
}

function capitalize(str) {
    if (!str) return "";
    return str.charAt(0).toUpperCase() + str.slice(1);
}
