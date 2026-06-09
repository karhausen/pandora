async function api(path, options = {}) {
  const response = await fetch(path, options);
  const text = await response.text();
  try {
    const data = JSON.parse(text);
    if (!response.ok) return {ok:false, error:data.detail || data};
    return data;
  } catch {
    return {ok: response.ok, text};
  }
}

function show(id, data) {
  document.getElementById(id).textContent = typeof data === "string" ? data : JSON.stringify(data, null, 2);
}

function summaryCard(label, value, cls="") {
  return `<div class="summary ${cls}"><div class="label">${label}</div><div class="value">${value}</div></div>`;
}

async function loadDashboard() {
  const data = await api("/api/gui/night-mode/dashboard");
  show("dashboardBox", data);
  document.getElementById("statusBadge").textContent = data.observe_only ? "Observe only" : "Unklar";
  const total = data.reports?.total ?? 0;
  const highRisk = data.review?.high_risk_count ?? 0;
  const pending = data.approval?.counts_by_status?.pending ?? 0;
  const allowed = data.maintenance?.next_window_decision?.allowed ? "Ja" : "Nein";
  document.getElementById("summaryCards").innerHTML = [
    summaryCard("Nachtberichte", total, total > 0 ? "ok" : "warn"),
    summaryCard("High Risk", highRisk, highRisk > 0 ? "danger" : "ok"),
    summaryCard("Pending", pending, pending > 0 ? "warn" : "ok"),
    summaryCard("Wartung jetzt erlaubt", allowed, allowed === "Ja" ? "ok" : "warn"),
    summaryCard("Auto Changes", data.auto_changes_made ? "Ja" : "Nein", data.auto_changes_made ? "danger" : "ok")
  ].join("");
  await loadReports();
}

async function previewNightMode() {
  const limit = Number(document.getElementById("previewLimit").value || 200);
  show("previewBox", {running:true, dry_run:true, limit});
  const data = await api("/api/gui/night-mode/maintenance/preview", {
    method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({limit})
  });
  show("previewBox", data);
}

async function loadReports() {
  const data = await api("/api/gui/night-mode/reports?limit=25");
  if (!data.reports || data.reports.length === 0) {
    document.getElementById("reportList").textContent = "Keine Nachtberichte gefunden.";
    return;
  }
  document.getElementById("reportList").innerHTML = data.reports.map(report => `
    <div class="report-item" onclick="showReport('${encodeURIComponent(report.id)}')">
      <div class="report-title">${report.title || report.id}</div>
      <div class="report-meta">${report.area_label || report.area} · ${report.modified_at || ""}</div>
      <span class="pill">${report.status || report.kind}</span>
    </div>
  `).join("");
}

async function showReport(encodedId) {
  const data = await api(`/api/gui/night-mode/reports/${encodedId}`);
  show("reportBox", data);
}

loadDashboard();
