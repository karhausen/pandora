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
  const data = await api("/api/gui/operations/dashboard");
  show("dashboardBox", data);
  const status = data.core_status || "unknown";
  document.getElementById("statusBadge").textContent = `${status} · ${data.version || "unknown"}`;
  const pending = data.approval?.pending_count ?? "?";
  const highRisk = data.review?.high_risk_count ?? "?";
  const locked = data.maintenance?.locked ? "Ja" : "Nein";
  const allowed = data.maintenance?.next_window_decision?.allowed ? "Ja" : "Nein";
  document.getElementById("summaryCards").innerHTML = [
    summaryCard("Core", status, status === "ok" ? "ok" : "warn"),
    summaryCard("Pending Approvals", pending, pending > 0 ? "warn" : "ok"),
    summaryCard("High Risk Items", highRisk, highRisk > 0 ? "danger" : "ok"),
    summaryCard("Maintenance Lock", locked, locked === "Ja" ? "danger" : "ok"),
    summaryCard("Jetzt erlaubt", allowed, allowed === "Ja" ? "ok" : "warn")
  ].join("");
}

async function previewMaintenance() {
  const limit = Number(document.getElementById("previewLimit").value || 200);
  show("previewBox", {running:true, dry_run:true, limit});
  const data = await api("/api/gui/operations/maintenance/preview", {
    method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({limit})
  });
  show("previewBox", data);
  await loadDashboard();
}

async function runMaintenance() {
  const limit = Number(document.getElementById("runLimit").value || 200);
  const force = document.getElementById("forceRun").checked;
  const confirmed = confirm("Maintenance erzeugt Reports/Vorschläge, aber installiert nichts. Fortfahren?");
  if (!confirmed) return;
  show("runBox", {running:true, force, limit});
  const data = await api("/api/gui/operations/maintenance/run", {
    method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({limit, force})
  });
  show("runBox", data);
  await loadDashboard();
}

loadDashboard();
