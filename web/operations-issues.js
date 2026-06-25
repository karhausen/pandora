async function api(path, options = {}) {
  const response = await fetch(path, options);
  const text = await response.text();
  try {
    const data = JSON.parse(text);
    if (!response.ok) return { ok: false, error: data.detail || data };
    return data;
  } catch {
    return { ok: response.ok, text };
  }
}

function esc(value) {
  return String(value ?? "").replace(/[&<>'"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;","'":"&#39;","\"":"&quot;"}[c]));
}

function show(id, data) {
  document.getElementById(id).textContent = typeof data === "string" ? data : JSON.stringify(data, null, 2);
}

function card(label, value, cls = "") {
  return `<div class="summary ${cls}"><div class="label">${esc(label)}</div><div class="value">${esc(value)}</div></div>`;
}

function issueRow(issue) {
  const p = issue.priority || issue.severity || "medium";
  const cls = p === "critical" || p === "high" ? "danger" : (p === "medium" ? "warn" : "ok");
  return `<div class="issue ${cls}">
    <div><strong>${esc(issue.title)}</strong><div class="detail">${esc(issue.detail)}</div></div>
    <div><span class="badge ${cls}">${esc(p)}</span><br><small>${esc(issue.area)}</small></div>
    <div>${esc(issue.recommended_action)}</div>
  </div>`;
}

function actionRow(action) {
  return `<div class="issue">
    <div><strong>${esc(action.title)}</strong><div class="detail">${esc(action.summary)}</div></div>
    <div><span class="badge">${esc(action.status)}</span><br><small>${esc(action.priority)}</small></div>
    <div>${esc(action.action_to_do || action.recommended_next_step)}</div>
  </div>`;
}

async function loadDashboard() {
  const data = await api("/api/gui/operations-issues/dashboard");
  show("rawBox", data);
  const counts = data.scan?.counts || {};
  document.getElementById("headlineText").textContent = `Issues: ${counts.total ?? 0} · Actions: ${data.actions?.count ?? 0}`;
  document.getElementById("summaryCards").innerHTML = [
    card("Issues", counts.total ?? 0, (counts.total || 0) > 0 ? "warn" : "ok"),
    card("Critical", counts.critical ?? 0, (counts.critical || 0) > 0 ? "danger" : "ok"),
    card("High", counts.high ?? 0, (counts.high || 0) > 0 ? "danger" : "ok"),
    card("Actions", data.actions?.count ?? 0, (data.actions?.count || 0) > 0 ? "warn" : "ok"),
  ].join("");
  document.getElementById("issueList").innerHTML = (data.scan?.issues || []).map(issueRow).join("") || "<p>Keine offenen Issues erkannt.</p>";
  document.getElementById("actionList").innerHTML = (data.actions?.actions || []).map(actionRow).join("") || "<p>Noch keine Issue Actions erzeugt.</p>";
}

async function createActions() {
  if (!confirm("Issue Actions erzeugen? Es werden nur reviewbare JSON-Proposals erstellt.")) return;
  const data = await api("/api/gui/operations-issues/create-actions", { method: "POST" });
  show("rawBox", data);
  await loadDashboard();
}

loadDashboard();
