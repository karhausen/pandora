let selectedItemId = null;

async function api(path, options = {}) {
  const response = await fetch(path, options);
  const text = await response.text();
  let data;
  try { data = JSON.parse(text); } catch { data = { raw: text }; }
  if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail || data));
  return data;
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, ch => ({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"}[ch]));
}

function metric(label, value) {
  return `<div class="metric"><strong>${escapeHtml(value)}</strong><span>${escapeHtml(label)}</span></div>`;
}

async function loadDashboard() {
  const data = await api("/api/gui/approval/dashboard?limit=200");
  document.getElementById("dashboardCards").innerHTML = [
    metric("offene Vorschläge", data.item_count ?? 0),
    metric("hohes Risiko", data.high_risk_count ?? 0),
    metric("Menschliche Freigabe", data.human_approval_required ? "ja" : "nein"),
    metric("Auto-Ausführung", data.observe_only ? "nein" : "prüfen")
  ].join("");
}

function renderItem(item) {
  const risk = item.risk_badge || { color: "green", label: item.risk || "unknown" };
  const selected = item.id === selectedItemId ? " selected" : "";
  return `<article class="item${selected}" onclick="selectItem('${escapeHtml(item.id)}')">
    <h3>${escapeHtml(item.title || item.id)}</h3>
    <p>${escapeHtml(item.summary || "Keine Zusammenfassung.")}</p>
    <div class="badges">
      <span class="badge">${escapeHtml(item.category || "unknown")}</span>
      <span class="badge ${escapeHtml(risk.color)}">${escapeHtml(risk.label)}</span>
      <span class="badge">${escapeHtml(item.status || "pending")}</span>
    </div>
  </article>`;
}

async function loadInbox() {
  const include = document.getElementById("includeReviewed").checked;
  const data = await api(`/api/gui/approval/inbox?limit=200&include_reviewed=${include}`);
  const items = data.items || [];
  document.getElementById("inboxList").innerHTML = items.length ? items.map(renderItem).join("") : `<div class="empty">Keine Vorschläge vorhanden.</div>`;
}

function row(key, value) {
  return `<div class="row"><div class="key">${escapeHtml(key)}</div><div>${escapeHtml(value)}</div></div>`;
}

async function selectItem(itemId) {
  selectedItemId = itemId;
  await loadInbox();
  const data = await api(`/api/gui/approval/inbox/${encodeURIComponent(itemId)}`);
  const item = data.item || {};
  const content = data.content || {};
  document.getElementById("detailBox").innerHTML = `
    <h3>${escapeHtml(item.title || item.id)}</h3>
    ${row("ID", item.id)}
    ${row("Kategorie", item.category)}
    ${row("Risiko", item.risk)}
    ${row("Status", item.status)}
    ${row("Erstellt", item.created_at)}
    <h4>Policy</h4>
    <pre>${escapeHtml(JSON.stringify(data.decision_policy || {}, null, 2))}</pre>
    <h4>Inhalt</h4>
    <pre>${escapeHtml(JSON.stringify(content, null, 2))}</pre>
    <p class="notice">${escapeHtml(data.safety_notice || "")}</p>`;
  document.getElementById("decisionBox").classList.remove("hidden");
  document.getElementById("decisionNote").value = "";
}

async function sendDecision(decision) {
  if (!selectedItemId) return;
  const note = document.getElementById("decisionNote").value;
  const result = await api(`/api/gui/approval/inbox/${encodeURIComponent(selectedItemId)}/decision`, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({decision, note, decided_by: "web-gui"})
  });
  document.getElementById("detailBox").innerHTML = `<h3>Entscheidung gespeichert</h3><pre>${escapeHtml(JSON.stringify(result, null, 2))}</pre>`;
  document.getElementById("decisionBox").classList.add("hidden");
  await refreshAll();
}

async function loadAudit() {
  document.getElementById("auditBox").textContent = JSON.stringify(await api("/api/gui/approval/audit?limit=100"), null, 2);
}

async function refreshAll() {
  await loadDashboard();
  await loadInbox();
  await loadAudit();
}

refreshAll().catch(err => {
  document.getElementById("detailBox").textContent = `Fehler: ${err.message}`;
});
