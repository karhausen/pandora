async function api(path) {
  const response = await fetch(path);
  const text = await response.text();
  try {
    const data = JSON.parse(text);
    if (!response.ok) return { ok: false, error: data.detail || data };
    return data;
  } catch {
    return { ok: response.ok, text };
  }
}

function show(id, data) {
  document.getElementById(id).textContent = typeof data === "string" ? data : JSON.stringify(data, null, 2);
}

function card(label, value, cls = "") {
  return `<div class="summary ${cls}"><div class="label">${label}</div><div class="value">${value}</div></div>`;
}

function esc(value) {
  return String(value ?? "").replace(/[&<>'"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;","'":"&#39;","\"":"&quot;"}[c]));
}

function checkRow(check) {
  return `<div class="check ${esc(check.status)}">
    <div class="area">${esc(check.area)}<br><strong>${esc(check.status)}</strong></div>
    <div><strong>${esc(check.title)}</strong><div class="detail">${esc(check.detail)}</div></div>
    <div><span class="badge ${check.status === "error" ? "danger" : ""}">${esc(check.severity)}</span></div>
  </div>`;
}

function recommendationRow(rec) {
  return `<div class="check ${esc(rec.level)}">
    <div class="area">${esc(rec.area)}<br><strong>${esc(rec.level)}</strong></div>
    <div><strong>${esc(rec.title)}</strong><div class="detail">${esc(rec.next_step)}</div></div>
    <div></div>
  </div>`;
}

async function loadHealth() {
  const data = await api("/api/gui/operations-health/status");
  show("rawBox", data);
  document.getElementById("headlineText").textContent = `Gesamtstatus: ${data.overall || "unknown"} · ${data.generated_at || ""}`;
  const c = data.counts || {};
  document.getElementById("summaryCards").innerHTML = [
    card("Gesamt", c.total ?? "?", "ok"),
    card("OK", c.ok ?? "?", "ok"),
    card("Warnungen", c.warning ?? "?", (c.warning || 0) > 0 ? "warn" : "ok"),
    card("Fehler", c.error ?? "?", (c.error || 0) > 0 ? "danger" : "ok"),
  ].join("");
  document.getElementById("checkList").innerHTML = (data.checks || []).map(checkRow).join("") || "<p>Keine Checks.</p>";
  document.getElementById("recommendations").innerHTML = (data.recommendations || []).map(recommendationRow).join("") || "<p>Keine Empfehlungen. Alles sieht gut aus.</p>";
}

loadHealth();
