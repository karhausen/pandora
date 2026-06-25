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

function show(id, data) {
  document.getElementById(id).textContent = typeof data === "string" ? data : JSON.stringify(data, null, 2);
}

function card(label, value, cls = "") {
  return `<div class="summary ${cls}"><div class="label">${label}</div><div class="value">${value}</div></div>`;
}

function link(label, href) {
  return `<a class="badge link" href="${href}">${label}</a>`;
}

async function loadCockpit() {
  const data = await api("/api/gui/operations-cockpit/dashboard");
  show("rawBox", data);
  const h = data.headline || {};
  document.getElementById("headlineText").textContent = `Core ${h.core_status || "unknown"} · Version ${h.pandora_version || "unknown"}`;
  document.getElementById("summaryCards").innerHTML = [
    card("Offene Actions", h.open_actions ?? "?", (h.open_actions || 0) > 0 ? "warn" : "ok"),
    card("Fehlerhafte Actions", h.failed_actions ?? "?", (h.failed_actions || 0) > 0 ? "danger" : "ok"),
    card("Aktive Workflows", h.active_workflows ?? "?", (h.active_workflows || 0) > 0 ? "warn" : "ok"),
    card("Blockierte Workflows", h.blocked_workflows ?? "?", (h.blocked_workflows || 0) > 0 ? "danger" : "ok"),
    card("Scheduler fällig", h.scheduler_due ? "Ja" : "Nein", h.scheduler_due ? "warn" : "ok"),
    card("Night Reports", h.night_reports ?? "?", "ok")
  ].join("");

  const attention = data.attention || [];
  document.getElementById("attentionList").innerHTML = attention.length
    ? attention.map(a => `<div class="attention ${a.level || ""}"><span>${a.title}</span><strong>${a.count}</strong><a class="badge link" href="${a.target}">Öffnen</a></div>`).join("")
    : `<div class="attention"><span>Keine dringenden Punkte.</span><strong>OK</strong></div>`;

  document.getElementById("quickLinks").innerHTML = (data.quick_links || []).map(q => link(q.label, q.href)).join("");
}

async function nightPreview() {
  const limit = Number(document.getElementById("nightLimit").value || 200);
  show("nightBox", { running: true, limit, write: false, create_actions: false });
  const data = await api("/api/gui/operations-cockpit/night-review-preview", {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ limit })
  });
  show("nightBox", data);
  await loadCockpit();
}

async function schedulerRun() {
  const limit = Number(document.getElementById("schedulerLimit").value || 200);
  if (!confirm("Manuellen Scheduler-Lauf starten? Es werden nur reviewbare Actions erzeugt.")) return;
  show("schedulerBox", { running: true, limit });
  const data = await api("/api/gui/operations-cockpit/scheduler-run", {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ limit })
  });
  show("schedulerBox", data);
  await loadCockpit();
}

loadCockpit();
