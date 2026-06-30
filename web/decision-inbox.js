let lastPreview = null;

async function api(path) {
  const response = await fetch(path);
  const text = await response.text();
  let data;
  try { data = JSON.parse(text); } catch { data = { raw: text }; }
  if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail || data));
  return data;
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, ch => ({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"}[ch]));
}

function fillExample(kind) {
  const examples = {
    tool: "Ich brauche ein Tool, das historische Aktienkurse analysiert.",
    core: "Pandora sollte Core-Verbesserungen für den Entscheidungsfluss vorschlagen.",
    note: "Was war meine letzte Notiz?"
  };
  document.getElementById("requestInput").value = examples[kind] || examples.tool;
}

async function loadStatus() {
  const data = await api("/api/cognitive/gui-decision-inbox/status");
  document.getElementById("statusBox").innerHTML = `<strong>${escapeHtml(data.mvp)}</strong> · ${escapeHtml(data.role)}<br>${escapeHtml(data.guarantee)}`;
}

async function previewDecision(action = null) {
  const request = document.getElementById("requestInput").value.trim();
  if (!request) {
    document.getElementById("detailBox").textContent = "Bitte zuerst eine Anfrage eingeben.";
    return;
  }
  const actionParam = action ? `&user_action=${encodeURIComponent(action)}` : "";
  const data = await api(`/api/cognitive/gui-decision-inbox/preview?query=${encodeURIComponent(request)}${actionParam}`);
  lastPreview = data;
  renderCards(data.cards || []);
  renderDetail(data.action_result || {}, data.decision || {});
  document.getElementById("traceBox").textContent = JSON.stringify(data, null, 2);
}

function renderCards(cards) {
  const box = document.getElementById("cardList");
  if (!cards.length) {
    box.innerHTML = `<div class="empty">Keine Decision Cards.</div>`;
    return;
  }
  box.innerHTML = cards.map(card => {
    const actions = (card.actions || []).map(action => {
      const cls = action.danger ? "danger" : (action.primary ? "" : "secondary");
      return `<button class="${cls}" onclick="previewDecision('${escapeHtml(action.id)}')">${escapeHtml(action.label)}</button>`;
    }).join(" ");
    return `<article class="decision-card">
      <h3>${escapeHtml(card.title)}</h3>
      <p>${escapeHtml(card.summary)}</p>
      <div class="badges">
        <span class="badge">${escapeHtml(card.decision_type)}</span>
        <span class="badge">${escapeHtml(card.execution_mode)}</span>
        <span class="badge">${card.requires_user_approval ? "Freigabe nötig" : "sicherer Pfad"}</span>
      </div>
      <div class="actions">${actions}</div>
      <p class="notice">${escapeHtml(card.safety_notice)}</p>
    </article>`;
  }).join("");
}

function renderDetail(actionResult, decision) {
  document.getElementById("detailBox").innerHTML = `
    <h3>${escapeHtml(actionResult.state || decision.status || "Status")}</h3>
    <p>${escapeHtml(actionResult.message || decision.summary || "Keine Details.")}</p>
    <div class="row"><span>Decision Type</span><strong>${escapeHtml(decision.decision_type)}</strong></div>
    <div class="row"><span>Next Step</span><strong>${escapeHtml(actionResult.next_step || decision.next_controlled_step)}</strong></div>
    <div class="row"><span>Approval</span><strong>${decision.requires_user_approval ? "ja" : "nein"}</strong></div>
    <h4>Handoff</h4>
    <pre>${escapeHtml(JSON.stringify(actionResult.handoff || {}, null, 2))}</pre>`;
}

loadStatus().catch(err => {
  document.getElementById("statusBox").textContent = `Fehler: ${err.message}`;
});
