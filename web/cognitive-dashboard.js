let lastDashboard = null;

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
    core: "Pandora sollte den Cognitive Decision Flow robuster machen.",
    review: "Prüfe den aktuellen Stand von Pandora und leite sinnvolle nächste Schritte ab."
  };
  document.getElementById("requestInput").value = examples[kind] || examples.review;
}

async function loadDashboard() {
  const request = document.getElementById("requestInput").value.trim();
  const cadence = document.getElementById("cadenceInput").value;
  if (!request) {
    document.getElementById("summaryBox").textContent = "Bitte zuerst eine Anfrage eingeben.";
    return;
  }
  const data = await api(`/api/cognitive/dashboard/preview?query=${encodeURIComponent(request)}&cadence=${encodeURIComponent(cadence)}`);
  lastDashboard = data;
  renderSummary(data);
  renderCards(data.cards || []);
  renderSections(data.sections || {});
  document.getElementById("traceBox").textContent = JSON.stringify(data.trace || data, null, 2);
}

function renderSummary(data) {
  document.getElementById("summaryBox").innerHTML = `<strong>${escapeHtml(data.status)}</strong><br>${escapeHtml(data.summary)}`;
}

function renderCards(cards) {
  const box = document.getElementById("cardGrid");
  if (!cards.length) {
    box.innerHTML = `<div class="empty">Keine Karten.</div>`;
    return;
  }
  box.innerHTML = cards.map(card => `<article class="metric-card ${escapeHtml(card.severity)}">
    <h3>${escapeHtml(card.title)}</h3>
    <div class="value">${escapeHtml(card.value)}</div>
    <p>${escapeHtml(card.summary)}</p>
    <span class="badge">${escapeHtml(card.action)}</span>
    ${card.requires_user_action ? '<span class="badge">Aktion nötig</span>' : '<span class="badge">Info</span>'}
  </article>`).join("");
}

function renderSections(sections) {
  const box = document.getElementById("sectionBox");
  const cards = Object.entries(sections).map(([name, value]) => {
    const title = name.replace(/_/g, " ");
    return `<article class="section-card">
      <h3>${escapeHtml(title)}</h3>
      <pre>${escapeHtml(JSON.stringify(value, null, 2))}</pre>
    </article>`;
  }).join("");
  box.innerHTML = `<div class="section-grid">${cards}</div>`;
}

api("/api/cognitive/dashboard/status").then(data => {
  document.getElementById("summaryBox").innerHTML = `<strong>${escapeHtml(data.mvp)}</strong><br>${escapeHtml(data.guarantee)}`;
}).catch(err => {
  document.getElementById("summaryBox").textContent = `Fehler: ${err.message}`;
});
