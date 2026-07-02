async function fetchJson(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) throw new Error(`${url}: ${response.status}`);
  return response.json();
}
function render(id, data) {
  document.getElementById(id).textContent = JSON.stringify(data, null, 2);
}
async function loadKnowledgeEvolution() {
  const status = document.getElementById("status");
  try {
    const [s, h, g, p] = await Promise.all([
      fetchJson("/api/knowledge-evolution/status"),
      fetchJson("/api/knowledge-evolution/health"),
      fetchJson("/api/knowledge-evolution/gaps"),
      fetchJson("/api/knowledge-evolution/proposals"),
    ]);
    status.innerHTML = `<strong>Status: ${s.ok ? "OK" : "Prüfen"}</strong> · Dateien: ${s.file_count} · Health: ${s.health_score} · Gaps: ${s.gap_count} · Vorschläge: ${s.proposal_candidate_count}<br><small>${s.policy}</small>`;
    render("health", h);
    render("gaps", g);
    render("proposals", p);
  } catch (error) {
    status.textContent = `Knowledge Evolution konnte nicht geladen werden: ${error}`;
  }
}
loadKnowledgeEvolution();
