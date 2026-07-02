async function fetchJson(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
  return response.json();
}

function item(title, text, className = '') {
  const div = document.createElement('div');
  div.className = `item ${className}`.trim();
  div.innerHTML = `<strong>${title}</strong><span>${text || ''}</span>`;
  return div;
}

function renderProposalPreview(result) {
  const proposal = result.proposal || {};
  const route = result.route || {};
  return JSON.stringify({
    id: proposal.id,
    type: proposal.type,
    title: proposal.title,
    priority: proposal.priority,
    confidence: proposal.confidence,
    impact: proposal.impact,
    risk: proposal.risk,
    status: proposal.status,
    route: route.label,
    safety: proposal.payload?.safety_contract,
  }, null, 2);
}

async function runFactoryPreview() {
  const request = document.getElementById('factoryRequest').value || '';
  const type = document.getElementById('factoryType').value || '';
  const params = new URLSearchParams({ request });
  if (type) params.set('type', type);
  const result = await fetchJson(`/api/evolution/factory/preview?${params.toString()}`);
  document.getElementById('factoryPreview').textContent = renderProposalPreview(result);
}

async function loadEvolution() {
  const [status, factoryStatus, routes, lifecycle, types, rules] = await Promise.all([
    fetchJson('/api/evolution/status'),
    fetchJson('/api/evolution/factory/status'),
    fetchJson('/api/evolution/factory/routes'),
    fetchJson('/api/evolution/lifecycle'),
    fetchJson('/api/evolution/types'),
    fetchJson('/api/evolution/rules'),
  ]);

  document.getElementById('statusPanel').innerHTML =
    `<strong>Generation ${status.genome_generation}</strong> · Genome valid: <strong>${status.genome_valid ? 'OK' : 'Prüfen'}</strong> · ` +
    `Factory: <strong>${factoryStatus.mode}</strong> · Routen: <strong>${factoryStatus.route_count}</strong> · ` +
    `Aktivierung: <strong>${factoryStatus.activates_changes ? 'aktiv' : 'nur Vorschläge'}</strong>`;

  const typeSelect = document.getElementById('factoryType');
  (types.types || []).forEach(type => {
    const option = document.createElement('option');
    option.value = type;
    option.textContent = type;
    typeSelect.appendChild(option);
  });

  const routesList = document.getElementById('factoryRoutes');
  routesList.innerHTML = '';
  (routes.routes || []).forEach(route => {
    routesList.appendChild(item(route.label, `${route.type} · Zielbereich: ${route.target_area} · Risiko: ${route.default_risk}`));
  });

  const timeline = document.getElementById('lifecycleSteps');
  timeline.innerHTML = '';
  (lifecycle.steps || []).forEach(step => timeline.appendChild(item(`${step.order}. ${step.title}`, step.purpose)));

  const chips = document.getElementById('proposalTypes');
  chips.innerHTML = '';
  (types.types || []).forEach(type => {
    const span = document.createElement('span');
    span.className = 'chip';
    span.textContent = type;
    chips.appendChild(span);
  });

  const ruleList = document.getElementById('rules');
  ruleList.innerHTML = '';
  (rules.rules || []).forEach(rule => ruleList.appendChild(item(rule.id, rule.title, rule.hard ? 'rule-hard' : '')));

  document.getElementById('factoryButton').addEventListener('click', () => {
    runFactoryPreview().catch(error => {
      document.getElementById('factoryPreview').textContent = `Factory Preview fehlgeschlagen: ${error}`;
    });
  });
}

loadEvolution().catch(error => {
  document.getElementById('statusPanel').textContent = `Evolution konnte nicht geladen werden: ${error}`;
});
