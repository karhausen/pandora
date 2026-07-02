async function fetchJson(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
  return response.json();
}

function item(title, text, className = '') {
  const div = document.createElement('div');
  div.className = `item ${className}`.trim();
  div.innerHTML = `<strong>${title}</strong><span>${text || ''}</span>`;
  return div;
}

async function loadEvolution() {
  const [status, genome, lifecycle, types, rules] = await Promise.all([
    fetchJson('/api/evolution/status'),
    fetchJson('/api/evolution/genome'),
    fetchJson('/api/evolution/lifecycle'),
    fetchJson('/api/evolution/types'),
    fetchJson('/api/evolution/rules'),
  ]);

  document.getElementById('statusPanel').innerHTML =
    `<strong>Generation ${status.genome_generation}</strong> · Genome valid: <strong>${status.genome_valid ? 'OK' : 'Prüfen'}</strong> · ` +
    `Proposal Model: <strong>${status.proposal_model}</strong> · Nächster Schritt: ${status.next_step}`;

  const sections = document.getElementById('genomeSections');
  sections.innerHTML = '';
  (genome.sections || []).forEach(section => sections.appendChild(item(section.title, section.description)));

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
}

loadEvolution().catch(error => {
  document.getElementById('statusPanel').textContent = `Evolution konnte nicht geladen werden: ${error}`;
});
