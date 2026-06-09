let selectedSkillId = null;
let selectedSkillPayload = null;

function setStatus(message) {
  document.getElementById('statusText').textContent = message;
}

function escapeHtml(value) {
  return String(value ?? '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#039;');
}

async function loadSkillCenter() {
  setStatus('Lade Skills ...');
  const filter = document.getElementById('statusFilter').value;
  const url = filter ? `/api/gui/skills?status=${encodeURIComponent(filter)}` : '/api/gui/skills';
  const [dashboardRes, listRes, candidatesRes, activationRes] = await Promise.all([
    fetch('/api/gui/skills/dashboard'),
    fetch(url),
    fetch('/api/gui/skills/candidates'),
    fetch('/api/gui/skills/activation-log'),
  ]);
  const dashboard = await dashboardRes.json();
  const list = await listRes.json();
  const candidates = await candidatesRes.json();
  const activations = await activationRes.json();
  renderSummary(dashboard);
  renderSkillList(list.skills || []);
  renderCandidates(candidates.proposals || []);
  renderActivationLog(activations.activations || []);
  setStatus(`${list.count || 0} Skill(s) geladen`);
}

function renderSummary(data) {
  const counts = data.status_counts || {};
  document.getElementById('skillCount').textContent = data.skill_count ?? 0;
  document.getElementById('activeCount').textContent = counts.ACTIVE ?? 0;
  document.getElementById('disabledCount').textContent = counts.DISABLED ?? 0;
  document.getElementById('candidateCount').textContent = data.proposal_count ?? 0;
}

function renderSkillList(skills) {
  const list = document.getElementById('skillList');
  if (!skills.length) {
    list.innerHTML = '<div class="skill-item"><h3>Keine Skills</h3><p>Für den Filter wurden keine Skills gefunden.</p></div>';
    return;
  }
  list.innerHTML = skills.map(skill => `
    <button class="skill-item ${skill.id === selectedSkillId ? 'selected' : ''}" onclick="showSkill('${escapeHtml(skill.id)}')">
      <h3>${escapeHtml(skill.name)} <span class="badge">${escapeHtml(skill.status)}</span></h3>
      <p>${escapeHtml(skill.description)}</p>
      <span class="badge">${escapeHtml(skill.security_level)}</span>
      <span class="badge">${skill.step_count ?? 0} Schritte</span>
    </button>
  `).join('');
}

async function showSkill(skillId) {
  selectedSkillId = skillId;
  setStatus(`Lade ${skillId} ...`);
  const res = await fetch(`/api/gui/skills/${encodeURIComponent(skillId)}`);
  if (!res.ok) {
    setStatus('Skill nicht gefunden');
    return;
  }
  selectedSkillPayload = await res.json();
  renderSkillDetail(selectedSkillPayload);
  await loadSkillCenter();
}

function renderSkillDetail(payload) {
  const skill = payload.skill || {};
  const tools = skill.required_tools || [];
  const detail = document.getElementById('skillDetail');
  detail.innerHTML = `
    <h2>${escapeHtml(skill.name || skill.id)}</h2>
    <p>${escapeHtml(skill.description || '')}</p>
    <div class="badge-row">
      <span class="badge primary">${escapeHtml(skill.status)}</span>
      <span class="badge">${escapeHtml(skill.security_level)}</span>
      <span class="badge">v${escapeHtml(skill.version)}</span>
    </div>
    <div class="meta-grid">
      <div class="meta-box"><span>ID</span>${escapeHtml(skill.id)}</div>
      <div class="meta-box"><span>Benötigte Tools</span>${tools.map(escapeHtml).join(', ') || '—'}</div>
      <div class="meta-box"><span>Schritte</span>${(skill.steps || []).length}</div>
      <div class="meta-box"><span>Input</span>${escapeHtml(JSON.stringify(skill.input_schema || {}))}</div>
    </div>
  `;
  document.getElementById('rawSkill').textContent = JSON.stringify(payload, null, 2);
  setStatus(`${skill.id} geladen`);
}

async function skillAction(action) {
  if (!selectedSkillId) {
    setStatus('Bitte zuerst einen Skill auswählen.');
    return;
  }
  setStatus(`${action} wird ausgeführt ...`);
  const res = await fetch(`/api/gui/skills/${encodeURIComponent(selectedSkillId)}/action`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ action }),
  });
  const payload = await res.json();
  if (!res.ok || payload.success === false) {
    setStatus(`Aktion fehlgeschlagen: ${payload.detail || payload.error || 'Unbekannter Fehler'}`);
    document.getElementById('rawSkill').textContent = JSON.stringify(payload, null, 2);
    return;
  }
  await showSkill(selectedSkillId);
  setStatus(payload.message || 'Aktion abgeschlossen');
}

function renderCandidates(proposals) {
  const container = document.getElementById('candidateList');
  if (!proposals.length) {
    container.innerHTML = '<div class="compact-item"><strong>Keine Kandidaten</strong><small>Der Maintenance-Lauf hat noch keine Skill-Kandidaten erzeugt.</small></div>';
    return;
  }
  container.innerHTML = proposals.slice(0, 8).map(item => {
    const skill = item.skill || {};
    return `<div class="compact-item"><strong>${escapeHtml(skill.name || item.id)}</strong><small>${escapeHtml(item.status || 'UNKNOWN')} · ${escapeHtml(item.id)}</small></div>`;
  }).join('');
}

function renderActivationLog(activations) {
  const container = document.getElementById('activationLog');
  if (!activations.length) {
    container.innerHTML = '<div class="compact-item"><strong>Keine Aktivierungen</strong><small>Noch keine Skill-Aktivierungen protokolliert.</small></div>';
    return;
  }
  container.innerHTML = activations.slice(0, 8).map(item => `
    <div class="compact-item"><strong>${escapeHtml(item.skill_id || item.proposal_id || 'Skill')}</strong><small>${item.activated ? 'aktiviert' : 'fehlgeschlagen'} · ${escapeHtml(item.created_at || '')}</small></div>
  `).join('');
}

window.addEventListener('DOMContentLoaded', loadSkillCenter);
