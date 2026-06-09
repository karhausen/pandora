async function api(path, options){
  const res = await fetch(path, options);
  if(!res.ok){ throw new Error(await res.text()); }
  return res.json();
}
function esc(v){ return String(v ?? '').replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c])); }
function readyText(p){ return p && p.ready ? 'bereit' : 'unvollständig'; }
function readyClass(p){ return p && p.ready ? 'status-ok' : 'status-bad'; }
async function loadAll(){
  const dash = await api('/api/gui/llm-profiles/dashboard');
  document.getElementById('activeProfile').textContent = dash.active_profile || 'nicht gesetzt';
  document.getElementById('profileHint').textContent = (dash.profile_purpose && dash.profile_purpose[dash.active_profile]) || 'Profil prüfen.';
  const cloud = dash.cloud_expert_provider || {};
  document.getElementById('cloudProvider').textContent = cloud.resolved_provider || '–';
  document.getElementById('cloudReady').innerHTML = `<span class="${readyClass(cloud)}">${readyText(cloud)}</span> · ${esc(cloud.model || 'kein Modell')}`;
  document.getElementById('securityStatus').textContent = dash.security && dash.security.ok ? 'OK' : 'Prüfen';
  document.getElementById('securityIssues').textContent = dash.security && dash.security.issues && dash.security.issues.length ? dash.security.issues.join(', ') : 'Keine Inline-Secrets gemeldet.';
  renderProfiles(await api('/api/gui/llm-profiles/profiles'));
  renderProviders(await api('/api/gui/llm-profiles/providers'));
  await loadRoutingEditor();
  document.getElementById('guardrails').innerHTML = (dash.guardrails || []).map(x=>`<li>${esc(x)}</li>`).join('');
}
function renderProfiles(data){
  document.getElementById('profiles').innerHTML = (data.profiles || []).map(p => `
    <div class="item">
      <div class="item-head">
        <div><h3>${esc(p.name)} ${p.active ? '<span class="badge primary">aktiv</span>' : ''}</h3><div class="meta">${esc(p.description)}</div></div>
        <button class="badge link" ${p.active ? 'disabled' : ''} onclick="setProfile('${esc(p.name)}')">Aktivieren</button>
      </div>
      <p>Cloud Expert: <strong>${esc(p.cloud_expert || 'nicht gesetzt')}</strong> · <span class="${readyClass(p.cloud_provider_status)}">${readyText(p.cloud_provider_status)}</span></p>
    </div>`).join('');
}
function renderProviders(data){
  document.getElementById('providers').innerHTML = (data.providers || []).map(p => `
    <div class="item">
      <div class="item-head"><h3>${esc(p.resolved_provider || p.requested_provider)}</h3><span class="${readyClass(p)}">${readyText(p)}</span></div>
      <div class="meta">Typ: ${esc(p.type)} · Modell: ${esc(p.model)} · Base URL: ${esc(p.base_url || '<nicht sichtbar/gesetzt>')}</div>
      <div class="meta">API-Key ENV: ${esc(p.api_key_env || '–')} · vorhanden: ${p.api_key_present ? 'ja' : 'nein'}</div>
    </div>`).join('');
}
let routingEditorState = null;

async function loadRoutingEditor(){
  const [status, data] = await Promise.all([
    api('/api/gui/llm-profiles/routing-editor/status'),
    api('/api/gui/llm-profiles/routing-editor/routes')
  ]);
  routingEditorState = data;
  document.getElementById('routingStatus').textContent = `${data.routes.length} Routing-Regeln · ${status.local_override_exists ? 'Local Override aktiv' : 'kein Local Override'} · Profil ${data.active_profile || 'unbekannt'}`;
  renderRoutes(data);
}

function renderRoutes(data){
  const providers = data.providers || [];
  document.getElementById('routes').innerHTML = (data.routes || []).map(route => {
    const providerOptions = providers.map(p => `<option value="${esc(p)}" ${p === route.provider ? 'selected' : ''}>${esc(p)}</option>`).join('');
    return `
      <div class="route-editor" data-purpose="${esc(route.purpose)}">
        <label><strong>${esc(route.purpose)}</strong><span>Aktuell: ${esc(route.resolved.provider_name)} · ${esc(route.resolved.model)}</span></label>
        <label>Provider<select class="route-provider">${providerOptions}</select></label>
        <label>Modell optional<input class="route-model" value="${esc(route.model || '')}" placeholder="Provider Default" /></label>
        <label>Grund<input class="route-reason" value="${esc(route.reason || '')}" placeholder="Warum diese Route?" /></label>
      </div>`;
  }).join('');
}

function collectRoutingUpdates(){
  return Array.from(document.querySelectorAll('.route-editor')).map(row => ({
    purpose: row.dataset.purpose,
    provider: row.querySelector('.route-provider').value,
    model: row.querySelector('.route-model').value.trim(),
    reason: row.querySelector('.route-reason').value.trim(),
  }));
}

async function previewRoutingChanges(){
  const data = await api('/api/gui/llm-profiles/routing-editor/preview', {
    method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({updates: collectRoutingUpdates()})
  });
  document.getElementById('routingPreview').textContent = JSON.stringify(data, null, 2);
}

async function applyRoutingChanges(){
  const preview = await api('/api/gui/llm-profiles/routing-editor/preview', {
    method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({updates: collectRoutingUpdates()})
  });
  if(!preview.ok){
    document.getElementById('routingPreview').textContent = JSON.stringify(preview, null, 2);
    return;
  }
  const message = `Routing wirklich speichern?\n\nWarnungen: ${(preview.warnings || []).length}`;
  if(!confirm(message)) return;
  const result = await api('/api/gui/llm-profiles/routing-editor/apply', {
    method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({updates: collectRoutingUpdates(), actor:'user-gui'})
  });
  document.getElementById('routingPreview').textContent = JSON.stringify(result, null, 2);
  await loadAll();
}

async function loadRoutingAudit(){
  const data = await api('/api/gui/llm-profiles/routing-editor/audit?limit=20');
  document.getElementById('routingPreview').textContent = JSON.stringify(data, null, 2);
}

async function rollbackRouting(){
  if(!confirm('Letztes Routing-Backup zurückspielen?')) return;
  const data = await api('/api/gui/llm-profiles/routing-editor/rollback', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({})});
  document.getElementById('routingPreview').textContent = JSON.stringify(data, null, 2);
  await loadAll();
}
async function setProfile(profile){
  await api('/api/gui/llm-profiles/profile', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({profile})});
  await loadAll();
}
async function smokePreview(){
  const data = await api('/api/gui/llm-profiles/smoke-preview', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({provider:'cloud_expert'})});
  alert(data.message || 'Smoke Preview abgeschlossen. Live-Aufruf wurde nicht ausgeführt.');
}
loadAll().catch(err => { document.body.insertAdjacentHTML('beforeend', `<pre>${esc(err.message)}</pre>`); });
