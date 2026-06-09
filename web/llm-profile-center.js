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
  renderRoutes(await api('/api/gui/llm-profiles/routes'));
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
function renderRoutes(data){
  const routes = data.routes || {};
  document.getElementById('routes').innerHTML = Object.entries(routes).map(([name, route]) => `
    <div class="route"><strong>${esc(name)}</strong><span class="meta">Provider: ${esc(route.provider || route.resolved_provider || '–')}</span><br><span class="meta">${esc(route.reason || '')}</span></div>`).join('');
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
