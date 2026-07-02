async function getJSON(url, options){ const r = await fetch(url, options); return await r.json(); }
async function loadStatus(){ document.getElementById('status').textContent = JSON.stringify(await getJSON('/api/goals/status'), null, 2); }
async function loadGoals(){ document.getElementById('output').textContent = JSON.stringify(await getJSON('/api/goals/list'), null, 2); }
async function evaluateGoals(){ document.getElementById('output').textContent = JSON.stringify(await getJSON('/api/goals/evaluate'), null, 2); }
loadStatus();
