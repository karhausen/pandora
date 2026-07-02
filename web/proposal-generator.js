async function loadStatus() {
  const box = document.getElementById('generatorStatus');
  try {
    const res = await fetch('/api/proposal-generator/status');
    const data = await res.json();
    box.innerHTML = `<strong>Status:</strong> ${data.ok ? 'OK' : 'Problem'} · Version ${data.version} · Modus: ${data.mode} · Queue: ${data.queue_status ? 'OK' : 'Problem'}`;
  } catch (err) {
    box.textContent = `Status konnte nicht geladen werden: ${err}`;
  }
}

function payload() {
  return {
    request: document.getElementById('proposalRequest').value,
    proposal_type: document.getElementById('proposalType').value || null,
    context: { source: 'proposal_generator_gui', mvp: '29.0' },
    use_llm: document.getElementById('useLlm').checked
  };
}

async function callGenerator(path) {
  const output = document.getElementById('proposalOutput');
  output.textContent = 'Arbeite ...';
  try {
    const res = await fetch(path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload())
    });
    const data = await res.json();
    output.textContent = JSON.stringify(data, null, 2);
  } catch (err) {
    output.textContent = `Fehler: ${err}`;
  }
}

document.getElementById('generateBtn').addEventListener('click', () => callGenerator('/api/proposal-generator/generate'));
document.getElementById('enqueueBtn').addEventListener('click', () => callGenerator('/api/proposal-generator/enqueue'));
loadStatus();
