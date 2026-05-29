let currentSessionId = localStorage.getItem("pandora_session_id") || null;

async function api(path, options = {}) {
  const response = await fetch(path, options);
  const text = await response.text();
  try {
    const data = JSON.parse(text);
    if (!response.ok) {
      return { success: false, error: data.detail || data.error || text };
    }
    return data;
  } catch {
    return { success: response.ok, answer: text };
  }
}

function addMessage(role, text) {
  const chat = document.getElementById("chat");
  const item = document.createElement("div");
  item.className = `message ${role}`;
  item.innerHTML = `<div class="role">${role === "user" ? "Du" : "Pandora"}</div><div class="answer"></div>`;
  item.querySelector(".answer").textContent = text;
  chat.prepend(item);
  chat.scrollTop = 0;
}

function clearChat() {
  document.getElementById("chat").innerHTML = "";
}

function setBusy(isBusy) {
  document.getElementById("runButton").disabled = isBusy;
  document.getElementById("statusText").textContent = isBusy ? "Pandora arbeitet..." : "Bereit";
}

function showDetails(result) {
  document.getElementById("details").classList.remove("hidden");
  document.getElementById("planBox").textContent = JSON.stringify(result.plan || {}, null, 2);
  document.getElementById("executionBox").textContent = JSON.stringify(result.execution || {}, null, 2);
}

async function ensureSession() {
  if (currentSessionId) return currentSessionId;
  const created = await api("/chat/sessions", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({title: "Neue Unterhaltung"})
  });
  currentSessionId = created.session_id;
  localStorage.setItem("pandora_session_id", currentSessionId);
  await loadSessions();
  return currentSessionId;
}

async function newSession() {
  const created = await api("/chat/sessions", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({title: "Neue Unterhaltung"})
  });
  currentSessionId = created.session_id;
  localStorage.setItem("pandora_session_id", currentSessionId);
  clearChat();
  document.getElementById("details").classList.add("hidden");
  await loadSessions();
}

async function loadSessions() {
  const data = await api("/chat/sessions");
  const select = document.getElementById("sessionSelect");
  if (!select || !data.sessions) return;

  select.innerHTML = "";
  for (const session of data.sessions) {
    const option = document.createElement("option");
    option.value = session.session_id;
    option.textContent = `${session.title || "Unterhaltung"} (${session.message_count})`;
    if (session.session_id === currentSessionId) option.selected = true;
    select.appendChild(option);
  }
}

async function switchSession() {
  const select = document.getElementById("sessionSelect");
  currentSessionId = select.value;
  localStorage.setItem("pandora_session_id", currentSessionId);
  await loadCurrentSession();
}

async function loadCurrentSession() {
  if (!currentSessionId) return;
  const session = await api(`/chat/sessions/${currentSessionId}`);
  clearChat();
  const messages = [...(session.messages || [])].reverse();
  for (const message of messages) {
    addMessage(message.role, message.content);
  }
}

async function runPandora() {
  const input = document.getElementById("taskInput");
  const task = input.value.trim();
  if (!task) return;

  await ensureSession();
  addMessage("user", task);
  setBusy(true);

  const result = await api("/chat/run", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({task, session_id: currentSessionId, provider_name: "mock", save: true})
  });

  if (result.success) {
    currentSessionId = result.session_id;
    localStorage.setItem("pandora_session_id", currentSessionId);
    addMessage("assistant", result.answer || "Erledigt.");
    showDetails(result);
    await loadSessions();
  } else {
    addMessage("assistant", `Fehler: ${result.error || "Unbekannter Fehler"}`);
    showDetails(result);
  }

  setBusy(false);
}

async function loadUserStatus() {
  const status = await api("/user/status");
  document.getElementById("statusText").textContent = status.ready ? "Bereit" : "Nicht bereit";
}

async function boot() {
  await loadUserStatus();
  await loadSessions();
  if (currentSessionId) {
    await loadCurrentSession();
  }
}

boot();
