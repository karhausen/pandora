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
  chat.appendChild(item);
  chat.scrollTop = chat.scrollHeight;
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

async function runPandora() {
  const input = document.getElementById("taskInput");
  const task = input.value.trim();
  if (!task) return;

  addMessage("user", task);
  setBusy(true);

  const result = await api("/user/run", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({task})
  });

  if (result.success) {
    addMessage("assistant", result.answer || "Erledigt.");
    showDetails(result);
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

loadUserStatus();
