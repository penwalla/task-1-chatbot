// Aether Chatbot Client-Side Logic
let currentConversationId = localStorage.getItem("aether_conversation_id") || null;
let currentStrategy = "sliding_window";
let maxMessages = 10;
let isGenerating = false;

// DOM Elements
const chatMessages = document.getElementById("chatMessages");
const welcomeScreen = document.getElementById("welcomeScreen");
const userInput = document.getElementById("userInput");
const chatForm = document.getElementById("chatForm");
const sendBtn = document.getElementById("sendBtn");
const newChatBtn = document.getElementById("newChatBtn");
const clearChatBtn = document.getElementById("clearChatBtn");
const refreshKeyBtn = document.getElementById("refreshKeyBtn");
const strategySelect = document.getElementById("strategySelect");
const maxMessagesInput = document.getElementById("maxMessagesInput");
const strategyBadge = document.getElementById("strategyBadge");
const capacityVal = document.getElementById("capacityVal");
const contextCount = document.getElementById("contextCount");
const prunedCount = document.getElementById("prunedCount");
const contextProgressBar = document.getElementById("contextProgressBar");
const memorySection = document.getElementById("memorySection");
const memorySummaryText = document.getElementById("memorySummaryText");
const apiStatusIndicator = document.getElementById("apiStatusIndicator");
const apiStatusText = document.getElementById("apiStatusText");
const alertBanner = document.getElementById("alertBanner");
const alertMessage = document.getElementById("alertMessage");
const closeAlertBtn = document.getElementById("closeAlertBtn");
const sessionSubtitle = document.getElementById("sessionSubtitle");
const sidebar = document.getElementById("sidebar");
const sidebarToggle = document.getElementById("sidebarToggle");

// Configure Marked for code highlighting
marked.setOptions({
  highlight: function(code, lang) {
    if (lang && hljs.getLanguage(lang)) {
      return hljs.highlight(code, { language: lang }).value;
    }
    return hljs.highlightAuto(code).value;
  },
  breaks: true
});

// Initialize on DOM load
document.addEventListener("DOMContentLoaded", async () => {
  setupEventListeners();
  await checkApiHealth();
  await initializeSession();
});

function setupEventListeners() {
  chatForm.addEventListener("submit", (e) => {
    e.preventDefault();
    handleSendMessage();
  });

  userInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  });

  userInput.addEventListener("input", () => {
    userInput.style.height = "auto";
    userInput.style.height = Math.min(userInput.scrollHeight, 140) + "px";
  });

  newChatBtn.addEventListener("click", () => startNewSession());
  clearChatBtn.addEventListener("click", () => clearCurrentSession());
  refreshKeyBtn.addEventListener("click", () => reloadEnvConfig());

  strategySelect.addEventListener("change", (e) => {
    currentStrategy = e.target.value;
    updateStrategyDisplay();
  });

  maxMessagesInput.addEventListener("change", (e) => {
    const val = parseInt(e.target.value, 10);
    if (val >= 2 && val <= 50) {
      maxMessages = val;
      capacityVal.textContent = `${maxMessages} msgs`;
    }
  });

  closeAlertBtn.addEventListener("click", () => {
    alertBanner.style.display = "none";
  });

  // Suggestion chips handler
  document.querySelectorAll(".chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      const prompt = chip.getAttribute("data-prompt");
      if (prompt && !isGenerating) {
        userInput.value = prompt;
        handleSendMessage();
      }
    });
  });

  if (sidebarToggle) {
    sidebarToggle.addEventListener("click", () => {
      sidebar.classList.toggle("open");
    });
  }
}

async function checkApiHealth() {
  const dot = apiStatusIndicator.querySelector(".status-dot");
  try {
    const res = await fetch("/api/health");
    if (!res.ok) throw new Error("Health check failed");
    const data = await res.json();
    
    if (data.api_key_configured) {
      dot.className = "status-dot active";
      apiStatusText.textContent = `Online: ${data.model}`;
      hideAlert();
    } else {
      dot.className = "status-dot inactive";
      apiStatusText.textContent = "API Key Missing";
      showAlert("⚠️ GEMINI_API_KEY is not configured. Please add your key to the .env file in the project root, then click 'Reload .env'.");
    }

    if (data.default_strategy) {
      currentStrategy = data.default_strategy;
      strategySelect.value = currentStrategy;
      updateStrategyDisplay();
    }
    if (data.max_context_messages) {
      maxMessages = data.max_context_messages;
      maxMessagesInput.value = maxMessages;
      capacityVal.textContent = `${maxMessages} msgs`;
    }
  } catch (err) {
    dot.className = "status-dot inactive";
    apiStatusText.textContent = "Server Offline";
    showAlert("Cannot connect to backend server. Make sure FastAPI server is running.");
  }
}

async function reloadEnvConfig() {
  try {
    const res = await fetch("/api/config/reload", { method: "POST" });
    const data = await res.json();
    await checkApiHealth();
    showAlert("Configuration reloaded from .env!", false);
    setTimeout(hideAlert, 3000);
  } catch (err) {
    showAlert("Failed to reload configuration: " + err.message);
  }
}

async function initializeSession() {
  if (currentConversationId) {
    try {
      const res = await fetch(`/api/conversations/${currentConversationId}`);
      if (res.ok) {
        const data = await res.json();
        sessionSubtitle.textContent = `Session: ${currentConversationId.slice(0, 8)}...`;
        renderHistory(data.messages);
        updateContextStats(data.stats, data.running_summary);
        return;
      }
    } catch (e) {
      console.warn("Could not restore previous conversation:", e);
    }
  }
  await startNewSession();
}

async function startNewSession() {
  try {
    const res = await fetch("/api/conversations/new", { method: "POST" });
    const data = await res.json();
    currentConversationId = data.conversation_id;
    localStorage.setItem("aether_conversation_id", currentConversationId);
    sessionSubtitle.textContent = `Session: ${currentConversationId.slice(0, 8)}...`;
    
    // Clear chat display & show welcome screen
    chatMessages.innerHTML = "";
    chatMessages.appendChild(welcomeScreen);
    welcomeScreen.style.display = "flex";
    
    updateContextStats(data.stats, "");
  } catch (err) {
    showAlert("Error creating new session: " + err.message);
  }
}

async function clearCurrentSession() {
  if (!currentConversationId) return;
  try {
    const res = await fetch(`/api/conversations/${currentConversationId}`, { method: "DELETE" });
    const data = await res.json();
    
    chatMessages.innerHTML = "";
    chatMessages.appendChild(welcomeScreen);
    welcomeScreen.style.display = "flex";
    
    updateContextStats(data.stats, "");
  } catch (err) {
    showAlert("Error clearing conversation: " + err.message);
  }
}

function updateStrategyDisplay() {
  if (currentStrategy === "sliding_window") {
    strategyBadge.textContent = "Sliding Window";
    memorySection.style.display = "none";
  } else {
    strategyBadge.textContent = "Summarization";
    memorySection.style.display = "flex";
  }
}

function renderHistory(messages) {
  if (!messages || messages.length === 0) {
    welcomeScreen.style.display = "flex";
    return;
  }
  welcomeScreen.style.display = "none";
  chatMessages.innerHTML = "";
  messages.forEach((msg) => {
    appendMessageBubble(msg.role, msg.content, false);
  });
  scrollToBottom();
}

async function handleSendMessage() {
  const text = userInput.value.trim();
  if (!text || isGenerating) return;

  // Hide welcome screen
  welcomeScreen.style.display = "none";

  // Append user message
  appendMessageBubble("user", text);
  userInput.value = "";
  userInput.style.height = "auto";

  // Append typing indicator
  const typingIndicator = createTypingIndicator();
  chatMessages.appendChild(typingIndicator);
  scrollToBottom();

  isGenerating = true;
  sendBtn.disabled = true;

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: text,
        conversation_id: currentConversationId,
        strategy: currentStrategy,
        max_messages: maxMessages,
      })
    });

    const data = await res.json();
    typingIndicator.remove();

    if (!res.ok || data.error) {
      const errMsg = data.message || `Error ${res.status}: Failed to get response.`;
      appendErrorBubble(errMsg, data.details);
      showAlert(errMsg);
    } else {
      currentConversationId = data.conversation_id;
      localStorage.setItem("aether_conversation_id", currentConversationId);
      sessionSubtitle.textContent = `Session: ${currentConversationId.slice(0, 8)}...`;
      
      appendMessageBubble("model", data.response);
      updateContextStats(data.stats);
    }
  } catch (err) {
    typingIndicator.remove();
    appendErrorBubble("Network failure or connection refused. Is the server running?", err.message);
    showAlert("Network error connecting to backend: " + err.message);
  } finally {
    isGenerating = false;
    sendBtn.disabled = false;
    userInput.focus();
    scrollToBottom();
  }
}

function appendMessageBubble(role, content, animate = true) {
  const row = document.createElement("div");
  row.className = `message-row ${role === "user" ? "user" : "bot"}`;
  if (!animate) row.style.animation = "none";

  const avatar = document.createElement("div");
  avatar.className = "message-avatar";
  avatar.textContent = role === "user" ? "👤" : "✨";

  const contentBox = document.createElement("div");
  contentBox.className = "message-content";

  const meta = document.createElement("div");
  meta.className = "message-meta";
  const now = new Date();
  meta.textContent = `${role === "user" ? "You" : "Aether"} • ${now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`;

  const bubble = document.createElement("div");
  bubble.className = "message-bubble";
  
  if (role === "user") {
    bubble.textContent = content;
  } else {
    // Render Markdown for assistant
    bubble.innerHTML = marked.parse(content);
  }

  contentBox.appendChild(meta);
  contentBox.appendChild(bubble);
  row.appendChild(avatar);
  row.appendChild(contentBox);

  chatMessages.appendChild(row);
  scrollToBottom();
}

function appendErrorBubble(message, details) {
  const row = document.createElement("div");
  row.className = "message-row bot";

  const avatar = document.createElement("div");
  avatar.className = "message-avatar";
  avatar.style.borderColor = "var(--danger-color)";
  avatar.textContent = "⚠️";

  const contentBox = document.createElement("div");
  contentBox.className = "message-content";

  const meta = document.createElement("div");
  meta.className = "message-meta";
  meta.textContent = "System Error";

  const bubble = document.createElement("div");
  bubble.className = "message-bubble";
  bubble.style.borderColor = "rgba(239, 68, 68, 0.4)";
  bubble.style.backgroundColor = "rgba(239, 68, 68, 0.1)";

  let html = `<strong style="color: #fca5a5;">Error:</strong> ${message}`;
  if (details && details !== message) {
    html += `<br><small style="color: #94a3b8; font-family: var(--font-mono);">${details}</small>`;
  }
  bubble.innerHTML = html;

  contentBox.appendChild(meta);
  contentBox.appendChild(bubble);
  row.appendChild(avatar);
  row.appendChild(contentBox);

  chatMessages.appendChild(row);
  scrollToBottom();
}

function createTypingIndicator() {
  const row = document.createElement("div");
  row.className = "message-row bot typing-indicator-row";

  const avatar = document.createElement("div");
  avatar.className = "message-avatar";
  avatar.textContent = "✨";

  const bubble = document.createElement("div");
  bubble.className = "message-bubble";
  bubble.innerHTML = `
    <div class="typing-dots">
      <div class="typing-dot"></div>
      <div class="typing-dot"></div>
      <div class="typing-dot"></div>
    </div>
  `;

  row.appendChild(avatar);
  row.appendChild(bubble);
  return row;
}

function updateContextStats(stats, runningSummary = null) {
  if (!stats) return;

  const inContext = stats.messages_in_context || 0;
  const total = stats.total_messages || 0;
  const pruned = stats.messages_pruned_or_summarized || 0;
  const capacity = stats.max_messages || maxMessages;

  contextCount.textContent = `${inContext} in win (${total} total)`;
  prunedCount.textContent = `${pruned} msgs`;
  capacityVal.textContent = `${capacity} msgs`;

  // Percentage of window used
  const percent = Math.min(100, Math.round((inContext / capacity) * 100));
  contextProgressBar.style.width = `${percent}%`;

  if (stats.strategy) {
    currentStrategy = stats.strategy;
    strategySelect.value = currentStrategy;
    updateStrategyDisplay();
  }

  // Update running memory summary
  if (runningSummary || stats.running_summary_preview) {
    memorySection.style.display = "flex";
    memorySummaryText.textContent = runningSummary || stats.running_summary_preview || "No summary yet.";
  } else if (currentStrategy !== "summarization") {
    memorySection.style.display = "none";
  }
}

function showAlert(msg, isError = true) {
  alertMessage.textContent = msg;
  alertBanner.style.backgroundColor = isError ? "rgba(239, 68, 68, 0.15)" : "rgba(16, 185, 129, 0.15)";
  alertBanner.style.borderColor = isError ? "rgba(239, 68, 68, 0.4)" : "rgba(16, 185, 129, 0.4)";
  alertBanner.style.color = isError ? "#fca5a5" : "#6ee7b7";
  alertBanner.style.display = "flex";
}

function hideAlert() {
  alertBanner.style.display = "none";
}

function scrollToBottom() {
  setTimeout(() => {
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }, 50);
}
