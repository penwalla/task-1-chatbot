# 🧠 Aether — Context-Aware Multi-Turn AI Chatbot

A production-grade, multi-turn AI chatbot powered by the **Google Gemini API** (`gemini-1.5-flash` / `gemini-2.0-flash`) featuring **active context window management**, personality-driven system prompting, resilient error handling, and dual interfaces (interactive Web UI & Terminal CLI).

---

## 🌟 Key Highlights & Requirements Checklist

| Requirement | Implementation Details |
|---|---|
| ✅ **Backend Endpoint** | FastAPI REST endpoints (`POST /api/chat`, `GET /api/conversations/{id}`, `DELETE /api/conversations/{id}`, `GET /api/health`). |
| ✅ **Multi-Turn Memory** | Per-conversation session isolation storing full turn histories; capable of cross-referencing facts established turns earlier. |
| ✅ **Context Window Management** | **Dual strategies supported:**<br>1. *Sliding Window Truncation* (preserves last $N$ turns cleanly starting with user turn).<br>2. *Progressive Summarization* (compresses older turns into a running memory summary prepended to active context). |
| ✅ **System Prompt & Persona** | **Aether**: An articulate, pragmatic Technical Mentor & Software Architect with clear guidelines for depth, memory continuity, and clarity. |
| ✅ **Robust Error Handling** | Catches missing API keys, invalid credentials (401/403), rate limits / quotas (429), timeouts, and network outages with friendly user messages and status codes. |
| ✅ **API Key Security** | Keys strictly loaded via environment variables (`.env`). `.gitignore` ensures keys are never committed to version control. |
| ✅ **Dual Interfaces** | Modern responsive Web interface (`/`) and an interactive Terminal CLI (`cli.py`). |

---

## 📐 Architecture & Context Management Strategy

When interacting with LLMs, sending unlimited conversation history indefinitely causes two problems:
1. **Context Window Overflow**: Exceeds the model's token limits.
2. **Cost & Latency Bloat**: Pricing and latency scale with prompt token size.

Aether solves this through two configurable strategies:

```
                          [ Incoming User Message ]
                                      │
                                      ▼
                      [ Check Conversation Memory Store ]
                                      │
               ┌──────────────────────┴──────────────────────┐
               ▼                                             ▼
    [ Sliding Window Strategy ]                  [ Summarization Strategy ]
  - Retains the most recent N turns            - Checks if turns > N
  - Older turns are pruned                     - Older turns are compressed into a
  - Sequence trimmed to start with 'user'        concise running memory summary
  - Lightweight & zero latency overhead        - Prepended to system instruction
               │                                             │
               └──────────────────────┬──────────────────────┘
                                      ▼
                   [ Model Query: Persona + Context ]
                                      │
                                      ▼
                        [ Reply Returned to Client ]
```

### 1. Sliding Window Truncation (`sliding_window`)
- Retains only the most recent $N$ messages (configurable via `MAX_CONTEXT_MESSAGES`, default: 10).
- Guarantees the prompt begins with a `user` turn to comply with conversational API invariants.
- Computes real-time telemetry (messages in context vs. total stored vs. pruned).

### 2. Progressive Summarization (`summarization`)
- When conversation length exceeds $N$, older turns are passed through a summarization pipeline.
- The summary updates iteratively: $\text{Summary}_{\text{new}} = f(\text{Summary}_{\text{old}}, \text{Older Turns})$.
- The summary is injected into the LLM's system instruction under `### Running Conversation Memory`, enabling lifelong memory recall without context bloat.

---

## 🎭 System Persona: Aether

Aether is configured as a seasoned Technical Mentor and Software Architect rather than a generic assistant:
- **Tone**: Warm, sharp, supportive, and intellectually honest.
- **Analogy-First Explanations**: Uses concrete mental models before diving into code.
- **Memory Continuity**: Explicitly tasked with referencing user-provided facts, programming languages, and project constraints across turns.
- **No Robotic Boilerplate**: Bypasses generic disclaimers ("As an AI...") and dives directly into the technical solution.

---

## 🚀 Quickstart Guide

### 1. Prerequisites
- Python 3.10+
- A Google Gemini API Key (Get a free key from [Google AI Studio](https://aistudio.google.com/app/apikey))

### 2. Clone and Setup Environment

```bash
# Clone the repository
git clone <your-repo-url>
cd task1

# Create and activate virtual environment
python -m venv .venv

# On Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# On Linux / macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure API Key

Copy the `.env.example` file to `.env`:

```bash
cp .env.example .env
```

Open `.env` in any text editor and paste your Gemini API key:

```env
GEMINI_API_KEY=AIzaSy...your_real_key_here
GEMINI_MODEL=gemini-1.5-flash
CONTEXT_STRATEGY=sliding_window
MAX_CONTEXT_MESSAGES=10
```

> ⚠️ **Security Note:** `.env` is already listed in `.gitignore`. Never commit `.env` to GitHub.

---

## 💻 Running the Application

### Option A: Web User Interface (Recommended)

Start the FastAPI server:

```bash
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Open your browser at **[http://127.0.0.1:8000](http://127.0.0.1:8000)**.

**Web UI Features:**
- Real-time Context Window Telemetry bar (active window capacity, message count, pruned count).
- Live Strategy Switcher (toggle between Sliding Window & Summarization).
- Visual running memory viewer.
- Built-in multi-turn test prompt chips.
- Markdown rendering with code syntax highlighting.
- Clear Session and New Session buttons.

---

### Option B: Terminal CLI

Run the interactive terminal interface:

```bash
python cli.py
```

**CLI Commands:**
- `/stats` — Displays active context window telemetry.
- `/strategy [sliding_window|summarization]` — Switch context strategy on the fly.
- `/reset` — Clear conversation history.
- `/exit` — Quit the session.

---

## 🧪 Automated Multi-Turn Test Suite

To verify context window management logic, session isolation, and (optionally) live 5-turn conversational recall:

```bash
python test_conversation.py
```

### Multi-Turn Test Sequence Walkthrough:

| Turn | Prompt | Expected Memory Validation |
|---|---|---|
| **Turn 1** | *"Hi Aether! My name is Jordan and I am developing an autonomous drone in Python."* | Establishes name, project, and language. |
| **Turn 2** | *"What is my name and what project am I working on?"* | Confirms immediate recall of `Jordan` and `autonomous drone`. |
| **Turn 3** | *"What are two critical sensors I should attach for obstacle avoidance?"* | Retains domain context (`drone`) and recommends sensors (e.g. LiDAR, Ultrasonic). |
| **Turn 4** | *"Why is the programming language I mentioned earlier suitable for these sensors?"* | Cross-references `Python` from Turn 1 with the sensors from Turn 3. |
| **Turn 5** | *"Please give a 2-bullet summary of our discussion so far."* | Synthesizes all 4 previous turns cohesively. |

---

## 📁 Repository Structure

```
├── .gitignore               # Strict ignore rules (.env, .venv, logs, etc.)
├── .env.example             # Safe environment variable template
├── requirements.txt         # Dependencies (FastAPI, google-genai, uvicorn, etc.)
├── cli.py                   # Terminal CLI interface
├── test_conversation.py     # Automated multi-turn test suite
├── README.md                # Comprehensive project documentation
├── app/
│   ├── __init__.py
│   ├── config.py            # Environment configuration & validation
│   ├── context_manager.py   # Sliding window & progressive summarization algorithms
│   ├── llm_client.py        # Gemini client wrapper with resilient error handling
│   ├── persona.py           # Aether system persona & prompts
│   └── main.py              # FastAPI application & REST endpoints
└── static/
    ├── index.html           # Modern web chat frontend
    ├── styles.css           # Responsive styling & themes
    └── app.js               # Client-side state, API calls, and telemetry
```

---

## 🛡️ Error Handling Details

The chatbot gracefully handles all standard failure conditions:

1. **Missing / Placeholder API Key**: Detects unconfigured keys before sending requests; displays actionable guidance pointing to Google AI Studio.
2. **Invalid API Key (401 / 403)**: Trapped cleanly and converted into a user-facing error message without server crashes.
3. **Rate Limits / Quota (429)**: Catches `RESOURCE_EXHAUSTED` and instructs the user to pause briefly.
4. **Timeouts / Network Outages**: Wrapped in `httpx` connection error handlers to ensure graceful degradation.

---

## 📜 License
MIT License. Built for Internship Submission Task 1.
