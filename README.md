# Tool-Use Agent

An agent that can call tools when it needs live data or exact computation,
instead of guessing: a calculator, current weather, Wikipedia search, and
current time by timezone. This is the mechanic behind every "AI agent" —
the LLM itself decides when a tool is needed and with what arguments.

- **LLM**: [Groq](https://groq.com) (`openai/gpt-oss-20b`, free tier), called
  directly via the `groq` SDK — no LangChain, so the tool-calling protocol
  itself is visible in [`agent.py`](agent.py) instead of hidden behind a
  framework.
- **Tools**: all free, no extra API keys needed beyond Groq —
  [Open-Meteo](https://open-meteo.com) for weather, Wikipedia's public REST
  API, a sandboxed arithmetic evaluator, and the standard library's
  `zoneinfo` for time.
- **UI**: [Gradio](https://www.gradio.app/) — each answer shows the tool
  calls the agent made along the way.

New to agents/tool-use? [`docs/HOW_IT_WORKS.md`](docs/HOW_IT_WORKS.md) (in
Spanish) walks through the agent loop step by step, including 2 real bugs
found while building it and how they were fixed.

## Local setup

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
cp .env.example .env
# edit .env and paste your free Groq API key from https://console.groq.com/keys
python app.py
```

## Deploying to Render

This repo includes a [`render.yaml`](render.yaml) Blueprint:

1. Sign in at https://dashboard.render.com.
2. **New → Blueprint** → pick this repo.
3. Paste your `GROQ_API_KEY` when prompted.
4. Deploy.
