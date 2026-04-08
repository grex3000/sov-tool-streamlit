# AI Share of Voice — Streamlit App

Web interface for the AI Share of Voice scanner. Fill in a form, click Run, get a full HTML report inline.

## Run locally

```bash
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# edit secrets.toml and add your OPENROUTER_API_KEY
streamlit run app.py
```

## Deploy to Streamlit Community Cloud (free)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app → select this repo
3. Set `app.py` as the main file
4. Under **Settings → Secrets**, add:
   ```toml
   OPENROUTER_API_KEY = "sk-or-v1-..."
   ```
5. Deploy — colleagues access it via the shared URL, no setup required

## Get an API key

Free key at [openrouter.ai](https://openrouter.ai) — one key covers all models (GPT, Claude, Gemini, Perplexity).
