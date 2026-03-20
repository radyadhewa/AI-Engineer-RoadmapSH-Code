# AI-basics

This folder contains focused, runnable Python examples that teach practical building blocks for working with modern language models and vector search. Each script demonstrates one concept (API vs SDK, prompt engineering, embeddings, vector search, parameter tuning) so you can read, run, and experiment.

Keep this README as your quick reference for what each file does, prerequisites, and how to run the examples locally.

---

## Quick file list

- `01-inferencing-via-API.py`  
  - Purpose: Learn raw HTTP integration with an LLM provider.
  - What to change: `payload["messages"]`, `payload["model"]`, or the prompt text.

- `02-inferencing-via-SDK.py`  
  - Purpose: Compare calling LLM via OpenRouter SDK vs raw HTTP.
  - What to change: `model` or `messages` to experiment with different prompts.
  - Demonstrates using a context manager and reading model output from the SDK response object.

- `03-prompt-engineering.py`  
  - Purpose: Practice prompt engineering techniques. System vs user roles, providing explicit constraints, and forcing output formatting
  - What to change: `SYSTEM_PROMPT` — add constraints, change roles or request structured JSON outputs to enforce formatting.

- `04-embeddings.py`  
  - Purpose: Generate sentence embeddings using Hugging Face inference client. Prints embedding dimensionality and a short preview.
  - What to change: `text` variable or switch `model_id` to try other models.

- `05-vector-db.py`  
  - Purpose: Small end-to-end vector search demonstration using FAISS. Building an in-memory FAISS index, and querying it with nearest-neighbor search
  - Implementation highlights:
    - `preprocess_text()` uses spaCy lemmatization and stopword handling.
    - `SentenceTransformer` encodes texts to float32 vectors suitable for FAISS.
    - `IndexFlatL2` is used for demonstration (no persistence).
  - What to change: Replace `example_documents()` with your corpus, then persist index and metadata for reuse (`faiss.write_index` and save metadata to JSON).

- `06-parameter-tuning.py`  
  - Purpose: Observe how `temperature`, `top_p`, and `max_tokens` shape outputs.
  - What to change: Try other prompts, adjust `temperature` between `0.0` and `1.0`, or change `max_tokens` to see truncation.

- `readme.md`  
  This file.

---

## Purpose & learning pattern

Each script follows a small learning loop:
1. Read the short header/comment at the top of the script to understand the learning goal.
2. Run the script with the required environment variables.
3. Modify one or two lines (prompt, parameter, or model id) and re-run to observe differences.
4. Combine scripts once comfortable (e.g., use `04-embeddings.py` + `05-vector-db.py` to create a contextual retrieval augmented generation pipeline).

This folder is intentionally minimal so you can focus on one concept at a time.

---

## Prerequisites

- Python 3.11+ 
- A working package manager (`pip` or `uv`)
- A `.env` file containing the required API keys (example below)
- Install the Python packages listed in the Install section

.env example (create `.env` in the project root or this folder):
```
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENROUTER_URL=https://api.openrouter.ai/v1/chat/completions
HUGGINGFACE_TOKEN=your_huggingface_token_here
```

Notes:
- `OPENROUTER_URL` should match the endpoint the SDK/examples expect. Example value is shown above.
- `HUGGINGFACE_TOKEN` is required for calling the Hugging Face inference API in `04-embeddings.py`.
- This API keys are free to make and we are using the free models. So no worries, go make one!

---

## Install (example)

A single-line pip install to get the main dependencies used by these examples:

```
python -m venv venv
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

or if you are using uv
```
uv venv
uv sync
python -m spacy download en_core_web_sm
```

---

## How to run each script

1. Ensure your `.env` file contains the required keys.
2. Install dependencies as shown above.
3. Run a script, for example:
   - `python 01-inferencing-via-API.py`
   - `python 02-inferencing-via-SDK.py`
   - `python 03-prompt-engineering.py`
   - `python 04-embeddings.py`
   - `python 05-vector-db.py`
   - `python 06-parameter-tuning.py`

Run scripts from the folder or the project root but ensure the working directory can locate the `.env` file.

---