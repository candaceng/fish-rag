# FishRAG

An AI-powered fishing research assistant built around retrieval-augmented generation (RAG).

## Problem and goals

Fishing advice is scattered across guides, regulations, and lake-specific resources. FishRAG aims to help a nontechnical competitive angler research unfamiliar water and prepare for tournaments through natural-language questions and source-grounded answers.

The initial focus is bass fishing on Rice Lake near Peterborough and other Ontario lakes. Topics will include techniques, seasonal behavior, lure selection, habitat, lake conditions, tournament preparation, and Ontario regulations. The broader goal is to teach transferable fishing strategies that apply across North America.

This personal ML engineering portfolio project emphasizes a simple, modular pipeline that is easy to understand, evaluate, and extend. Planned answers will cite trustworthy source documents; regulation answers will need current official sources and clear geographic and date context.

## Planned architecture and stack

```text
Source documents → ingestion and chunking → local embeddings → vector store
Question → retrieval → retrieved context + OpenAI generation → cited answer
```

| Component | Planned tool |
| --- | --- |
| Runtime and packaging | Python 3.11+, setuptools |
| Ingestion and retrieval | LlamaIndex |
| Local embeddings | SentenceTransformers |
| Vector storage | Chroma |
| Answer generation | OpenAI API |
| Frontend | Streamlit |
| Development | pytest, Ruff, JupyterLab |
| Evaluation | Retrieval Recall@k and MRR; answer faithfulness |

Only development dependencies are configured today. RAG and frontend dependencies will be added as their components are built.

## Project structure

```text
src/fishrag/       Python package
data/documents/   Local source documents (contents ignored by Git)
notebooks/        Initial RAG experiments
tests/            Future tests
evals/            Future benchmark questions and evaluation scripts
.env.example      Environment variable template
pyproject.toml    Package and development tool configuration
```

## Local setup

With Python 3.11 or newer installed, run these commands from the repository root:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
python -c "import fishrag; print(fishrag.__name__)"
```

On macOS/Linux, activate with `source .venv/bin/activate` and copy the template with `cp .env.example .env`.

Keep API keys in your local `.env` file and never commit them. No key is needed for the current scaffold. Environment-file loading and API integration are not implemented yet.

Run `ruff check .` for linting and `jupyter lab` for experiments. Use `pytest` once tests are added; the empty test suite currently collects no tests.

## Roadmap

1. Build `notebooks/01_document_ingestion.ipynb` using real Ontario fishing resources. Record source URLs, publication dates, access dates, and usage permissions.
2. Explore document loading and chunking, then add local embeddings and Chroma retrieval.
3. Create a small, reviewed question set and measure Recall@k and MRR.
4. Add source-cited answer generation and assess answer faithfulness, including unsupported questions.
5. Add a simple Streamlit interface and expand lake coverage.

## Current status

Initial repository setup only. No ingestion, embeddings, retrieval, LLM calls, frontend, or deployment is implemented. No documents have been downloaded and no evaluation results are available.

## License

Code is licensed under the [MIT License](LICENSE). Third-party source documents retain their own licenses and usage terms.
