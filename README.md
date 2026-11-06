# FishRAG

An AI-powered fishing research assistant built around retrieval-augmented generation (RAG).

## Problem and goals

Fishing advice is scattered across guides, regulations, and lake-specific resources. FishRAG aims to help a nontechnical competitive angler research unfamiliar water and prepare for tournaments through natural-language questions and source-grounded answers.

The initial focus is bass fishing on Rice Lake near Peterborough and other Ontario lakes. Topics will include techniques, seasonal behaviour, lure selection, habitat, lake conditions, tournament preparation, and Ontario regulations. The broader goal is to teach transferable fishing strategies that apply across North America.

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

The optional `ingestion` extra adds only LlamaIndex core and Beautiful Soup for local HTML loading and chunking. Embedding, vector storage, generation, and frontend integrations will be added later.

## Project structure

```text
src/fishrag/       Python package
data/documents/   Local source documents (contents ignored by Git)
data/sources.json Tracked provenance and usage review (no source content)
notebooks/        Initial RAG experiments
tests/            Future tests
evals/            Future benchmark questions and evaluation scripts
.env.example      Environment variable template
pyproject.toml    Package and development tool configuration
```

## License

Code is licensed under the [MIT License](LICENSE). Third-party source documents retain their own licenses and usage terms.
