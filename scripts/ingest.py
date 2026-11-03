"""Gera o índice: chunks (index/chunks.jsonl) e embeddings normalizados (index/embeddings.npy).

Uso: uv run python scripts/ingest.py
"""

import json

import numpy as np
from sentence_transformers import SentenceTransformer

from project_nov26_p3_rag.busca import INDICE, MODELO_EMBEDDING
from project_nov26_p3_rag.chunks import todos_os_chunks


def main() -> None:
    chunks = todos_os_chunks()
    modelo = SentenceTransformer(MODELO_EMBEDDING, device="cpu")
    # A família e5 exige os prefixos "passage: " (documentos) e "query: " (perguntas).
    E = modelo.encode(
        [f"passage: {c['texto']}" for c in chunks],
        normalize_embeddings=True,
        batch_size=32,
        show_progress_bar=False,
    )
    INDICE.mkdir(exist_ok=True)
    np.save(INDICE / "embeddings.npy", E.astype(np.float32))
    with open(INDICE / "chunks.jsonl", "w") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"{len(chunks)} chunks · embeddings {E.shape}")


if __name__ == "__main__":
    main()
