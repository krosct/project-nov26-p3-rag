"""Busca densa (embeddings), lexical (BM25) e híbrida (Reciprocal Rank Fusion).

Com algumas centenas de chunks, a busca exata por força bruta (produto escalar
numa matriz numpy) leva microssegundos; um índice aproximado (FAISS, pgvector)
só compensa na casa dos milhões de vetores.
"""

import json
import re
import unicodedata
from functools import cached_property
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

INDICE = Path(__file__).resolve().parents[2] / "index"
MODELO_EMBEDDING = "intfloat/multilingual-e5-small"
K_RRF = 60


def tokenizar(texto: str) -> list[str]:
    sem_acento = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode().lower()
    return [t for t in re.split(r"\W+", sem_acento) if t]


def rrf(*rankings: list[int], k: int = K_RRF) -> list[int]:
    """Funde rankings pela posição: score = Σ 1/(k + posição); ignora a escala dos scores."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for posicao, idx in enumerate(ranking, start=1):
            scores[idx] = scores.get(idx, 0.0) + 1 / (k + posicao)
    return sorted(scores, key=scores.get, reverse=True)


class Buscador:
    def __init__(self, pasta: Path = INDICE):
        self.chunks = [json.loads(linha) for linha in (pasta / "chunks.jsonl").open()]
        self.embeddings = np.load(pasta / "embeddings.npy")
        self.bm25 = BM25Okapi([tokenizar(c["texto"]) for c in self.chunks])

    @cached_property
    def modelo(self):
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(MODELO_EMBEDDING, device="cpu")

    def densa(self, pergunta: str, k: int = 5) -> list[int]:
        q = self.modelo.encode([f"query: {pergunta}"], normalize_embeddings=True)[0]
        return np.argsort(-(self.embeddings @ q))[:k].tolist()

    def lexical(self, pergunta: str, k: int = 5) -> list[int]:
        return np.argsort(-self.bm25.get_scores(tokenizar(pergunta)))[:k].tolist()

    def hibrida(self, pergunta: str, k: int = 5, profundidade: int = 20) -> list[int]:
        return rrf(self.densa(pergunta, profundidade), self.lexical(pergunta, profundidade))[:k]

    def buscar(self, pergunta: str, modo: str = "hibrida", k: int = 5) -> list[dict]:
        indices = {"densa": self.densa, "lexical": self.lexical, "hibrida": self.hibrida}[modo](
            pergunta, k
        )
        return [self.chunks[i] for i in indices]
