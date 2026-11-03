"""Compara densa × lexical (BM25) × híbrida (RRF) em consultas difíceis (sem LLM).

Uso: uv run python scripts/avaliar_busca.py   →  reports/busca_dificil.json
"""

import json
from pathlib import Path

import numpy as np

from project_nov26_p3_rag.busca import Buscador

RAIZ = Path(__file__).resolve().parents[1]


def main() -> None:
    consultas = [json.loads(linha) for linha in open(RAIZ / "data/busca_dificil.jsonl")]
    b = Buscador()
    modos = {"densa": b.densa, "lexical": b.lexical, "hibrida": b.hibrida}
    resultado = {}
    for nome, buscar in modos.items():
        r1, r5, rr = [], [], []
        for q in consultas:
            ids = [b.chunks[i]["id"] for i in buscar(q["consulta"], 5)]
            rel = set(q["chunks_relevantes"])
            r1.append(float(ids[0] in rel))
            r5.append(float(bool(rel & set(ids))))
            rr.append(next((1 / p for p, i in enumerate(ids, 1) if i in rel), 0.0))
        resultado[nome] = {
            "acerto@1": round(float(np.mean(r1)), 3),
            "acerto@5": round(float(np.mean(r5)), 3),
            "mrr": round(float(np.mean(rr)), 3),
        }
    resultado["n_consultas"] = len(consultas)
    (RAIZ / "reports/busca_dificil.json").write_text(json.dumps(resultado, indent=2))
    print(json.dumps(resultado, indent=2))


if __name__ == "__main__":
    main()
