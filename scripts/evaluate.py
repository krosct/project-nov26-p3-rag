"""Avaliação do assistente em data/avaliacao.jsonl (50 perguntas).

Uso: uv run python scripts/evaluate.py
1. Busca (sem LLM): recall@5 e MRR, densa × híbrida; recusas do guardrail em cada modo.
2. Pipeline completo (busca híbrida + LLM + ferramentas): recusa correta, correção e
   fidelidade (LLM juiz de outro fornecedor), latência p50/p95 e tokens.
Saídas: reports/busca.json, reports/respostas.jsonl, reports/resumo.json
"""

import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from project_nov26_p3_rag import guardrail
from project_nov26_p3_rag.assistente import responder
from project_nov26_p3_rag.busca import Buscador
from project_nov26_p3_rag.llm import MODELO_GERADOR, MODELO_JUIZ, completar

RAIZ = Path(__file__).resolve().parents[1]
K = 5

PROMPT_JUIZ = """Você avalia respostas de um assistente de cardápios de delivery.

Pergunta: {pergunta}
Resposta esperada (referência): {esperada}

Dados disponíveis para o assistente (contexto recuperado e resultados de ferramentas):
{dados}

Resposta do assistente: {resposta}

Avalie:
- "correta": a resposta atende à pergunta e é consistente com a resposta esperada e com os dados?
  (Listas podem ter ordem diferente; pequenas omissões em listas longas são aceitáveis.)
- "fiel": TODA afirmação da resposta está suportada pelos dados acima (nada inventado)?

Responda SOMENTE com JSON: {{"correta": true|false, "fiel": true|false, "motivo": "<frase curta>"}}"""  # noqa: E501


def metricas_busca(perguntas: list[dict], buscador: Buscador, modo: str) -> dict:
    recalls, rrs, recusas_armadilha, recusas_indevidas = [], [], 0, 0
    for q in perguntas:
        indices = {"densa": buscador.densa, "hibrida": buscador.hibrida}[modo](q["pergunta"], K)
        ids = [buscador.chunks[i]["id"] for i in indices]
        if q["chunks_relevantes"]:
            relevantes = set(q["chunks_relevantes"])
            recalls.append(len(relevantes & set(ids)) / len(relevantes))
            rrs.append(next((1 / p for p, i in enumerate(ids, 1) if i in relevantes), 0.0))
        recusa = guardrail.verificar(q["pergunta"], [buscador.chunks[i] for i in indices])
        if recusa and q["deve_recusar"]:
            recusas_armadilha += 1
        elif recusa:
            recusas_indevidas += 1
    return {
        "recall@5": round(float(np.mean(recalls)), 3),
        "mrr": round(float(np.mean(rrs)), 3),
        "n_perguntas_com_relevantes": len(recalls),
        "armadilhas_recusadas_pelo_guardrail": recusas_armadilha,
        "recusas_indevidas_do_guardrail": recusas_indevidas,
    }


def julgar(q: dict, r: dict) -> dict:
    ferramentas = [
        (f["nome"], f["argumentos"], json.dumps(f["resultado"], ensure_ascii=False))
        for f in r["ferramentas"]
    ]
    dados = r["contexto"] + "".join(f"\nFerramenta {n}({a}): {res}" for n, a, res in ferramentas)
    prompt = PROMPT_JUIZ.format(
        pergunta=q["pergunta"], esperada=q["resposta_esperada"], dados=dados, resposta=r["resposta"]
    )
    saida = completar([{"role": "user", "content": prompt}], MODELO_JUIZ)
    m = re.search(r"\{.*\}", saida["texto"], flags=re.DOTALL)
    try:
        return (
            json.loads(m.group(0)) if m else {"correta": None, "fiel": None, "motivo": "sem JSON"}
        )
    except json.JSONDecodeError:
        return {"correta": None, "fiel": None, "motivo": "JSON inválido"}


def avaliar_pergunta(q: dict, buscador: Buscador) -> dict:
    r = responder(q["pergunta"], buscador, "hibrida", K)
    linha = {
        **q,
        **{
            k: r[k]
            for k in (
                "resposta",
                "chunks",
                "citados",
                "ferramentas",
                "recusou",
                "guardrail",
                "latencia_s",
                "tokens",
                "chamadas_llm",
            )
        },
    }
    if q["deve_recusar"]:
        linha["correta"] = r["recusou"]
        linha["fiel"] = True if r["recusou"] else None
    elif r["recusou"]:
        linha["correta"], linha["fiel"] = False, True
    else:
        julgamento = julgar(q, r)
        linha.update(
            correta=julgamento.get("correta"),
            fiel=julgamento.get("fiel"),
            motivo_juiz=julgamento.get("motivo"),
        )
    print(
        f"[{q['id']:2}] {q['tipo']:9} correta={linha['correta']} recusou={r['recusou']}", flush=True
    )
    return linha


def main() -> None:
    perguntas = [json.loads(linha) for linha in open(RAIZ / "data/avaliacao.jsonl")]
    buscador = Buscador()

    busca = {modo: metricas_busca(perguntas, buscador, modo) for modo in ("densa", "hibrida")}
    (RAIZ / "reports/busca.json").write_text(json.dumps(busca, indent=2))
    print(json.dumps(busca, indent=2))

    buscador.densa("aquecimento")  # carrega o modelo antes das threads
    with ThreadPoolExecutor(max_workers=4) as executor:
        linhas = list(executor.map(lambda q: avaliar_pergunta(q, buscador), perguntas))
    with open(RAIZ / "reports/respostas.jsonl", "w") as f:
        for linha in linhas:
            f.write(
                json.dumps(
                    {k: v for k, v in linha.items() if k != "ferramentas"}
                    | {"ferramentas": [x["nome"] for x in linha["ferramentas"]]},
                    ensure_ascii=False,
                )
                + "\n"
            )

    armadilhas = [x for x in linhas if x["deve_recusar"]]
    normais = [x for x in linhas if not x["deve_recusar"]]
    respondidas = [x for x in normais if not x["recusou"]]
    lat = [x["latencia_s"] for x in linhas]
    por_tipo = {}
    for tipo in dict.fromkeys(x["tipo"] for x in normais):
        grupo = [x for x in normais if x["tipo"] == tipo]
        por_tipo[tipo] = round(sum(bool(x["correta"]) for x in grupo) / len(grupo), 3)
    resumo = {
        "modelo_gerador": MODELO_GERADOR,
        "modelo_juiz": MODELO_JUIZ,
        "busca": busca,
        "armadilhas_recusadas": f"{sum(x['recusou'] for x in armadilhas)}/{len(armadilhas)}",
        "recusas_indevidas": f"{sum(x['recusou'] for x in normais)}/{len(normais)}",
        "acuracia_respostas_normais": round(
            sum(bool(x["correta"]) for x in normais) / len(normais), 3
        ),
        "acuracia_por_tipo": por_tipo,
        "fidelidade": round(
            sum(bool(x["fiel"]) for x in respondidas) / max(len(respondidas), 1), 3
        ),
        "uso_de_ferramenta_nas_perguntas_de_filtro_e_pedido": round(
            sum(bool(x["ferramentas"]) for x in normais if x["tipo"] in ("filtro", "pedido"))
            / sum(x["tipo"] in ("filtro", "pedido") for x in normais),
            3,
        ),
        "latencia_p50_s": round(float(np.percentile(lat, 50)), 2),
        "latencia_p95_s": round(float(np.percentile(lat, 95)), 2),
        "tokens_medios_por_pergunta": round(float(np.mean([x["tokens"] for x in linhas])), 0),
    }
    (RAIZ / "reports/resumo.json").write_text(json.dumps(resumo, indent=2, ensure_ascii=False))
    print(json.dumps(resumo, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
