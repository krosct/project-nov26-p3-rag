"""Orquestra: busca → guardrail → LLM com ferramentas → resposta com citações."""

import json
import re
import time

from project_nov26_p3_rag import guardrail
from project_nov26_p3_rag.busca import Buscador
from project_nov26_p3_rag.ferramentas import TOOLS, executar
from project_nov26_p3_rag.llm import MODELO_GERADOR, completar

RECUSA_PADRAO = "Não encontrei essa informação nos cardápios."
MAX_VOLTAS = 3

SISTEMA = f"""Você é o assistente de cardápios de uma plataforma de delivery.

Regras:
1. Responda em português usando SOMENTE o contexto numerado e os resultados das ferramentas.
2. Cite a fonte de cada informação do contexto com [n].
3. Se a resposta não estiver no contexto nem nas ferramentas, responda exatamente: "{RECUSA_PADRAO}"
4. Alergias: nunca afirme que um item é seguro se os alérgenos dele constam como "não informados";
   nesse caso, diga que o cardápio não informa e recomende confirmar com o restaurante.
5. Para listar itens por preço máximo, alergênico, categoria ou vegetariano, use a ferramenta
   filtrar_pratos. Para status de pedido, use consultar_pedido.
6. Seja breve: no máximo 5 frases ou uma lista curta."""


def montar_contexto(chunks: list[dict]) -> str:
    return "\n".join(f"[{i}] {c['texto']}" for i, c in enumerate(chunks, start=1))


def responder(pergunta: str, buscador: Buscador, modo: str = "hibrida", k: int = 5) -> dict:
    inicio = time.perf_counter()
    chunks = buscador.buscar(pergunta, modo, k)
    base = {
        "chunks": [c["id"] for c in chunks],
        "contexto": montar_contexto(chunks),
        "ferramentas": [],
        "tokens": 0,
        "chamadas_llm": 0,
    }

    recusa = guardrail.verificar(pergunta, chunks)
    if recusa:
        return {
            **base,
            "resposta": recusa,
            "citados": [],
            "recusou": True,
            "guardrail": True,
            "latencia_s": round(time.perf_counter() - inicio, 3),
        }

    mensagens = [
        {"role": "system", "content": SISTEMA},
        {
            "role": "user",
            "content": f"Contexto:\n{montar_contexto(chunks)}\n\nPergunta: {pergunta}",
        },
    ]
    resposta = ""
    for _ in range(MAX_VOLTAS):
        r = completar(mensagens, MODELO_GERADOR, tools=TOOLS)
        base["chamadas_llm"] += 1
        base["tokens"] += r["tokens_entrada"] + r["tokens_saida"]
        if not r["tool_calls"]:
            resposta = r["texto"].strip()
            break
        mensagens.append(
            {
                "role": "assistant",
                "content": r["texto"] or None,
                "tool_calls": [
                    {
                        "id": c["id"],
                        "type": "function",
                        "function": {"name": c["nome"], "arguments": c["argumentos"]},
                    }
                    for c in r["tool_calls"]
                ],
            }
        )
        for c in r["tool_calls"]:
            resultado = executar(c["nome"], c["argumentos"])
            base["ferramentas"].append(
                {"nome": c["nome"], "argumentos": c["argumentos"], "resultado": resultado}
            )
            mensagens.append(
                {
                    "role": "tool",
                    "tool_call_id": c["id"],
                    "content": json.dumps(resultado, ensure_ascii=False),
                }
            )
    else:
        resposta = RECUSA_PADRAO

    citados = sorted(
        {int(n) for n in re.findall(r"\[(\d+)\]", resposta) if 0 < int(n) <= len(chunks)}
    )
    return {
        **base,
        "resposta": resposta,
        "citados": [chunks[n - 1]["id"] for n in citados],
        "recusou": RECUSA_PADRAO.lower().rstrip(".") in resposta.lower()
        or guardrail.MARCADOR_RECUSA in resposta.lower(),
        "guardrail": False,
        "latencia_s": round(time.perf_counter() - inicio, 3),
    }
