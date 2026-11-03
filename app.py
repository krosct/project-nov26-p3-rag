"""Demo Gradio do assistente de cardápios (Hugging Face Spaces).

Executar localmente: uv run --group app python app.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import gradio as gr  # noqa: E402

from project_nov26_p3_rag.assistente import responder  # noqa: E402
from project_nov26_p3_rag.busca import Buscador  # noqa: E402

LIMITE_POR_SESSAO = 10
EXEMPLOS = [
    "Quais combos custam até R$ 50?",
    "O Frango Xadrez do Dragão Dourado tem amendoim?",
    "O X-Tudo do Burger Brasa tem glúten?",
    "Qual o status do pedido 1002?",
    "Quais sobremesas sem lactose existem?",
    "Veio um item faltando no meu pedido, o que eu faço?",
]

buscador = Buscador()
textos = {c["id"]: c["texto"] for c in buscador.chunks}


def perguntar(pergunta: str, usadas: int):
    if not pergunta.strip():
        return "Digite uma pergunta.", "", usadas
    if usadas >= LIMITE_POR_SESSAO:
        return f"Limite de {LIMITE_POR_SESSAO} perguntas por sessão atingido.", "", usadas
    r = responder(pergunta, buscador)
    fontes = r["citados"] or r["chunks"][:3]
    detalhes = ["**Fontes citadas:**" if r["citados"] else "**Trechos recuperados:**"]
    detalhes += [f"- `{i}` — {textos[i]}" for i in fontes]
    if r["ferramentas"]:
        detalhes += ["", "**Ferramentas chamadas:**"]
        detalhes += [f"- `{f['nome']}({f['argumentos']})`" for f in r["ferramentas"]]
    if r["guardrail"]:
        detalhes += ["", "🛡️ Resposta bloqueada pelo guardrail de alergênicos (sem chamar o LLM)."]
    detalhes += ["", f"⏱️ {r['latencia_s']:.1f} s"]
    return r["resposta"], "\n".join(detalhes), usadas + 1


with gr.Blocks(title="Assistente de cardápios") as demo:
    gr.Markdown(
        "# 🍽️ Assistente de cardápios\n"
        "RAG híbrido (embeddings + BM25) sobre 20 cardápios fictícios e políticas de uma "
        "plataforma de delivery, com ferramentas (filtros exatos e consulta de pedidos) e "
        "guardrail de alergênicos. [Código e avaliação](https://github.com/krosct/project-nov26-p3-rag)"
    )
    usadas = gr.State(0)
    pergunta = gr.Textbox(label="Pergunta", placeholder="Ex.: Quais pizzas custam até R$ 50?")
    botao = gr.Button("Perguntar", variant="primary")
    resposta = gr.Markdown(label="Resposta")
    detalhes = gr.Markdown()
    gr.Examples(EXEMPLOS, inputs=pergunta)
    botao.click(perguntar, [pergunta, usadas], [resposta, detalhes, usadas])
    pergunta.submit(perguntar, [pergunta, usadas], [resposta, detalhes, usadas])

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=2).launch()
