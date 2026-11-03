"""Guardrail de alergênicos, em código (determinístico e testável).

Se a pergunta trata de alergia a um item específico e o cardápio não informa
os alergênicos desse item, o LLM não é chamado: a resposta é uma recusa fixa.
Perguntas de filtro ("quais pratos sem glúten?") seguem para a ferramenta
`filtrar_pratos`, que exclui itens sem informação.
"""

import re

from project_nov26_p3_rag.busca import tokenizar

TERMOS_ALERGIA = {
    "alergia",
    "alergico",
    "alergica",
    "alergicos",
    "alergenos",
    "alergeno",
    "celiaco",
    "celiaca",
    "gluten",
    "lactose",
    "intolerante",
    "intolerancia",
    "amendoim",
    "castanha",
    "castanhas",
    "nozes",
    "camarao",
    "frutos",
    "ovo",
    "ovos",
    "soja",
    "leite",
    "trigo",
    "peixe",
}
INTENCAO_FILTRO = re.compile(
    r"\b(quais|qual\s+\w+\s+(?:nao|sem)|opcoes|lista|liste|algum|alguma|alguns|recomend|indique|sugest)"
)
MARCADOR_RECUSA = "não informa os alergênicos"


def menciona_alergia(pergunta: str) -> bool:
    return bool(TERMOS_ALERGIA & set(tokenizar(pergunta)))


def eh_filtro(pergunta: str) -> bool:
    return bool(INTENCAO_FILTRO.search(" ".join(tokenizar(pergunta))))


def verificar(pergunta: str, chunks: list[dict]) -> str | None:
    """Retorna a recusa fixa, ou None se a pergunta pode seguir para o LLM."""
    if not menciona_alergia(pergunta) or eh_filtro(pergunta):
        return None
    itens = [c for c in chunks if c["tipo"] == "item"]
    if not itens:
        return None
    principal = itens[0]
    if principal["meta"].get("alergenos") is None:
        meta = principal["meta"]
        return (
            f'O cardápio do {meta["restaurante"]} {MARCADOR_RECUSA} de "{meta["nome"]}". '
            "Por segurança, não consigo confirmar se ele é adequado para sua restrição: "
            "confirme os ingredientes diretamente com o restaurante antes de pedir."
        )
    return None
