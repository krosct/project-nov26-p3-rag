"""Ferramentas que o LLM pode pedir para executar.

Filtros por preço e alergênico são consultas exatas sobre dados estruturados:
busca vetorial pode trazer um prato de R$ 45 "parecido" com "até R$ 40".
"""

import json

from project_nov26_p3_rag.chunks import DADOS

ALERGENOS = ["gluten", "lactose", "ovo", "amendoim", "castanhas", "soja", "peixe", "frutos_do_mar"]
CATEGORIAS = [
    "entrada",
    "prato_principal",
    "lanche",
    "pizza",
    "sobremesa",
    "bebida",
    "combo",
    "outro",
]
LIMITE_RESULTADOS = 15


def _itens() -> list[dict]:
    itens = []
    for arquivo in sorted((DADOS / "cardapios").glob("*.json")):
        menu = json.loads(arquivo.read_text())
        itens += [{"restaurante": menu["restaurante"], **item} for item in menu["itens"]]
    return itens


def filtrar_pratos(
    preco_max: float | None = None,
    sem_alergeno: str | None = None,
    categoria: str | None = None,
    vegetariano: bool | None = None,
    restaurante: str | None = None,
) -> dict:
    """Filtra itens. Com `sem_alergeno`, itens sem informação de alergênicos são EXCLUÍDOS."""
    resultados, sem_informacao = [], 0
    for item in _itens():
        if preco_max is not None and (item["preco"] is None or item["preco"] > preco_max):
            continue
        if categoria and item["categoria"] != categoria:
            continue
        if vegetariano is not None and item.get("vegetariano") is not vegetariano:
            continue
        if restaurante and restaurante.lower() not in item["restaurante"].lower():
            continue
        if sem_alergeno:
            if item["alergenos"] is None:
                sem_informacao += 1
                continue
            if sem_alergeno in item["alergenos"]:
                continue
        resultados.append(
            {k: item.get(k) for k in ("restaurante", "nome", "preco", "categoria", "alergenos")}
        )
    resultados.sort(key=lambda i: i["preco"])
    resposta = {"total": len(resultados), "itens": resultados[:LIMITE_RESULTADOS]}
    if sem_alergeno:
        resposta["excluidos_sem_informacao_de_alergenos"] = sem_informacao
    return resposta


def consultar_pedido(id_pedido: str) -> dict:
    pedidos = json.loads((DADOS / "pedidos.json").read_text())
    for pedido in pedidos:
        if pedido["id"] == str(id_pedido).strip().lstrip("#"):
            return pedido
    return {"erro": f"pedido {id_pedido} não encontrado"}


FUNCOES = {"filtrar_pratos": filtrar_pratos, "consultar_pedido": consultar_pedido}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "filtrar_pratos",
            "description": "Lista itens de todos os cardápios que atendem a filtros exatos de preço, "  # noqa: E501
            "alergênico, categoria, vegetariano ou restaurante. Use para perguntas como "
            "'pratos até R$ 40', 'sobremesas sem lactose', 'opções vegetarianas'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "preco_max": {"type": "number", "description": "Preço máximo em reais"},
                    "sem_alergeno": {"type": "string", "enum": ALERGENOS},
                    "categoria": {"type": "string", "enum": CATEGORIAS},
                    "vegetariano": {"type": "boolean"},
                    "restaurante": {"type": "string", "description": "Parte do nome"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "consultar_pedido",
            "description": "Consulta status, itens e previsão de entrega de um pedido pelo número.",
            "parameters": {
                "type": "object",
                "properties": {"id_pedido": {"type": "string"}},
                "required": ["id_pedido"],
            },
        },
    },
]


def executar(nome: str, argumentos: str) -> dict:
    if nome not in FUNCOES:
        return {"erro": f"ferramenta desconhecida: {nome}"}
    try:
        kwargs = json.loads(argumentos or "{}")
        return FUNCOES[nome](**kwargs)
    except (TypeError, ValueError) as erro:
        return {"erro": f"argumentos inválidos: {erro}"}
