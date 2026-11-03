"""Transforma cardápios e políticas em chunks: um por item e um por seção de política."""

import json
import re
from pathlib import Path

DADOS = Path(__file__).resolve().parents[2] / "data"


def _preco(valor: float | None) -> str:
    return "sob consulta" if valor is None else f"R$ {valor:.2f}".replace(".", ",")


def _alergenos(lista: list[str] | None) -> str:
    if lista is None:
        return "não informados pelo restaurante"
    return ", ".join(lista) if lista else "nenhum declarado"


def chunks_cardapios(pasta: Path = DADOS / "cardapios") -> list[dict]:
    chunks = []
    for arquivo in sorted(pasta.glob("*.json")):
        menu = json.loads(arquivo.read_text())
        prefixo = arquivo.stem.replace("_", "")  # menu_02 -> menu02
        for i, item in enumerate(menu["itens"], start=1):
            partes = [
                f"Restaurante: {menu['restaurante']}",
                f"Categoria: {item['categoria']}",
                f"Item: {item['nome']}",
                f"Preço: {_preco(item['preco'])}",
            ]
            if item.get("descricao"):
                partes.append(f"Descrição: {item['descricao']}")
            partes.append(f"Alérgenos: {_alergenos(item['alergenos'])}")
            if item.get("vegetariano") is not None:
                partes.append(f"Vegetariano: {'sim' if item['vegetariano'] else 'não'}")
            chunks.append(
                {
                    "id": f"{prefixo}-item{i:02d}",
                    "tipo": "item",
                    "texto": " | ".join(partes),
                    "meta": {"restaurante": menu["restaurante"], "fonte": arquivo.name, **item},
                }
            )
    return chunks


def chunks_politicas(pasta: Path = DADOS / "politicas") -> list[dict]:
    chunks = []
    for arquivo in sorted(pasta.glob("*.md")):
        texto = arquivo.read_text()
        titulo = texto.splitlines()[0].lstrip("# ").strip()
        secoes = re.split(r"^## ", texto, flags=re.MULTILINE)[1:]
        for i, secao in enumerate(secoes, start=1):
            subtitulo, _, corpo = secao.partition("\n")
            chunks.append(
                {
                    "id": f"politica-{arquivo.stem}-{i}",
                    "tipo": "politica",
                    "texto": f"Política: {titulo} — {subtitulo.strip()}. {' '.join(corpo.split())}",
                    "meta": {"fonte": arquivo.name},
                }
            )
    return chunks


def todos_os_chunks() -> list[dict]:
    return chunks_cardapios() + chunks_politicas()
