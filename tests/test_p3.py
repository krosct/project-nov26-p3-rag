import json

from project_nov26_p3_rag import assistente, guardrail
from project_nov26_p3_rag.busca import rrf, tokenizar
from project_nov26_p3_rag.chunks import chunks_cardapios, chunks_politicas
from project_nov26_p3_rag.ferramentas import consultar_pedido, executar, filtrar_pratos


def item_chunk(nome, alergenos, restaurante="R"):
    return {
        "id": nome,
        "tipo": "item",
        "texto": nome,
        "meta": {"restaurante": restaurante, "nome": nome, "alergenos": alergenos},
    }


# --- chunks e busca -------------------------------------------------------------


def test_um_chunk_por_item_com_restaurante_e_alergenos_no_texto():
    chunks = chunks_cardapios()
    assert len(chunks) == 153
    x_tudo = next(c for c in chunks if c["id"] == "menu03-item01")
    assert "Burger Brasa" in x_tudo["texto"]
    assert "não informados" in x_tudo["texto"]


def test_politicas_divididas_por_secao():
    ids = {c["id"] for c in chunks_politicas()}
    assert {"politica-reembolso-1", "politica-taxa_entrega-2"} <= ids


def test_tokenizar_remove_acentos():
    assert tokenizar("Pão de Açúcar!") == ["pao", "de", "acucar"]


def test_rrf_prioriza_consenso_entre_rankings():
    assert rrf([1, 2, 3], [3, 1, 4])[:2] == [1, 3]


# --- ferramentas ----------------------------------------------------------------


def test_filtro_por_alergeno_exclui_itens_sem_informacao():
    r = filtrar_pratos(restaurante="Burger Brasa", sem_alergeno="gluten")
    assert r["total"] == 0
    assert r["excluidos_sem_informacao_de_alergenos"] == 9


def test_filtro_por_preco_e_categoria():
    r = filtrar_pratos(preco_max=50, categoria="combo")
    assert [i["nome"] for i in r["itens"]] == ["Combo Feira", "Combo Brasa"]


def test_consultar_pedido():
    assert consultar_pedido("#1002")["status"] == "saiu para entrega"
    assert "erro" in consultar_pedido("9999")


def test_executar_argumentos_invalidos_nao_quebra():
    assert "erro" in executar("filtrar_pratos", '{"inexistente": 1}')
    assert "erro" in executar("apagar_tudo", "{}")


# --- guardrail ------------------------------------------------------------------


def test_guardrail_recusa_item_sem_informacao():
    recusa = guardrail.verificar("O X-Tudo tem glúten?", [item_chunk("X-Tudo", None)])
    assert recusa and guardrail.MARCADOR_RECUSA in recusa


def test_guardrail_deixa_passar_item_com_informacao_e_filtros():
    assert (
        guardrail.verificar(
            "O Frango Xadrez tem amendoim?", [item_chunk("Frango Xadrez", ["amendoim"])]
        )
        is None
    )
    assert guardrail.verificar("Quais pratos sem glúten?", [item_chunk("X-Tudo", None)]) is None
    assert guardrail.verificar("Quanto custa o X-Tudo?", [item_chunk("X-Tudo", None)]) is None


# --- assistente (LLM e busca falsos) ----------------------------------------------


class BuscadorFalso:
    def buscar(self, pergunta, modo, k):
        return [item_chunk("Hot Roll", ["peixe"], "Sushi Kento")]


def test_assistente_executa_ferramenta_e_cita_fonte(monkeypatch):
    respostas = iter(
        [
            {
                "texto": "",
                "tool_calls": [
                    {"id": "c1", "nome": "consultar_pedido", "argumentos": '{"id_pedido": "1002"}'}
                ],
                "latencia_s": 0,
                "tokens_entrada": 10,
                "tokens_saida": 5,
            },
            {
                "texto": "Seu pedido saiu para entrega. O Hot Roll custa R$ 32,90 [1].",
                "tool_calls": [],
                "latencia_s": 0,
                "tokens_entrada": 20,
                "tokens_saida": 10,
            },
        ]
    )
    vistas = []

    def llm_falso(mensagens, modelo, tools=None):
        vistas.append(json.loads(json.dumps(mensagens)))
        return next(respostas)

    monkeypatch.setattr(assistente, "completar", llm_falso)
    r = assistente.responder("Status do pedido 1002?", BuscadorFalso())
    assert r["ferramentas"][0]["resultado"]["status"] == "saiu para entrega"
    assert vistas[1][-1]["role"] == "tool"
    assert r["citados"] == ["Hot Roll"]
    assert r["recusou"] is False


def test_assistente_detecta_recusa(monkeypatch):
    monkeypatch.setattr(
        assistente,
        "completar",
        lambda *a, **k: {
            "texto": assistente.RECUSA_PADRAO,
            "tool_calls": [],
            "latencia_s": 0,
            "tokens_entrada": 1,
            "tokens_saida": 1,
        },
    )
    assert assistente.responder("Vocês vendem enguia?", BuscadorFalso())["recusou"] is True
