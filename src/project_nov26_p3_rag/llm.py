"""Cliente de LLM (API compatível com a OpenAI) com suporte a tool calling e cache em disco."""

import hashlib
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import APIConnectionError, APIStatusError, OpenAI, RateLimitError

load_dotenv()

BASE_URL = os.environ.get("LLM_BASE_URL", "https://openrouter.ai/api/v1")
MODELO_GERADOR = os.environ.get("LLM_MODEL", "nvidia/nemotron-3-ultra-550b-a55b:free")
MODELO_JUIZ = os.environ.get("LLM_JUDGE_MODEL", "dots-studio/dots-3-note-preview:free")
CACHE = Path(os.environ.get("LLM_CACHE_DIR", ".cache/llm"))


def _client() -> OpenAI:
    return OpenAI(base_url=BASE_URL, api_key=os.environ["OPENROUTER_API_KEY"])


def completar(
    mensagens: list[dict],
    modelo: str = MODELO_GERADOR,
    tools: list[dict] | None = None,
    usar_cache: bool = True,
    max_tentativas: int = 6,
) -> dict:
    """Retorna {texto, tool_calls, latencia_s, tokens_entrada, tokens_saida}."""
    chave = hashlib.sha256(json.dumps([mensagens, modelo, tools], sort_keys=True).encode())
    arquivo = CACHE / f"{chave.hexdigest()}.json"
    if usar_cache and arquivo.exists():
        return json.loads(arquivo.read_text())

    extra = {"tools": tools} if tools else {}
    for tentativa in range(max_tentativas):
        try:
            inicio = time.perf_counter()
            resp = _client().chat.completions.create(
                model=modelo, messages=mensagens, temperature=0, **extra
            )
            msg = resp.choices[0].message
            chamadas = [
                {"id": c.id, "nome": c.function.name, "argumentos": c.function.arguments}
                for c in (msg.tool_calls or [])
            ]
            if not msg.content and not chamadas:
                raise ValueError("resposta vazia")
            resultado = {
                "texto": msg.content or "",
                "tool_calls": chamadas,
                "latencia_s": round(time.perf_counter() - inicio, 3),
                "tokens_entrada": resp.usage.prompt_tokens if resp.usage else 0,
                "tokens_saida": resp.usage.completion_tokens if resp.usage else 0,
            }
            break
        except (RateLimitError, APIConnectionError, APIStatusError, ValueError):
            if tentativa == max_tentativas - 1:
                raise
            time.sleep(min(2 ** (tentativa + 1), 60))

    if usar_cache:
        CACHE.mkdir(parents=True, exist_ok=True)
        arquivo.write_text(json.dumps(resultado, ensure_ascii=False))
    return resultado
