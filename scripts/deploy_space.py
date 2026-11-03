"""Publica a demo no Hugging Face Spaces.

Uso: HF_TOKEN=... OPENROUTER_API_KEY=... uv run --group app python scripts/deploy_space.py
A chave do LLM vai como *secret* do Space (nunca como arquivo).
"""

import os
from pathlib import Path

from huggingface_hub import HfApi

RAIZ = Path(__file__).resolve().parents[1]
SPACE = "trebuchetly/assistente-cardapios"


def main() -> None:
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(SPACE, repo_type="space", space_sdk="gradio", exist_ok=True)
    api.add_space_secret(SPACE, "OPENROUTER_API_KEY", os.environ["OPENROUTER_API_KEY"])
    for origem, destino in [
        ("space/README.md", "README.md"),
        ("space/requirements.txt", "requirements.txt"),
        ("app.py", "app.py"),
    ]:
        api.upload_file(
            path_or_fileobj=str(RAIZ / origem),
            path_in_repo=destino,
            repo_id=SPACE,
            repo_type="space",
        )
    for pasta in ("src", "data", "index"):
        api.upload_folder(
            folder_path=str(RAIZ / pasta),
            path_in_repo=pasta,
            repo_id=SPACE,
            repo_type="space",
            ignore_patterns=["__pycache__/*", "*.pyc"],
        )
    print(f"https://huggingface.co/spaces/{SPACE}")


if __name__ == "__main__":
    main()
