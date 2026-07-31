"""
Camada fina sobre a API REST do GitHub, usada pelo pipeline de auto-atualização
da Liss (ver `executar_auto_atualizacao` em main.py).

Variáveis de ambiente necessárias:
  GITHUB_TOKEN  — Personal Access Token com escopo "repo" (ou "contents" se
                  for um token fine-grained). NUNCA coloque isso em código.
  GITHUB_REPO   — "usuario/repositorio", ex: "gatao/liss"
  GITHUB_BRANCH — branch alvo dos commits (padrão: "main")
"""
import os
import base64
import requests

GITHUB_API = "https://api.github.com"
_TIMEOUT = 25


def _headers() -> dict:
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise RuntimeError("GITHUB_TOKEN não configurado no ambiente.")
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def _repo() -> str:
    repo = os.environ.get("GITHUB_REPO")
    if not repo:
        raise RuntimeError("GITHUB_REPO não configurado (formato: usuario/repositorio).")
    return repo


def _branch() -> str:
    return os.environ.get("GITHUB_BRANCH", "main")


def github_configurado() -> bool:
    return bool(os.environ.get("GITHUB_TOKEN") and os.environ.get("GITHUB_REPO"))


def github_list_arquivos() -> list[str]:
    """Lista (recursivamente) todos os caminhos de arquivo do branch alvo."""
    url_ref = f"{GITHUB_API}/repos/{_repo()}/git/ref/heads/{_branch()}"
    r = requests.get(url_ref, headers=_headers(), timeout=_TIMEOUT)
    r.raise_for_status()
    commit_sha = r.json()["object"]["sha"]

    url_tree = f"{GITHUB_API}/repos/{_repo()}/git/trees/{commit_sha}"
    r = requests.get(url_tree, headers=_headers(), params={"recursive": "1"}, timeout=_TIMEOUT)
    r.raise_for_status()
    return [item["path"] for item in r.json().get("tree", []) if item["type"] == "blob"]


def github_read_file(path: str) -> tuple[str | None, str | None]:
    """Retorna (conteudo, sha) ou (None, None) se o arquivo não existir."""
    url = f"{GITHUB_API}/repos/{_repo()}/contents/{path}"
    r = requests.get(url, headers=_headers(), params={"ref": _branch()}, timeout=_TIMEOUT)
    if r.status_code == 404:
        return None, None
    r.raise_for_status()
    data = r.json()
    conteudo = base64.b64decode(data["content"]).decode("utf-8")
    return conteudo, data["sha"]


def github_write_file(path: str, conteudo: str, mensagem: str, sha: str | None = None) -> dict:
    """Cria (sha=None) ou atualiza (sha informado) um arquivo em um único commit."""
    url = f"{GITHUB_API}/repos/{_repo()}/contents/{path}"
    payload = {
        "message": mensagem,
        "content": base64.b64encode(conteudo.encode("utf-8")).decode("utf-8"),
        "branch": _branch(),
    }
    if sha:
        payload["sha"] = sha
    r = requests.put(url, headers=_headers(), json=payload, timeout=_TIMEOUT)
    r.raise_for_status()
    return r.json()


def github_delete_file(path: str, sha: str, mensagem: str) -> dict:
    url = f"{GITHUB_API}/repos/{_repo()}/contents/{path}"
    payload = {"message": mensagem, "sha": sha, "branch": _branch()}
    r = requests.delete(url, headers=_headers(), json=payload, timeout=_TIMEOUT)
    r.raise_for_status()
    return r.json()
