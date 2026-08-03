"""
Módulo de autoevolução da Liss.

Fluxo:
  1. Groq (modelo de código) gera o(s) arquivo(s) atualizado(s) a partir da
     instrução + código-fonte atual.
  2. Sandbox local isolado testa: sintaxe (ast + py_compile) e depois sobe o
     servidor de verdade num processo filho isolado, numa porta descartável,
     pra pegar erro de import/execução que só aparece em runtime.
  3. Só se passar em tudo: cria uma branch de checkpoint no GitHub, commita,
     abre um Pull Request pra main e faz o merge automático. O merge dispara
     o deploy automático nativo do Render.
  4. Se qualquer etapa falhar, NADA chega ao GitHub. O log de erro do sandbox
     alimenta a próxima tentativa (até MAX_TENTATIVAS), pra ela tentar se
     corrigir sozinha antes de desistir.

Nota sobre timeout do sandbox: main.py sobe um servidor web, que por design
fica rodando pra sempre. Por isso a regra aqui é a oposta de "timeout =
sempre falha": se o processo MORRE SOZINHO dentro do timeout, é crash e
reprova; se ele CONTINUA DE PÉ até o timeout, é sinal de que subiu com
sucesso e aprova (o processo de teste é então encerrado).
"""
import os
import re
import ast
import json
import random
import logging
import tempfile
import subprocess
from pathlib import Path
from typing import Optional

from groq import Groq
from github import Github, GithubException

log = logging.getLogger("liss.autoevolucao")

# ---------------------------------------------------------------------------
# Modelos Groq para geração de código. "qwen-2.5-coder-32b" nunca esteve no
# catálogo da Groq; "llama-3.3-70b-versatile" foi descontinuado pela Groq
# (anúncio 17/jun/2026, desligamento em ago/2026). Os dois abaixo são a
# recomendação atual da própria Groq para workloads de código/raciocínio.
# Ajustável sem editar código via a env var GROQ_CODE_MODEL.
# ---------------------------------------------------------------------------
CASCATA_MODELOS_GROQ = [
    os.environ.get("GROQ_CODE_MODEL", "openai/gpt-oss-120b"),
    "openai/gpt-oss-20b",
]

MAX_TENTATIVAS = 3
TIMEOUT_SANDBOX_SEGUNDOS = 10
ARQUIVOS_BASE_CONTEXTO = ("main.py", "index.html", "requirements.txt", "github_ops.py", "autoevolucao.py")

PROMPT_SISTEMA = """Você é o módulo de autoevolução da Liss: uma IA que reescreve o
próprio código-fonte. Você recebe o conteúdo atual dos arquivos do projeto,
uma instrução de quem criou o projeto, e — se uma tentativa anterior falhou —
o log de erro exato dessa tentativa.

Responda SOMENTE com um JSON válido, sem cercas de markdown, sem texto fora do JSON:
{
  "commit_message": "mensagem curta em português descrevendo a mudança",
  "arquivos": {
    "caminho/arquivo.py": "conteúdo COMPLETO do arquivo já atualizado"
  }
}

Regras obrigatórias:
- Cada valor em "arquivos" é o ARQUIVO INTEIRO, nunca um diff ou trecho.
- Só inclua arquivos que realmente precisam mudar.
- Nunca escreva segredos, tokens ou chaves de API no código — sempre leia de
  variáveis de ambiente via os.environ.
- O código Python precisa compilar e executar sem erro de import.
- Se recebeu um log de erro de tentativa anterior, corrija exatamente esse
  problema — não repita o mesmo erro.
- Mantenha o estilo, os comentários em português e a arquitetura já usados no
  projeto, a menos que a instrução peça o contrário.
"""


class Autoevolucao:
    """Orquestra o ciclo geração (Groq) -> validação (sandbox) -> deploy (GitHub)."""

    def __init__(self, github_ops_module):
        # Reaproveita github_ops.py só pra LEITURA de contexto (já testado);
        # escrita/branch/PR/merge aqui usam PyGithub, como pedido.
        self._github_ops = github_ops_module
        self._groq = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self._gh_client: Optional[Github] = None

    # ------------------------------------------------------------------ util
    @staticmethod
    def _extrair_json(texto: str) -> dict:
        texto = texto.strip()
        texto = re.sub(r"^```(json)?|```$", "", texto, flags=re.MULTILINE).strip()
        return json.loads(texto)

    def _repo_pygithub(self):
        if self._gh_client is None:
            token = os.environ.get("GITHUB_TOKEN")
            if not token:
                raise RuntimeError("GITHUB_TOKEN não configurado no ambiente.")
            self._gh_client = Github(token)
        nome_repo = os.environ.get("GITHUB_REPO")
        if not nome_repo:
            raise RuntimeError("GITHUB_REPO não configurado (formato: usuario/repositorio).")
        return self._gh_client.get_repo(nome_repo)

    # ---------------------------------------------------------- 1) geração
    def _gerar_codigo(self, instrucao: str, contexto_arquivos: dict, log_erro_anterior: Optional[str]) -> dict:
        prompt = (
            "ARQUIVOS ATUAIS:\n" + json.dumps(contexto_arquivos, ensure_ascii=False)
            + "\n\nINSTRUÇÃO:\n" + instrucao
        )
        if log_erro_anterior:
            prompt += (
                "\n\nA TENTATIVA ANTERIOR FALHOU NO SANDBOX COM ESTE ERRO — CORRIJA "
                "EXATAMENTE ISSO:\n" + log_erro_anterior
            )

        ultimo_erro = None
        for modelo in CASCATA_MODELOS_GROQ:
            try:
                resposta = self._groq.chat.completions.create(
                    model=modelo,
                    messages=[
                        {"role": "system", "content": PROMPT_SISTEMA},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.2,
                )
                texto = resposta.choices[0].message.content
                return self._extrair_json(texto)
            except Exception as e:
                log.warning(f"⚠️ Groq falhou com {modelo}: {e}")
                ultimo_erro = e
        raise ultimo_erro or RuntimeError("Nenhum modelo Groq respondeu.")

    # ---------------------------------------------------------- 2) sandbox
    def _testar_sandbox(self, contexto_arquivos: dict, arquivos_novos: dict) -> Optional[str]:
        """Retorna None se passou em tudo, ou uma string de log de erro se falhou."""
        merged = {**contexto_arquivos, **arquivos_novos}

        with tempfile.TemporaryDirectory(prefix="liss_sandbox_") as tmp:
            tmp_path = Path(tmp)
            for caminho, conteudo in merged.items():
                destino = tmp_path / caminho
                destino.parent.mkdir(parents=True, exist_ok=True)
                destino.write_text(conteudo, encoding="utf-8")

            # 2a) sintaxe de cada arquivo NOVO/MODIFICADO
            for caminho, conteudo in arquivos_novos.items():
                if caminho.endswith(".py"):
                    try:
                        ast.parse(conteudo)
                    except SyntaxError as e:
                        return f"Erro de sintaxe em {caminho}, linha {e.lineno}: {e.msg}"

            # 2b) py_compile isolado em processo filho, com timeout
            for caminho in arquivos_novos:
                if caminho.endswith(".py"):
                    try:
                        resultado = subprocess.run(
                            ["python3", "-m", "py_compile", str(tmp_path / caminho)],
                            capture_output=True, text=True, timeout=TIMEOUT_SANDBOX_SEGUNDOS,
                        )
                    except subprocess.TimeoutExpired:
                        return f"py_compile travou (timeout) em {caminho}"
                    if resultado.returncode != 0:
                        return f"py_compile falhou em {caminho}:\n{resultado.stderr[-1500:]}"

            # 2c) sobe o servidor de verdade, isolado, numa porta descartável.
            # Morreu sozinho dentro do timeout = crash (reprova).
            # Continuou de pé até o timeout = subiu com sucesso (aprova).
            if "main.py" in merged:
                erro_smoke = self._smoke_test_processo(tmp_path)
                if erro_smoke:
                    return erro_smoke

        return None

    @staticmethod
    def _smoke_test_processo(tmp_path: Path) -> Optional[str]:
        porta_teste = random.randint(20000, 40000)
        env = os.environ.copy()
        env["PORT"] = str(porta_teste)
        env.pop("AUTO_RESTART_LOCAL", None)  # o sandbox nunca reinicia o processo real
        try:
            processo = subprocess.Popen(
                ["python3", "main.py"], cwd=str(tmp_path), env=env,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            )
        except Exception as e:
            return f"Não consegui nem iniciar o processo de teste: {e}"

        try:
            _, stderr = processo.communicate(timeout=TIMEOUT_SANDBOX_SEGUNDOS)
            # comunicate() retornou = o processo já terminou sozinho = crash
            return f"O servidor caiu sozinho no teste (código {processo.returncode}):\n{stderr[-1500:]}"
        except subprocess.TimeoutExpired:
            processo.kill()
            processo.wait()
            return None  # ainda de pé no timeout = subiu com sucesso

    # ------------------------------------------------- 3) branch + PR + merge
    def _publicar_no_github(self, arquivos: dict, commit_message: str) -> str:
        repo = self._repo_pygithub()
        branch_base = os.environ.get("GITHUB_BRANCH", "main")
        nome_branch = f"auto-update-{os.urandom(4).hex()}"

        sha_base = repo.get_branch(branch_base).commit.sha
        repo.create_git_ref(ref=f"refs/heads/{nome_branch}", sha=sha_base)

        for caminho, conteudo in arquivos.items():
            try:
                arquivo_atual = repo.get_contents(caminho, ref=nome_branch)
                repo.update_file(
                    caminho, f"{commit_message} ({caminho})", conteudo,
                    arquivo_atual.sha, branch=nome_branch,
                )
            except GithubException as e:
                if e.status == 404:
                    repo.create_file(caminho, f"{commit_message} ({caminho})", conteudo, branch=nome_branch)
                else:
                    raise

        pr = repo.create_pull(
            title=f"🤖 {commit_message}",
            body="Auto-atualização gerada pela Liss (Groq) e validada em sandbox isolado antes do merge.",
            head=nome_branch,
            base=branch_base,
        )
        pr.merge(commit_message=commit_message, merge_method="squash")
        return pr.html_url

    # ------------------------------------------------------------ pipeline
    def evoluir(self, instrucao: str) -> list[str]:
        """Executa o ciclo completo. Retorna uma lista de mensagens de progresso."""
        logs: list[str] = []

        contexto_arquivos = {}
        for caminho in ARQUIVOS_BASE_CONTEXTO:
            conteudo, _ = self._github_ops.github_read_file(caminho)
            if conteudo is not None:
                contexto_arquivos[caminho] = conteudo

        log_erro_anterior: Optional[str] = None

        for tentativa in range(1, MAX_TENTATIVAS + 1):
            logs.append(f"🧠 Tentativa {tentativa}/{MAX_TENTATIVAS}: gerando código com a Groq…")
            try:
                plano = self._gerar_codigo(instrucao, contexto_arquivos, log_erro_anterior)
            except Exception as e:
                logs.append(f"⚠️ A Groq não respondeu: {e}")
                return logs

            arquivos_novos = plano.get("arquivos", {})
            commit_message = plano.get("commit_message", "Auto-atualização via Liss")

            if not arquivos_novos:
                logs.append("Não encontrei nenhuma mudança de código concreta pra essa instrução.")
                return logs

            logs.append(f"🧪 Testando {len(arquivos_novos)} arquivo(s) em sandbox isolado…")
            try:
                erro = self._testar_sandbox(contexto_arquivos, arquivos_novos)
            except Exception as e:
                erro = f"Erro inesperado rodando o sandbox: {e}"

            if erro is None:
                logs.append("✅ Passou no sandbox (sintaxe + processo isolado real).")
                try:
                    url_pr = self._publicar_no_github(arquivos_novos, commit_message)
                    logs.append(f"🔀 Checkpoint criado, PR aberto e merge feito: {url_pr}")
                    logs.append("🚀 O merge na main deve disparar o deploy automático do Render agora.")
                except Exception as e:
                    logs.append(f"⚠️ Passou no sandbox, mas o GitHub recusou o merge: {e}")
                return logs

            logs.append(f"❌ Sandbox reprovou: {erro}")
            log_erro_anterior = erro

        logs.append(f"⚠️ Não consegui uma versão que passasse no sandbox depois de {MAX_TENTATIVAS} tentativas. Nada foi commitado.")
        return logs
