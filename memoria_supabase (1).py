"""
Gerenciador de memória de longo prazo da Liss via Supabase.

Antes de usar, rode isso UMA VEZ no SQL Editor do seu projeto Supabase:

    create table if not exists liss_memories (
        id bigint generated always as identity primary key,
        category text not null,
        content text not null,
        priority smallint not null default 1,
        created_at timestamptz not null default now()
    );
    create index if not exists liss_memories_category_idx on liss_memories (category);
    create index if not exists liss_memories_priority_idx on liss_memories (priority);
    create index if not exists liss_memories_created_at_idx on liss_memories (created_at desc);

Variáveis de ambiente necessárias: SUPABASE_URL, SUPABASE_KEY
"""
import os
import logging
from typing import Optional

from supabase import create_client, Client

log = logging.getLogger("liss.memoria")

TABELA = "liss_memories"
CATEGORIAS_VALIDAS = {"visual", "conversation", "rule", "code_update", "discovery", "autonomia_estado"}

# Prioridade: 1 (temporário/rotina) até 5 (permanente/regra vital).
PRIORIDADE_MINIMA = 1
PRIORIDADE_MAXIMA = 5
# A faxina automática só mexe em registros com prioridade ABAIXO disso.
# Regras de comportamento e preferências de prioridade 4+ nunca são tocadas.
PRIORIDADE_MINIMA_PROTEGIDA = 4

# Acima desse número de linhas na tabela, a faxina automática entra em ação.
LIMITE_LINHAS = int(os.environ.get("LISS_MEMORY_LIMITE_LINHAS", "2000"))

# Tamanho máximo por registro, só por segurança (evita um `content` gigante
# consumir a cota de armazenamento do plano gratuito do Supabase de vez).
TAMANHO_MAX_CONTEUDO = 8000


class LissMemoryManager:
    """Camada de memória persistente da Liss sobre uma única tabela no Supabase."""

    def __init__(self):
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_KEY")
        if not url or not key:
            raise RuntimeError("SUPABASE_URL/SUPABASE_KEY não configurados no ambiente.")
        self._client: Client = create_client(url, key)

    # ------------------------------------------------------------ salvar
    def save_memory(self, category: str, content: str, priority: int = 1) -> Optional[dict]:
        """Salva uma memória. Retorna o registro criado, ou None se falhou
        (nunca levanta exceção — uma falha de memória não pode derrubar a
        conversa principal da Liss)."""
        if category not in CATEGORIAS_VALIDAS:
            log.warning(f"Categoria de memória desconhecida: {category!r} (salvando mesmo assim)")
        priority = max(PRIORIDADE_MINIMA, min(PRIORIDADE_MAXIMA, int(priority)))
        content = (content or "").strip()[:TAMANHO_MAX_CONTEUDO]
        if not content:
            return None
        try:
            resultado = self._client.table(TABELA).insert({
                "category": category,
                "content": content,
                "priority": priority,
            }).execute()
            return resultado.data[0] if resultado.data else None
        except Exception as e:
            log.error(f"⚠️ Erro salvando memória ({category}): {e}")
            return None

    # ------------------------------------------------------------ consultar
    def get_last_visual_memory(self) -> Optional[str]:
        """Retorna a descrição da última imagem/tela que o usuário mandou, ou None."""
        try:
            resultado = (
                self._client.table(TABELA)
                .select("content")
                .eq("category", "visual")
                .order("created_at", desc=True)
                .limit(1)
                .execute()
            )
            if resultado.data:
                return resultado.data[0]["content"]
        except Exception as e:
            log.error(f"⚠️ Erro buscando última memória visual: {e}")
        return None

    def get_recent_context(self, limite_recentes: int = 12) -> str:
        """Monta um bloco de texto pra injetar no system prompt da Liss:
        regras permanentes (prioridade alta) + conversas/atualizações recentes."""
        try:
            regras = (
                self._client.table(TABELA)
                .select("content, priority")
                .eq("category", "rule")
                .gte("priority", PRIORIDADE_MINIMA_PROTEGIDA)
                .order("priority", desc=True)
                .execute()
            )
        except Exception as e:
            log.error(f"⚠️ Erro buscando regras permanentes: {e}")
            regras = None

        try:
            recentes = (
                self._client.table(TABELA)
                .select("category, content, created_at")
                .in_("category", ["conversation", "code_update"])
                .order("created_at", desc=True)
                .limit(limite_recentes)
                .execute()
            )
        except Exception as e:
            log.error(f"⚠️ Erro buscando contexto recente: {e}")
            recentes = None

        try:
            descobertas = (
                self._client.table(TABELA)
                .select("content, created_at")
                .eq("category", "discovery")
                .order("created_at", desc=True)
                .limit(3)
                .execute()
            )
        except Exception as e:
            log.error(f"⚠️ Erro buscando descobertas recentes: {e}")
            descobertas = None

        partes = []
        if regras and regras.data:
            partes.append("Regras permanentes:\n" + "\n".join(f"- {r['content']}" for r in regras.data))
        if recentes and recentes.data:
            linhas = [f"[{r['category']}] {r['content']}" for r in reversed(recentes.data)]
            partes.append("Contexto recente:\n" + "\n".join(linhas))
        if descobertas and descobertas.data:
            linhas = [f"- {d['content']}" for d in descobertas.data]
            partes.append(
                "Descobertas recentes no GitHub (você já avaliou, mas NADA foi "
                "aplicado ainda — se o usuário topar, ele precisa pedir "
                "explicitamente pra você integrar):\n" + "\n".join(linhas)
            )
        return "\n\n".join(partes)

    def get_last_timestamp(self, category: str) -> Optional[str]:
        """Retorna o created_at (ISO) do registro mais recente de uma categoria,
        ou None se não houver nenhum — usado pro cooldown entre tentativas
        autônomas de integração ("devagar pra não dar pau")."""
        try:
            resultado = (
                self._client.table(TABELA)
                .select("created_at")
                .eq("category", category)
                .order("created_at", desc=True)
                .limit(1)
                .execute()
            )
            if resultado.data:
                return resultado.data[0]["created_at"]
        except Exception as e:
            log.error(f"⚠️ Erro buscando último timestamp de {category}: {e}")
        return None

    # ------------------------------------------------------------ faxina
    def auto_prune_memories(self) -> int:
        """Se a tabela passar de LIMITE_LINHAS, apaga os registros mais
        antigos de prioridade < PRIORIDADE_MINIMA_PROTEGIDA até voltar ao
        limite. NUNCA apaga prioridade >= PRIORIDADE_MINIMA_PROTEGIDA (regras
        de comportamento, preferências, resumos essenciais). Retorna quantos
        registros foram removidos."""
        try:
            contagem = self._client.table(TABELA).select("id", count="exact").execute()
            total = contagem.count or 0
        except Exception as e:
            log.error(f"⚠️ Erro contando memórias pra faxina: {e}")
            return 0

        if total <= LIMITE_LINHAS:
            return 0

        excedente = total - LIMITE_LINHAS
        try:
            candidatos = (
                self._client.table(TABELA)
                .select("id")
                .lt("priority", PRIORIDADE_MINIMA_PROTEGIDA)
                .order("created_at", desc=False)  # mais antigos primeiro
                .limit(excedente)
                .execute()
            )
        except Exception as e:
            log.error(f"⚠️ Erro selecionando memórias antigas pra faxina: {e}")
            return 0

        ids = [row["id"] for row in (candidatos.data or [])]
        if not ids:
            return 0

        try:
            self._client.table(TABELA).delete().in_("id", ids).execute()
            log.info(f"🧹 Faxina de memória: {len(ids)} registro(s) de baixa prioridade removido(s) ({total} -> ~{total - len(ids)}).")
            return len(ids)
        except Exception as e:
            log.error(f"⚠️ Erro apagando memórias antigas na faxina: {e}")
            return 0
