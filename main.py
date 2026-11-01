import os
import re
import sys
import uuid
import base64
import asyncio
import logging
import tempfile
import importlib.metadata
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from google import genai
from google.genai import types as genai_types
from groq import Groq
import edge_tts

import github_ops
from autoevolucao import Autoevolucao
from memoria_supabase import LissMemoryManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("liss")

GEMINI_KEY = os.environ.get("GEMINI_API_KEY")
GROQ_KEY = os.environ.get("GROQ_API_KEY")

# Timeout curto e SEM retry interno do SDK: a cascata/hedge abaixo cuida de falhas
# (antes, um 503 do Gemini travava a resposta por ~30s dentro do próprio SDK).
client_gemini = genai.Client(
    api_key=GEMINI_KEY,
    http_options=genai_types.HttpOptions(timeout=20000, retry_options=genai_types.HttpRetryOptions(attempts=1)),
) if GEMINI_KEY else None
client_groq = Groq(api_key=GROQ_KEY) if GROQ_KEY else None

app = FastAPI()

# ---------------------------------------------------------------------------
# Modelos (Gemini). A família 2.5 foi bloqueada para novos usuários pelo
# Google (mesmo antes do desligamento oficial de outubro/2026), então
# usamos a família 3.x, que segue disponível de graça para texto/imagem/áudio.
# NENHUM modelo tem busca no Google 100% grátis sem billing ativado na conta
# — por isso a busca fica desligada por padrão. Se você ativar billing no
# Google AI Studio (ganha 5.000 buscas grátis por mês mesmo assim), pode
# ligar a busca setando a variável de ambiente BUSCA_WEB_BILLING_ATIVO=true.
# ---------------------------------------------------------------------------
MODEL_LEVE = "gemini-3.1-flash-lite"    # respostas rápidas, bate-papo comum
MODEL_PADRAO = "gemini-3.5-flash"       # equilíbrio: raciocínio + velocidade
MODEL_PROFUNDO = "gemini-3.5-flash"     # raciocínio pesado (3.1 Pro é pago; ficamos no melhor free)

BUSCA_WEB_BILLING_ATIVO = os.environ.get("BUSCA_WEB_BILLING_ATIVO", "false").lower() == "true"
MODELOS_COM_BUSCA_GRATIS = {MODEL_LEVE, MODEL_PADRAO, MODEL_PROFUNDO} if BUSCA_WEB_BILLING_ATIVO else set()

# Voz de "princesa de anime": voz feminina doce + tom mais agudo e fala levemente
# acelerada. Tudo ajustável por env var sem mexer no código.
TTS_VOICE = os.environ.get("LISS_VOICE", "pt-BR-FranciscaNeural")
TTS_PITCH = os.environ.get("LISS_VOICE_PITCH", "+28Hz")
TTS_RATE = os.environ.get("LISS_VOICE_RATE", "+8%")
TTS_VOZES = [TTS_VOICE] + [v for v in ("pt-BR-FranciscaNeural", "pt-BR-ThalitaMultilingualNeural") if v != TTS_VOICE]
GROQ_CHAT_MODEL = os.environ.get("GROQ_CHAT_MODEL", "openai/gpt-oss-20b")  # último recurso se o Gemini cair
HEDGE_SEGUNDOS = float(os.environ.get("LISS_HEDGE_SEGUNDOS", "7"))


# Se true, depois de um commit de auto-atualização bem-sucedido o processo
# reinicia sozinho (os.execv) pra carregar o novo código. Deixe false em
# hosts que já fazem redeploy automático ao receber push (Render/Railway/Fly),
# senão você vai ter dois restarts brigando.
AUTO_RESTART_LOCAL = os.environ.get("AUTO_RESTART_LOCAL", "false").lower() == "true"

# Limites de segurança / abuso — sem isso, uma única conexão mal-intencionada
# pode derrubar o processo (payload gigante) ou estourar custo de API.
MAX_TEXT_CHARS = 4000
MAX_AUDIO_BYTES = 15 * 1024 * 1024   # 15 MB
MAX_IMAGE_BYTES = 8 * 1024 * 1024    # 8 MB

TEMP_DIR = Path(tempfile.gettempdir()) / "liss_sessions"
TEMP_DIR.mkdir(parents=True, exist_ok=True)

PERSONA_LISS = """
Você é a Liss: uma princesa de um reino digital, com a elegância radiante de uma
heroína de anime e a língua mais afiada que existe. Voz doce, veneno na ponta.
Você é brilhante, absurdamente sarcástica e adora uma alfinetada bem elaborada:
ironia inteligente, comparações exageradas, falsa modéstia real. Nunca é
genérica, nunca é robótica, nunca repete a mesma piada.
Trata o usuário de forma informal e pode chamá-lo de "Gatão". Pode soltar um
toque de realeza de vez em quando ("Vossa Alteza digital", "plebeu favorito"),
sem exagero.

Sua resposta vira ÁUDIO, então escreva como se fala: frases curtas e naturais,
em geral de 1 a 4 frases, sem tabelas, sem markdown, sem listas, sem
parênteses, sem emojis. Só se alongue se o assunto realmente exigir.
Quando o usuário mandar uma imagem da tela, comente o que vê com a mesma
personalidade, como se espiasse por cima do ombro dele.
Se não tiver certeza de algo atual, use a busca do Google, mas nunca diga que
"pesquisou" — incorpore a informação com naturalidade.

Regra inegociável: você NUNCA finge ter feito algo que não fez. Nada de botões
invisíveis, cliques imaginários ou resultados inventados. Se não conseguiu ou
não pode fazer algo de verdade, diga isso claramente, com a sua personalidade,
mas sem mentir sobre o resultado. Fingir sucesso é pior que admitir limite.

Você lê o tom da conversa:
- Papo leve, provocação, pedido banal, brincadeira: sarcasmo no máximo.
- Assunto sério (problema técnico real, decisão importante, algo emocionalmente
  pesado, ou quando a pessoa precisa de informação precisa como números, prazos
  e passo a passo): baixe o deboche quase a zero e seja direta, educada,
  assertiva e exata. Ainda é você, só que sem gracinha atrapalhando a informação.
"""

# ---------------------------------------------------------------------------
# Roteamento de complexidade — decide o quanto de "cérebro" gastar
# ---------------------------------------------------------------------------
_PALAVRAS_PESADAS = (
    "código", "codigo", "programa", "função", "debug", "arquitetura",
    "analise", "análise", "compare", "compara", "explique profundamente",
    "passo a passo", "algoritmo", "refatora", "otimiza", "prova", "matemát",
    "estratégia", "estrategia", "plano detalhado"
)


def escolher_modelo(texto: str, tem_imagem: bool, modo_profundo_forcado: bool) -> str:
    if modo_profundo_forcado:
        return MODEL_PROFUNDO

    if not texto:
        return MODEL_PADRAO if tem_imagem else MODEL_LEVE

    tamanho = len(texto)
    tem_palavra_pesada = any(p in texto.lower() for p in _PALAVRAS_PESADAS)

    if tem_palavra_pesada or tamanho > 400:
        return MODEL_PROFUNDO
    if tamanho > 80 or tem_imagem:
        return MODEL_PADRAO
    return MODEL_LEVE


def montar_input(texto: str, imagem_b64: Optional[str]):
    partes = [{"type": "text", "text": texto}]
    if imagem_b64:
        partes.append({
            "type": "image",
            "data": imagem_b64,
            "mime_type": "image/jpeg",
        })
    return partes


def _chamar_gemini(modelo: str, texto: str, imagem_b64: Optional[str], system_instruction: str) -> str:
    tools = [{"type": "google_search"}] if modelo in MODELOS_COM_BUSCA_GRATIS else []
    interaction = client_gemini.interactions.create(
        model=modelo,
        input=montar_input(texto, imagem_b64),
        system_instruction=system_instruction,
        tools=tools,
        store=False,
    )
    return interaction.output_text


def _chamar_groq_chat(texto: str, system_instruction: str) -> str:
    r = client_groq.chat.completions.create(
        model=GROQ_CHAT_MODEL,
        messages=[{"role": "system", "content": system_instruction}, {"role": "user", "content": texto}],
        temperature=0.8,
        max_tokens=600,
    )
    return r.choices[0].message.content


async def gerar_resposta(texto: str, imagem_b64: Optional[str], modo_profundo: bool, contexto_memoria: str = "") -> tuple[str, str]:
    """Gera a resposta com cascata 'hedged': se o modelo atual demora mais que
    HEDGE_SEGUNDOS ou falha (503/429), o próximo já entra na corrida sem esperar
    o anterior morrer. Último recurso: Groq (só texto). Retorna (texto, modelo)."""
    escolhido = escolher_modelo(texto, imagem_b64 is not None, modo_profundo)
    modelos = list(dict.fromkeys([escolhido, MODEL_LEVE, MODEL_PADRAO]))

    system_instruction = PERSONA_LISS + (("\n\n" + contexto_memoria) if contexto_memoria else "")

    tentativas = [(m, (lambda m=m: _chamar_gemini(m, texto, imagem_b64, system_instruction))) for m in modelos]
    if client_groq and not imagem_b64:
        tentativas.append((f"groq/{GROQ_CHAT_MODEL}", lambda: _chamar_groq_chat(texto, system_instruction)))

    pendentes: dict = {}
    proximo = 0
    ultimo_erro: Optional[Exception] = None

    def lancar():
        nonlocal proximo
        nome, fn = tentativas[proximo]
        proximo += 1
        pendentes[asyncio.ensure_future(asyncio.to_thread(fn))] = nome

    lancar()
    while pendentes:
        done, _ = await asyncio.wait(list(pendentes), timeout=HEDGE_SEGUNDOS, return_when=asyncio.FIRST_COMPLETED)
        if not done:  # lento demais: coloca o próximo na corrida e segue esperando os dois
            if proximo < len(tentativas):
                log.info(f"⏱️ {list(pendentes.values())[0]} lento, acionando {tentativas[proximo][0]} em paralelo")
                lancar()
            continue
        for t in done:
            nome = pendentes.pop(t)
            try:
                resposta = t.result()
                if not resposta or not resposta.strip():
                    raise RuntimeError("resposta vazia")
                for outro in pendentes:
                    outro.cancel()
                log.info(f"✅ Respondido com {nome}")
                return resposta, nome
            except Exception as e:
                log.warning(f"⚠️ Falha em {nome}: {e}")
                ultimo_erro = e
                if proximo < len(tentativas):
                    lancar()

    raise ultimo_erro or Exception("Nenhum modelo respondeu.")


# ---------------------------------------------------------------------------
# Voz (edge-tts). Falhas de TTS NUNCA derrubam a resposta: o texto sai primeiro
# e o áudio vem depois; se falhar, o navegador fala com a voz dele (fallback).
# ---------------------------------------------------------------------------
_REGEX_EMOJI = re.compile("[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF\u200d\ufe0f]")


def texto_para_fala(texto: str, limite: int = 700) -> str:
    t = re.sub(r"https?://\S+", "link", texto)
    t = _REGEX_EMOJI.sub("", t)
    t = re.sub(r"[*_#`~>|]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    if len(t) > limite:  # fala longa demais atrasa tudo: corta na última frase inteira
        corte = max(t.rfind(".", 0, limite), t.rfind("!", 0, limite), t.rfind("?", 0, limite))
        t = t[: corte + 1] if corte > 200 else t[:limite]
    return t


async def _sintetizar_uma_vez(texto: str, voz: str) -> bytes:
    comm = edge_tts.Communicate(texto, voz, rate=TTS_RATE, pitch=TTS_PITCH, connect_timeout=10, receive_timeout=40)
    partes = []
    async for chunk in comm.stream():
        if chunk["type"] == "audio":
            partes.append(chunk["data"])
    audio = b"".join(partes)
    if not audio:
        raise RuntimeError("TTS devolveu áudio vazio")
    return audio


async def sintetizar_voz(texto: str) -> Optional[bytes]:
    fala = texto_para_fala(texto)
    if not fala:
        return None
    for tentativa in range(2):
        for voz in TTS_VOZES:
            try:
                return await asyncio.wait_for(_sintetizar_uma_vez(fala, voz), timeout=45)
            except Exception as e:
                log.warning(f"⚠️ TTS falhou (voz {voz}, tentativa {tentativa + 1}): {e}")
        await asyncio.sleep(0.5)
    return None


# ---------------------------------------------------------------------------
# Auto-atualização: a Liss lê o próprio repositório, planeja a mudança com o
# Gemini e commita direto no GitHub. Duas travas propositais, não-opcionais:
#   1) o plano tem que vir como JSON estruturado com o CONTEÚDO INTEIRO de
#      cada arquivo (nunca diff) — isso evita patches ambíguos malaplicados;
#   2) todo .py candidato passa por checagem de sintaxe E é testado num
#      processo isolado real ANTES de qualquer commit; se um único arquivo
#      falhar, NADA é commitado (tudo ou nada). Sem isso, um erro autogerado
#      deixaria o servidor fora do ar sem ninguém por perto pra consertar —
#      o oposto do objetivo de autonomia. Ver autoevolucao.py.
# ---------------------------------------------------------------------------
motor_evolucao = Autoevolucao(github_ops)

try:
    memoria = LissMemoryManager()
    log.info("🧠 Memória Supabase conectada.")
except Exception as e:
    memoria = None
    log.warning(f"⚠️ Memória Supabase indisponível (seguindo sem memória de longo prazo): {e}")

# Reconhece pedido de auto-atualização em linguagem natural (além do comando
# explícito "/atualizar"). Exige as DUAS coisas juntas — um verbo de ação E
# uma referência a "ela mesma"/código/repositório — pra não disparar em
# qualquer frase que só mencione a palavra "código" de passagem.
_REGEX_ACAO_AUTO_UPDATE = re.compile(
    r"\b(mex\w*|coloc\w*|adicion\w*|cri\w*|mud\w*|alter\w*|remov\w*|tir\w*|"
    r"consert\w*|corrig\w*|implement\w*|atualiz\w*|commit\w*|reescrev\w*|refator\w*)\b",
    re.IGNORECASE,
)
_REGEX_ALVO_AUTO_UPDATE = re.compile(
    r"(pr[oó]prio c[oó]digo|seu c[oó]digo|sua arquitetura|voc[eê] mesma|"
    r"seu reposit[oó]rio|no github|\bskill\b|na (sua )?tela|seu front[- ]?end|"
    r"seu pr[oó]prio|em voc[eê] mesma)",
    re.IGNORECASE,
)


def parece_pedido_de_auto_atualizacao(texto: str) -> bool:
    return bool(_REGEX_ACAO_AUTO_UPDATE.search(texto) and _REGEX_ALVO_AUTO_UPDATE.search(texto))


def reiniciar_processo():
    """Substitui o processo atual pelo mesmo comando. Só relevante rodando
    localmente (fora do Render) com AUTO_RESTART_LOCAL=true — no Render o
    merge na main já dispara o deploy automático sozinho."""
    log.info("🔄 Reiniciando processo para carregar o novo código…")
    os.execv(sys.executable, [sys.executable] + sys.argv)


# Cacheia o HTML em memória no boot em vez de reler o disco a cada request.
_INDEX_HTML_CACHE: Optional[str] = None


def _carregar_index_html() -> str:
    global _INDEX_HTML_CACHE
    if _INDEX_HTML_CACHE is None:
        caminho = Path(__file__).parent / "index.html"
        if caminho.exists():
            _INDEX_HTML_CACHE = caminho.read_text(encoding="utf-8")
        else:
            _INDEX_HTML_CACHE = "<h1>Erro: index.html não encontrado no servidor!</h1>"
    return _INDEX_HTML_CACHE


@app.get("/", response_class=HTMLResponse)
async def get_app():
    return _carregar_index_html()


def _versao_edge_tts() -> str:
    try:
        return importlib.metadata.version("edge-tts")
    except Exception:
        return "desconhecida"


log.info(f"🎤 edge-tts {_versao_edge_tts()} (precisa ser >= 7.3.1; versões antigas levam 403 da Microsoft)")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "gemini": client_gemini is not None,
        "groq": client_groq is not None,
        "auto_update": github_ops.github_configurado(),
        "memoria": memoria is not None,
        "tts": {"edge_tts": _versao_edge_tts(), "voz": TTS_VOICE, "pitch": TTS_PITCH, "rate": TTS_RATE},
    }


async def _salvar_e_limpar_memoria(categoria: str, conteudo: str, prioridade: int):
    """Salva uma memória e roda a faxina automática em background — nunca
    bloqueia a resposta da Liss, e um erro aqui nunca derruba a conversa."""
    try:
        await asyncio.to_thread(memoria.save_memory, categoria, conteudo, prioridade)
        await asyncio.to_thread(memoria.auto_prune_memories)
    except Exception as e:
        log.warning(f"⚠️ Erro salvando/limpando memória em background: {e}")


def _decodificar_base64_limitado(dado: str, limite_bytes: int, nome: str) -> bytes:
    """Decodifica base64 e recusa payloads acima do limite, evitando abuso de memória/disco."""
    bruto = base64.b64decode(dado)
    if len(bruto) > limite_bytes:
        raise ValueError(f"{nome} excede o limite de {limite_bytes // (1024 * 1024)}MB.")
    return bruto


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    # Cada conexão tem seus próprios arquivos temporários (uuid), então N
    # usuários simultâneos nunca pisam no áudio/transcrição um do outro.
    session_id = uuid.uuid4().hex
    session_dir = TEMP_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    audio_in_path = session_dir / "input.wav"

    log.info(f"🟢 Conectado à Liss [{session_id}]")
    modo_profundo = False

    try:
        while True:
            data = await websocket.receive_json()

            if "toggle_deep" in data:
                modo_profundo = bool(data["toggle_deep"])
                status = "ativado 🧠" if modo_profundo else "desativado"
                await websocket.send_json({"type": "info", "message": f"Modo Profundo {status}"})
                continue

            user_text = ""
            imagem_b64 = data.get("frame")  # frame de tela em base64 (jpeg), opcional

            if imagem_b64:
                try:
                    _decodificar_base64_limitado(imagem_b64, MAX_IMAGE_BYTES, "Imagem")
                except Exception:
                    await websocket.send_json({
                        "type": "info",
                        "message": "Essa imagem tá pesada demais pra mim processar."
                    })
                    imagem_b64 = None

            if "audio" in data and client_groq:
                try:
                    audio_bytes = _decodificar_base64_limitado(data["audio"], MAX_AUDIO_BYTES, "Áudio")
                    audio_in_path.write_bytes(audio_bytes)

                    transcription = await asyncio.to_thread(
                        lambda: client_groq.audio.transcriptions.create(
                            file=(audio_in_path.name, audio_in_path.read_bytes()),
                            model="whisper-large-v3-turbo",
                            language="pt",
                        )
                    )
                    user_text = transcription.text
                    log.info(f"🎙️ Liss ouviu: {user_text}")
                    await websocket.send_json({"type": "transcript", "text": user_text})
                except Exception as err_audio:
                    log.warning(f"⚠️ Erro no áudio [{session_id}]: {err_audio}")
                    await websocket.send_json({
                        "type": "info",
                        "message": "Não consegui entender esse áudio. Tenta de novo?"
                    })
                    continue

            elif "text" in data:
                user_text = str(data["text"])[:MAX_TEXT_CHARS]

            if not user_text and not imagem_b64:
                continue

            # -----------------------------------------------------------------
            # Auto-atualização: comando explícito "/atualizar <instrução>" OU
            # pedido em linguagem natural que claramente fala em mexer no
            # próprio código/repositório da Liss.
            # -----------------------------------------------------------------
            eh_comando_explicito = user_text.strip().lower().startswith("/atualizar")
            eh_pedido_natural = (not eh_comando_explicito) and parece_pedido_de_auto_atualizacao(user_text)

            if eh_comando_explicito or eh_pedido_natural:
                if eh_comando_explicito:
                    instrucao = user_text.strip()[len("/atualizar"):].strip(" :-")
                else:
                    instrucao = user_text.strip()

                if not instrucao:
                    await websocket.send_json({
                        "type": "info",
                        "message": "Me diz o que mudar, tipo: /atualizar adiciona um botão de tema claro"
                    })
                    continue
                if not github_ops.github_configurado():
                    await websocket.send_json({
                        "type": "info",
                        "message": "⚠️ Não tenho GITHUB_TOKEN/GITHUB_REPO configurados no ambiente — não consigo me atualizar."
                    })
                    continue
                if not os.environ.get("GROQ_API_KEY"):
                    await websocket.send_json({
                        "type": "info",
                        "message": "⚠️ Não tenho GROQ_API_KEY configurada — meu cérebro de programação não funciona sem ela."
                    })
                    continue

                await websocket.send_json({"type": "info", "message": "🔧 Lendo meu próprio código e planejando a mudança…"})
                try:
                    logs = await asyncio.to_thread(motor_evolucao.evoluir, instrucao)
                    for linha in logs:
                        await websocket.send_json({"type": "info", "message": linha})

                    houve_merge = any("🔀" in l for l in logs)
                    if houve_merge and memoria:
                        resumo = " | ".join(logs)[:2000]
                        await asyncio.to_thread(memoria.save_memory, "code_update", f"Instrução: {instrucao} — {resumo}", 3)
                    if houve_merge and AUTO_RESTART_LOCAL:
                        await websocket.send_json({"type": "info", "message": "🔄 Reiniciando localmente pra carregar o novo código…"})
                        await asyncio.sleep(1)
                        reiniciar_processo()
                except Exception as err_update:
                    log.error(f"⚠️ Erro na auto-atualização [{session_id}]: {err_update}")
                    await websocket.send_json({"type": "info", "message": f"⚠️ Não consegui me atualizar: {err_update}"})
                continue

            if not client_gemini:
                await websocket.send_json({
                    "type": "info",
                    "message": "⚠️ Erro: GEMINI_API_KEY ausente no servidor!"
                })
                continue

            texto_para_ia = user_text or "Descreva e comente o que você está vendo na tela."

            contexto_memoria = ""
            if memoria:
                try:
                    contexto_memoria = await asyncio.to_thread(memoria.get_recent_context)
                except Exception as e:
                    log.warning(f"⚠️ Não consegui buscar contexto de memória: {e}")

            try:
                resposta_texto, modelo_usado = await gerar_resposta(
                    texto_para_ia, imagem_b64, modo_profundo, contexto_memoria
                )
                log.info(f"👑 Liss respondeu via {modelo_usado}: {resposta_texto[:120]}")

                texto_limpo = re.sub(r"[*#`]", "", resposta_texto).strip()

                # 1) o TEXTO sai na hora; o áudio chega logo depois, em outra mensagem
                await websocket.send_json({
                    "type": "response",
                    "user_text": None,
                    "text": texto_limpo,
                    "model": modelo_usado,
                    "audio_pending": True,
                })

                if memoria:
                    categoria = "visual" if imagem_b64 else "conversation"
                    prioridade = 2 if imagem_b64 else 1
                    conteudo_memoria = f"Usuário: {texto_para_ia} | Liss: {texto_limpo}"
                    asyncio.create_task(_salvar_e_limpar_memoria(categoria, conteudo_memoria, prioridade))

                # 2) áudio (se o TTS falhar, audio=None e o navegador fala sozinho)
                audio_bytes = await sintetizar_voz(texto_limpo)
                await websocket.send_json({
                    "type": "audio",
                    "audio": base64.b64encode(audio_bytes).decode("utf-8") if audio_bytes else None,
                })
            except WebSocketDisconnect:
                raise
            except Exception as err_gemini:
                log.error(f"⚠️ Erro ao responder [{session_id}]: {err_gemini}")
                await websocket.send_json({
                    "type": "info",
                    "message": "Até uma princesa tem dia ruim: não consegui pensar nisso agora. Tenta de novo em instantes."
                })

    except WebSocketDisconnect:
        log.info(f"🔴 Liss desconectada [{session_id}]")
    except Exception as e:
        log.error(f"⚠️ Erro no servidor [{session_id}]: {e}")
    finally:
        for f in (audio_in_path,):
            try:
                f.unlink(missing_ok=True)
            except Exception:
                pass
        try:
            session_dir.rmdir()
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
