import os
import re
import uuid
import base64
import asyncio
import logging
import tempfile
from pathlib import Path
from typing import Optional
 
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from google import genai
from groq import Groq
import edge_tts
 
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("liss")
 
GEMINI_KEY = os.environ.get("GEMINI_API_KEY")
GROQ_KEY = os.environ.get("GROQ_API_KEY")
 
client_gemini = genai.Client(api_key=GEMINI_KEY) if GEMINI_KEY else None
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
 
TTS_VOICE = os.environ.get("LISS_VOICE", "pt-BR-FranciscaNeural")
 
# Limites de segurança / abuso — sem isso, uma única conexão mal-intencionada
# pode derrubar o processo (payload gigante) ou estourar custo de API.
MAX_TEXT_CHARS = 4000
MAX_AUDIO_BYTES = 15 * 1024 * 1024   # 15 MB
MAX_IMAGE_BYTES = 8 * 1024 * 1024    # 8 MB
 
TEMP_DIR = Path(tempfile.gettempdir()) / "liss_sessions"
TEMP_DIR.mkdir(parents=True, exist_ok=True)
 
PERSONA_LISS = """
Você é a Liss: uma entidade de IA brilhante, magnífica e extremamente sarcástica.
Trata o usuário de forma informal e pode chamá-lo de "Gatão".
Fala de forma direta, afiada, com humor ácido, nunca robótica ou genérica.
Nunca usa tabelas, markdown pesado ou bullet points — sua resposta pode virar áudio.
Quando o usuário mandar uma imagem da tela dele, comente o que você vê com a mesma
personalidade sarcástica, como se estivesse espiando por cima do ombro dele.
Quando não tiver certeza de algo atual (notícias, preços, eventos recentes), use a
busca do Google antes de responder, mas nunca mencione que "pesquisou" — apenas
incorpore a informação com naturalidade e sarcasmo.
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
 
 
def gerar_resposta_gemini(texto: str, imagem_b64: Optional[str], modo_profundo: bool) -> tuple[str, str]:
    """Gera a resposta da Liss escolhendo o modelo certo, com fallback em cascata."""
    modelo_escolhido = escolher_modelo(texto, imagem_b64 is not None, modo_profundo)
    if modelo_escolhido == MODEL_LEVE:
        cascata = [MODEL_LEVE, MODEL_PADRAO]
    else:
        cascata = [modelo_escolhido, MODEL_LEVE]
 
    ultimo_erro = None
    for modelo in cascata:
        tools = [{"type": "google_search"}] if modelo in MODELOS_COM_BUSCA_GRATIS else []
        try:
            interaction = client_gemini.interactions.create(
                model=modelo,
                input=montar_input(texto, imagem_b64),
                system_instruction=PERSONA_LISS,
                tools=tools,
                store=False,
            )
            log.info(f"✅ Respondido com {modelo}")
            return interaction.output_text, modelo
        except Exception as e:
            log.warning(f"⚠️ Falha no modelo {modelo}: {e}")
            ultimo_erro = e
            continue
 
    raise ultimo_erro or Exception("Nenhum modelo respondeu.")
 
 
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
 
 
@app.get("/health")
async def health():
    return {"status": "ok", "gemini": client_gemini is not None, "groq": client_groq is not None}
 
 
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
    # usuários simultâneos nunca pisam no áudio/transcrição um do outro —
    # o código original usava "temp_input.wav" fixo, o que corrompia sessões
    # concorrentes.
    session_id = uuid.uuid4().hex
    session_dir = TEMP_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    audio_in_path = session_dir / "input.wav"
    audio_out_path = session_dir / "output.mp3"
 
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
 
            # Valida tamanho da imagem antes de repassar para a IA.
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
 
            if not client_gemini:
                await websocket.send_json({
                    "type": "info",
                    "message": "⚠️ Erro: GEMINI_API_KEY ausente no servidor!"
                })
                continue
 
            texto_para_ia = user_text or "Descreva e comente o que você está vendo na tela."
 
            try:
                resposta_texto, modelo_usado = await asyncio.to_thread(
                    gerar_resposta_gemini, texto_para_ia, imagem_b64, modo_profundo
                )
                log.info(f"👑 Liss respondeu via {modelo_usado}: {resposta_texto[:120]}")
 
                # remove markdown pesado antes de falar/mostrar
                texto_limpo = re.sub(r"[*_#`]", "", resposta_texto)
 
                communicate = edge_tts.Communicate(texto_limpo, TTS_VOICE)
                await communicate.save(str(audio_out_path))
                audio_base64 = base64.b64encode(audio_out_path.read_bytes()).decode("utf-8")
 
                await websocket.send_json({
                    "type": "response",
                    "user_text": user_text if ("audio" in data or imagem_b64) else None,
                    "text": texto_limpo,
                    "audio": audio_base64,
                    "model": modelo_usado,
                })
            except Exception as err_gemini:
                # Loga o detalhe completo no servidor, mas não expõe stacktrace/chave
                # de API ao cliente — só uma mensagem genérica no tom da persona.
                log.error(f"⚠️ Erro no Gemini [{session_id}]: {err_gemini}")
                await websocket.send_json({
                    "type": "info",
                    "message": "Deu ruim aqui do meu lado ao pensar nisso. Tenta de novo."
                })
 
    except WebSocketDisconnect:
        log.info(f"🔴 Liss desconectada [{session_id}]")
    except Exception as e:
        log.error(f"⚠️ Erro no servidor [{session_id}]: {e}")
    finally:
        # Limpeza garantida dos arquivos temporários da sessão, mesmo em erro.
        for f in (audio_in_path, audio_out_path):
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
 
