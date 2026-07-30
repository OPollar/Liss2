import os
import re
import base64
import asyncio
import logging
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
# Modelos (Gemini, camada 100% gratuita, sem cartão cadastrado) — a Liss
# escolhe sozinha qual usar. IMPORTANTE: os modelos da família Gemini 3.x
# (3.5/3.6/3.1) NÃO têm busca no Google grátis — só a família 2.5 tem,
# e só nos modelos Flash/Flash-Lite (o 2.5 Pro não tem busca grátis).
# A Google já avisou que a família 2.5 será desligada em outubro de 2026 —
# quando chegar perto disso, troque para gemini-3.5-flash e ative billing
# se quiser manter a busca funcionando.
# ---------------------------------------------------------------------------
MODEL_LEVE = "gemini-2.5-flash-lite"    # respostas rápidas, bate-papo comum
MODEL_PADRAO = "gemini-2.5-flash"       # equilíbrio: raciocínio + velocidade
MODEL_PROFUNDO = "gemini-2.5-pro"       # raciocínio pesado, código complexo, análise longa

# modelos que têm busca no Google liberada de graça (sem billing ativado)
MODELOS_COM_BUSCA_GRATIS = {MODEL_LEVE, MODEL_PADRAO}

TTS_VOICE = "pt-BR-FranciscaNeural"

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
    cascata = {
        MODEL_PROFUNDO: [MODEL_PROFUNDO, MODEL_PADRAO, MODEL_LEVE],
        MODEL_PADRAO: [MODEL_PADRAO, MODEL_LEVE],
        MODEL_LEVE: [MODEL_LEVE, MODEL_PADRAO],
    }[modelo_escolhido]

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


@app.get("/", response_class=HTMLResponse)
async def get_app():
    if os.path.exists("index.html"):
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Erro: index.html não encontrado no servidor!</h1>"


@app.get("/health")
async def health():
    return {"status": "ok", "gemini": client_gemini is not None, "groq": client_groq is not None}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    log.info("🟢 Conectado à Liss!")

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

            if "audio" in data and client_groq:
                try:
                    audio_bytes = base64.b64decode(data["audio"])
                    with open("temp_input.wav", "wb") as f:
                        f.write(audio_bytes)

                    with open("temp_input.wav", "rb") as file:
                        transcription = client_groq.audio.transcriptions.create(
                            file=("temp_input.wav", file.read()),
                            model="whisper-large-v3-turbo",
                            language="pt"
                        )
                    user_text = transcription.text
                    log.info(f"🎙️ Liss ouviu: {user_text}")
                except Exception as err_audio:
                    log.warning(f"⚠️ Erro no áudio: {err_audio}")
                    await websocket.send_json({
                        "type": "info",
                        "message": "Não consegui entender esse áudio. Tenta de novo?"
                    })
                    continue

            elif "text" in data:
                user_text = data["text"]

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

                audio_file = "temp_output.mp3"
                communicate = edge_tts.Communicate(texto_limpo, TTS_VOICE)
                await communicate.save(audio_file)

                with open(audio_file, "rb") as f:
                    audio_base64 = base64.b64encode(f.read()).decode('utf-8')

                await websocket.send_json({
                    "type": "response",
                    "user_text": user_text if ("audio" in data or imagem_b64) else None,
                    "text": texto_limpo,
                    "audio": audio_base64,
                    "model": modelo_usado,
                })
            except Exception as err_gemini:
                log.error(f"⚠️ Erro no Gemini: {err_gemini}")
                await websocket.send_json({
                    "type": "info",
                    "message": f"Erro da Liss ao pensar: {err_gemini}"
                })

    except WebSocketDisconnect:
        log.info("🔴 Liss desconectada.")
    except Exception as e:
        log.error(f"⚠️ Erro no servidor: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
