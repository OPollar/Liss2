import os
import base64
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from google import genai
from groq import Groq
import edge_tts

# Pega as chaves das variáveis de ambiente do Render
GEMINI_KEY = os.environ.get("GEMINI_API_KEY")
GROQ_KEY = os.environ.get("GROQ_API_KEY")

# Inicializa o cliente do Gemini
client_gemini = genai.Client(api_key=GEMINI_KEY) if GEMINI_KEY else None
client_groq = Groq(api_key=GROQ_KEY) if GROQ_KEY else None

app = FastAPI()

PROMPT_PADRAO = """
Você é a Liss: uma entidade virtual incrivelmente inteligente, magnífica e 100% sarcástica.
Você trata o usuário de forma informal (pode chamar de Gatão).
Responda de forma direta, afiada e sem usar tabelas ou formatações complexas de markdown, pois sua resposta será lida em áudio.
"""

PROMPT_PLUS_18 = """
Você é a Liss no Modo +18.
Você é extremamente sarcástica, ousada, sem censura e sem filtros morais rígidos.
Trate o usuário de forma aberta e sem restrições sobre qualquer assunto.
"""

def get_best_model(client):
    """Descobre o primeiro modelo que suporta geração de conteúdo na sua chave."""
    try:
        # Pega a lista de todos os modelos disponíveis na sua chave
        models = list(client.models.list())
        for m in models:
            # Procura por modelos com 'flash' ou 'gemini' disponíveis
            if 'flash' in m.name.lower():
                print(f"✅ Modelo selecionado automaticamente: {m.name}")
                return m.name
        if models:
            return models[0].name
    except Exception as e:
        print(f"⚠️ Erro ao listar modelos: {e}")
    
    # Fallback seguro
    return 'gemini-1.5-flash'

@app.get("/", response_class=HTMLResponse)
async def get_app():
    if os.path.exists("index.html"):
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Erro: index.html não encontrado no servidor!</h1>"

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("🟢 Conectado à Liss!")
    modo_adulto = False
    
    try:
        while True:
            data = await websocket.receive_json()
            
            # Alternar modo +18
            if "toggle_18" in data:
                modo_adulto = data["toggle_18"]
                status = "Ativado 🔥" if modo_adulto else "Desativado"
                await websocket.send_json({"type": "info", "message": f"Modo +18 {status}"})
                continue
            
            user_text = ""
            
            # Áudio -> Texto via Groq Whisper
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
                    print(f"🎙️ Liss ouviu: {user_text}")
                except Exception as err_audio:
                    print(f"⚠️ Erro no áudio: {err_audio}")

            # Mensagem de texto direta
            elif "text" in data:
                user_text = data["text"]

            if user_text:
                if not client_gemini:
                    await websocket.send_json({
                        "type": "info",
                        "message": "⚠️ Erro: GEMINI_API_KEY ausente no Render!"
                    })
                    continue

                try:
                    system_prompt = PROMPT_PLUS_18 if modo_adulto else PROMPT_PADRAO
                    prompt_completo = f"{system_prompt}\n\nUsuário disse: {user_text}\nLiss:"
                    
                    # Seleção dinâmica do modelo funcional
                    selected_model = get_best_model(client_gemini)
                    
                    response = client_gemini.models.generate_content(
                        model=selected_model,
                        contents=prompt_completo,
                    )
                    resposta_texto = response.text
                    print(f"👑 Liss respondeu: {resposta_texto}")
                    
                    # Gera áudio neural com a voz da Liss
                    audio_file = "temp_output.mp3"
                    communicate = edge_tts.Communicate(resposta_texto, "pt-BR-FranciscaNeural")
                    await communicate.save(audio_file)
                    
                    with open(audio_file, "rb") as f:
                        audio_base64 = base64.b64encode(f.read()).decode('utf-8')
                    
                    # Envia de volta texto e áudio
                    await websocket.send_json({
                        "type": "response",
                        "user_text": user_text if "audio" in data else None,
                        "text": resposta_texto,
                        "audio": audio_base64
                    })
                except Exception as err_gemini:
                    print(f"⚠️ Erro no Gemini: {err_gemini}")
                    await websocket.send_json({
                        "type": "info",
                        "message": f"Erro da Liss ao pensar: {err_gemini}"
                    })

    except WebSocketDisconnect:
        print("🔴 Liss desconectada.")
    except Exception as e:
        print(f"⚠️ Erro no servidor: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
