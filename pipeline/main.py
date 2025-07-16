"""
Main entry point for the Voice-to-Voice application pipeline
Structured FastAPI server with modular handlers
"""

import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import base64
import json
import os
from tts import TTSHandler
from stt import STTHandler
from rag import RAGHandler

from llama_index.llms.gemini import Gemini
gemini_llm = Gemini(model_name="models/gemini-2.5-flash")

def translate_nepali_to_english(text):
    # Use Gemini API for translation
    prompt = f"Translate the following Nepali text to English. Only return the translation, no explanations:\n\n{text}"
    result = gemini_llm.complete(prompt)
    return str(result).strip()

def translate_english_to_nepali(text):
    # Use Gemini API for translation and summarization
    prompt = (
        "Translate the following English text to Nepali for a phone call. "
        "Summarize and keep the response short and clear. "
        "Do not use any digits in the Nepali response; write numbers in Nepali words. "
        "Only return the Nepali summary, no explanations, and do not repeat the original question.\n\n"
        f"English response: {text}"
    )
    result = gemini_llm.complete(prompt)
    return str(result).strip()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

tts_handler = TTSHandler()
stt_handler = STTHandler()
rag_handler = RAGHandler()

app = FastAPI(title="Voice-to-Voice Chat", version="1.0.0")
app.mount("/static", StaticFiles(directory=".."), name="static")

@app.get("/")
async def get_client():
    """Serve the web client"""
    return HTMLResponse(open(os.path.join("..", "voice_chat_client.html")).read())

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info(f"New client connected: {websocket.client}")
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            if message.get("type") == "audio":
                try:
                    audio_data = base64.b64decode(message["audio"])
                    await websocket.send_text(json.dumps({
                        "type": "status",
                        "message": "Processing your voice message..."
                    }))
                    text = stt_handler.transcribe(audio_data)
                    logger.info(f"STT output: {text}")
                    english_query = translate_nepali_to_english(text)
                    logger.info(f"Translated to English: {english_query}")
                    rag_result = rag_handler.query(english_query)
                    if isinstance(rag_result, tuple):
                        response, context = rag_result
                    else:
                        response, context = rag_result, None
                    logger.info(f"RAG response: {response}")
                    if context:
                        logger.info(f"RAG context: {context}")
                    nepali_response = translate_english_to_nepali(response)
                    logger.info(f"Translated to Nepali: {nepali_response}")
                    audio_response = tts_handler.synthesize(nepali_response)
                    await websocket.send_text(json.dumps({
                        "type": "text_response",
                        "text": nepali_response,
                        "rag_context": context
                    }))
                    response_audio_b64 = base64.b64encode(audio_response).decode('utf-8')
                    await websocket.send_text(json.dumps({
                        "type": "audio_response",
                        "audio": response_audio_b64
                    }))
                except Exception as e:
                    logger.error(f"Error processing audio: {e}")
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": f"Audio processing error: {str(e)}"
                    }))
            elif message.get("type") == "text":
                try:
                    input_text = message.get("text", "")
                    await websocket.send_text(json.dumps({
                        "type": "status",
                        "message": "Processing your question..."
                    }))
                    logger.info(f"Query to RAG (from text): {input_text}")
                    rag_result = rag_handler.query(input_text)
                    if isinstance(rag_result, tuple):
                        response, context = rag_result
                    else:
                        response, context = rag_result, None
                    logger.info(f"RAG response: {response}")
                    if context:
                        logger.info(f"RAG context: {context}")
                    await websocket.send_text(json.dumps({
                        "type": "text_response",
                        "text": response,
                        "rag_context": context
                    }))
                    audio_response = tts_handler.synthesize(response)
                    if audio_response:
                        response_audio_b64 = base64.b64encode(audio_response).decode('utf-8')
                        await websocket.send_text(json.dumps({
                            "type": "audio_response",
                            "audio": response_audio_b64
                        }))
                except Exception as e:
                    logger.error(f"Error processing text: {e}")
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": f"Text processing error: {str(e)}"
                    }))
            elif message.get("type") == "ping":
                await websocket.send_text(json.dumps({
                    "type": "pong",
                    "message": "Server is alive"
                }))
            else:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "Unknown message type"
                }))
    except WebSocketDisconnect:
        logger.info(f"Client disconnected: {websocket.client}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": f"Server error: {str(e)}"
            }))
        except:
            pass

if __name__ == "__main__":
    import uvicorn
    logger.info("🎙️  Starting Voice-to-Voice FastAPI Server...")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
