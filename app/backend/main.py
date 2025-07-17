import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import base64
import json
import os
import numpy as np
import soundfile as sf
import io
from dotenv import load_dotenv
from agent import Agent
from stt_handler import STTHandler
from tts_handler import TTSHandler
from rag_handler import RAGHandler
from vad_handler import VADHandler, AudioProcessor
from translation import translate_nepali_to_english, translate_english_to_nepali
from llm import summarize_rag_to_nepali

# Load environment variables from .env file
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Voice Call Bot", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static frontend files
app.mount("/static", StaticFiles(directory="../frontend"), name="static")

# Initialize handlers
agent = Agent()
stt_handler = STTHandler()
tts_handler = TTSHandler()
rag_handler = RAGHandler()

# Store VAD handlers per client
client_vad_handlers = {}

@app.get("/")
async def root():
    with open(os.path.join(os.path.dirname(__file__), "../frontend/index.html")) as f:
        return HTMLResponse(f.read())

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    client_id = id(websocket)
    logger.info(f"New client connected: {websocket.client} (ID: {client_id})")

    # Initialize VAD handler for this client with higher threshold for loud speech
    vad_handler = VADHandler(
        energy_threshold=0.08,  # Higher threshold - need to speak louder
        silence_duration=1.5,   # 1.5 seconds of silence to end speech
        min_speech_duration=0.5  # Minimum 0.5 seconds for valid speech
    )
    audio_processor = AudioProcessor(chunk_duration=0.1, sample_rate=16000)
    client_vad_handlers[client_id] = {
        'vad': vad_handler,
        'processor': audio_processor,
        'current_time': 0.0
    }

    # Send model status on connect
    await websocket.send_text(json.dumps({
        "type": "status",
        "models": {
            "stt": True,
            "tts": True,
            "rag": rag_handler.retriever is not None,
            "gemini": True
        },
        "vad_enabled": True
    }))
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            step_log = []
            agent_decision = None
            rag_context = None
            if message.get("type") == "audio_chunk":
                # Handle continuous audio chunks for VAD
                try:
                    audio_data = base64.b64decode(message["audio"])
                    client_data = client_vad_handlers.get(client_id)

                    if client_data:
                        try:
                            # Convert raw audio bytes to numpy array
                            # The frontend sends Int16Array as bytes, so we need to convert it back
                            audio_int16 = np.frombuffer(audio_data, dtype=np.int16)
                            # Convert to float32 and normalize to [-1, 1]
                            audio_array = audio_int16.astype(np.float32) / 32768.0

                            # Process audio chunks
                            chunks = client_data['processor'].add_audio(audio_array)

                            for chunk in chunks:
                                is_speaking, complete_speech = client_data['vad'].process_audio_chunk(
                                    chunk, client_data['current_time']
                                )
                                client_data['current_time'] += 0.1  # 100ms chunks

                                if complete_speech is not None:
                                    # Complete speech detected, process it
                                    await process_complete_speech(websocket, complete_speech, step_log=[])

                        except Exception as e:
                            logger.error(f"Error processing audio chunk: {e}")
                            # Log more details for debugging
                            logger.error(f"Audio data length: {len(audio_data)}, type: {type(audio_data)}")

                except Exception as e:
                    logger.error(f"Error handling audio chunk: {e}")

            elif message.get("type") == "audio":
                try:
                    audio_data = base64.b64decode(message["audio"])
                    step_log.append("Received audio from user.")
                    nepali_text = stt_handler.transcribe(audio_data)
                    step_log.append(f"STT output: {nepali_text}")
                    agent_decision = agent.classify(nepali_text)
                    step_log.append(f"Agent decision: {agent_decision}")
                    if agent_decision == "rag":
                        await websocket.send_text(json.dumps({
                            "type": "text_response",
                            "text": "Please wait till I look up the information",
                            "agent_decision": agent_decision,
                            "step_log": step_log
                        }))
                        english_query = translate_nepali_to_english(nepali_text)
                        step_log.append(f"Translated to English: {english_query}")
                        rag_context = rag_handler.query(english_query)
                        step_log.append(f"RAG context: {rag_context}")
                        nepali_response = summarize_rag_to_nepali(rag_context, nepali_text)
                        step_log.append(f"LLM summarized to Nepali: {nepali_response}")
                    else:
                        nepali_response = agent.normal_conversation(nepali_text)
                        step_log.append(f"LLM conversational response: {nepali_response}")
                    audio_response = tts_handler.synthesize(nepali_response)
                    step_log.append("TTS synthesized audio.")
                    response_audio_b64 = base64.b64encode(audio_response).decode('utf-8')
                    await websocket.send_text(json.dumps({
                        "type": "text_response",
                        "text": nepali_response,
                        "agent_decision": agent_decision,
                        "rag_context": rag_context,
                        "step_log": step_log
                    }))
                    await websocket.send_text(json.dumps({
                        "type": "audio_response",
                        "audio": response_audio_b64
                    }))
                except Exception as e:
                    logger.error(f"Error processing audio: {e}")
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": f"Audio processing error: {str(e)}",
                        "step_log": step_log
                    }))
            elif message.get("type") == "text":
                try:
                    input_text = message.get("text", "")
                    step_log.append(f"Received text: {input_text}")
                    agent_decision = agent.classify(input_text)
                    step_log.append(f"Agent decision: {agent_decision}")
                    if agent_decision == "rag":
                        await websocket.send_text(json.dumps({
                            "type": "text_response",
                            "text": "Please wait till I look up the information",
                            "agent_decision": agent_decision,
                            "step_log": step_log
                        }))
                        english_query = translate_nepali_to_english(input_text)
                        step_log.append(f"Translated to English: {english_query}")
                        rag_context = rag_handler.query(english_query)
                        step_log.append(f"RAG context: {rag_context}")
                        nepali_response = summarize_rag_to_nepali(rag_context, input_text)
                        step_log.append(f"LLM summarized to Nepali: {nepali_response}")
                    else:
                        nepali_response = agent.normal_conversation(input_text)
                        step_log.append(f"LLM conversational response: {nepali_response}")
                    audio_response = tts_handler.synthesize(nepali_response)
                    step_log.append("TTS synthesized audio.")
                    response_audio_b64 = base64.b64encode(audio_response).decode('utf-8')
                    await websocket.send_text(json.dumps({
                        "type": "text_response",
                        "text": nepali_response,
                        "agent_decision": agent_decision,
                        "rag_context": rag_context,
                        "step_log": step_log
                    }))
                    await websocket.send_text(json.dumps({
                        "type": "audio_response",
                        "audio": response_audio_b64
                    }))
                except Exception as e:
                    logger.error(f"Error processing text: {e}")
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": f"Text processing error: {str(e)}",
                        "step_log": step_log
                    }))
            elif message.get("type") == "vad_sensitivity":
                # Adjust VAD sensitivity
                try:
                    sensitivity = float(message.get("sensitivity", 0.5))
                    client_data = client_vad_handlers.get(client_id)
                    if client_data:
                        client_data['vad'].set_sensitivity(sensitivity)
                        await websocket.send_text(json.dumps({
                            "type": "vad_sensitivity_updated",
                            "sensitivity": sensitivity
                        }))
                except Exception as e:
                    logger.error(f"Error setting VAD sensitivity: {e}")

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
        # Clean up client VAD handler
        if client_id in client_vad_handlers:
            del client_vad_handlers[client_id]
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        # Clean up client VAD handler
        if client_id in client_vad_handlers:
            del client_vad_handlers[client_id]
        try:
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": f"Server error: {str(e)}"
            }))
        except:
            pass

async def process_complete_speech(websocket: WebSocket, audio_data: np.ndarray, step_log: list):
    """Process a complete speech segment detected by VAD"""
    try:
        # Convert numpy array to audio bytes for STT
        with io.BytesIO() as wav_buffer:
            sf.write(wav_buffer, audio_data, 16000, format='WAV')
            wav_buffer.seek(0)
            audio_bytes = wav_buffer.read()

        step_log.append("VAD detected complete speech segment.")

        # Process the speech using existing pipeline
        nepali_text = stt_handler.transcribe(audio_bytes)
        step_log.append(f"STT output: {nepali_text}")

        agent_decision = agent.classify(nepali_text)
        step_log.append(f"Agent decision: {agent_decision}")

        rag_context = None
        if agent_decision == "rag":
            # Send "please wait" message immediately
            await websocket.send_text(json.dumps({
                "type": "text_response",
                "text": "ठीक छ, कृपया पर्खनुहोस्। म जानकारी खोज्दै छु...",
                "agent_decision": "searching",
                "step_log": step_log
            }))

            english_query = translate_nepali_to_english(nepali_text)
            step_log.append(f"Translated to English: {english_query}")
            rag_context = rag_handler.query(english_query)
            step_log.append(f"RAG context: {rag_context}")
            nepali_response = summarize_rag_to_nepali(rag_context, nepali_text)
            step_log.append(f"LLM summarized to Nepali: {nepali_response}")
        else:
            nepali_response = agent.normal_conversation(nepali_text)
            step_log.append(f"LLM conversational response: {nepali_response}")

        audio_response = tts_handler.synthesize(nepali_response)
        step_log.append("TTS synthesized audio.")
        response_audio_b64 = base64.b64encode(audio_response).decode('utf-8')

        await websocket.send_text(json.dumps({
            "type": "text_response",
            "text": nepali_response,
            "agent_decision": agent_decision,
            "rag_context": rag_context,
            "step_log": step_log,
            "input_text": nepali_text  # Include the detected speech
        }))
        await websocket.send_text(json.dumps({
            "type": "audio_response",
            "audio": response_audio_b64
        }))

    except Exception as e:
        logger.error(f"Error processing complete speech: {e}")
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": f"Speech processing error: {str(e)}",
            "step_log": step_log
        }))

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Voice Call Bot FastAPI Server...")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    ) 