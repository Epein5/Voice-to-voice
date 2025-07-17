const ws = new WebSocket('ws://' + window.location.host + '/ws');
const conversation = document.getElementById('conversation');
const recordBtn = document.getElementById('recordBtn');
const textInput = document.getElementById('textInput');
const sendTextBtn = document.getElementById('sendTextBtn');
const audioPlayer = document.getElementById('audioPlayer');
const sensitivitySlider = document.getElementById('sensitivitySlider');
const sensitivityValue = document.getElementById('sensitivityValue');
const agentDecisionDiv = document.getElementById('agentDecision');
const ragContextPre = document.getElementById('ragContext');
const stepLogUl = document.getElementById('stepLog');
const sttStatus = document.getElementById('sttStatus');
const ttsStatus = document.getElementById('ttsStatus');
const ragStatus = document.getElementById('ragStatus');
const geminiStatus = document.getElementById('geminiStatus');

let mediaRecorder;
let audioChunks = [];
let audioContext;
let microphone;
let processor;
let isListening = false;
let vadEnabled = false;

function appendMessage(text, sender) {
    const div = document.createElement('div');
    div.className = sender === 'user' ? 'user-message' : 'bot-message';
    div.textContent = text;
    conversation.appendChild(div);
    conversation.scrollTop = conversation.scrollHeight;
}

function updateModelStatus(models) {
    sttStatus.className = 'status-dot ' + (models.stt ? 'on' : 'off');
    ttsStatus.className = 'status-dot ' + (models.tts ? 'on' : 'off');
    ragStatus.className = 'status-dot ' + (models.rag ? 'on' : 'off');
    geminiStatus.className = 'status-dot ' + (models.gemini ? 'on' : 'off');
}

function updateAgentDecision(decision) {
    agentDecisionDiv.textContent = decision ? decision : '-';
    agentDecisionDiv.style.color = decision === 'rag' ? '#fdcb6e' : '#00b894';
}

function updateRagContext(context) {
    ragContextPre.textContent = context ? context : '-';
}

function updateStepLog(logArr) {
    stepLogUl.innerHTML = '';
    if (!logArr || !logArr.length) {
        stepLogUl.innerHTML = '<li>-</li>';
        return;
    }
    logArr.forEach(step => {
        const li = document.createElement('li');
        li.textContent = step;
        stepLogUl.appendChild(li);
    });
}

// WebSocket events
ws.onopen = () => {
    appendMessage('Connected to bot.', 'bot');
};
ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    if (msg.type === 'status' && msg.models) {
        updateModelStatus(msg.models);
        if (msg.vad_enabled) {
            vadEnabled = true;
            initializeVAD();
        }
    } else if (msg.type === 'text_response') {
        if (msg.input_text) {
            appendMessage(msg.input_text, 'user');
        }
        appendMessage(msg.text, 'bot');
        updateAgentDecision(msg.agent_decision);
        updateRagContext(msg.rag_context);
        updateStepLog(msg.step_log);
    } else if (msg.type === 'audio_response') {
        audioPlayer.src = 'data:audio/wav;base64,' + msg.audio;
        audioPlayer.style.display = 'block';
        audioPlayer.play();
    } else if (msg.type === 'error') {
        appendMessage('Error: ' + msg.message, 'bot');
        updateStepLog(msg.step_log);
    } else if (msg.type === 'vad_sensitivity_updated') {
        console.log('VAD sensitivity updated to:', msg.sensitivity);
    }
};
ws.onclose = () => {
    appendMessage('Disconnected from bot.', 'bot');
    stopListening();
};

// Voice Activity Detection
async function initializeVAD() {
    if (!navigator.mediaDevices) {
        alert('Audio recording not supported.');
        return;
    }

    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            audio: {
                sampleRate: 16000,
                channelCount: 1,
                echoCancellation: true,
                noiseSuppression: true
            }
        });

        audioContext = new (window.AudioContext || window.webkitAudioContext)({
            sampleRate: 16000
        });

        microphone = audioContext.createMediaStreamSource(stream);

        // Create a script processor for continuous audio processing
        processor = audioContext.createScriptProcessor(4096, 1, 1);

        processor.onaudioprocess = (event) => {
            if (isListening) {
                const inputData = event.inputBuffer.getChannelData(0);
                sendAudioChunk(inputData);
            }
        };

        microphone.connect(processor);
        processor.connect(audioContext.destination);

        // Start listening automatically
        startListening();

    } catch (error) {
        console.error('Error initializing VAD:', error);
        appendMessage('Error: Could not access microphone for voice detection.', 'bot');
    }
}

function startListening() {
    if (!isListening) {
        isListening = true;
        recordBtn.classList.add('listening');
        recordBtn.textContent = '🎧 Listening...';
        recordBtn.onclick = stopListening;
        appendMessage('Voice detection started. Speak naturally - no need to press buttons!', 'bot');
    }
}

function stopListening() {
    if (isListening) {
        isListening = false;
        recordBtn.classList.remove('listening');
        recordBtn.textContent = '🎤 Start Voice Detection';
        recordBtn.onclick = startListening;
        appendMessage('Voice detection stopped.', 'bot');
    }
}

function sendAudioChunk(audioData) {
    // Convert Float32Array to Int16Array for better compression
    const int16Array = new Int16Array(audioData.length);
    for (let i = 0; i < audioData.length; i++) {
        int16Array[i] = Math.max(-32768, Math.min(32767, audioData[i] * 32768));
    }

    // Convert to base64
    const bytes = new Uint8Array(int16Array.buffer);
    const base64Audio = btoa(String.fromCharCode.apply(null, bytes));

    // Send to server for VAD processing
    ws.send(JSON.stringify({
        type: 'audio_chunk',
        audio: base64Audio
    }));
}

// Legacy recording function (fallback)
async function legacyRecord() {
    if (mediaRecorder && mediaRecorder.state === 'recording') {
        mediaRecorder.stop();
        recordBtn.classList.remove('recording');
        recordBtn.textContent = '🎤 Record';
        return;
    }
    if (!navigator.mediaDevices) {
        alert('Audio recording not supported.');
        return;
    }
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(stream);
    audioChunks = [];
    mediaRecorder.ondataavailable = (e) => {
        audioChunks.push(e.data);
    };
    mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
        const reader = new FileReader();
        reader.onloadend = () => {
            const base64Audio = reader.result.split(',')[1];
            ws.send(JSON.stringify({ type: 'audio', audio: base64Audio }));
            appendMessage('[You sent a voice message]', 'user');
        };
        reader.readAsDataURL(audioBlob);
    };
    mediaRecorder.start();
    recordBtn.classList.add('recording');
    recordBtn.textContent = '⏹️ Stop';
}

// Text sending
sendTextBtn.onclick = () => {
    const text = textInput.value.trim();
    if (!text) return;
    ws.send(JSON.stringify({ type: 'text', text }));
    appendMessage(text, 'user');
    textInput.value = '';
};
textInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') sendTextBtn.onclick();
});

// Sensitivity control
sensitivitySlider.addEventListener('input', (e) => {
    const sensitivity = parseFloat(e.target.value);
    sensitivityValue.textContent = sensitivity.toFixed(1);

    // Send sensitivity to server
    ws.send(JSON.stringify({
        type: 'vad_sensitivity',
        sensitivity: sensitivity
    }));
});