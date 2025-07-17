// WebSocket connection
const ws = new WebSocket('ws://' + window.location.host + '/ws');

// DOM elements
const conversation = document.getElementById('conversation');
const recordBtn = document.getElementById('recordBtn');
const textInput = document.getElementById('textInput');
const sendTextBtn = document.getElementById('sendTextBtn');
const audioPlayer = document.getElementById('audioPlayer');
const sensitivitySlider = document.getElementById('sensitivitySlider');
const sensitivityValue = document.getElementById('sensitivityValue');
const connectionStatus = document.getElementById('connectionStatus');
const voiceIndicator = document.getElementById('voiceIndicator');
const loadingOverlay = document.getElementById('loadingOverlay');

// Status elements
const sttStatus = document.getElementById('sttStatus');
const ttsStatus = document.getElementById('ttsStatus');
const ragStatus = document.getElementById('ragStatus');
const geminiStatus = document.getElementById('geminiStatus');

// Info panel elements
const agentDecision = document.getElementById('agentDecision');
const decisionDescription = document.getElementById('decisionDescription');
const ragContext = document.getElementById('ragContext');
const stepLog = document.getElementById('stepLog');

// Real-time processing elements
let currentProcessingTask = null;
let processingStartTime = null;

// Collapse buttons
const ragCollapseBtn = document.getElementById('ragCollapseBtn');
const logCollapseBtn = document.getElementById('logCollapseBtn');
const ragContent = document.getElementById('ragContent');
const logContent = document.getElementById('logContent');

// Audio variables
let mediaRecorder;
let audioChunks = [];
let audioContext;
let microphone;
let processor;
let isListening = false;
let vadEnabled = false;

// UI State Management
class UIManager {
    static showLoading(message = 'Processing your request...') {
        loadingOverlay.querySelector('p').textContent = message;
        loadingOverlay.classList.add('active');
    }

    static hideLoading() {
        loadingOverlay.classList.remove('active');
    }

    static updateConnectionStatus(connected) {
        if (connected) {
            connectionStatus.classList.add('connected');
            connectionStatus.querySelector('span').textContent = 'Connected';
        } else {
            connectionStatus.classList.remove('connected');
            connectionStatus.querySelector('span').textContent = 'Disconnected';
        }
    }

    static showVoiceActivity(active) {
        if (active) {
            voiceIndicator.classList.add('active');
        } else {
            voiceIndicator.classList.remove('active');
        }
    }

    static updateVoiceButton(state) {
        const icon = recordBtn.querySelector('i');
        const text = recordBtn.querySelector('span');
        
        recordBtn.className = 'voice-btn';
        
        switch (state) {
            case 'listening':
                recordBtn.classList.add('listening');
                icon.className = 'fas fa-ear-listen';
                text.textContent = 'Listening...';
                break;
            case 'recording':
                recordBtn.classList.add('recording');
                icon.className = 'fas fa-stop';
                text.textContent = 'Stop Recording';
                break;
            default:
                icon.className = 'fas fa-microphone';
                text.textContent = 'Start Voice Detection';
        }
    }
}

// Message handling
function appendMessage(text, sender) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}`;
    messageDiv.textContent = text;
    conversation.appendChild(messageDiv);
    conversation.scrollTop = conversation.scrollHeight;
}

// Status updates
function updateModelStatus(models) {
    updateStatusIndicator(sttStatus, models.stt);
    updateStatusIndicator(ttsStatus, models.tts);
    updateStatusIndicator(ragStatus, models.rag);
    updateStatusIndicator(geminiStatus, models.gemini);
}

function updateStatusIndicator(element, status) {
    element.className = 'status-indicator';
    if (status === true) {
        element.classList.add('connected');
    } else if (status === 'partial') {
        element.classList.add('partial');
    }
}

// Agent decision updates
function updateAgentDecision(decision) {
    const badge = agentDecision;
    const description = decisionDescription;

    badge.className = 'decision-badge';

    switch (decision) {
        case 'normal':
            badge.classList.add('normal');
            badge.textContent = 'Normal Chat';
            description.textContent = 'Having a regular conversation';
            break;
        case 'rag':
            badge.classList.add('rag');
            badge.textContent = 'Knowledge Search';
            description.textContent = 'Searching knowledge base for information';
            break;
        case 'searching':
            badge.classList.add('searching');
            badge.textContent = 'Searching...';
            description.textContent = 'Looking up information, please wait';
            break;
        case 'stt':
            badge.classList.add('searching');
            badge.textContent = 'Speech-to-Text';
            description.textContent = 'Converting speech to text...';
            break;
        case 'agent':
            badge.classList.add('searching');
            badge.textContent = 'Agent Decision';
            description.textContent = 'Determining response type...';
            break;
        case 'translation':
            badge.classList.add('searching');
            badge.textContent = 'Translation';
            description.textContent = 'Translating to English...';
            break;
        case 'llm':
            badge.classList.add('searching');
            badge.textContent = 'AI Processing';
            description.textContent = 'Generating response...';
            break;
        case 'tts':
            badge.classList.add('searching');
            badge.textContent = 'Text-to-Speech';
            description.textContent = 'Converting text to speech...';
            break;
        case 'processing':
            badge.classList.add('searching');
            badge.textContent = 'Processing...';
            description.textContent = 'Starting to process your request';
            break;
        default:
            badge.textContent = 'Waiting...';
            description.textContent = 'Ready to process your request';
    }
}

// RAG context updates
function updateRagContext(context) {
    if (context && context.trim() !== '-') {
        ragContext.textContent = context;
        ragContent.style.maxHeight = '200px';
    } else {
        ragContext.textContent = 'No knowledge retrieved yet...';
    }
}

// Real-time processing updates
function handleRealtimeUpdate(update) {
    const { task, status, message, duration, result, timestamp } = update;

    // Update agent decision area with current task
    if (status === 'processing') {
        updateAgentDecision(task);
    }

    // Create or update the processing item
    let processingItem = document.getElementById(`processing-${task}`);

    if (!processingItem) {
        processingItem = document.createElement('div');
        processingItem.id = `processing-${task}`;
        processingItem.className = 'log-item processing-item';
        stepLog.appendChild(processingItem);
    }

    // Update the processing item based on status
    if (status === 'processing') {
        processingItem.classList.add('processing');
        processingItem.classList.remove('completed', 'error');
        processingItem.innerHTML = `
            <div class="log-icon processing-spinner">
                <i class="fas fa-spinner fa-spin"></i>
            </div>
            <div class="processing-content">
                <div class="processing-message">${message}</div>
                <div class="processing-timer" id="timer-${task}">0.0s</div>
            </div>
        `;

        // Start timer for this task
        startProcessingTimer(task, timestamp);

    } else if (status === 'completed') {
        processingItem.classList.remove('processing');
        processingItem.classList.add('completed');
        processingItem.innerHTML = `
            <div class="log-icon success">
                <i class="fas fa-check-circle"></i>
            </div>
            <div class="processing-content">
                <div class="processing-message">${message}</div>
                ${result ? `<div class="processing-result">${result}</div>` : ''}
            </div>
        `;

        // Stop timer for this task
        stopProcessingTimer(task);

    } else if (status === 'error') {
        processingItem.classList.remove('processing');
        processingItem.classList.add('error');
        processingItem.innerHTML = `
            <div class="log-icon error">
                <i class="fas fa-exclamation-triangle"></i>
            </div>
            <div class="processing-content">
                <div class="processing-message">${message}</div>
            </div>
        `;

        stopProcessingTimer(task);
    }

    // Auto-scroll to bottom
    stepLog.scrollTop = stepLog.scrollHeight;
}

// Timer management for processing tasks
const activeTimers = {};

function startProcessingTimer(task, startTimestamp) {
    const timerElement = document.getElementById(`timer-${task}`);
    if (!timerElement) return;

    const startTime = startTimestamp || Date.now();

    activeTimers[task] = setInterval(() => {
        const elapsed = (Date.now() - startTime) / 1000;
        timerElement.textContent = `${elapsed.toFixed(1)}s`;
    }, 100);
}

function stopProcessingTimer(task) {
    if (activeTimers[task]) {
        clearInterval(activeTimers[task]);
        delete activeTimers[task];
    }
}

// Clear all processing items
function clearProcessingLog() {
    // Stop all active timers
    Object.keys(activeTimers).forEach(task => {
        stopProcessingTimer(task);
    });

    // Clear the log
    stepLog.innerHTML = `
        <div class="log-item">
            <div class="log-icon"><i class="fas fa-clock"></i></div>
            <div>Ready to process your request...</div>
        </div>
    `;
}

// Step log updates (legacy support)
function updateStepLog(steps) {
    if (!steps || steps.length === 0) {
        return;
    }

    // Add traditional step log items after processing items
    steps.forEach((step, index) => {
        const logItem = document.createElement('div');
        logItem.className = 'log-item traditional-step';

        // Determine log type based on content
        let logType = 'info';
        let icon = 'fas fa-info-circle';

        if (step.includes('Error') || step.includes('error')) {
            logType = 'error';
            icon = 'fas fa-exclamation-triangle';
        } else if (step.includes('completed') || step.includes('success')) {
            logType = 'success';
            icon = 'fas fa-check-circle';
        }

        logItem.classList.add(logType);
        logItem.innerHTML = `
            <div class="log-icon"><i class="${icon}"></i></div>
            <div>${step}</div>
        `;

        stepLog.appendChild(logItem);
    });

    // Auto-scroll to bottom
    stepLog.scrollTop = stepLog.scrollHeight;
}

// WebSocket event handlers
ws.onopen = () => {
    UIManager.updateConnectionStatus(true);
    appendMessage('Connected to Voice AI Assistant', 'bot');
};

ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);

    switch (msg.type) {
        case 'status':
            updateModelStatus(msg.models);
            if (msg.vad_enabled) {
                vadEnabled = true;
                initializeVAD();
            }
            break;

        case 'realtime_update':
            handleRealtimeUpdate(msg);
            break;

        case 'clear_processing_log':
            clearProcessingLog();
            break;

        case 'text_response':
            UIManager.hideLoading();
            if (msg.input_text) {
                appendMessage(msg.input_text, 'user');
            }
            appendMessage(msg.text, 'bot');
            updateAgentDecision(msg.agent_decision);
            updateRagContext(msg.rag_context);
            updateStepLog(msg.step_log);

            // Show processing time if available
            if (msg.processing_time) {
                const timeItem = document.createElement('div');
                timeItem.className = 'log-item processing-summary';
                timeItem.innerHTML = `
                    <div class="log-icon success">
                        <i class="fas fa-stopwatch"></i>
                    </div>
                    <div>Total processing time: ${msg.processing_time.toFixed(1)}s</div>
                `;
                stepLog.appendChild(timeItem);
                stepLog.scrollTop = stepLog.scrollHeight;
            }
            break;

        case 'audio_response':
            audioPlayer.src = 'data:audio/wav;base64,' + msg.audio;
            audioPlayer.style.display = 'block';
            audioPlayer.play();
            break;

        case 'error':
            UIManager.hideLoading();
            appendMessage('Error: ' + msg.message, 'bot');
            updateStepLog(msg.step_log);
            break;

        case 'vad_sensitivity_updated':
            console.log('VAD sensitivity updated to:', msg.sensitivity);
            break;
    }
};

ws.onclose = () => {
    UIManager.updateConnectionStatus(false);
    appendMessage('Disconnected from server', 'bot');
    stopListening();
};

ws.onerror = (error) => {
    console.error('WebSocket error:', error);
    UIManager.updateConnectionStatus(false);
    appendMessage('Connection error occurred', 'bot');
};

// Text input handling
sendTextBtn.onclick = () => {
    const text = textInput.value.trim();
    if (text) {
        startNewProcessing();
        UIManager.showLoading('Processing your message...');
        ws.send(JSON.stringify({ type: 'text', text: text }));
        appendMessage(text, 'user');
        textInput.value = '';
    }
};

textInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
        sendTextBtn.onclick();
    }
});

// Sensitivity control
sensitivitySlider.addEventListener('input', (e) => {
    const sensitivity = parseFloat(e.target.value);
    sensitivityValue.textContent = sensitivity.toFixed(1);
    
    ws.send(JSON.stringify({ 
        type: 'vad_sensitivity', 
        sensitivity: sensitivity 
    }));
});

// Collapse functionality
ragCollapseBtn.addEventListener('click', () => {
    ragContent.classList.toggle('collapsed');
    const icon = ragCollapseBtn.querySelector('i');
    icon.className = ragContent.classList.contains('collapsed') ? 
        'fas fa-chevron-down' : 'fas fa-chevron-up';
});

logCollapseBtn.addEventListener('click', () => {
    logContent.classList.toggle('collapsed');
    const icon = logCollapseBtn.querySelector('i');
    icon.className = logContent.classList.contains('collapsed') ? 
        'fas fa-chevron-down' : 'fas fa-chevron-up';
});

// Voice Activity Detection
async function initializeVAD() {
    if (!navigator.mediaDevices) {
        appendMessage('Error: Audio recording not supported in this browser.', 'bot');
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
        processor = audioContext.createScriptProcessor(4096, 1, 1);

        processor.onaudioprocess = (event) => {
            if (isListening) {
                const inputData = event.inputBuffer.getChannelData(0);
                sendAudioChunk(inputData);

                // Show voice activity based on audio level
                const audioLevel = calculateAudioLevel(inputData);
                UIManager.showVoiceActivity(audioLevel > 0.01);
            }
        };

        microphone.connect(processor);
        processor.connect(audioContext.destination);

        // Start listening automatically
        startListening();

    } catch (error) {
        console.error('Error initializing VAD:', error);
        appendMessage('Error: Could not access microphone. Please check permissions.', 'bot');
    }
}

function calculateAudioLevel(audioData) {
    let sum = 0;
    for (let i = 0; i < audioData.length; i++) {
        sum += audioData[i] * audioData[i];
    }
    return Math.sqrt(sum / audioData.length);
}

function startListening() {
    if (!isListening) {
        isListening = true;
        UIManager.updateVoiceButton('listening');
        recordBtn.onclick = stopListening;
        appendMessage('🎧 Voice detection started. Speak naturally!', 'bot');
    }
}

function stopListening() {
    if (isListening) {
        isListening = false;
        UIManager.updateVoiceButton('idle');
        UIManager.showVoiceActivity(false);
        recordBtn.onclick = startListening;
        appendMessage('Voice detection stopped.', 'bot');
    }
}

function sendAudioChunk(audioData) {
    // Convert Float32Array to Int16Array
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

// Clear processing log when starting new request
function startNewProcessing() {
    clearProcessingLog();
    updateAgentDecision('processing');
}

// Legacy recording function (fallback)
async function legacyRecord() {
    if (mediaRecorder && mediaRecorder.state === 'recording') {
        mediaRecorder.stop();
        UIManager.updateVoiceButton('idle');
        return;
    }

    if (!navigator.mediaDevices) {
        appendMessage('Error: Audio recording not supported.', 'bot');
        return;
    }

    try {
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
                UIManager.showLoading('Processing your voice message...');
                ws.send(JSON.stringify({ type: 'audio', audio: base64Audio }));
                appendMessage('[Voice message sent]', 'user');
            };
            reader.readAsDataURL(audioBlob);
        };

        mediaRecorder.start();
        UIManager.updateVoiceButton('recording');

    } catch (error) {
        console.error('Error starting recording:', error);
        appendMessage('Error: Could not start recording.', 'bot');
    }
}

// Initialize UI
document.addEventListener('DOMContentLoaded', () => {
    updateAgentDecision('waiting');
    updateRagContext('');
    updateStepLog([]);

    // Set initial button state
    UIManager.updateVoiceButton('idle');
    recordBtn.onclick = () => {
        if (vadEnabled) {
            startListening();
        } else {
            legacyRecord();
        }
    };
});
