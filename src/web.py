
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import threading
import os
import sys
import uuid
import time
from datetime import datetime

try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from model import (
        DeepNovaAI, Transformer, ProductionTokenizer, ModelArgs,
        load_model, get_best_device, get_memory_info, 
        cleanup_memory, logger, torch
    )
    MODEL_AVAILABLE = True
except ImportError as e:
    print(f"Error: {e}")
    MODEL_AVAILABLE = False
    sys.exit(1)

app = Flask(__name__)
CORS(app)

deepnova = None
model_loaded = False
model_loading = False
load_progress = 0
conversation_id = str(uuid.uuid4())[:8]

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DeepNova AI</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            background: #ffffff;
            height: 100vh;
            overflow: hidden;
        }

        ::-webkit-scrollbar {
            width: 6px;
            height: 6px;
        }

        ::-webkit-scrollbar-track {
            background: #f1f1f1;
        }

        ::-webkit-scrollbar-thumb {
            background: #c1c1c1;
            border-radius: 3px;
        }

        ::-webkit-scrollbar-thumb:hover {
            background: #a8a8a8;
        }

        .app {
            display: flex;
            height: 100vh;
        }

        .sidebar {
            width: 260px;
            background: #f9f9fb;
            border-right: 1px solid #e5e5ea;
            display: flex;
            flex-direction: column;
        }

        .sidebar-header {
            padding: 20px;
            border-bottom: 1px solid #e5e5ea;
        }

        .sidebar-header h1 {
            font-size: 18px;
            font-weight: 600;
            color: #1a1a1a;
            margin-bottom: 4px;
        }

        .sidebar-header p {
            font-size: 12px;
            color: #8e8e93;
        }

        .new-chat {
            margin: 16px;
            padding: 10px;
            background: #2196f3;
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 14px;
            font-weight: 500;
            cursor: pointer;
            transition: background 0.2s;
        }

        .new-chat:hover {
            background: #1976d2;
        }

        .model-panel {
            padding: 16px;
            border-bottom: 1px solid #e5e5ea;
        }

        .model-card {
            background: white;
            border-radius: 10px;
            padding: 12px;
            border: 1px solid #e5e5ea;
        }

        .status-row {
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 10px;
        }

        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #c6c6c8;
        }

        .status-dot.ready {
            background: #34c759;
        }

        .status-dot.loading {
            background: #ff9500;
            animation: pulse 1s infinite;
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }

        .status-text {
            font-size: 13px;
            font-weight: 500;
            color: #1c1c1e;
        }

        .model-detail {
            font-size: 11px;
            color: #8e8e93;
            margin-bottom: 10px;
        }

        .progress-bar {
            height: 3px;
            background: #e5e5ea;
            border-radius: 2px;
            overflow: hidden;
            margin-bottom: 10px;
        }

        .progress-fill {
            height: 100%;
            background: #2196f3;
            width: 0%;
            transition: width 0.3s;
        }

        .load-btn {
            width: 100%;
            padding: 8px;
            background: #f2f2f7;
            border: 1px solid #e5e5ea;
            border-radius: 6px;
            font-size: 12px;
            cursor: pointer;
            transition: all 0.2s;
        }

        .load-btn:hover {
            background: #e5e5ea;
        }

        .load-btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }

        .sidebar-footer {
            margin-top: auto;
            padding: 16px;
            border-top: 1px solid #e5e5ea;
        }

        .stat-item {
            display: flex;
            justify-content: space-between;
            font-size: 11px;
            color: #8e8e93;
            margin-bottom: 6px;
        }

        .main {
            flex: 1;
            display: flex;
            flex-direction: column;
        }

        .chat-header {
            padding: 16px 24px;
            border-bottom: 1px solid #e5e5ea;
        }

        .chat-header h2 {
            font-size: 16px;
            font-weight: 600;
            color: #1c1c1e;
        }

        .chat-header p {
            font-size: 12px;
            color: #8e8e93;
            margin-top: 4px;
        }

        .messages {
            flex: 1;
            overflow-y: auto;
            padding: 24px;
        }

        .message {
            display: flex;
            margin-bottom: 24px;
            animation: fadeIn 0.3s ease;
        }

        @keyframes fadeIn {
            from {
                opacity: 0;
                transform: translateY(10px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        .message-avatar {
            width: 32px;
            height: 32px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            font-size: 14px;
            flex-shrink: 0;
            margin-right: 12px;
        }

        .message.user .message-avatar {
            background: #2196f3;
            color: white;
        }

        .message.assistant .message-avatar {
            background: #34c759;
            color: white;
        }

        .message-content {
            flex: 1;
            line-height: 1.5;
            font-size: 14px;
            color: #1c1c1e;
        }

        .message-content pre {
            background: #f2f2f7;
            padding: 12px;
            border-radius: 8px;
            overflow-x: auto;
            margin: 8px 0;
            font-size: 13px;
        }

        .message-content code {
            background: #f2f2f7;
            padding: 2px 6px;
            border-radius: 4px;
            font-family: monospace;
            font-size: 12px;
        }

        .message-time {
            font-size: 11px;
            color: #c6c6c8;
            margin-top: 6px;
        }

        .input-area {
            padding: 16px 24px 24px;
            border-top: 1px solid #e5e5ea;
        }

        .input-wrapper {
            display: flex;
            gap: 12px;
            align-items: flex-end;
            background: #f2f2f7;
            border-radius: 12px;
            padding: 10px 14px;
            border: 1px solid #e5e5ea;
        }

        .input-wrapper:focus-within {
            border-color: #2196f3;
        }

        .message-input {
            flex: 1;
            border: none;
            background: transparent;
            resize: none;
            font-family: inherit;
            font-size: 14px;
            line-height: 1.4;
            outline: none;
            max-height: 120px;
        }

        .send-btn {
            width: 34px;
            height: 34px;
            border-radius: 50%;
            background: #2196f3;
            border: none;
            color: white;
            cursor: pointer;
            font-size: 16px;
            transition: background 0.2s;
            flex-shrink: 0;
        }

        .send-btn:hover:not(:disabled) {
            background: #1976d2;
        }

        .send-btn:disabled {
            background: #c6c6c8;
            cursor: not-allowed;
        }

        .stop-btn {
            width: 34px;
            height: 34px;
            border-radius: 50%;
            background: #ff9500;
            border: none;
            color: white;
            cursor: pointer;
            flex-shrink: 0;
        }

        .typing {
            display: flex;
            gap: 4px;
            padding: 8px 0;
        }

        .typing span {
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background: #c6c6c8;
            animation: typing 1.4s infinite;
        }

        .typing span:nth-child(2) { animation-delay: 0.2s; }
        .typing span:nth-child(3) { animation-delay: 0.4s; }

        @keyframes typing {
            0%, 60%, 100% { transform: translateY(0); opacity: 0.4; }
            30% { transform: translateY(-6px); opacity: 1; }
        }

        .hint {
            font-size: 11px;
            color: #c6c6c8;
            text-align: center;
            margin-top: 8px;
        }
    </style>
</head>
<body>
    <div class="app">
        <div class="sidebar">
            <div class="sidebar-header">
                <h1>DeepNova</h1>
                <p>AI Assistant</p>
            </div>
            
            <button class="new-chat" onclick="newChat()">New Chat</button>
            
            <div class="model-panel">
                <div class="model-card">
                    <div class="status-row">
                        <div class="status-dot" id="statusDot"></div>
                        <span class="status-text" id="statusText">Not Ready</span>
                    </div>
                    <div class="model-detail" id="modelDetail">Model not loaded</div>
                    <div class="progress-bar" id="progressBar" style="display:none">
                        <div class="progress-fill" id="progressFill"></div>
                    </div>
                    <button class="load-btn" id="loadBtn" onclick="loadModel()">Load Model</button>
                </div>
            </div>
            
            <div class="sidebar-footer">
                <div class="stat-item">
                    <span>Messages</span>
                    <span id="msgCount">0</span>
                </div>
                <div class="stat-item">
                    <span>Device</span>
                    <span id="deviceInfo">-</span>
                </div>
            </div>
        </div>
        
        <div class="main">
            <div class="chat-header">
                <h2>DeepNova AI</h2>
                <p>Advanced Intelligence</p>
            </div>
            
            <div class="messages" id="messages">
                <div class="message assistant">
                    <div class="message-avatar">AI</div>
                    <div class="message-content">
                        Hello! I am DeepNova, an intelligent AI assistant.<br><br>
                        I can help answer questions, learn from text files, remember conversations, and recall knowledge.<br><br>
                        How can I help you today?
                        <div class="message-time">Now</div>
                    </div>
                </div>
            </div>
            
            <div class="input-area">
                <div class="input-wrapper">
                    <textarea class="message-input" id="messageInput" placeholder="Ask me anything..." rows="1" onkeydown="handleKey(event)"></textarea>
                    <button class="send-btn" id="sendBtn" onclick="sendMessage()">→</button>
                    <button class="stop-btn" id="stopBtn" onclick="stopGeneration()" style="display:none">■</button>
                </div>
                <div class="hint">Enter to send, Shift+Enter for new line</div>
            </div>
        </div>
    </div>

    <script>
        let isGenerating = false;
        let currentMessageDiv = null;
        let statusInterval = null;
        
        const textarea = document.getElementById('messageInput');
        textarea.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = Math.min(this.scrollHeight, 120) + 'px';
        });
        
        function handleKey(event) {
            if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault();
                sendMessage();
            }
        }
        
        function checkStatus() {
            fetch('/api/status')
                .then(res => res.json())
                .then(data => {
                    const dot = document.getElementById('statusDot');
                    const statusText = document.getElementById('statusText');
                    const modelDetail = document.getElementById('modelDetail');
                    const loadBtn = document.getElementById('loadBtn');
                    const progressBar = document.getElementById('progressBar');
                    const progressFill = document.getElementById('progressFill');
                    const deviceInfo = document.getElementById('deviceInfo');
                    
                    if (data.loaded) {
                        dot.className = 'status-dot ready';
                        statusText.textContent = 'Ready';
                        modelDetail.textContent = data.device ? `Running on ${data.device.toUpperCase()}` : 'Ready';
                        loadBtn.textContent = 'Loaded';
                        loadBtn.disabled = true;
                        progressBar.style.display = 'none';
                        deviceInfo.textContent = data.device ? data.device.toUpperCase() : 'CPU';
                    } else if (data.loading) {
                        dot.className = 'status-dot loading';
                        statusText.textContent = 'Loading...';
                        modelDetail.textContent = `${data.progress || 0}%`;
                        loadBtn.textContent = 'Loading...';
                        loadBtn.disabled = true;
                        progressBar.style.display = 'block';
                        progressFill.style.width = (data.progress || 0) + '%';
                    } else {
                        dot.className = 'status-dot';
                        statusText.textContent = 'Not Ready';
                        modelDetail.textContent = 'Click Load Model';
                        loadBtn.textContent = 'Load Model';
                        loadBtn.disabled = false;
                        progressBar.style.display = 'none';
                    }
                });
        }
        
        function loadModel() {
            fetch('/api/load', { method: 'POST' })
                .then(res => res.json())
                .then(data => {
                    if (data.success) {
                        addSystemMessage('Loading model...');
                        if (!statusInterval) {
                            statusInterval = setInterval(checkStatus, 1000);
                        }
                        checkStatus();
                    }
                });
        }
        
        function newChat() {
            fetch('/api/clear', { method: 'POST' })
                .then(() => {
                    document.getElementById('messages').innerHTML = '';
                    addSystemMessage('Chat cleared. Starting fresh!');
                });
        }
        
        function sendMessage() {
            if (isGenerating) return;
            
            const input = document.getElementById('messageInput');
            const message = input.value.trim();
            if (!message) return;
            
            input.value = '';
            input.style.height = 'auto';
            
            addMessage('user', message);
            
            isGenerating = true;
            const sendBtn = document.getElementById('sendBtn');
            const stopBtn = document.getElementById('stopBtn');
            sendBtn.disabled = true;
            sendBtn.style.display = 'none';
            stopBtn.style.display = 'block';
            
            const loadingId = addTypingIndicator();
            
            fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: message })
            })
            .then(res => res.json())
            .then(data => {
                removeTypingIndicator(loadingId);
                if (data.response) {
                    addMessage('assistant', data.response);
                } else if (data.error) {
                    addSystemMessage('Error: ' + data.error);
                }
            })
            .catch(err => {
                removeTypingIndicator(loadingId);
                addSystemMessage('Error: ' + err.message);
            })
            .finally(() => {
                isGenerating = false;
                sendBtn.disabled = false;
                sendBtn.style.display = 'block';
                stopBtn.style.display = 'none';
                updateStats();
            });
        }
        
        function stopGeneration() {
            if (!isGenerating) return;
            fetch('/api/stop', { method: 'POST' })
                .then(() => {
                    isGenerating = false;
                    const sendBtn = document.getElementById('sendBtn');
                    const stopBtn = document.getElementById('stopBtn');
                    sendBtn.disabled = false;
                    sendBtn.style.display = 'block';
                    stopBtn.style.display = 'none';
                    addSystemMessage('Generation stopped');
                });
        }
        
        function addMessage(role, content) {
            const messagesDiv = document.getElementById('messages');
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${role}`;
            messageDiv.innerHTML = `
                <div class="message-avatar">${role === 'user' ? 'U' : 'AI'}</div>
                <div class="message-content">
                    ${escapeHtml(content)}
                    <div class="message-time">${new Date().toLocaleTimeString()}</div>
                </div>
            `;
            messagesDiv.appendChild(messageDiv);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }
        
        function addSystemMessage(content) {
            const messagesDiv = document.getElementById('messages');
            const messageDiv = document.createElement('div');
            messageDiv.className = 'message assistant';
            messageDiv.innerHTML = `
                <div class="message-avatar">●</div>
                <div class="message-content">
                    <em>${escapeHtml(content)}</em>
                    <div class="message-time">${new Date().toLocaleTimeString()}</div>
                </div>
            `;
            messagesDiv.appendChild(messageDiv);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }
        
        function addTypingIndicator() {
            const messagesDiv = document.getElementById('messages');
            const id = 'typing-' + Date.now();
            const div = document.createElement('div');
            div.id = id;
            div.className = 'message assistant';
            div.innerHTML = `
                <div class="message-avatar">AI</div>
                <div class="message-content">
                    <div class="typing"><span></span><span></span><span></span></div>
                </div>
            `;
            messagesDiv.appendChild(div);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
            return id;
        }
        
        function removeTypingIndicator(id) {
            const element = document.getElementById(id);
            if (element) element.remove();
        }
        
        function updateStats() {
            fetch('/api/stats')
                .then(res => res.json())
                .then(data => {
                    document.getElementById('msgCount').textContent = data.message_count || 0;
                });
        }
        
        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }
        
        checkStatus();
        statusInterval = setInterval(checkStatus, 3000);
        setInterval(updateStats, 5000);
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/api/status')
def status():
    return jsonify({
        'loaded': model_loaded,
        'loading': model_loading,
        'progress': load_progress,
        'device': get_best_device() if model_loaded else None
    })

@app.route('/api/load', methods=['POST'])
def load():
    global model_loading, load_progress
    
    if model_loaded or model_loading:
        return jsonify({'success': True, 'message': 'Model already loading or loaded'})
    
    def load_thread():
        global deepnova, model_loaded, model_loading, load_progress
        
        model_loading = True
        load_progress = 10
        
        try:
            load_progress = 20
            args = ModelArgs.deepseek_v3_lite()
            args.device = get_best_device()
            
            load_progress = 40
            tokenizer = ProductionTokenizer()
            
            load_progress = 60
            model = Transformer(args)
            model = model.to(torch.device(args.device))
            model.eval()
            
            load_progress = 80
            deepnova = DeepNovaAI(model, tokenizer, args)
            
            load_progress = 100
            model_loaded = True
            
            print(f"Model loaded on {args.device}")
            
        except Exception as e:
            print(f"Load error: {e}")
            model_loaded = False
        
        finally:
            model_loading = False
    
    threading.Thread(target=load_thread, daemon=True).start()
    return jsonify({'success': True})

@app.route('/api/chat', methods=['POST'])
def chat():
    global deepnova, model_loaded
    
    if not model_loaded:
        return jsonify({'error': 'Model not loaded'}), 503
    
    data = request.json
    message = data.get('message', '')
    
    if not message:
        return jsonify({'error': 'Empty message'}), 400
    
    try:
        response = deepnova.chat(
            message,
            max_new_tokens=500,
            temperature=0.7
        )
        return jsonify({'response': response})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/clear', methods=['POST'])
def clear():
    global deepnova
    if deepnova:
        deepnova.clear_context(keep_important=True)
    return jsonify({'success': True})

@app.route('/api/stop', methods=['POST'])
def stop():
    global deepnova
    return jsonify({'success': True})

@app.route('/api/stats')
def stats():
    if deepnova:
        stats = deepnova.get_stats()
        return jsonify({
            'message_count': stats.get('total_messages', 0),
            'token_count': stats.get('total_tokens_generated', 0)
        })
    return jsonify({'message_count': 0, 'token_count': 0})

if __name__ == '__main__':
    print("\n" + "="*50)
    print("DeepNova Web Server")
    print("="*50)
    print("Open browser at: http://localhost:5000")
    print("="*50 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)