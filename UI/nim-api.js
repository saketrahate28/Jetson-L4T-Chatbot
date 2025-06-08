/**
 * NVIDIA Inference Microservices (NIM) API Client
 *
 * This file provides functions for interacting with NVIDIA NIM services.
 * It depends on nim-config.js being loaded first.
 */

// Check if NIM_CONFIG is defined
if (typeof NIM_CONFIG === 'undefined') {
    console.error("NIM_CONFIG is not defined. Make sure nim-config.js is loaded before nim-api.js");
    throw new Error("NIM_CONFIG is not defined");
}

/**
 * NIM API Client
 */
const NimAPI = {
    /**
     * Get standard headers for all requests
     * @private
     * @returns {Object} - Headers object with content type
     */
    _getHeaders() {
        // For server-side authentication, we only need Content-Type
        return {
            'Content-Type': 'application/json'
        };
    },

    /**
     * Send a chat completion request through the backend
     * @param {Array} messages - Array of message objects with role and content
     * @param {Object} options - Optional parameters for the request
     * @returns {Promise} - Promise that resolves to the response
     */
    async chatCompletion(messages, options = {}) {
        if (!Array.isArray(messages) || messages.length === 0) {
            throw new Error("Messages must be a non-empty array");
        }

        // Use the direct endpoint properly formatted
        const url = NIM_CONFIG.ENDPOINTS.BASE_URL + "/api/chat";

        const response = await fetch(url, {
            method: 'POST',
            headers: this._getHeaders(),
            body: JSON.stringify({
                messages: messages,
                model: options.model || NIM_CONFIG.MODELS.CHAT,
                temperature: options.temperature || NIM_CONFIG.DEFAULTS.TEMPERATURE,
                max_tokens: options.max_tokens || NIM_CONFIG.DEFAULTS.MAX_TOKENS,
                stream: options.stream !== undefined ? options.stream : NIM_CONFIG.DEFAULTS.STREAM
            })
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({ error: { message: `Server error: ${response.status}` }}));
            throw new Error(errorData.error?.message || `API error: ${response.status}`);
        }

        return await response.json();
    },

    /**
     * Send a RAG-enhanced chat request
     * @param {string} query - User query for RAG processing
     * @param {Object} options - Optional parameters
     * @returns {Promise} - Promise that resolves to the RAG response
     */
    async ragChat(query, options = {}) {
        if (!query || typeof query !== 'string') {
            throw new Error("Query must be a non-empty string");
        }

        // This is already correct, using the direct endpoint
        const url = NIM_CONFIG.ENDPOINTS.RAG_CHAT;

        const response = await fetch(url, {
            method: 'POST',
            headers: this._getHeaders(),
            body: JSON.stringify({ query: query })
        });

        if (!response.ok) {
            // Better error handling with more detailed messages
            try {
                const errorData = await response.json();
                throw new Error(errorData.error?.message || `API error: ${response.status}`);
            } catch (jsonError) {
                // If the response isn't valid JSON, use a generic error
                throw new Error(`API error: ${response.status}`);
            }
        }

        return await response.json();
    },

    /**
     * Generate embeddings for a text or array of texts through the backend
     * @param {string|Array} input - Text or array of texts to generate embeddings for
     * @param {Object} options - Optional parameters for the request
     * @returns {Promise} - Promise that resolves to the embeddings
     */
    async createEmbeddings(input, options = {}) {
        const texts = Array.isArray(input) ? input : [input];

        // Use direct endpoint properly formatted
        const url = NIM_CONFIG.ENDPOINTS.BASE_URL + "/api/embeddings";

        try {
            const response = await fetch(url, {
                method: 'POST',
                headers: this._getHeaders(),
                body: JSON.stringify({
                    texts: texts,
                    model: options.model || NIM_CONFIG.MODELS.EMBEDDINGS
                })
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({ error: { message: `Server error: ${response.status}` }}));
                throw new Error(errorData.error?.message || `API error: ${response.status}`);
            }

            const data = await response.json();
            return data.embeddings || data.data || data;
        } catch (error) {
            console.error("Embeddings error:", error);
            throw error;
        }
    },

    /**
     * Send a text completion request through the backend
     * @param {string} prompt - The prompt to generate completion for
     * @param {Object} options - Optional parameters for the request
     * @returns {Promise} - Promise that resolves to the response
     */
    async textCompletion(prompt, options = {}) {
        if (!prompt || typeof prompt !== 'string') {
            throw new Error("Prompt must be a non-empty string");
        }

        // Use direct endpoint properly formatted
        const url = NIM_CONFIG.ENDPOINTS.BASE_URL + "/api/generator";

        try {
            const response = await fetch(url, {
                method: 'POST',
                headers: this._getHeaders(),
                body: JSON.stringify({
                    prompt: prompt,
                    model: options.model || NIM_CONFIG.MODELS.TEXT,
                    temperature: options.temperature || NIM_CONFIG.DEFAULTS.TEMPERATURE,
                    max_tokens: options.max_tokens || NIM_CONFIG.DEFAULTS.MAX_TOKENS
                })
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({ error: { message: `Server error: ${response.status}` }}));
                throw new Error(errorData.error?.message || `API error: ${response.status}`);
            }

            return await response.json();
        } catch (error) {
            console.error("Text completion error:", error);
            throw error;
        }
    },

    /**
     * Check the status of the backend API and RAG system
     * @returns {Promise} - Promise that resolves to the status information
     */
    async checkStatus() {
        // This is already correct, using direct endpoint
        const url = NIM_CONFIG.ENDPOINTS.STATUS;

        try {
            const response = await fetch(url, {
                method: 'GET',
                headers: this._getHeaders()
            });

            if (!response.ok) {
                throw new Error(`Server error: ${response.status}`);
            }

            return await response.json();
        } catch (error) {
            console.error("Error checking backend status:", error);
            throw error;
        }
    }
};

// Enhanced message sending function
window.sendMessage = function() {
    console.log("Send button clicked");

    // Get the user input element
    const userInput = document.getElementById('user-input');
    if (!userInput) {
        console.error("User input element not found");
        return;
    }

    const messageText = userInput.value.trim();
    if (!messageText) {
        console.log("Empty message, not sending");
        return;
    }

    console.log("Sending message:", messageText);

    // Add user message to chat
    addMessageToChat(messageText, 'user');

    // Clear the input
    userInput.value = '';

    // Show typing indicator
    const typingIndicator = document.getElementById('typing-indicator');
    if (typingIndicator) {
        typingIndicator.style.display = 'block';
    }

    // Add styling for formatted HTML messages if not already added
    if (!document.getElementById('formatted-message-styles')) {
        const style = document.createElement('style');
        style.id = 'formatted-message-styles';
        style.textContent = `
            .message.bot ul, .message.bot ol {
                padding-left: 20px;
                margin: 8px 0;
            }
            .message.bot li {
                margin-bottom: 4px;
            }
            .message.bot p {
                margin: 8px 0;
            }
            .message.bot strong, .message.bot b {
                color: #ffcc00;
                font-weight: bold;
            }
            .message.bot h1, .message.bot h2, .message.bot h3, .message.bot h4 {
                margin-top: 16px;
                margin-bottom: 8px;
                color: #76b900;
            }
            .message.bot code {
                background-color: #2a2a2a;
                padding: 2px 4px;
                border-radius: 3px;
                font-family: monospace;
            }
            .message.bot pre {
                background-color: #2a2a2a;
                padding: 8px;
                border-radius: 5px;
                overflow-x: auto;
                margin: 8px 0;
            }
            .message.bot table {
                border-collapse: collapse;
                margin: 16px 0;
                width: 100%;
            }
            .message.bot th, .message.bot td {
                border: 1px solid #444;
                padding: 6px 10px;
                text-align: left;
            }
            .message.bot th {
                background-color: #333;
                color: #fff;
            }
        `;
        document.head.appendChild(style);
    }

    // Use the NimAPI.ragChat method to handle the request
    NimAPI.ragChat(messageText)
        .then(data => {
            console.log("Response data:", data);

            // Hide typing indicator
            if (typingIndicator) {
                typingIndicator.style.display = 'none';
            }

            // Add the bot's response to the chat
            if (data.status === 'success' && data.message) {
                // Check if the response format is HTML
                const isHtml = data.format === 'html';

                // For HTML content, use innerHTML, otherwise use textContent
                addMessageToChat(data.message, 'bot');

                // Log RAG debug info if available
                if (data.debug_info && data.debug_info.used_rag) {
                    console.log("RAG information:", data.debug_info);
                }
            } else {
                addMessageToChat("Error: " + (data.message || "Unknown error"), 'bot');
            }
        })
        .catch(error => {
            console.error("Error:", error);

            // Hide typing indicator
            if (typingIndicator) {
                typingIndicator.style.display = 'none';
            }

            // Show error message
            addMessageToChat(`Error: ${error.message}`, 'bot');
        });
};

// Add enhanced CSS styles for better message formatting
document.addEventListener("DOMContentLoaded", function() {
    // Add custom styles for better chat formatting
    const style = document.createElement('style');
    style.textContent = `
        /* Base message styling */
        .message.bot {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            line-height: 1.4;
            color: #333;
        }

        /* Academic/informational styling for better presentation of factual content */
        .message.bot.academic {
            padding-left: 16px;
            border-left: 3px solid #0066cc;
            line-height: 1.5;
        }

        /* Paragraph styling - reduced margins */
        .message.bot p {
            margin: 0.4em 0;
        }

        /* Header styling for articles */
        .message.bot h1, .message.bot h2, .message.bot h3, .message.bot h4 {
            margin: 0.8em 0 0.4em 0;
            color: #0066cc;
            font-weight: 600;
            border-bottom: 1px solid #eee;
            padding-bottom: 4px;
        }

        .message.bot h1 { font-size: 1.5em; }
        .message.bot h2 { font-size: 1.3em; }
        .message.bot h3 { font-size: 1.1em; }
        .message.bot h4 { font-size: 1em; }

        /* Horizontal rule for section dividers */
        .message.bot hr {
            border: 0;
            height: 1px;
            background: #ddd;
            margin: 1em 0;
        }

        /* Notes and callouts */
        .message.bot .note {
            background-color: #f8f9fa;
            border-left: 4px solid #0066cc;
            padding: 8px 12px;
            margin: 8px 0;
            font-size: 0.9em;
            color: #555;
        }

        /* Numbered steps with visual distinction */
        .message.bot .numbered-step {
            margin-top: 0.8em;
            margin-bottom: 0.2em;
        }

        .message.bot .step-number {
            font-weight: bold;
            color: #0066cc;
        }

        /* List styling - tighter spacing */
        .message.bot ul, .message.bot ol {
            margin-top: 0.3em;
            margin-bottom: 0.3em;
            padding-left: 1.5em;
        }

        .message.bot li {
            margin-bottom: 0.2em;
        }

        /* Command block styling */
        .message.bot code {
            background-color: #f0f0f0;
            padding: 2px 4px;
            border-radius: 3px;
            font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
            font-size: 0.9em;
            white-space: nowrap;
            color: #d14;
        }

        /* Larger command blocks */
        .message.bot pre {
            background-color: #f6f8fa;
            border: 1px solid #e1e4e8;
            border-radius: 4px;
            padding: 8px 12px;
            margin: 0.4em 0;
            overflow-x: auto;
        }

        .message.bot pre code {
            background: none;
            border: none;
            padding: 0;
            white-space: pre;
            color: #24292e;
            font-size: 0.9em;
            line-height: 1.5;
        }

        /* Command block styling */
        .message.bot .command-block {
            background-color: #1e1e1e;
            color: #f0f0f0;
            border-left: 3px solid #76b900;
        }

        .message.bot .command-block code {
            color: #f0f0f0;
            background: none;
        }

        /* Strong/bold text emphasis */
        .message.bot strong, .message.bot b {
            font-weight: 600;
            color: #0066cc;
        }

        /* Section title styling */
        .message.bot p strong:first-child {
            display: block;
            margin-top: 0.7em;
            color: #0066cc;
        }

        /* Blockquote styling */
        .message.bot blockquote {
            border-left: 3px solid #dfe2e5;
            padding-left: 1em;
            color: #6a737d;
            margin: 0.5em 0;
        }

        /* Fix spacing in sequential commands */
        .message.bot pre + pre {
            margin-top: 0.3em;
        }

        /* Format citations and references */
        .message.bot .citation {
            font-size: 0.8em;
            color: #666;
            vertical-align: super;
            text-decoration: none;
        }

        /* Special formatting for note sections */
        .message.bot p:last-child:contains("Note:") {
            font-style: italic;
            color: #666;
            border-top: 1px solid #eee;
            padding-top: 8px;
            margin-top: 12px;
        }
    `;
    document.head.appendChild(style);
});

// Helper function to add messages to chat
function addMessageToChat(text, sender) {
    const chatMessages = document.getElementById('chat-messages');
    if (!chatMessages) {
        console.error("Chat messages container not found");
        return;
    }

    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}`;

    if (sender === 'bot') {
        // Render Markdown as HTML using marked.js
        messageDiv.innerHTML = marked.parse(text);
    } else {
        messageDiv.textContent = text;
    }

    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return messageDiv;
}

// Event listeners
document.addEventListener('DOMContentLoaded', function() {
    // Handle Enter key press for input
    const userInput = document.getElementById('user-input');
    if (userInput) {
        userInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });
    }

    // Add click handler for send button
    const sendButton = document.getElementById('send-button');
    if (sendButton) {
        sendButton.addEventListener('click', sendMessage);
    }

    // Initialize typing indicator
    const typingIndicator = document.getElementById('typing-indicator');
    if (typingIndicator) {
        typingIndicator.style.display = 'none';
    }

    // Check RAG system status
    NimAPI.checkStatus()
        .then(status => {
            console.log("NIM Status:", status);
            // Update UI based on status if needed
            updateConnectionStatus(status.embeddings_ready ? 'connected' : 'connecting');
        })
        .catch(error => {
            console.error("Failed to check NIM status:", error);
            updateConnectionStatus('error');
        });

    // Check if we have an API key, if not, try to verify with backend
    if (!NIM_CONFIG.API_KEY) {
        console.log("No API key found in local storage, checking backend configuration...");

        fetch('/api/get_api_key')
            .then(response => response.json())
            .then(data => {
                if (data.key_configured) {
                    console.log("API key configured on server:", data.masked_key);
                    // We don't get the actual key, but we know it exists on the server
                    // Set a special marker in NIM_CONFIG that tells us to rely on server auth
                    NIM_CONFIG.SERVER_AUTH = true;

                    // Update connection status
                    updateConnectionStatus('connected');
                } else {
                    console.error("No API key configured on server");
                    updateConnectionStatus('error');
                }
            })
            .catch(error => {
                console.error("Failed to check API key configuration:", error);
                updateConnectionStatus('error');
            });
    }
});

// Update connection status in UI
function updateConnectionStatus(status) {
    const connectionStatus = document.getElementById('connection-status');
    const connectionText = document.getElementById('connection-text');

    // Skip if elements don't exist
    if (!connectionStatus || !connectionText) return;

    switch (status) {
        case 'connected':
            connectionStatus.style.backgroundColor = '#10b981'; // Green
            connectionText.textContent = 'Connected';
            break;
        case 'connecting':
            connectionStatus.style.backgroundColor = '#f59e0b'; // Amber
            connectionText.textContent = 'Connecting...';
            break;
        case 'error':
            connectionStatus.style.backgroundColor = '#ef4444'; // Red
            connectionText.textContent = 'Connection Error';
            break;
        default:
            connectionStatus.style.backgroundColor = '#6b7280'; // Gray
            connectionText.textContent = 'Unknown Status';
    }
}

async function sendChatToBackend(messageText, useRag = true) {
    try {
        const endpoint = useRag
            ? NIM_CONFIG.ENDPOINTS.RAG_CHAT
            : NIM_CONFIG.ENDPOINTS.BASE_URL + "/api/chat";

        const requestBody = useRag
            ? { query: messageText }
            : { messages: [{ role: 'user', content: messageText }] };

        const response = await fetch(endpoint, {
            method: 'POST',
            headers: NimAPI._getHeaders(),
            body: JSON.stringify(requestBody)
        });

        if (!response.ok) {
            throw new Error(`Server error: ${response.status} ${response.statusText}`);
        }

        return await response.json();
    } catch (error) {
        console.error("Error sending message:", error);
        throw error;
    }
}

// Add CSS for .highlight if not present
if (!document.getElementById('highlight-style')) {
    const style = document.createElement('style');
    style.id = 'highlight-style';
    style.textContent = `.highlight { color: #ffcc00; font-weight: bold; }`;
    document.head.appendChild(style);
}

// Add/replace CSS for clean, modern bot message formatting
if (!document.getElementById('bot-message-style')) {
    const style = document.createElement('style');
    style.id = 'bot-message-style';
    style.textContent = `
.message.bot {
    font-family: 'Segoe UI', Arial, sans-serif;
    color: #e6e6e6;
    background: #232323;
    border-radius: 8px;
    padding: 16px 20px;
    margin: 8px 0;
    font-size: 16px;
    line-height: 1.7;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
}
.message.bot h1, .message.bot h2, .message.bot h3 {
    color: #32CD32;
    margin-top: 1em;
    margin-bottom: 0.5em;
    font-weight: 700;
}
.message.bot h1 { font-size: 1.5em; border-bottom: 1px solid #333; }
.message.bot h2 { font-size: 1.25em; }
.message.bot h3 { font-size: 1.1em; }
.message.bot p {
    margin-bottom: 1em;
}
.message.bot ul, .message.bot ol {
    margin-left: 2em;
    margin-bottom: 1em;
}
.message.bot li {
    margin-bottom: 0.4em;
}
/* Fix for code block visibility */
.message.bot pre,
.message.bot pre code,
.message.bot code {
    background: #181818 !important;
    color: #fff !important;
    border-radius: 4px;
    padding: 2px 6px;
    font-family: 'Fira Mono', monospace;
    font-size: 0.97em;
}
`;
    document.head.appendChild(style);
}