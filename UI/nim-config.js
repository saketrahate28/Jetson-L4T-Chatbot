/**
 * NVIDIA Inference Microservices (NIM) Configuration
 * This file contains settings for connecting to NVIDIA NIM services.
 */


// NIM Configuration object
const NIM_CONFIG = {

    ENDPOINTS: {
        BASE_URL: "http://127.0.0.1:3000",
        RAG_CHAT: "/api/rag_chat",
        GENERATOR: "/api/generator",
        RETRIEVER: "/api/retriever",
        STATUS: "/api/status",
        DEBUG: "/api/debug",
        NIM_STATUS: "/api/nim/status"
    },

    MODELS: {
        CHAT: "nvdev/meta/llama-3.1-70b-instruct",
        TEXT: "nvdev/meta/llama-3.1-70b-instruct",
        EMBEDDINGS: "nvidia/nv-embed-v1"
    },

    // Default parameters for API calls
    DEFAULTS: {
        MAX_TOKENS: 4096,
        TEMPERATURE: 0.6,
        TOP_P: 0.7,
        FREQUENCY_PENALTY: 0,
        PRESENCE_PENALTY: 0,
        STREAM: false
    }
};

/**
 * Test backend connection
 * @returns {Promise<boolean>} Whether the connection was successful
 */
async function testNimConnection() {
    try {
        const response = await fetch(NIM_CONFIG.ENDPOINTS.STATUS, {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json'
            }
        });

        if (!response.ok) {
            console.error("Backend connection test failed:", response.status);
            return false;
        }

        const data = await response.json();
        console.log("Backend connection test successful:", data);
        return data.nim_api === 'configured';
    } catch (error) {
        console.error("Backend connection test error:", error);
        return false;
    }
}

/**
 * Test if the RAG system is ready
 * @returns {Promise<boolean>} Whether the RAG system is ready
 */
async function testRagStatus() {
    try {
        const response = await fetch(NIM_CONFIG.ENDPOINTS.STATUS, {
            method: 'GET',
            headers: { 'Content-Type': 'application/json' }
        });

        if (!response.ok) {
            return false;
        }

        const data = await response.json();
        return data.embeddings_ready === true;
    } catch (error) {
        console.error("RAG status check error:", error);
        return false;
    }
}

/**
 * Get standard headers for all requests - NO authorization
 * @returns {Object} Headers object with content type
 */
function getStandardHeaders() {
    return {
        'Content-Type': 'application/json'
    };
}

/**
 * Standardized function to send chat messages to the backend
 * @param {string} messageText - The message to send
 * @param {boolean} useRag - Whether to use RAG or regular chat
 * @returns {Promise<Object>} The response from the backend
 */
async function sendChatToBackend(messageText, useRag = true) {
    try {
        const endpoint = useRag
            ? NIM_CONFIG.ENDPOINTS.RAG_CHAT
            : `${BACKEND_URL}/api/chat`;

        const requestBody = useRag
            ? { query: messageText }
            : { messages: [{ role: 'user', content: messageText }] };

        const response = await fetch(endpoint, {
            method: 'POST',
            headers: getStandardHeaders(),
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

// Export configuration and functions if using modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        NIM_CONFIG,
        testNimConnection,
        testRagStatus,
        getStandardHeaders,
        sendChatToBackend
    };
}
