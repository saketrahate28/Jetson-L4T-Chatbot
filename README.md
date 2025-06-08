# NVIDIA Jetson L4T Chatbot with RAG

A sophisticated AI-powered chatbot specifically designed for NVIDIA Jetson devices, featuring Retrieval Augmented Generation (RAG) capabilities for intelligent document-based question answering.

![NVIDIA Jetson](https://img.shields.io/badge/NVIDIA-Jetson-76B900?style=for-the-badge&logo=nvidia&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-2.3.3-green?style=for-the-badge&logo=flask&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-Latest-orange?style=for-the-badge)

## 🚀 Overview

The Jetson L4T Chatbot is an intelligent conversational AI system that combines the power of NVIDIA's NIM (NVIDIA Inference Microservices) with advanced document retrieval capabilities. Built specifically for Jetson developers and users, it provides instant, accurate answers about Jetson hardware, software, configuration, and troubleshooting.

### ✨ Key Features

- **🧠 RAG-Enhanced Responses**: Utilizes document embeddings and vector search for contextually accurate answers
- **📚 Comprehensive Knowledge Base**: Pre-loaded with extensive Jetson documentation including:
  - Jetson AGX Orin, Orin NX, and Orin Nano specifications
  - L4T (Linux for Tegra) configuration guides
  - Hardware setup and troubleshooting manuals
  - Graphics programming and multimedia guides
- **🎨 Modern Web Interface**: Clean, responsive chat UI with dark theme
- **⚡ Real-time Processing**: Low-latency responses optimized for edge computing
- **🔒 Secure API Integration**: Server-side API key management
- **📊 Interactive Visualizations**: Built-in charts and status indicators
- **💾 Chat History Management**: Organized conversation history with folder support
- **🔍 Multiple Document Formats**: Supports PDF, DOCX, TXT, and Excel files

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Frontend UI   │◄──►│   Flask Backend  │◄──►│  NVIDIA NIM API │
│   (HTML/JS)     │    │   (Python)       │    │   (LLM Service) │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                        ┌──────────────────┐
                        │   Vector Store   │
                        │   (FAISS + Doc   │
                        │   Embeddings)    │
                        └──────────────────┘
```

### 🧩 Components

1. **Frontend**: Modern chat interface with real-time messaging
2. **Backend**: Flask API server with RAG implementation
3. **Document Processing**: LangChain-based document loaders and text splitters
4. **Vector Database**: FAISS for efficient similarity search
5. **LLM Integration**: NVIDIA NIM API for natural language generation

## 🛠️ Installation & Setup

### Prerequisites

- Python 3.8 or higher
- NVIDIA API key (for NIM services)
- Git (for cloning the repository)

### Step 1: Clone the Repository

```bash
git clone https://github.com/saketrahate28/Jetson-L4T-Chatbot.git
cd Jetson-L4T-Chatbot
```

### Step 2: Install Dependencies

```bash
cd Model
pip install -r requirements.txt
```

### Step 3: Environment Configuration

1. Create a `.env` file in the `Model` directory:
```bash
touch Model/.env
```

2. Add your NVIDIA API key to the `.env` file:
```env
NVIDIA_API_KEY=your_nvidia_api_key_here
```

To get an NVIDIA API key:
- Visit [NVIDIA NGC](https://catalog.ngc.nvidia.com/)
- Sign up/Sign in
- Navigate to "API Keys" section
- Generate a new API key

### Step 4: Add Your Documents (Optional)

Place your documents in the `Data` folder:
- Supported formats: PDF, DOCX, TXT, XLSX
- The system will automatically process and index these documents
- Pre-loaded Jetson documentation is already included

### Step 5: Run the Application

```bash
cd Model
python app1.py
```

The application will start on `http://localhost:3000`

## 🎯 Usage

### Basic Chat
1. Open your browser and navigate to `http://localhost:3000`
2. Start chatting with the bot about Jetson-related topics
3. Ask questions like:
   - "How do I flash Jetson AGX Orin?"
   - "What are the power specifications for Orin Nano?"
   - "How to configure GPIO on Jetson?"

### Features Tour

#### 🗂️ Chat History & Folders
- **New Chat**: Start fresh conversations
- **Folder Organization**: Organize chats by topic (General, Work, Personal)
- **Auto-save**: Conversations are automatically saved
- **Search History**: Quickly find previous conversations

#### 🔖 Bookmark & Share
- **Bookmark**: Save important conversations
- **Share**: Generate shareable links for conversations

#### 🧪 Testing Interface
Visit `http://localhost:3000/test-rag` for:
- RAG system testing
- Debug information
- System status monitoring

#### 📊 NIM Visualization
Visit `http://localhost:3000/nim-visualization` for:
- Interactive benefits overview
- Visual system architecture

## 📁 Project Structure

```
Jetson-L4T-Chatbot/
├── Data/                          # Document knowledge base
│   ├── Jetson AGX Orin/          # AGX Orin specific docs
│   ├── Jetson Orin NX/           # Orin NX specific docs
│   ├── Jetson Orin Nano/         # Orin Nano specific docs
│   └── *.docx, *.pdf, *.xlsx     # Various documentation files
├── Model/                         # Backend application
│   ├── app1.py                   # Main Flask application
│   ├── requirements.txt          # Python dependencies
│   └── .env                      # Environment variables (not in repo)
├── UI/                           # Frontend assets
│   ├── chatbot.html             # Main chat interface
│   ├── nim-api.js               # API integration
│   ├── nim-config.js            # Configuration
│   ├── styles.css               # Styling
│   ├── test-rag.html            # Testing interface
│   ├── nim-visualization.html    # Visualization page
│   └── NVLogo_2D.jpg            # NVIDIA logo
├── .gitignore                    # Git ignore rules
└── README.md                     # This file
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `NVIDIA_API_KEY` | Your NVIDIA NIM API key | Yes |

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main chat interface |
| `/api/rag_chat` | POST | RAG-enhanced chat |
| `/api/chat` | POST | Basic chat without RAG |
| `/api/status` | GET | System status |
| `/api/debug` | GET | Debug information |
| `/test-rag` | GET | Testing interface |
| `/nim-visualization` | GET | Visualization page |

## 🧪 Testing

### Manual Testing
1. Visit `/test-rag` for the testing interface
2. Enter test queries
3. Monitor response quality and system status

### API Testing
```bash
# Test basic connectivity
curl -X GET http://localhost:3000/api/status

# Test RAG functionality
curl -X POST http://localhost:3000/api/rag_chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What is Jetson AGX Orin?"}'
```

## 🔍 Troubleshooting

### Common Issues

1. **API Key Errors**
   - Ensure `.env` file exists with valid `NVIDIA_API_KEY`
   - Check API key permissions and quotas

2. **Document Loading Issues**
   - Verify document formats are supported
   - Check file permissions in the `Data` directory

3. **Connection Issues**
   - Ensure port 3000 is available
   - Check firewall settings

4. **Memory Issues**
   - Large document collections may require more RAM
   - Consider reducing batch sizes in `app1.py`

### Debug Mode
Enable debug logging by checking the console output when running `python app1.py`

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit your changes: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature-name`
5. Submit a pull request

### Development Guidelines
- Follow Python PEP 8 style guidelines
- Add comments for complex logic
- Test your changes thoroughly
- Update documentation as needed

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **NVIDIA** for providing the NIM API and Jetson documentation
- **LangChain** for the RAG framework
- **FAISS** for vector similarity search
- **Flask** for the web framework

## 📞 Support

For issues and questions:
- Create an issue on GitHub
- Check existing documentation in the `Data` folder
- Review the troubleshooting section above

## 🔮 Future Enhancements

- [ ] Voice input/output capabilities
- [ ] Multi-language support
- [ ] Advanced chat analytics
- [ ] Integration with Jetson device monitoring
- [ ] Mobile-responsive improvements
- [ ] Custom model fine-tuning options

---

**Built with ❤️ for the NVIDIA Jetson Community**

*Powered by NVIDIA NIM • Enhanced with RAG • Optimized for Edge Computing*