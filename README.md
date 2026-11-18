# 🔥 AI Fire Safety Trainer

An AI-powered fire and emergency safety training assistant designed to provide interactive, intelligent, and multilingual fire safety guidance. The system uses Artificial Intelligence, Retrieval-Augmented Generation (RAG), speech processing, and text-to-speech technology to help users understand fire hazards, emergency procedures, and safety precautions.

---

## 📌 Project Overview

The **AI Fire Safety Trainer** is an intelligent virtual training assistant that provides real-time responses to fire safety and emergency-related questions.

Instead of relying only on predefined responses, the system retrieves relevant information from a fire safety knowledge base and generates contextual answers using Large Language Models (LLMs).

The project aims to make fire safety training more accessible, interactive, and effective through AI-powered conversations.

### 🎯 Objectives

- Provide AI-based fire and emergency safety guidance.
- Answer questions using a structured fire safety knowledge base.
- Implement Retrieval-Augmented Generation (RAG).
- Support voice-based interaction.
- Enable multilingual communication.
- Provide a foundation for integration with VR-based safety training environments.

---

## ✨ Key Features

- **AI-Powered Question Answering:** Generates contextual answers to fire safety questions.
- **RAG-Based Knowledge Retrieval:** Retrieves relevant information from the fire safety knowledge base before generating responses.
- **Voice Interaction:** Supports speech input and audio responses.
- **Text-to-Speech:** Converts generated responses into spoken audio.
- **Speech Recognition:** Converts user voice input into text.
- **Multilingual Support:** Designed for English, Hindi, and Hinglish interactions.
- **FastAPI Backend:** Provides API endpoints for communication with the application.
- **Safety-Focused Responses:** Designed to prioritize relevant fire safety instructions.
- **VR Integration Foundation:** Supports future integration with virtual training environments and digital human avatars.

---

## 🏗️ System Architecture

The application follows an AI-powered conversational pipeline.

```text
              USER
                |
                v
       Voice / Text Input
                |
                v
       Speech Recognition
                |
                v
       Language & Intent Detection
                |
                v
       Knowledge Retrieval (RAG)
                |
                v
          ChromaDB
                |
                v
       Large Language Model
                |
                v
       Safety Response Validation
                |
                v
        Generated Response
                |
                v
         Text-to-Speech
                |
                v
        Voice / Text Output
                |
                v
          VR AI Trainer
```

---

## 🧠 Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core development |
| FastAPI | Backend API framework |
| Uvicorn | ASGI server |
| Large Language Models | AI response generation |
| ChromaDB | Vector database |
| RAG | Knowledge-based response generation |
| Speech Recognition | Voice-to-text processing |
| Piper TTS | Text-to-speech generation |
| Qwen | Language model |
| Whisper / Faster-Whisper | Speech recognition |
| Git & GitHub | Version control |

---

## 📂 Project Structure

```text
AI-Fire-Safety-Trainer/
│
├── ai/
│   ├── backend/
│   │   └── app.py
│   │
│   ├── config/
│   │
│   ├── rag/
│   │
│   ├── stt/
│   │
│   ├── tts/
│   │
│   └── ...
│
├── config/
│
├── data/
│   └── knowledge_base/
│       └── Fire_Safety_Training.txt
│
├── models/
│
├── scripts/
│
├── process.py
│
├── requirements.txt
│
├── translate_requirements.txt
│
└── README.md
```

*Note: This is a high-level structure. Update the directory tree to match the final contents of your repository.*

---

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/AI-Fire-Safety-Trainer.git
```

### 2. Navigate to the Project

```bash
cd AI-Fire-Safety-Trainer
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Environment

**Windows:**

```powershell
venv\Scripts\activate
```

**Linux / macOS:**

```bash
source venv/bin/activate
```

### 5. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For additional translation dependencies, if required:

```bash
pip install -r translate_requirements.txt
```

---

## 🚀 Running the Application

Run the following command from the project root directory:

```bash
python -m uvicorn ai.backend.app:app --host 0.0.0.0 --port 8000 --reload
```

The backend will start at:

```text
http://localhost:8000
```

### API Documentation

FastAPI automatically provides interactive API documentation.

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

---

## 📚 Knowledge Base

The system uses a fire safety knowledge base to retrieve relevant information for user queries.

Example knowledge-base file:

```text
data/knowledge_base/

Fire_Safety_Training.txt
```

The knowledge base can contain information about:

- Fire prevention procedures
- Fire classifications
- Fire extinguisher types
- Emergency evacuation procedures
- Workplace fire hazards
- Personal protective equipment (PPE)
- Emergency response protocols
- Industrial fire safety practices

The RAG pipeline retrieves relevant knowledge and uses it as context for generating AI responses.

---

## 🔄 AI Processing Workflow

1. **Input Processing:** Receives user questions through text or voice.
2. **Speech Recognition:** Converts audio input into text.
3. **Language Detection:** Identifies the language of the user's query.
4. **Knowledge Retrieval:** Searches relevant fire safety information from the vector database.
5. **AI Generation:** Uses the language model to generate a contextual response.
6. **Safety Validation:** Checks the generated response against safety requirements.
7. **Speech Generation:** Converts the response into audio using TTS.
8. **Output Delivery:** Returns the response through the application interface or connected VR environment.

---

## 🛡️ Safety Considerations

This application is intended for educational and training purposes.

- AI-generated responses should be grounded in approved safety documentation.
- Emergency instructions must be reviewed against applicable workplace procedures.
- The system should not replace certified fire safety professionals.
- During an actual fire emergency, users should follow site emergency procedures and contact emergency services.

---

## 🔮 Future Enhancements

- Interactive 3D human-like AI trainer.
- Real-time VR fire safety simulations.
- Improved multilingual voice conversations.
- Real-time speech response generation.
- Advanced fire scenario simulations.
- User performance tracking and assessment.
- AI-driven emergency response training.
- Integration with industrial safety training platforms.

---

## 👨‍💻 Development

This project is developed using Python and AI technologies, with a modular architecture that supports backend services, knowledge retrieval, language models, speech processing, and future VR integration.

---

## 📄 License

Add your chosen open-source license here before publishing the repository.

---

## ⭐ Acknowledgements

This project brings together open-source AI technologies, language models, speech processing tools, and knowledge retrieval techniques to support intelligent fire and emergency safety training.

**AI Fire Safety Trainer – Making Fire Safety Training Smarter, Interactive, and Accessible.**
