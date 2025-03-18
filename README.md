
# 🤖 Ollama Chatbot

Welcome to the **Ollama Chatbot RAG**! This Python-based chatbot runs locally using the **Ollama Local API** (`http://localhost:11434/api/generate`) and the `llama3.2:latest` model to provide fast, reliable, and offline assistance for your queries.

---

## 🌟 Features

- 🏡 **Runs Locally**: No internet or external API required—everything is handled on your local machine.
- 🤖 **AI-Powered Chat**: Smart and context-aware responses.
- 💻 **Code Generation**: Get Python code snippets and technical help.
- ⚡ **Lightweight and Fast**: Uses the `llama3.2:latest` model for efficient performance.

---

## 📋 Getting Started

Follow these steps to set up and run the chatbot:

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/ImasHF26/Ollama-Chatbot-RAG.git
cd ollama-chatbot-RAG
```

### 2️⃣ Install Dependencies

Make sure Python 3.6 or higher is installed. Set up a virtual environment and install the required libraries:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use .venv\Scripts\activate
pip install -r requirements.txt
```

### 3️⃣ Ensure Local API Is Running 🖥️

The chatbot uses a **local API** to generate responses.  
Verify that the API is running at `http://localhost:11434/api/generate`.

- **Model used**: `llama3.2:latest`  
- **Why this model?** It provides balanced performance, speed, and accuracy for general-purpose conversations.

If the API is not running, refer to the setup guide provided with your local server.

### 4️⃣ Run the Chatbot 🚀

Now, you can start the chatbot:

```bash
python main.py
```

---

## 🛠️ Project Structure

Here’s an overview of the project files:

```
Ollama Chatbot/
│
├── .venv/              # Virtual environment
├── ollama_api.py       # Handles interaction with the local API
├── rag_chatbot.py      # Contains chatbot logic
├── main.py             # Main entry point to run the chatbot
├── fileprocessor.py    # Main entry point to upload multi type files
└── requirements.txt    # List of dependencies
```

---

## 💻 Usage Instructions

1. Launch the chatbot using the `python main.py` command.
2. Start chatting! For example:
   - "Can you generate a Python function to calculate factorial?"
   - "How does the local model work?"

---

## 📌 Important Notes

- This project runs entirely on your local machine without needing an external API.  
- The **local API endpoint** is `http://localhost:11434/api/generate`.  
- The **model used** is `llama3.2:latest` for its high performance in generating accurate and efficient responses.  

---

## 🤝 Contribution

Contributions are welcome! 🛠️  
Fork the repository, create a branch, and submit a pull request.

---


### **Updates Added**:
1. **Local API Endpoint**: Explained the usage of `http://localhost:11434/api/generate`.
2. **Model Details**: Described `llama3.2:latest` and its purpose in providing a lightweight and fast experience.
3. **Removed External Dependencies**: Highlighted that the chatbot doesn’t require external APIs.
4. **Clear Setup Process**: Provided instructions to ensure the local API is running properly.
