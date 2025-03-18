import os
from ollama_api import OllamaAPI
from rag_chatbot import RAGChatbot

def main():
    ollama_api = OllamaAPI()
    chatbot = RAGChatbot(ollama_api)

    print("Bienvenue dans le système RAG !")
    print("Commandes disponibles :")
    print("- 'index chemin_du_fichier' : pour indexer un fichier (txt, pdf, docx, json)")
    print("- 'chat' : pour démarrer une conversation")
    print("- 'exit' : pour quitter")

    while True:
        command = input("\nEntrez une commande : ").strip()

        if command.lower() == "exit":
            print("Au revoir !")
            break
        elif command.lower() == "chat":
            chatbot.chat()
        elif command.lower().startswith("index "):
            file_path = command[6:].strip()
            if file_path:
                if os.path.exists(file_path):
                    chatbot.index_file(file_path)
                else:
                    print(f"Le fichier {file_path} n'existe pas ou n'est pas accessible.")
            else:
                print("Veuillez fournir un chemin de fichier valide.")
        else:
            print("Commande non reconnue. Veuillez réessayer.")

if __name__ == "__main__":
    main()