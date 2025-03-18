import sys
import time
import os
import pickle
import faiss
import numpy as np
from colorama import Fore, Style
from sentence_transformers import SentenceTransformer
from ollama_api import OllamaAPI
from file_processor import FileProcessor

class RAGChatbot:
    def __init__(self, ollama_api, chunk_size=1024, chunk_overlap=128, faiss_index_file='faiss_index.faiss', metadata_file='metadata.pickle', hashes_file='hashes.pickle'):
        self.ollama_api = ollama_api
        self.embedding_model = SentenceTransformer('all-mpnet-base-v2')  # Modèle performant
        self.file_processor = FileProcessor(chunk_size, chunk_overlap)
        self.dimension = 768  # Dimension pour 'all-mpnet-base-v2'
        self.faiss_index_file = faiss_index_file
        self.metadata_file = metadata_file
        self.hashes_file = hashes_file

        self.index = self.load_or_initialize_index()
        self.metadata = self.load_or_initialize_metadata()
        self.load_processed_hashes()

    def normalize_embedding(self, embedding):
        norm = np.linalg.norm(embedding)
        return embedding / norm if norm > 0 else embedding

    def load_or_initialize_index(self):
        if os.path.exists(self.faiss_index_file):
            return faiss.read_index(self.faiss_index_file)
        else:
            return faiss.IndexFlatIP(self.dimension)

    def load_or_initialize_metadata(self):
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'rb') as f:
                return pickle.load(f)
        else:
            return []

    def load_processed_hashes(self):
        if os.path.exists(self.hashes_file):
            with open(self.hashes_file, 'rb') as f:
                self.file_processor.processed_hashes = pickle.load(f)

    def save_state(self):
        faiss.write_index(self.index, self.faiss_index_file)
        with open(self.metadata_file, 'wb') as f:
            pickle.dump(self.metadata, f)
        with open(self.hashes_file, 'wb') as f:
            pickle.dump(self.file_processor.processed_hashes, f)

    def preprocess_text(self, text):
        return text.lower().strip()

    def index_file(self, file_path):
        try:
            chunks, file_hash = self.file_processor.process_file(file_path)
            if chunks is None:
                return

            embeddings = []
            preprocessed_chunks = [self.preprocess_text(chunk) for chunk in chunks]
            for chunk in preprocessed_chunks:
                embedding = self.embedding_model.encode([chunk])[0]
                normalized_embedding = self.normalize_embedding(embedding)
                embeddings.append(normalized_embedding)

            embeddings = np.array(embeddings, dtype='float32')
            start_index = self.index.ntotal
            self.index.add(embeddings)

            for i, chunk in enumerate(chunks):
                self.metadata.append((file_hash, i, chunk))

            print("Chunks indexés :")
            for i, chunk in enumerate(chunks):
                print(f"Chunk {i} : {chunk}")

            self.save_state()
            print(f"Fichier {file_path} indexé avec succès. {len(chunks)} chunks ajoutés.")
        except Exception as e:
            print(f"Erreur lors de l'indexation du fichier {file_path} : {str(e)}")

    def find_relevant_context(self, user_query, top_k=3, similarity_threshold=0.6):
        preprocessed_query = self.preprocess_text(user_query)
        query_embedding = self.embedding_model.encode([preprocessed_query])[0]
        normalized_query = self.normalize_embedding(query_embedding).reshape(1, -1)

        if self.index.ntotal == 0:
            print("Aucun contexte indexé dans Faiss.")
            return None

        distances, indices = self.index.search(normalized_query, top_k)
        relevant_chunks = []

        print(f"Requête : {user_query}")
        print(f"Distances (similarités) : {distances[0]}")
        print(f"Indices : {indices[0]}")

        for i, distance in zip(indices[0], distances[0]):
            if i >= 0 and distance >= similarity_threshold:
                chunk = self.metadata[i][2]
                #print(f"Chunk trouvé (similarité {distance}) : {chunk}")
                relevant_chunks.append(chunk)

        if not relevant_chunks:
            print("Aucun chunk pertinent trouvé (similarité inférieure au seuil).")

        return relevant_chunks if relevant_chunks else None

    def generate_response(self, user_query):
        context = self.find_relevant_context(user_query)
        if context:
            context_text = "\n".join(context)
            prompt = f"Contexte :\n{context_text}\n\nQuestion : {user_query}\nRéponse :"
        else:
            prompt = f"Vous êtes un assistant autonome. Répondez à la question suivante de manière concise et précise, en vous basant sur vos connaissances générales si nécessaire.\n\nQuestion : {user_query}\nRéponse :"
        # \n{prompt}
        print(f"Prompt envoyé à ollama :")
        response = self.ollama_api.chat_with_ollama(prompt)
        return response

    def simulate_typing(self, text, delay=0.009):
        for char in text:
            sys.stdout.write(char)
            sys.stdout.flush()
            time.sleep(delay)
        print()

    def format_response(self, response):
        formatted_response = ""
        for char in response:
            if char.isdigit():
                formatted_response += Fore.BLUE + char + Style.RESET_ALL
            elif char == '`':
                formatted_response += Fore.GREEN + char + Style.RESET_ALL
            else:
                formatted_response += char
        return formatted_response

    def display_welcome_message(self):
        welcome_message = Fore.GREEN + "Chatbot: Bonjour ! Tapez 'exit' pour quitter." + Style.RESET_ALL
        self.simulate_typing(welcome_message)

    def display_exit_message(self):
        exit_message = Fore.RED + "Chatbot: Au revoir ! À bientôt !" + Style.RESET_ALL
        self.simulate_typing(exit_message)

    def chat(self):
        self.display_welcome_message()
        while True:
            user_input = input("\n" + Fore.YELLOW + "Vous : " + Style.RESET_ALL)

            if user_input.lower() == "exit":
                self.display_exit_message()
                break

            try:
                response = self.generate_response(user_input)
                formatted_response = self.format_response(response)
                self.simulate_typing(Fore.CYAN + f"Chatbot : {formatted_response}" + Style.RESET_ALL)
            except Exception as e:
                self.simulate_typing(Fore.RED + f"Chatbot : Désolé, une erreur s'est produite : {str(e)}" + Style.RESET_ALL)