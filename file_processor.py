import os
import hashlib
import json
import PyPDF2
from docx import Document

class FileProcessor:
    def __init__(self, chunk_size=1024, chunk_overlap=128):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.processed_hashes = set()  # Pour stocker les hashes des fichiers déjà traités

    def calculate_hash(self, content):
        """
        Calcule le hash SHA256 du contenu d'un fichier.
        """
        return hashlib.sha256(content.encode('utf-8')).hexdigest()

    def read_file(self, file_path):
        """
        Lit le contenu d'un fichier selon son type (txt, pdf, docx, json).
        """
        file_extension = os.path.splitext(file_path)[1].lower()
        content = ""

        try:
            if file_extension == '.txt':
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
            elif file_extension == '.pdf':
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page in pdf_reader.pages:
                        content += page.extract_text() + "\n"
            elif file_extension == '.docx':
                doc = Document(file_path)
                for para in doc.paragraphs:
                    content += para.text + "\n"
            elif file_extension == '.json':
                with open(file_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    # Convertir tout le JSON en texte (peut être affiné selon la structure)
                    content = json.dumps(data, ensure_ascii=False)
            else:
                raise ValueError(f"Type de fichier non pris en charge : {file_extension}")
            
            return content.strip()
        except Exception as e:
            raise Exception(f"Erreur lors de la lecture du fichier {file_path} : {str(e)}")

    def split_into_chunks(self, text):
        """
        Divise le texte en chunks avec chevauchement.
        """
        if not text:
            return []

        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = start + self.chunk_size
            chunk = text[start:end].strip()
            if chunk:  # Ignorer les chunks vides
                chunks.append(chunk)
            start += self.chunk_size - self.chunk_overlap  # Décalage avec chevauchement

        return chunks

    def process_file(self, file_path):
        """
        Traite un fichier : lit, calcule le hash, vérifie s'il est déjà traité, et retourne les chunks.
        """
        content = self.read_file(file_path)
        file_hash = self.calculate_hash(content)

        if file_hash in self.processed_hashes:
            print(f"Le fichier {file_path} a déjà été traité.")
            return None, file_hash

        self.processed_hashes.add(file_hash)
        chunks = self.split_into_chunks(content)
        return chunks, file_hash