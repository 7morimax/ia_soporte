import os
import requests
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Paso 1: Cargar y dividir el único archivo por ticket
def load_tickets_from_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Separa los tickets usando un delimitador, como === o [TICKET]
    raw_tickets = content.split("=== Ticket")
    docs = []

    for raw in raw_tickets:
        lines = raw.strip().splitlines()
        if not lines:
            continue
        ticket_text = "\n".join(lines)
        docs.append(Document(page_content=ticket_text))

    return docs

# Paso 2: Dividir en chunks (si cada ticket es largo)
def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(docs)

# Paso 3: Embeddings y vector store con Chroma
def create_vectorstore(chunks):
    embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(chunks, embedding_model, persist_directory="db")
    return vectorstore

# Paso 4: Consulta a Ollama con contexto
def ask_deepseek(prompt, context):
    full_prompt = f"""Responde basándote en el siguiente contexto técnico, tienes que ser breve, al final de cada respuesta indicar que el area de Sistemas se comunicará:\n\n{context}\n\nPregunta: {prompt}"""
    
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "deepseek-r1:7b",
            "prompt": full_prompt,
            "stream": False
        }
    )
    
    print(response.json())  # 👈 Esto te mostrará qué devuelve exactamente
    return response.json()


# --- FLUJO PRINCIPAL ---
def main():
    file_path = "./tickets.txt"
    
    print("📄 Cargando y procesando tickets...")
    docs = load_tickets_from_file(file_path)
    chunks = split_documents(docs)
    
    print("🧠 Indexando con embeddings...")
    vectorstore = create_vectorstore(chunks)

    while True:
        question = input("\n❓ Ingresa tu pregunta (o escribe 'salir'): ")
        if question.lower() == "salir":
            break

        docs = vectorstore.similarity_search(question, k=3)
        context = "\n".join([doc.page_content for doc in docs])

        print("🤖 DeepSeek responde...")
        respuesta = ask_deepseek(question, context)
        print(f"\n📝 Respuesta: {respuesta}")

if __name__ == "__main__":
    main()
