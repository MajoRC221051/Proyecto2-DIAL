import os
import streamlit as st
from pinecone import Pinecone, ServerlessSpec
import openai
from dotenv import load_dotenv
from typing import List

# Cargar variables de entorno
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

# Inicializar cliente Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
INDEX_NAME = "consulta-tecnica"

# Crear índice si no existe
if INDEX_NAME not in [idx.name for idx in pc.list_indexes()]:
    pc.create_index(
        name=INDEX_NAME,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

index = pc.Index(INDEX_NAME)

def embed_text(texts: List[str]) -> List[List[float]]:
    response = openai.embeddings.create(
        model="text-embedding-ada-002",
        input=texts
    )
    return [item.embedding for item in response.data]

def insert_content(contents: List[str]):
    embeddings = embed_text(contents)
    stats = index.describe_index_stats()  # <- CORREGIDO AQUÍ
    existing_count = stats.namespaces.get("", {}).get("vector_count", 0)
    vectors = [(str(existing_count + i), embeddings[i]) for i in range(len(contents))]
    index.upsert(vectors=vectors)

def query_pinecone(question: str, top_k: int = 5):
    question_embedding = embed_text([question])[0]
    res = index.query(vector=question_embedding, top_k=top_k, include_metadata=False)
    return [match['id'] for match in res['matches']]

def load_contents():
    if not os.path.exists("data/contenido.txt"):
        return []
    with open("data/contenido.txt", "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def save_content(new_content: str):
    with open("data/contenido.txt", "a", encoding="utf-8") as f:
        f.write(new_content.strip() + "\n")

def generate_response(question: str, context: str):
    prompt = f"""
Eres un asistente técnico y profesional. Usa el siguiente contexto para responder a la pregunta de forma clara y precisa.
Contexto: {context}
Pregunta: {question}
Respuesta:
"""
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return response.choices[0].message.content

# Estilos visuales
st.markdown("""
<style>
:root {
  --bg: #ffffff;
  --text-primary: #111827;
  --text-secondary: #6b7280;
  --font-sans: 'Inter', sans-serif;
}
body, .block-container {
  background-color: var(--bg);
  color: var(--text-primary);
  font-family: var(--font-sans);
  max-width: 1200px;
  margin: auto;
  padding: 4rem 2rem 6rem 2rem;
}
h1 {
  font-weight: 800;
  font-size: 3rem;
  margin-bottom: 0.5rem;
}
p.subtitle {
  color: var(--text-secondary);
  font-size: 1.125rem;
  margin-top: 0;
  margin-bottom: 3rem;
}
section.card {
  background: #f9fafb;
  border-radius: 0.75rem;
  padding: 2rem;
  box-shadow: 0 2px 8px rgb(0 0 0 / 0.05);
  margin-bottom: 2rem;
}
button.streamlit-button {
  background-color: var(--text-primary) !important;
  border-radius: 0.5rem !important;
  font-weight: 700 !important;
  padding: 0.6rem 1.2rem !important;
  transition: background-color 0.3s ease !important;
}
button.streamlit-button:hover {
  background-color: #374151 !important;
}
textarea, input[type=text] {
  border: 1px solid #d1d5db !important;
  border-radius: 0.5rem !important;
  font-size: 1rem !important;
  padding: 0.75rem 1rem !important;
  width: 100% !important;
  margin-bottom: 1rem !important;
  box-sizing: border-box !important;
}
h2 {
  font-weight: 600;
  font-size: 1.75rem;
  margin-top: 3rem;
  margin-bottom: 1rem;
  color: var(--text-primary);
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>Asistente de Consulta Técnica</h1>", unsafe_allow_html=True)
st.markdown('<p class="subtitle">Haz preguntas y recibe respuestas basadas en contenido técnico cargado por estudiantes.</p>', unsafe_allow_html=True)

# Panel de contenido
with st.expander("Insertar nuevo contenido para ampliar la base de conocimiento"):
    new_content = st.text_area("Ingresa texto técnico o explicación relevante")
    boton_agregar = st.button("Agregar contenido")
    if boton_agregar:
        if new_content.strip():
            save_content(new_content.strip())
            insert_content([new_content.strip()])
            st.success("Contenido agregado con éxito. Recarga la página para actualizar la base.")
        else:
            st.error("Por favor ingresa contenido antes de agregar.")

contents = load_contents()
if len(contents) < 70:
    st.warning(f"Actualmente hay {len(contents)} registros, se recomienda cargar al menos 70 para mejores resultados.")

# Panel de preguntas
question = st.text_input("Pregúntame algo sobre los temas técnicos disponibles:")
boton_enviar = st.button("Enviar")
if boton_enviar:
    if not question.strip():
        st.error("Debe escribir una pregunta para continuar.")
    elif len(contents) == 0:
        st.error("No hay contenido en la base, agrega textos técnicos primero.")
    else:
        matched_ids = query_pinecone(question, top_k=5)
        matched_texts = [contents[int(i)] for i in matched_ids if int(i) < len(contents)]
        context = "\n".join(matched_texts)
        answer = generate_response(question, context)
        st.markdown("<h2>Respuesta</h2>", unsafe_allow_html=True)
        st.write(answer)
        st.markdown("<h2>Contexto utilizado</h2>", unsafe_allow_html=True)
        for i, txt in enumerate(matched_texts, 1):
            st.markdown(f"{i}. {txt}")
