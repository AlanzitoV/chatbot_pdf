from PyPDF2 import PdfReader
import os 
import streamlit as st

# Importar las librerías necesarias de langchain
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from streamlit_chat import message

# Configurar la clave de la API de OpenAI
OPENAI_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Crear llaves de sesión para almacenar respuestas y solicitudes en la conversación
if 'responses' not in st.session_state:
    st.session_state['responses'] = ["¡Hola!, ¿En qué puedo ayudarte?"]

if 'requests'not in st.session_state:
    st.session_state['requests'] = []

# Configurar la página de Streamlit
st.set_page_config(page_title="Chatbot con PDF", layout="wide")
st.markdown("""<style>.block-container {padding-top: 1rem;}</style>""", unsafe_allow_html=True)

# Función para crear embeddings a partir de un archivo PDF
def create_embeddings(pdf):
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # Dividir el texto en fragmentos
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        
        embeddings = OpenAIEmbeddings()
        embeddings_pdf = FAISS.from_texts(chunks, embeddings)
        
        return embeddings_pdf

# Barra lateral para cargar el archivo PDF
st.sidebar.markdown("<h1 style='text-align: center; color: #176B87;'>Cargar Archivo PDF</h1>", unsafe_allow_html=True)
st.sidebar.write("Carga el archivo .pdf con el cual quieres interactuar")
pdf_doc = st.sidebar.file_uploader("", type="pdf")
st.sidebar.write("---")
clear_button = st.sidebar.button("Limpiar Conversación", key="clear")

# Crear embeddings a partir del PDF cargado
embeddings_pdf = create_embeddings(pdf_doc)

# Sección de chat
st.markdown("<h2 style='text-align: center; color: #176887; text-decoration: underline;'><strong>Interactúa con el BOT sobre tu documento</strong></h2>", unsafe_allow_html=True)
st.write("---")

# Contenedor para el historial del chat
response_container = st.container()

# Contenedor para el cuadro de texto
textcontainer = st.container()

# Crear el campo para ingresar la pregunta del usuario
with textcontainer:
    # Formulario de entrada de texto
    with st.form(key='my_form', clear_on_submit=True):
        query = st.text_area("Tu:", key='input', height=100)
        submit_button = st.form_submit_button(label='Enviar')
        
    if query:
        with st.spinner("Escribiendo..."):
            # Calcular similitud coseno con los embeddings del PDF
            docs = embeddings_pdf.similarity_search(query)
            
            # Realizar una pregunta al modelo de OpenAI
            llm = OpenAI(model_name="text-davinci-003")
            chain = load_qa_chain(llm, chain_type="stuff")
            
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                print(cb)
                
        # Almacenar la pregunta y la respuesta en la conversación
        st.session_state.requests.append(query)
        st.session_state.responses.append(response)

# Configurar el contenedor de respuestas para mostrar el historial del chat
with response_container:
    if st.session_state['responses']:
        for i in range(len(st.session_state['responses'])):
            message(st.session_state['responses'][i], key=str(i), avatar_style="pixel-art")
            if i < len(st.session_state['requests']):
                message(st.session_state["requests"][i], is_user=True, key=str(i) + '_user')