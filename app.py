import streamlit as st
import pandas as pd
import os
import google.generativeai as genai
from langchain.document_loaders import PyPDFLoader
from PyPDF2 import PdfReader
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.agents import Tool
from langchain.embeddings import HuggingFaceEmbeddings
from io import BytesIO
from uuid import uuid4
from dotenv import main
import gc

# Load environment variables
# main.load_dotenv()

#=================
# CSS and Page Configuration
#=================
st.set_page_config(page_title="QUILLERY-DeepMindCraft", page_icon=":brain:", layout="wide")

css = '''
<style>
    .stApp { background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); }
    .css-1d391kg { background-color: rgba(255, 255, 255, 0.1); backdrop-filter: blur(10px); border-radius: 20px; padding: 2rem; box-shadow: 20px 20px 60px #bebebe, -20px -20px 60px #ffffff; }
    .css-1v0mbdj { background-color: rgba(255, 255, 255, 0.05); border-radius: 15px; padding: 1rem; box-shadow: inset 5px 5px 10px #bebebe, inset -5px -5px 10px #ffffff; }
    .stButton>button { border-radius: 10px; border: none; padding: 10px 20px; background: linear-gradient(145deg, #e6e6e6, #ffffff); box-shadow: 5px 5px 10px #bebebe, -5px -5px 10px #ffffff; transition: all 0.3s ease; }
    .stButton>button:hover { box-shadow: inset 5px 5px 10px #bebebe, inset -5px -5px 10px #ffffff; }
</style>
'''

st.markdown(css, unsafe_allow_html=True)

#=================
# Main and Sidebar Content
#=================
st.markdown("<h1 style='text-align: center; color: #1E1E1E;'>QUILLERY</h1>", unsafe_allow_html=True)

with st.sidebar:
    try:
        image_url = "Image/logo.png"
        st.image(image_url, caption="", use_column_width=True)
    except FileNotFoundError:
        st.warning("Logo image not found. Please check the path.")

# Placeholder for chat history
st.markdown("<div style='margin-top: 2rem;'>", unsafe_allow_html=True)
st.markdown("<p><strong>AI:</strong> Hello! How can I assist you today?</p>", unsafe_allow_html=True)
st.markdown("<p><strong>You:</strong> Can you explain what RAG means in the context of AI?</p>", unsafe_allow_html=True)
st.markdown("<p><strong>AI:</strong> Certainly! RAG stands for Retrieval-Augmented Generation. It's an AI technique that combines...</p>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

#=================
# API Key and File Upload
#=================
gemini_api_key = st.sidebar.text_input("Gemini API Key", type="password")
genai.configure(api_key=gemini_api_key)

file_format = st.sidebar.selectbox("Select File Format", ["CSV", "PDF", "TXT"])
if file_format == "TXT":
    file_format = "plain"

uploaded_files = st.sidebar.file_uploader("Upload a file", type=["csv", "txt", "pdf"], accept_multiple_files=True)

def validate_format(file_format, uploaded_files):
    for file in uploaded_files:
        if str(file_format).lower() not in str(file.type).lower():
            return False
    return True

def select_pdf_analysis():
    type_pdf = st.selectbox("Select Analysis Type on PDFs", ["Compare", "Merge"])
    if type_pdf == "Compare":
        st.write("Analysis Comparing PDFs")
        return "Compare"
    else:
        st.write("Analysis Merging PDFs")
        return "Merge"

def save_uploaded_file(uploadedfile):
    with open(os.path.join(uploadedfile.name), "wb") as f:
        f.write(uploadedfile.getbuffer())
    return st.success("Saved File")

def history_func(answer, q):
    if 'history' not in st.session_state:
        st.session_state.history = ''

    value = f'Q: {q} \nA: {answer}'
    st.session_state.history = f'{value} \n {"-" * 100} \n {st.session_state.history}'
    h = st.session_state.history

    st.text_area(label='Chat History', value=h, key='history', height=400)

def csv_analysis(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        st.header("Dataframe Overview")
        st.dataframe(df)
        save_uploaded_file(uploaded_file)
        file_name = uploaded_file.name
        st.write(f"Filename: {file_name}")
        
        # Displaying more detailed statistics about the data
        st.header("Dataframe Summary Statistics")
        st.write(df.describe())
        
        # Allow the user to input a query to analyze the CSV
        user_query = st.text_input('Enter your query')

        if st.button("Answer My Question"):
            st.write("Running the query: ", user_query)
            model = genai.GenerativeModel('gemini-pro')

            # Providing more context to the model for better responses
            context = f"Here is a CSV data preview:\n{df.head().to_string()}\n\nSummary statistics:\n{df.describe().to_string()}"
            response = model.generate_content(f"Based on the following CSV data context:\n{context}\n\nQuery: {user_query}")

            st.text_area('LLM Answer:', value=response.text, height=400)
            history_func(response.text, user_query)

            # Clearing memory
            del df
            gc.collect()
    except Exception as e:
        st.error(f"Error processing CSV file: {e}")

def merge_pdf_analysis(uploaded_files):
    raw_text = ''
    for file in uploaded_files:
        pdf_reader = PdfReader(file)
        temp_text = ''
        for page in pdf_reader.pages:
            text = page.extract_text()
            if text:
                temp_text += text
        raw_text += temp_text

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )

    texts = text_splitter.split_text(raw_text)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': False})
    docsearch = FAISS.from_texts(texts, embeddings)

    st.subheader("Enter a question:")
    question = st.text_input("Question")

    if st.button("Answer My Question"):
        docs = docsearch.similarity_search(question)
        model = genai.GenerativeModel('gemini-pro')
        context = "\n".join([doc.page_content for doc in docs])
        response = model.generate_content(f"Context: {context}\n\nQuestion: {question}")
        st.subheader("Answer:")
        st.text_area('LLM Answer: ', value=response.text, height=400)
        history_func(response.text, question)
        del raw_text, texts, embeddings, docsearch
        gc.collect()

def compare_pdf_analysis(uploaded_files):
    tools = []
    llm = genai.GenerativeModel('gemini-pro')
    for file in uploaded_files:
        st.write("File name is ", file.name)
        save_uploaded_file(file)
        loader = PyPDFLoader(file.name)
        pages = loader.load_and_split()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = text_splitter.split_documents(pages)
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': False})
        retriever = FAISS.from_documents(docs, embeddings).as_retriever()
        function_name = file.name.replace('.pdf', '').replace(' ', '_')[:64]
        tools.append(Tool(name=function_name, description=f"useful when you want to answer questions about {function_name}", func=RetrievalQA.from_chain_type(llm=llm, retriever=retriever)))

    agent = initialize_agent(
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        tools=tools,
        llm=llm,
        verbose=True,
    )

    question = st.text_input("Question")
    if st.button("Answer My Question"):
        st.write("Running the query")
        response = agent.run(question)
        st.text_area('LLM Answer: ', value=response, height=400)
        history_func(response, question)
        del tools, llm, agent
        gc.collect()

def text_analysis(uploaded_files):
    raw_text = ''
    for file in uploaded_files:
        temp_text = file.read().decode("utf-8")
        raw_text += temp_text

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    texts = text_splitter.split_text(raw_text)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': False})

    docsearch = FAISS.from_texts(texts, embeddings)

    st.subheader("Enter a question:")
    question = st.text_input("Question")

    if st.button("Answer My Question"):
        docs = docsearch.similarity_search(question)
        model = genai.GenerativeModel('gemini-pro')
        context = "\n".join([doc.page_content for doc in docs])
        response = model.generate_content(f"Context: {context}\n\nQuestion: {question}")
        st.subheader("Answer:")
        st.text_area('LLM Answer: ', value=response.text, height=400)
        history_func(response.text, question)
        del raw_text, texts, embeddings, docsearch
        gc.collect()

#=================
# Answer Generation
#=================
if uploaded_files:
    if validate_format(file_format, uploaded_files):
        if file_format == "CSV":
            if len(uploaded_files) > 1:
                st.write("Only 1 CSV file can be uploaded")
            else:
                for file in uploaded_files:
                    csv_analysis(file)
        elif file_format == "PDF":
            if len(uploaded_files) > 1:
                select = select_pdf_analysis()
                if select == "Compare":
                    compare_pdf_analysis(uploaded_files)
                else:
                    merge_pdf_analysis(uploaded_files)
            else:
                merge_pdf_analysis(uploaded_files)
        else:
            text_analysis(uploaded_files)
    else:
        st.write("Formats are not valid")

gc.collect()
