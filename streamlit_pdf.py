# import das bibliotecas
import streamlit as st
import dotenv, os
from library import pdf_functions

from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import LLMChain
from langchain.llms import AzureOpenAI
from langchain.chat_models import AzureChatOpenAI



# carregando as variáveis da API
dotenv.load_dotenv()


# Titulo do aplicativo
st.title("*** Streamlit & LangChain ***")
st.subheader("Pergunte sobre seu PDF")


# 01 - Upload de arquivos e extração do texto
pdf = st.file_uploader("Escolha um arquivo PDF:", type="pdf")


if pdf is not None:
    # função que carrega o pdf e retorna o texto
    text = pdf_functions.tratamento_pdf(pdf)

    # 02 - split into chunks
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    chunks = text_splitter.split_text(text)


    # 03 - criação de embeddings - vetores
    embeddings = OpenAIEmbeddings(
        model=os.getenv("EMBEDDING_MODEL_NAME"),
        deployment=os.getenv("EMBEDDING_DEPLOYMENT_NAME"),
        chunk_size=1
    )
    knowledge_base = FAISS.from_texts(chunks, embeddings)

    # 04 - pergunta do usuário
    user_question = st.text_input("Faça uma pergunta sobre seu PDF:")
    if user_question:
        docs = knowledge_base.similarity_search(user_question)


        llm = AzureChatOpenAI(
            openai_api_base=os.getenv("OPENAI_API_BASE"),
            openai_api_version=os.getenv("OPENAI_API_VERSION"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_api_type=os.getenv("OPENAI_API_TYPE"),
            deployment_name=os.getenv("DEPLOYMENT_NAME"),
            temperature=0.0,
        )
        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents=docs, question=user_question)
        st.write(response)