import os
import pickle

from dotenv import load_dotenv
from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from streamlit import streamlit as st
from langchain_community.vectorstores import FAISS


class App:
    __urls = []
    __openai_embeddings = None
    __submit_clicked = False
    __main_placeholder = st.empty()
    __llm = None
    __file_path = 'faiss_store_openai.pkl'

    def __init__(self):
        load_dotenv()
        # self.__openai_embeddings = OpenAIEmbeddings()
        self.__llm = ChatOpenAI(model="gpt-4o", temperature=0, max_tokens=1000)
        self.__openai_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    def run(self):
        self.__build_ui()

    def __build_ui(self):
        st.title("News Research Tool")

        st.sidebar.title("News Articles URL")

        for i in range(3):
            url = st.sidebar.text_input(f"News Article URL {i}")
            self.__urls.append(url)

        self.submit_clicked = st.sidebar.button("Submit")
        if os.path.exists(self.__file_path):
            self.__run_query()
        elif self.submit_clicked:
            self.__make_pickle_file()
            self.__run_query()


    def __make_pickle_file(self):
        loader = UnstructuredURLLoader(urls=self.__urls)
        self.__main_placeholder.text("Data Loading started...")
        data = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', ','],
            chunk_size=1000
        )

        documents = text_splitter.split_documents(data)

        # Create embeddings

        vectors = FAISS.from_documents(documents, embedding=self.__openai_embeddings)
        self.__main_placeholder.text("Embedding started...")

        # save the FASIS index to pickle file
        with open(self.__file_path, "wb") as f:
            pickle.dump(vectors, f)
        self.__main_placeholder.text("Embedding saved...")


    def __run_query(self):
        query = st.text_input("Questions");
        if query:
            with open(self.__file_path, "rb") as f:
                vector_store = pickle.load(f)
            chain = RetrievalQAWithSourcesChain.from_llm(llm=self.__llm, retriever=vector_store.as_retriever())
            results = chain({"question": query}, return_only_outputs=True)

            st.header("Answers:")
            st.subheader(results["answer"])

            sources = results.get("sources", "")
            source_list = sources.split("\n")
            for source in source_list:
                st.write(source)
