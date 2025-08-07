from langchain_community.document_loaders import TextLoader,CSVLoader, UnstructuredURLLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss as fs
import numpy as np


class SampleReader:
    def __init__(self):
        pass

    def  run(self):
        loader = UnstructuredURLLoader([
            "https://www.moneycontrol.com/news/business/tata-motors-mahindra-gain-certificates-for-production-linked-payouts-11281691.html",
            "https://www.moneycontrol.com/news/business/stocks/buy-tata-motors-target-of-rs-743-kr-choksey-11080811.html"
        ])
        data = loader.load()
        print(data[0].page_content)


    def split_chunks_using_character_text_splitter(self):
        textLoader = TextLoader('./resources/intestellar.txt')
        text = textLoader.load()[0].page_content
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=200,
            chunk_overlap=0
        )
        chunks = text_splitter.split_text(text)
        print(len(chunks))
        for chunk in chunks:
            print(len(chunk))

    def split_text_using_recursive_character_text_splitter(self):
        textLoader = TextLoader('./resources/intestellar.txt')
        text = textLoader.load()[0].page_content
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", " ", " "],
            chunk_size=200,
            chunk_overlap=0
        )
        chunks = text_splitter.split_text(text)
        print(len(chunks))
        for chunk in chunks:
            print(len(chunk))


    def embeddings(self):
        pd.set_option('display.max_colwidth', 100)

        df = pd.read_csv('./resources/sample_text.csv')

        encoder = SentenceTransformer("all-mpnet-base-v2")
        vectors = encoder.encode(df.text)
        dim = vectors.shape[1]

        index = fs.IndexFlatL2(dim)
        index.add(vectors)

        # search_query = "I want to buy a polo t shirt"
        # search_query = "An apple keeps away from doctor"
        # search_query = "I want to visit to UK"
        search_query = "Some random text"

        vec = encoder.encode(search_query)
        svec = np.array(vec).reshape(1, -1)

        distance, I = index.search(svec, k=2)
        print(df.loc[I[0]])

