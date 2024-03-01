import PyPDF2
import sentence_transformers
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.chroma import Chroma


# import chromadb
# from chromadb.config import Settings
# client = chromadb.PersistentClient(path="./chroma", settings=Settings(allow_reset=True))

def text_extraction(path):
    with open(path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        list_text = [page.extract_text() for page in reader.pages]
        text = "".join(list_text)
        return text
        

def split_text(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False
    )
    docs = text_splitter.split_text(text)
    return docs 



def embedding(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    Chroma.from_texts(chunks, embeddings,persist_directory="./ chroma", metadatas=[{"source": f"{i}-pl"} for i in range(len(chunks))])
    

pdf_text = text_extraction("data/Indian Polity Laxmikanth 6th edition.pdf")
print("...1...")
chunks = split_text(pdf_text)
print("...2...")
embedding(chunks)
print("...3...")