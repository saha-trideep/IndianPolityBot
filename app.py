import os
import time
import streamlit as st
import chromadb
from dotenv import load_dotenv
from langchain.vectorstores.chroma import Chroma
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceEndpoint
from langchain.chains import RetrievalQA
from langchain.schema import retriever
from langchain.memory import ConversationBufferWindowMemory

from chromadb.config import Settings
client = chromadb.PersistentClient(path="./chroma", settings=Settings(allow_reset=False))

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


# Load environment variables
load_dotenv()


# Load HuggingFace API token
#HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN") 
HUGGINGFACEHUB_API_TOKEN = st.secrets["HUGGINGFACEHUB_API_TOKEN"]

embed_model = SentenceTransformer("all-MiniLM-L6-v2")
# Set up Chroma
# embedding = HuggingFaceEmbeddings(model_name=embed_model)
db = Chroma(persist_directory="./chroma", embedding_function=embed_model, client=client)

# Set up LLM
repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
llm = HuggingFaceEndpoint(
    repo_id=repo_id, 
    max_length=512, 
    temperature=0.5, 
    token=HUGGINGFACEHUB_API_TOKEN,
    add_to_git_credential=True
)

# Set up memory
memory = ConversationBufferWindowMemory(k=5, return_messages=True)

# Set up retriever
retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 2})

# Set up chain
chain = RetrievalQA.from_chain_type(llm=llm, 
                                    chain_type="stuff", 
                                    retriever=retriever,
                                    memory=memory, 
                                    )


assistant_logo = "data/constitution-clipart-consitution-19.png"
constitution_cover = "data/Constitution_of_India_Cover.png"

# Configure Streamlit page
st.set_page_config(
    page_title="Indian Polity Chatbot",
    page_icon="data/constitution_of_india.png",
)


st.sidebar.image(constitution_cover, caption="Indian Polity by M. Laxmikant", use_column_width=True)

st.title("Indian Polity Chatbot")
st.sidebar.markdown("Welcome to the Indian Polity Chatbot! This chatbot is based on 'Indian Polity by M Laxmikant' PDF, mainly for UPSC exam preparation question and answers.")

# Suggestions in the sidebar
st.sidebar.markdown("### Suggestions")
suggested_questions = ["What is Article 370?", "Explain the concept of Directive Principles of State Policy and their significance in the Indian Constitution.?",\
    "Explain the Preamble of the Constitution", "What are the Fundamental Rights?",\
    "Discuss the composition and functions of the Election Commission of India.",\
    "What is the process of impeachment of the President of India?",\
    "Differentiate between Fundamental Rights and Directive Principles of State Policy."]
for question in suggested_questions:
    st.sidebar.markdown(f"- {question}")


if 'messages' not in st.session_state:
    # Start with first message from assistant
    st.session_state['messages'] = [{"role": "assistant", 
                                  "content": "Constituent Assembly of India📗."}]

for message in st.session_state.messages:
    if message["role"] == 'assistant':
        with st.chat_message(message["role"], avatar=assistant_logo):
            st.markdown(message["content"])
    else:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Chat logic
if query := st.chat_input("Ask me about Constitution of India"):

        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": query})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant", avatar=assistant_logo):
            message_placeholder = st.empty()
            

            # Send user's question to our chain
            result = chain.invoke(query)
            response = result['result']
            full_response = ""

            # Simulate stream of response with milliseconds delay
            for chunk in response.split():
                full_response += chunk + " "
                time.sleep(0.05)
                # Add a blinking cursor to simulate typing
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)

        # Add assistant message to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})