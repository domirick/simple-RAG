import sys
import nest_asyncio
import streamlit as st
from dotenv import load_dotenv

from rag_chain import RAGChain

nest_asyncio.apply()

APP_NAME = "Simple RAG"
FAVICON = "https://icons.iconarchive.com/icons/pictogrammers/material/128/robot-icon.png"

# Init page
st.set_page_config(
    page_title=APP_NAME, 
    page_icon=FAVICON, 
    layout="centered", 
    initial_sidebar_state="auto", 
    menu_items=None
)

# Load & apply CSS
with open("app/components/theme.css", "r") as f:
    theme = f.read()
st.markdown(f"<style>{theme}</style>", unsafe_allow_html=True)

st.title(APP_NAME)

# Check args
if len(sys.argv) > 1 and sys.argv[1] == "load_env":
    try:
        load_dotenv()
    except Exception as e:
        st.error(f"Error loading environment variables: {e}")
        st.stop()

# Init RAG chain
if "rag" not in st.session_state:
    st.session_state["rag"] = RAGChain()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Clear chat
def clear_chat():
    del st.session_state.messages

def reindex_database():
    st.session_state["rag"].reindex_database()

# Sidebar
with st.sidebar:
    st.button("Clear chat", on_click=clear_chat)
    st.button("Reindex database", on_click=reindex_database)

# Chat display
for msg in st.session_state.messages:
    if not msg["role"] == "system":
        st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    msg = st.session_state["rag"].inference(st.session_state.messages)
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)