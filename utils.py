import os
import random
import streamlit as st

#decorator
def enable_chat_history(func):
    
    if os.environ.get("OPENAI_API_KEY"):
       

        # to clear chat history after swtching chatbot
        current_page = func.__qualname__
        if "current_page" not in st.session_state:
            st.session_state["current_page"] = current_page
        if st.session_state["current_page"] != current_page:
            try:
                st.cache_resource.clear()
                del st.session_state["current_page"]
                del st.session_state["chat_history"]
                del st.session_state["history_cpt"]
                del st.session_state["history_pos"]
                del st.session_state["history_rev"]
                del st.session_state['temp'] 
            except:
                pass

        # to show chat history on ui
        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = [{"role": "assistant", "content": "How can I help you?"}]
        for msg in st.session_state["chat_history"]:
            print("Yes-----------------")
            st.chat_message(msg["role"]).write(msg["content"])

    def execute(*args, **kwargs):
        func(*args, **kwargs)
    return execute

def display_msg(msg, author):
    """Method to display message on the UI

    Args:
        msg (str): message to display
        author (str): author of the message -user/assistant
    """
    st.session_state.chat_history.append({"role": author, "content": msg})
    st.chat_message(author).write(msg)

def configure_openai_api_key():
    openai_api_key = st.sidebar.text_input(
        label="OpenAI API Key",
        type="password",
        # value=st.session_state['OPENAI_API_KEY'] if 'OPENAI_API_KEY' in st.session_state else  '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',
        value=st.session_state['OPENAI_API_KEY'] if 'OPENAI_API_KEY' in st.session_state else  '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',

        placeholder="sk-..."
        )
    
    if 'history_cpt' not in st.session_state:
            st.session_state['history_cpt'] = []
    if 'history_pos' not in st.session_state:
            st.session_state['history_pos'] = []
    if 'history_rev' not in st.session_state:
            st.session_state['history_rev'] = []
    if 'temp' not in st.session_state:
            st.session_state['temp'] = []
    if openai_api_key:
        st.session_state['OPENAI_API_KEY'] = openai_api_key
        # os.environ['OPENAI_API_KEY'] = openai_api_key
       
    else:
        st.error("Please add your OpenAI API key to continue.")
        st.info("Obtain your key from this link: https://platform.openai.com/account/api-keys")
        st.stop()
    return openai_api_key