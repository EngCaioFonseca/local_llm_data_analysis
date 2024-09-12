import streamlit as st
import pandas as pd
import io
from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Initialize the Ollama model
@st.cache_resource
def load_model():
    return Ollama(
        model="llama2:3.1",
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    )

llm = load_model()

st.title("LLM Data Analysis Assistant")

# File uploader
uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"], key="file_uploader")

if 'df' not in st.session_state:
    st.session_state.df = None

if uploaded_file is not None:
    # Read the file
    if uploaded_file.name.endswith('.csv'):
        st.session_state.df = pd.read_csv(uploaded_file)
    else:
        st.session_state.df = pd.read_excel(uploaded_file)

if st.session_state.df is not None:
    st.write("Data Preview:")
    st.dataframe(st.session_state.df.head())

    # Data info
    st.write("Data Info:")
    buffer = io.StringIO()
    st.session_state.df.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)

    # User input for analysis
    user_input = st.text_area("Ask a question about the data:", height=100, key="data_question")

    if st.button("Analyze", key="analyze_button"):
        if user_input:
            # Prepare context for the LLM
            context = f"Here's some information about the dataset:\n"
            context += f"Columns: {', '.join(st.session_state.df.columns)}\n"
            context += f"Shape: {st.session_state.df.shape}\n"
            context += f"Data types:\n{st.session_state.df.dtypes.to_string()}\n"
            context += f"Summary statistics:\n{st.session_state.df.describe().to_string()}\n"

            # Create a prompt template
            prompt_template = PromptTemplate(
                input_variables=["context", "question"],
                template="Based on the following information about a dataset:\n{context}\n\nPlease answer the following question: {question}\n\nAnswer:",
            )

            # Create an LLMChain
            chain = LLMChain(llm=llm, prompt=prompt_template)

            # Generate response from the LLM
            with st.spinner('Analyzing...'):
                response = chain.run(context=context, question=user_input)
            
            st.write("Analysis Result:")
            st.write(response)
        else:
            st.warning("Please enter a question about the data.")

# Chat interface
st.sidebar.header("Chat with the LLM")
chat_input = st.sidebar.text_input("Ask a general question:", key="chat_input")

if st.sidebar.button("Send", key="chat_button"):
    if chat_input:
        chat_prompt = PromptTemplate(
            input_variables=["question"],
            template="Human: {question}\n\nAssistant:",
        )
        chat_chain = LLMChain(llm=llm, prompt=chat_prompt)
        with st.sidebar.spinner('Thinking...'):
            response = chat_chain.run(question=chat_input)
        st.sidebar.write("LLM Response:")
        st.sidebar.write(response)
    else:
        st.sidebar.warning("Please enter a question.")
