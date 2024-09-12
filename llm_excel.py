import streamlit as st
import pandas as pd
import io
import requests

# Hugging Face API setup
API_URL = "https://api-inference.huggingface.co/models/meta-llama/Llama-2-7b-chat-hf"

# Check if the API key is set
if 'HUGGINGFACE_API_KEY' not in st.secrets:
    st.error("HUGGINGFACE_API_KEY is not set in the Streamlit secrets. Please set it up as described in the README.")
    st.stop()

headers = {"Authorization": f"Bearer {st.secrets['HUGGINGFACE_API_KEY']}"}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

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

        # Create the prompt
            prompt = f"Based on the following information about a dataset:\n{context}\n\nPlease answer the following question: {user_input}\n\nAnswer:"

            # Generate response from the LLM
            with st.spinner('Analyzing...'):
                try:
                    output = query({
                        "inputs": prompt,
                        "parameters": {"max_new_tokens": 500}
                    })
                
                    st.write("Analysis Result:")
                    if isinstance(output, list) and len(output) > 0 and 'generated_text' in output[0]:
                        st.write(output[0]['generated_text'])
                    elif isinstance(output, dict) and 'generated_text' in output:
                        st.write(output['generated_text'])
                    else:
                        st.write(str(output))  # Fallback: write the raw output
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
        else:
            st.warning("Please enter a question about the data.")

# Chat interface
st.sidebar.header("Chat with the LLM")
chat_input = st.sidebar.text_input("Ask a general question:", key="chat_input")

if st.sidebar.button("Send", key="chat_button"):
    if chat_input:
        with st.sidebar.spinner('Thinking...'):
            try:
                output = query({
                    "inputs": f"Human: {chat_input}\n\nAssistant:",
                    "parameters": {"max_new_tokens": 500}
                })
                st.sidebar.write("LLM Response:")
                if isinstance(output, list) and len(output) > 0 and 'generated_text' in output[0]:
                    st.sidebar.write(output[0]['generated_text'])
                elif isinstance(output, dict) and 'generated_text' in output:
                    st.sidebar.write(output['generated_text'])
                else:
                    st.sidebar.write(str(output))  # Fallback: write the raw output
            except Exception as e:
                st.sidebar.error(f"An error occurred: {str(e)}")
    else:
        st.sidebar.warning("Please enter a question.")
