# LLM Data Analysis Assistant

## Overview

This Streamlit-based web application leverages the power of local Large Language Models (LLMs) to analyze CSV and Excel datasets. It uses Llama 3.1 via Ollama and LangChain to provide intelligent data analysis and general question answering capabilities.

## Features

- Upload and preview CSV or Excel files
- Display basic dataset information
- LLM-powered data analysis based on user questions
- General chat interface for broader queries
- Local execution of Llama 3.1 model using Ollama
- Flexible prompt engineering with LangChain

## Requirements

- Python 3.7+
- Streamlit
- Pandas
- Openpyxl
- LangChain
- Ollama (with Llama 3.1 model)

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/llm-data-analysis-assistant.git
   cd llm-data-analysis-assistant
   ```

2. Install the required Python packages:
   ```
   pip install streamlit pandas openpyxl langchain
   ```

3. Install Ollama by following the instructions at [Ollama's official website](https://ollama.ai/).

4. Pull the Llama 3.1 model in Ollama:
   ```
   ollama pull llama2:3.1
   ```

## Usage

1. Run the Streamlit app:
   ```
   streamlit run llm_data_analysis_app.py
   ```

2. Open your web browser and go to the URL provided by Streamlit (usually `http://localhost:8501`).

3. Use the application:
   - Upload a CSV or Excel file using the file uploader.
   - View the data preview and information.
   - Ask questions about the data in the text area and click "Analyze".
   - Use the sidebar to ask general questions to the LLM.

## How It Works

1. **Data Loading**: The app uses Pandas to read and process CSV and Excel files.

2. **LLM Initialization**: Llama 3.1 is initialized using Ollama through LangChain's interface.

3. **Data Analysis**: 
   - When a user asks a question about the data, the app prepares a context containing information about the dataset.
   - This context, along with the user's question, is sent to the LLM using a structured prompt.
   - The LLM generates a response based on the context and question.

4. **General Chat**: 
   - The sidebar provides a general chat interface where users can ask broader questions.
   - These questions are sent directly to the LLM without additional context.

## Customization

- To use a different Ollama model, change the `model` parameter in the `load_model` function.
- Adjust the prompt templates in the `PromptTemplate` objects to customize how the LLM interprets questions and generates responses.

## Limitations

- The analysis quality depends on the capabilities of the Llama 3.1 model and the specificity of the questions asked.
- Large datasets may impact performance, as the entire dataset info is included in the LLM's context.
- The app runs the LLM locally, so performance will depend on your machine's capabilities.

## Contributing

Contributions to improve the LLM Data Analysis Assistant are welcome. Please follow these steps:
1. Fork the repository
2. Create a new branch for your feature
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

This project is open-source and available under the MIT License.

## Acknowledgments

- Streamlit for the web app framework
- LangChain for LLM integration
- Ollama for local LLM deployment
- The Llama 3.1 model developers
