# Book-Wizard
## Overview
This project delivers a comprehensive Retrieval-Augmented Generation (RAG) system for local deployment, powered entirely by Hugging Face's ecosystem. It processes PDF documents (e.g., "Harry Potter and the Sorcerer's Stone") to build a searchable vector database using Chroma, and employs Hugging Face's LLaMA-3.2-1B model for efficient question-answering. Designed as an end-to-end solution, it outperforms alternatives like Ollama in inference speed and deployment simplicity, leveraging Hugging Face's optimized models and tools.

## Features
- **PDF Text Extraction**: Extracts text from PDF files using PyPDF2 and LlamaIndex.
- **Vector Store**: Creates a Chroma vector database with SentenceTransformer embeddings for rapid retrieval.
- **Question-Answering**: Utilizes Hugging Face's LLaMA-3.2-1B model for fast, context-aware responses.
- **End-to-End Local Deployment**: Runs offline with Hugging Face models, requiring no external dependencies beyond initial setup.
- **Custom Prompting**: Generates JSON-formatted question-answer sets with tailored prompts.
- **Performance Superiority**: Outperforms Ollama in inference time due to Hugging Face's optimized pipelines and CUDA support.
- **Source Tracking**: Provides metadata (page labels, file names) for retrieved document sources.

## Prerequisites
- Python 3.8+
- CUDA-enabled GPU (recommended for optimal inference speed)
- Required Libraries:
  - `chromadb`
  - `httpx`
  - `tldextract`
  - `sanic`
  - `llama_index`
  - `sentence_transformers`
  - `jsonify`
  - `langchain==0.1.14`
  - `langchain-community==0.0.31`
  - `PyPDF2`
  - `transformers`
  - `torch`
  - `huggingface_hub`
- Hugging Face API Token (for model download)

## Installation

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd local-rag-hf
   ```

2. **Set Up Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -q chromadb httpx tldextract sanic llama_index sentence_transformers jsonify
   pip install langchain==0.1.14
   pip install langchain-community==0.0.31
   pip install PyPDF2
   ```

4. **Configure Hugging Face Access**:
   - Obtain a Hugging Face token from [Hugging Face](https://huggingface.co/settings/tokens).
   - Run the login command in the script or set the token as an environment variable:
     ```bash
     export HUGGINGFACEHUB_API_TOKEN=hf_your_token_here
     ```
   - Alternatively, use the interactive login:
     ```python
     from huggingface_hub import login
     login(token="hf_your_token_here")
     ```

5. **Prepare Data**:
   - Place PDF files (e.g., `/home/cpatwadityasharma/hf_rag/documents/Harry Potter and the Sorcerer's Stone.pdf`) in the specified directory.

## Usage

1. **Run the Script**:
   ```bash
   python your_script_name.py
   ```
   - Replace `your_script_name.py` with the main script file name.

2. **Process PDF and Build Vector Store**:
   - The script extracts text, splits it into chunks, and creates a Chroma vector store in `/home/cpatwadityasharma/hf_rag/chroma_db/docs_cosine`.

3. **Query the System**:
   - Use the `qa_chain_without_mem` to ask questions. Example:
     ```python
     query1 = "Create a JSON object with fields as five short question, correct answer as correctAnswer, four incorrect answers as incorrectAnswers as object as {ic1, ic2, ic3, ic4}, and an explanation. The question should be from ${category}, and the explanation as questionExplanation should of the answer in few words, and Don't write anything else, except JSON data"
     result1 = qa_chain_without_mem(query1)
     qa_post_processing(result1)
     ```
   - Output includes the answer and sources.

## Project Structure
- **`your_script_name.py`**: Main script for PDF processing, vector store creation, and question-answering.
- **`/home/cpatwadityasharma/hf_rag/documents/`**: Directory for input PDF files.
- **`/home/cpatwadityasharma/hf_rag/chroma_db/docs_cosine/`**: Persisted Chroma vector store.

## Performance Comparison with Ollama
- **Inference Speed**: Hugging Face's LLaMA-3.2-1B, when run with CUDA and optimized pipelines (e.g., `temperature=0.01`, `max_new_tokens=512`), delivers faster inference than Ollama due to pre-trained model efficiency and hardware acceleration.
- **End-to-End Efficiency**: Hugging Face provides a seamless workflow from text extraction to response generation, reducing setup complexity compared to Ollama's additional configuration requirements.
- **Resource Utilization**: Hugging Face's integration with PyTorch and CUDA minimizes memory overhead, outperforming Ollama's inference on similar hardware.

## Configuration
- **Model**: Default is `meta-llama/Llama-3.2-1B` from Hugging Face.
- **Embedding**: Uses `sentence-transformers/all-mpnet-base-v2` with cosine similarity.
- **Paths**: Update `pdf_path` and `persist_directory` to match your system.
- **Today's Date and Time**: The system is operational as of 02:40 PM IST on Wednesday, August 20, 2025.

## Optimization
- **Inference Time Reduction**: 
  - Enable CUDA with `model_kwargs = {"device": "cuda"}` for GPU acceleration.
  - Use `torch.backends.cuda.enable_mem_efficient_sdp(False)` and `torch.backends.cuda.enable_flash_sdp(False)` for memory-efficient processing.
  - Adjust `chunk_size=700` and `chunk_overlap=70` in `RecursiveCharacterTextSplitter` for optimal chunking.
- **Scalability**: Increase `k` in `search_kwargs = {"k": 3}` for more context retrieval if needed.

## Notes
- Ensure sufficient GPU memory for model loading (e.g., LLaMA-3.2-1B requires ~2GB).
- The vector store persists locally; clear it manually if retraining is needed.
- Hugging Face's ecosystem eliminates the need for external model servers, unlike Ollama.

## Troubleshooting
- **CUDA Errors**: Verify GPU drivers and CUDA toolkit installation.
- **Model Load Failure**: Check Hugging Face token and internet connectivity.
- **Slow Inference**: Reduce `max_new_tokens` or ensure GPU is utilized.
- **PDF Processing**: Ensure PDFs are text-extractable; use OCR if needed.




