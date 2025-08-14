# !pip install -q chromadb httpx tldextract sanic llama_index jsonify sentence_transformers
# !pip install langchain==0.1.14
# !pip install langchain-community==0.0.31
# !pip install PyPDF2

import numpy as np 
import pandas as pd 
import os
from langchain.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import PyPDFLoader

import torch
import transformers
from transformers import AutoTokenizer
from time import time
from langchain.llms import HuggingFacePipeline
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma

import glob
import textwrap

from time import time

import os
from langchain.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import PyPDFLoader

import torch
import transformers
from transformers import AutoTokenizer
from time import time
from langchain.llms import HuggingFacePipeline
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
from PyPDF2 import PdfReader
from llama_index.core.schema import Document
from time import time

# Path to the PDF file
pdf_path = "/home/cpatwadityasharma/hf_rag/documents/Harry Potter and the Sorcerers Stone.pdf"

time_start = time()

# Extract text from the PDF
reader = PdfReader(pdf_path)
pdf_text = "\n".join(page.extract_text() for page in reader.pages)

# Wrap the text as a document for LlamaIndex
documents = [Document(text=pdf_text)]

time_end = time()
print(f"Loaded {len(documents)} docs")


time_duration = time_end - time_start
# report the duration
print(f'Took {time_duration} seconds')

documents = [doc.to_langchain_format() for doc in documents]
#documents

time_start = time()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=70)
texts = text_splitter.split_documents(documents)
time_end = time()

time_duration = time_end - time_start
# report the duration
print(f'Took {time_duration} seconds')


model_name =   "sentence-transformers/all-mpnet-base-v2"   
model_kwargs = {"device": "cuda"}

embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

time_start = time()

vectordb = Chroma.from_documents(documents=texts, embedding=embeddings, collection_metadata={"hnsw:space": "cosine"}, persist_directory="/home/cpatwadityasharma/hf_rag/chroma_db/docs_cosine")

time_end = time()

time_duration = time_end - time_start
# report the duration
print(f'Took {time_duration} seconds')


load_vector_store = Chroma(persist_directory="/home/cpatwadityasharma/hf_rag/chroma_db/docs_cosine", embedding_function=embeddings)

def extract_unique_dicts(list_of_dicts):
    unique_dicts = []
    seen_dicts = set()

    for d in list_of_dicts:
        # Convert the dictionary to a frozenset of items (since dictionaries are not hashable)
        dict_representation = frozenset(d.items())

        # Check if the dictionary representation is unique
        if dict_representation not in seen_dicts:
            unique_dicts.append(d)
            seen_dicts.add(dict_representation)

    return unique_dicts

def vector_search_source(query ):
    docs = load_vector_store.similarity_search_with_score(query=query, k=5)
    
    print("Sources:")
    src = []
    for meta in docs:
        #print(list(list(list(meta)[0])[0])[1]) # content
        keys = ['page_label','file_name'] # "file_path","file_type" ,"creation_date"
        src.append(dict(filter(lambda item:item[0] in keys , list(list(list(meta)[0])[1])[1].items()))) # source

    sources = extract_unique_dicts(src)
    sources = filter(None,sources)

    for meta in sources:
        print(meta)
        
        
retriever = load_vector_store.as_retriever(search_kwargs = {"k": 3} )
retriever.get_relevant_documents("what is deep learning" )



from huggingface_hub import login
login(token="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")        



# from torch import cuda, bfloat16
# import transformers
# from transformers import AutoTokenizer, AutoModelForCausalLM
# import torch
# from transformers import StoppingCriteria, StoppingCriteriaList

# # Load model and tokenizer
# model_name = "meta-llama/Llama-3.2-1B"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     trust_remote_code=True,
#     torch_dtype="auto"
# )
# model.eval()
# model.to(device)
# print(f"Model loaded on {device}")

# Define custom stopping criteria
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from transformers import StoppingCriteria, StoppingCriteriaList


device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'


# Load model and tokenizer
model_name = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype="auto"
)
model.eval()
model.to(device)
print(f"Model loaded on {device}")



import torch
from transformers import StoppingCriteria, StoppingCriteriaList

# gpt-j-6b is trained to add "<|endoftext|>" at the end of generations
stop_token_ids = tokenizer.convert_tokens_to_ids(["<|endoftext|>"])

# define custom stopping criteria object
class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_id in stop_token_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

stopping_criteria = StoppingCriteriaList([StopOnTokens()])
stopping_criteria


generate_text = transformers.pipeline(
    model=model, tokenizer=tokenizer,
    return_full_text=True,  # langchain expects the full text
    task='text-generation',
    device='cuda:0',
    # we pass model parameters here too
    stopping_criteria=stopping_criteria,  # without this model will ramble
    temperature=0.01,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
    top_p=0.15,  # select from top tokens whose probability add up to 15%
    top_k=0,  # select from top 0 tokens (because zero, relies on top_p)
    max_new_tokens=512,  # mex number of tokens to generate in the output
    repetition_penalty=1.1 , # without this output begins repeating
    do_sample=True,
)



from langchain import PromptTemplate, LLMChain
from langchain.llms import HuggingFacePipeline

llm = HuggingFacePipeline(pipeline=generate_text)




prompt_template = """
Don't try to make up an answer, if you don't know just say that you don't know.
Answer in the same language the question was asked.
Use only the following pieces of context to answer the question at the end.

{context}

Question: {question}


Answer:"""

PROMPT = PromptTemplate(
    template = prompt_template, 
    input_variables = ["context", "question" ]
)

from langchain import PromptTemplate, LLMChain
llm_chain = LLMChain(prompt=PROMPT, llm=llm)
llm_chain




from langchain.chains import RetrievalQA
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)

retriever = load_vector_store.as_retriever(search_kwargs = {"k": 3})

qa_chain_without_mem = RetrievalQA.from_chain_type(
    llm = llm,
    chain_type = "stuff", # map_reduce, map_rerank, stuff, refine
    retriever = retriever, 
    #chain_type_kwargs = {"prompt": PROMPT},
    return_source_documents = True,
    verbose = False
)


def extract_unique_dicts(list_of_dicts):
    unique_dicts = []
    seen_dicts = set()

    for d in list_of_dicts:
        # Convert the dictionary to a frozenset of items (since dictionaries are not hashable)
        dict_representation = frozenset(d.items())

        # Check if the dictionary representation is unique
        if dict_representation not in seen_dicts:
            unique_dicts.append(d)
            seen_dicts.add(dict_representation)

    return unique_dicts

def qa_post_processing(raw_result):

    print("Answer")
    
    context = raw_result['result']
    start_index = context.find("Helpful Answer:")
    extracted_string = context[start_index:].strip().replace("Helpful Answer: " , "")
    print(extracted_string)
    
    print(" ")
    print("Sources:")
    src=[]
    for meta in raw_result['source_documents']:
        keys = ['page_label','file_name'] # "file_path","file_type" ,"creation_date"
        src.append(dict(filter(lambda item:item[0] in keys , list(list(meta)[1])[1].items()))) # source

    sources = extract_unique_dicts(src)
    sources = filter(None,sources)

    for meta in sources:
        print(meta)
        
#input your query         
        
query1 = "Create a JSON object with fields as five short question, correct answer as correctAnswer, four incorrect answers as incorrectAnswers as object as {ic1, ic2, ic3, ic4}, and an explanation. The question should be from ${category}, and the explanation as questionExplanation should of the answer in few words, and Don't write anything else, except JSON data"
result1 = qa_chain_without_mem(query1)
qa_post_processing(result1)





        







