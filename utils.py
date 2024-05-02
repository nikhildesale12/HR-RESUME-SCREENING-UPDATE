from langchain_community.vectorstores import Pinecone
from langchain_community.llms import OpenAI
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.schema import Document
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from pypdf import PdfReader
from langchain_community.llms.openai import OpenAI
from langchain.chains.summarize import load_summarize_chain
import time
import os


#Extract Information from PDF file
# Input - list of pdf files
# Output - returns text with all pages
def get_pdf_text(pdf_doc):
    text = ""
    pdf_reader = PdfReader(pdf_doc)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text



# iterate over files in 
# that user uploaded PDF files, one by one
# Input - list of pdf , unique id of vectore
def create_docs(user_pdf_list, unique_id):
    docs=[]
    for filename in user_pdf_list:
        
        chunks=get_pdf_text(filename)

        #Adding items to our list - Adding data & its metadata
        docs.append(Document(
            page_content=chunks,
            metadata={"name": filename.name,"id":filename.file_id,"type=":filename.type,"size":filename.size,"unique_id":unique_id},
        ))

    return docs


#Create embeddings instance
# It convert text into neumerical 
def create_embeddings_load_data():
    # embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    # embeddings = SentenceTransformerEmbeddings(model_name="recobo/agri-sentence-transformer")
    return embeddings


#Function to push data to Vector Store - Pinecone here
def push_to_pinecone(pinecone_index_name,embeddings,docs):

    # Pinecone.init(
    # api_key=pinecone_apikey,
    # environment=pinecone_environment
    # )

    # pc = Pinecone(api_key='887d84f4-b20c-4255-a1c9-529b2f52a153')

    
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    PineconeVectorStore.from_documents(docs, embeddings, index_name=pinecone_index_name)
    


#Function to pull infrmation from Vector Store - Pinecone here
def pull_from_pinecone(pinecone_apikey,pinecone_index_name,embeddings):
    # For some of the regions allocated in pinecone which are on free tier, the data takes upto 10secs for it to available for filtering
    #so I have introduced 20secs here, if its working for you without this delay, you can remove it :)
    #https://docs.pinecone.io/docs/starter-environment
    print("20secs delay...")
    time.sleep(20)
    # pinecone.init(
    # api_key=pinecone_apikey,
    # environment=pinecone_environment
    # )

    pc = Pinecone(api_key=pinecone_apikey)

    index_name = pinecone_index_name

    index = PineconeVectorStore.from_existing_index(index_name, embeddings)
    return index



#Function to help us get relavant documents from vector store - based on user input
def similar_docs(job_description,document_count,pinecone_index_name,embeddings,unique_id):

    # pinecone.init(
    # api_key=pinecone_apikey,
    # environment=pinecone_environment
    # )

    pinecone_apikey = os.environ.get("PINECONE_API_KEY")
    pc = Pinecone(api_key=pinecone_apikey)

    index_name = pinecone_index_name

    index = pull_from_pinecone(pinecone_apikey,index_name,embeddings)
    similar_docs = index.similarity_search_with_score(job_description, int(document_count),{"unique_id":unique_id})
    #print(similar_docs)
    return similar_docs


# Helps us get the summary of a document
def get_summary(current_doc):
    # llm = OpenAI(temperature=0)
    llm = HuggingFaceHub(repo_id="Falconsai/text_summarization", model_kwargs={"temperature":1e-10})
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    summary = chain.run([current_doc])

    return summary




    
