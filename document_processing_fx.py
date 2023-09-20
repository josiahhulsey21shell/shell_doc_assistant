from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

import chromadb
import tqdm
import glob
import boto3
import json

def create_file_list(path, extension = None):

    if extension:
        file_folder = f"{path}/*{extension}"
    else:
        file_folder = f"{path}/*"


    file_list = []
    for file in glob.glob(file_folder):
        file_list.append(file)

    return file_list




def query_endpoint(encoded_text):
    '''
    Function that will send text to the USE endpoint.
    '''

    endpoint_name = 'jumpstart-dft-sentence-encoder-cmlm-en-large-1'
    client = boto3.client('runtime.sagemaker')
    response = client.invoke_endpoint(EndpointName=endpoint_name, ContentType='application/x-text', Body=encoded_text, Accept='application/json;verbose')
    return response

def parse_response(query_response):
    '''
    Function that will return the embedded text from the USE endpoint.
    '''

    model_predictions = json.loads(query_response['Body'].read())
    embedding, model_output = model_predictions['embedding'], model_predictions['model_output']
    return embedding, model_output

def embed_question(question):
    '''
    Function that will embed a users question using the USE model 
    '''


    univ_sen_embedding = query_endpoint(question)
    parsed_embedding = parse_response(univ_sen_embedding)
    
    return parsed_embedding



def chunk_and_embed_documents(file_list, chunk_size= 500, chunk_overlap= 25):
    '''
    Function that will iterate over a list of file paths for pdfs, load them to memory, chunk them, and then use the USE encoder to retrun the encodings of the chunks.

    At the moment only works for PDFS!!!

    '''
    #lists that the relevant data will be appended to in the workflow
    ids = []
    chunk_docs = []
    embeddings = []

    #make an instance of the recursive splitter with the given chunk size and chunk overlap.
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    #start a counter for use in the doc id for chroma
    counter = 0
    #iterate over each file in the file list
    for i in tqdm.tqdm(file_list):
        #open the pdf
        loader = PyPDFLoader(i)
        
        #put this in a try statement because there were some bad files that came through and were causing failures here
        try:
            #load the pages and split them into individual documents
            pages = loader.load_and_split()
            
            #iterate over each page and chunk it
            for p in pages:
                #chunk the page
                chunks = splitter.split_text(p.page_content)

                #iterate over each chunk and embed the chunk using the USE encoder
                for c in chunks:
                    #increase doc id count
                    counter = counter + 1

                    #append doc id and chunk to the output lists
                    ids.append(f'id{counter}')
                    chunk_docs.append(c)

                    #embed the chunk
                    query_response = query_endpoint(c.encode('utf-8'))
                    parsed_response  =  parse_response(query_response)
                    #comes back as a tuple. Need the first element of it.
                    embeddings.append(parsed_response[0])
                
        except:
            print(f"Error Splitting Pages for {i}")
            next
    

    return [ids,chunk_docs,embeddings]


def create_chroma_db(path):
    '''
    Function that will create a chroma db whereever you provide a path to. MAKE SURE YOU ARE CONSISTANT WITH YOUR PATH CALLS
    '''
    client = chromadb.PersistentClient(path=path)

    return client

def create_collection(client, collection_name):
    '''
    Function that will create a collection in the given Chroma db.
    '''
    #create the collection
    collection = client.create_collection(name=collection_name)

    return collection

def add_data_to_collection(collection, chunked_docs, ids, embeddings):
    ''' 
    formatted to take the results of the chunk_and_embed_documents function and store it in the given collection
    
    '''
    #add the documents and ids and embeddings to the db
    collection.add(documents = chunked_docs, ids = ids, embeddings = embeddings)    


def load_chroma_db(path):
    '''
    Function that will load a chroma db from disk. MAKE SURE YOU ARE CONSISTANT WITH YOUR PATH CALLS
    '''
    client = chromadb.PersistentClient(path=path)

    return client


def get_similair_documents(chromadb_path, collection_name, question, ndocs):
    '''
    Function that will return N ammount of similair documents from the current collection
    chromadb_path = path to your chroma db
    collection name = name of the collection you are pulling from
    question = the question that will be asked
    ndocs = number of similair documents to return


    '''
    #embed the users question using the USE endpoint
    user_question_emb = embed_question(question)

    #connect to the db
    client = chromadb.PersistentClient(path=chromadb_path)
    #connect to the collection
    collection = client.get_collection(collection_name)

    #find the most alike documents
    sim_dox = collection.query(
        #return the first element from this function 
        query_embeddings = user_question_emb[0] ,
        n_results = ndocs)
   
    return sim_dox


