import boto3
import json
import document_processing_fx as dp

def create_prompt(context, question, documents = None):
    '''
    Function that will create the prompt to be sent to LLama. This will add in all the context and other instructions for the model. You should probably make different flavors of these!!

    Context should be fed directly from the similarity_search_fx.
    quetion = the question to be asked   
        
    '''
    
    #might be redundant
    context_list = []
    for i in context:
        context_list.append(i)
    


    prompt_guidance = f'''
    You are going to be asked a geological question by a user. You are to respond using the context below and your own knowledge. If you are unsure of your answer, do not make anything up. Just return back that you are unsure. Do not thank the user
    for providing context. Just give your answer.

    Here is some context for the question in the form of a python list
    {context_list}

    the users question is below:
    {question}

    '''
    
    return prompt_guidance


def construct_payload_for_llama(prompt):
    '''
    Constructs the payload to send to LLama
    prompt = the prompt that was constructed by whichever create_prompt fx you chose
    '''

    payload = {
        "inputs": [[
            {"role": "user", "content":  prompt},
        ]],
        "parameters": {"max_new_tokens": 1000, "temperature": 0.01}
    }
    
    return payload



def ask_llama_a_question(payload):
    '''
    Constructs the payload to send to LLama
    payload = the payload constructed by the construct_payload_for_llama fx
    '''

    sagemaker = boto3.client('sagemaker-runtime')

    response = sagemaker.invoke_endpoint(
      EndpointName="jumpstart-dft-meta-textgeneration-llama-2-7b-f",
      ContentType = "application/json",
      Body=json.dumps(payload),
      CustomAttributes= "accept_eula=true"
    )

    output = json.loads(response['Body'].read())
    
    return output






def uaq_workflow(chromadb_path, collection_name, question, ndocs, max_tokens = 1000, temperature = .001):
    '''
    User Asks Question Workflow
    Workflow that will take a users question all the way through the process and get an answer. You need to have created a chromadb and collection with embeddings to be able to use this function!!!!

    chromadb_path = the path to your chromadb
    collection_name = the name of your collection with your dox and embeddings
    question = the question the user has asked
    ndocs = the number of similair documents you want returned
    # max_tokens = NOT IMPLEMENTED the maximum number of tokens the model gets to answer the question
    # temperature = NOT IMPLEMENTED how creative the model can be. Lower = less.

    '''

    context_data = dp.get_similair_documents(chromadb_path, collection_name, question, ndocs)

    prompt = create_prompt(context_data["documents"][0], question)

    payload = construct_payload_for_llama(prompt)

    output = ask_llama_a_question(payload)

    return output





