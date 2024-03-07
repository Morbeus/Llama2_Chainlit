from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.chains.llm import LLMChain
from langchain_community.llms import CTransformers
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
import chainlit as cl

DB_FAISS_PATH = 'vectorstore/db_faiss'

custom_prompt = """
Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompts():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompts = PromptTemplate(template=custom_prompt,
                            input_variables=['context', 'question'])
    return prompts

#Retrieval QA Chain
def retriever_qa_chain(llm, prompt, db):
    # llmchain = LLMChain(llm = llm, prompt = prompt, callbacks = None)
    # document_prompt = PromptTemplate(
    # input_variables=["page_content"],
    # template="{page_content}")
    # document_variable_name = "context"
    # combine_documents_chain = StuffDocumentsChain(
    #     llm_chain=llmchain,
    #     document_variable_name="context",
    #     document_prompt=document_prompt,
    #     callbacks=None,
    # )
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                           retriever=db.as_retriever(search_kwargs={'k': 2}),
                                           return_source_documents=True,
                                           chain_type_kwargs={'prompt': prompt}
                                           )
    return qa_chain

#Loading the model
def load_llm_model():
    # Load the locally downloaded model here
    llm = CTransformers(
        model = "TheBloke/Llama-2-7B-Chat-GGML",
        model_type="llama",
        max_new_tokens = 1024,
        temperature = 0.5
    )
    return llm

#QA Model Function
def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings)
    llm = load_llm_model()
    qa_prompt = set_custom_prompts()
    qa = retriever_qa_chain(llm, qa_prompt, db)

    return qa

#output function
def result(query):
    qa_result = qa_bot()
    response = qa_result({'query': query})
    return response

#chainlit code
@cl.on_chat_start
async def start():
    chain = qa_bot()
    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    msg.content = "Hi, What is your query?"
    await msg.update()

    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain") 
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True
    res = await chain.acall(message.content, callbacks=[cb])
    answer = res["result"]
    sources = res["source_documents"]

    if sources:
        answer += f"\nSources:" + str(sources)
    else:
        answer += "\nNo sources found"

    await cl.Message(content=answer).send()