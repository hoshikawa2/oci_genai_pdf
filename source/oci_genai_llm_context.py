import chromadb
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.chat_models.oci_generative_ai import ChatOCIGenAI
from langchain_core.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.embeddings import OCIGenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import FAISS

def chat():

    caminhos_pdf = [    './Manuals/using-integrations-oracle-integration-3.pdf',
                        './Manuals/SOASUITE.pdf',
                        './Manuals/SOASUITEHL7.pdf'
                        ]

    pages = []
    ids = []
    counter = 1
    for caminho_pdf in caminhos_pdf:
        doc_pages = PyPDFLoader(caminho_pdf).load_and_split()

        pages.extend(doc_pages)
        ids.append(str(counter))
        counter = counter + 1

    llm = ChatOCIGenAI(
        model_id="meta.llama-3.1-405b-instruct",
        service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
        compartment_id="ocid1.compartment.oc1..aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        auth_profile="DEFAULT",  # replace with your profile name,
        model_kwargs={"temperature": 0.7, "top_p": 0.75, "max_tokens": 1000},
    )

    embeddings = OCIGenAIEmbeddings(
        model_id="cohere.embed-multilingual-v3.0",
        service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
        compartment_id="ocid1.compartment.oc1..aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        auth_profile="DEFAULT",  # replace with your profile name
    )

    vectorstore = FAISS.from_documents(
        pages,
        embedding=embeddings
    )
    retriever = vectorstore.as_retriever()

    template = """ 
        If the query in question is not a comparison between SOA SUITE and OIC, consider only the documents relevant to the subject,
         that is, if the question is about SOA SUITE, consider only the SOA SUITE documents. If the question is about OIC,
          consider only the OIC document. If the question is a comparison between SOA SUITE and OIC, consider all documents. 
          Inform at the beginning which tool is being discussed    : {input} 
    """
    prompt = PromptTemplate.from_template(template)


    chain = (
            {"context": retriever,
             "input": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )

    while (True):
        query = input()
        if query == "quit":
            break
        print(chain.invoke(query))


chat()

