from langchain_community.chat_models.oci_generative_ai import ChatOCIGenAI
from langchain_core.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_community.embeddings import OCIGenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema.runnable import RunnableMap
from langchain_community.document_loaders import PyPDFLoader, UnstructuredPowerPointLoader, UnstructuredPDFLoader, PyMuPDFLoader
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from tqdm import tqdm
import os
import pickle

INDEX_PATH = "./faiss_index"
PROCESSED_DOCS_FILE = os.path.join(INDEX_PATH, "processed_docs.pkl")

def read_pdfs(pdf_path):
    if "-ocr" in pdf_path:
        doc_pages = PyMuPDFLoader(str(pdf_path)).load()
    else:
        doc_pages = UnstructuredPDFLoader(str(pdf_path)).load()
    full_text = "\n".join([page.page_content for page in doc_pages])
    return full_text

def smart_split_text(text, max_chunk_size=2000):
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + max_chunk_size, text_length)
        split_point = max(
            text.rfind('.', start, end),
            text.rfind('!', start, end),
            text.rfind('?', start, end),
            text.rfind('\n\n', start, end)
        )

        if split_point == -1 or split_point <= start:
            split_point = end
        else:
            split_point += 1

        chunk = text[start:split_point].strip()
        if chunk:
            chunks.append(chunk)

        start = split_point

    return chunks

def load_previously_indexed_docs():
    if os.path.exists(PROCESSED_DOCS_FILE):
        with open(PROCESSED_DOCS_FILE, "rb") as f:
            return pickle.load(f)
    return set()

def save_indexed_docs(docs):
    with open(PROCESSED_DOCS_FILE, "wb") as f:
        pickle.dump(docs, f)

def chat():
    llm = ChatOCIGenAI(
        model_id="meta.llama-3.1-405b-instruct",
        service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
        compartment_id="ocid1.compartment.oc1..aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        auth_profile="DEFAULT",
        model_kwargs={"temperature": 0.7, "top_p": 0.75, "max_tokens": 4000},
    )

    embeddings = OCIGenAIEmbeddings(
        model_id="cohere.embed-multilingual-v3.0",
        service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
        compartment_id="ocid1.compartment.oc1..aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        auth_profile="DEFAULT",
    )

    pdf_paths = [
        './Manuals/using-integrations-oracle-integration-3.pdf',
        './Manuals/SOASE.pdf',
        './Manuals/SOASUITEHL7.pdf'
    ]

    already_indexed_docs = load_previously_indexed_docs()
    updated_docs = set()

    try:
        vectorstore = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        print("✔️ FAISS index loaded.")
    except Exception:
        print("⚠️ FAISS index not found, creating a new one.")
        vectorstore = None

    new_chunks = []

    for pdf_path in tqdm(pdf_paths, desc="📄 Processing PDFs"):
        print(f" {os.path.basename(pdf_path)}")
        if pdf_path in already_indexed_docs:
            print(f"✅ Already indexed: {pdf_path}")
            continue

        full_text = read_pdfs(pdf_path=pdf_path)
        text_chunks = smart_split_text(full_text, max_chunk_size=2000)

        for chunk_text in tqdm(text_chunks, desc=f"📄 Splitting text", dynamic_ncols=True, leave=False):
            doc = Document(page_content=chunk_text, metadata={"source": pdf_path})
            new_chunks.append(doc)
            print(f"✅ Indexed chunk with {len(chunk_text)} chars.")

        updated_docs.add(str(pdf_path))

    if new_chunks:
        if vectorstore:
            vectorstore.add_documents(new_chunks)
        else:
            vectorstore = FAISS.from_documents(new_chunks, embedding=embeddings)

        vectorstore.save_local(INDEX_PATH)
        save_indexed_docs(already_indexed_docs.union(updated_docs))
        print(f"💾 {len(new_chunks)} chunks saved to FAISS index.")
    else:
        print("📁 No new documents to index.")

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 50, "fetch_k": 100})

    template = """ 
        Document context:
        {context}
        
        Question:
        {input}
        
        Interpretation rules:
        Rule 1: SOA SUITE documents: `SOASUITE.pdf` and `SOASUITEHL7.pdf`
        Rule 2: Oracle Integration (OIC) document: `using-integrations-oracle-integration-3.pdf`
        Rule 3: If not a comparison between SOA SUITE and OIC, only consider documents relevant to the product.
        Rule 4: If the question compares SOA SUITE and OIC, compare both.
        Mention at the beginning which tool is being addressed: {input}
    """
    prompt = PromptTemplate.from_template(template)

    def get_context(x):
        query = x.get("input") if isinstance(x, dict) else x
        return retriever.invoke(query)

    chain = (
            RunnableMap({
                "context": RunnableLambda(get_context),
                "input": lambda x: x.get("input") if isinstance(x, dict) else x
            })
            | prompt
            | llm
            | StrOutputParser()
    )

    print("READY")
    while True:
        query = input()
        if query == "quit":
            break
        response = chain.invoke(query)
        print(response)

chat()