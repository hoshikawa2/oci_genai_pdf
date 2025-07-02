import os
import pickle
import re
import atexit
import subprocess
import socket
import time
from tqdm import tqdm
from neo4j import GraphDatabase
from langchain_community.chat_models.oci_generative_ai import ChatOCIGenAI
from langchain_core.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_community.embeddings import OCIGenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema.runnable import RunnableMap
from langchain_community.document_loaders import UnstructuredPDFLoader, PyMuPDFLoader
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda


# =========================
# Graphos Database: Neo4J
# =========================

def is_port_open(host, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(2)
        return sock.connect_ex((host, port)) == 0


def start_neo4j_if_not_running():
    if is_port_open('127.0.0.1', 7687):
        print("🟢 Neo4j is already running on port 7687.")
        return

    print("🟡 Neo4j not found on port 7687. Starting via Docker...")

    try:
        subprocess.run([
            "docker", "run", "--name", "neo4j-graphrag",
            "-p", "7474:7474", "-p", "7687:7687",
            "-d",
            "-e", f"NEO4J_AUTH={NEO4J_USER}/{NEO4J_PASSWORD}",
            "--restart", "unless-stopped",
            "neo4j:5"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print("🚫 Error starting Neo4j via Docker:", e)
        raise

    print("⏳ Waiting for Neo4j to start...", end="")
    for _ in range(10):
        if is_port_open('127.0.0.1', 7687):
            print("✅ Neo4j is ready!")
            return
        print(".", end="", flush=True)
        time.sleep(2)

    print("\n❌ Failed to connect to Neo4j.")
    raise ConnectionError("Neo4j did not start correctly.")


# =========================
# Global Configurations
# =========================
INDEX_PATH = "./faiss_index"
PROCESSED_DOCS_FILE = os.path.join(INDEX_PATH, "processed_docs.pkl")

chapter_separator_regex = r"^(#{1,6} .+|\*\*.+\*\*)$"

# Neo4j Configuration
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "your_password_here"

#LLM Definitions
llm = ChatOCIGenAI(
    model_id="meta.llama-3.1-405b-instruct",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id="ocid1.compartment.oc1..aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    auth_profile="DEFAULT",
    model_kwargs={"temperature": 0.7, "top_p": 0.75, "max_tokens": 4000},
)

llm_for_rag = ChatOCIGenAI(
    model_id="meta.llama-3.1-405b-instruct",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id="ocid1.compartment.oc1..aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    auth_profile="DEFAULT",
)

embeddings = OCIGenAIEmbeddings(
    model_id="cohere.embed-multilingual-v3.0",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id="ocid1.compartment.oc1..aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    auth_profile="DEFAULT",
)

start_neo4j_if_not_running()

graph_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
atexit.register(lambda: graph_driver.close())


# =========================
# Helper Functions
# =========================
def create_knowledge_graph(chunks):
    with graph_driver.session() as session:
        for doc in chunks:
            text = doc.page_content
            source = doc.metadata.get("source", "unknown")

            if not text.strip():
                print(f"⚠️ Skipping empty chunk from {source}")
                continue

            prompt = f"""
            You are an expert in knowledge extraction.

            Given the following technical text:

            {text}

            Extract key entities (systems, components, processes, protocols, APIs, services) and their relationships in the following format:
            - Entity1 -[RELATION]-> Entity2

            Use UPPERCASE for RELATION types, replacing spaces with underscores.

            Example output:
            SOA Suite -[HAS_COMPONENT]-> BPEL Process
            Integration Flow -[USES]-> REST API
            Order Service -[CALLS]-> Inventory Service

            Important:
            - If there are no entities or relationships, return: NONE
            - Only output the list, no explanation.
            """

            response = llm_for_rag.invoke(prompt)

            if not hasattr(response, "content"):
                print("[ERROR] Failed to get graph triples.")
                continue

            result = response.content.strip()

            if result.upper() == "NONE":
                print(f"ℹ️ No entities found in chunk from {source}")
                continue

            triples = result.splitlines()

            for triple in triples:
                parts = triple.split("-[")
                if len(parts) != 2:
                    print(f"⚠️ Skipping malformed triple: {triple}")
                    continue

                right_part = parts[1].split("]->")
                if len(right_part) != 2:
                    print(f"⚠️ Skipping malformed relation part: {parts[1]}")
                    continue

                relation, entity2 = right_part
                relation = relation.strip().replace(" ", "_").upper()
                entity1 = parts[0].strip()
                entity2 = entity2.strip()

                session.run(
                    f"""
                    MERGE (e1:Entity {{name: $entity1}})
                    MERGE (e2:Entity {{name: $entity2}})
                    MERGE (e1)-[:{relation} {{source: $source}}]->(e2)
                    """,
                    entity1=entity1,
                    entity2=entity2,
                    source=source
                )


# =========================
# GraphRAG - Query the graph
# =========================
def query_knowledge_graph(query_text):
    with graph_driver.session() as session:
        result = session.run(
            """
            MATCH (e1)-[r]->(e2)
            WHERE toLower(e1.name) CONTAINS toLower($search)
               OR toLower(e2.name) CONTAINS toLower($search)
               OR toLower(type(r)) CONTAINS toLower($search)
            RETURN e1.name AS from, type(r) AS relation, e2.name AS to
            LIMIT 20
            """,
            search=query_text
        )
        relations = []
        for record in result:
            relations.append(f"{record['from']} -[{record['relation']}]-> {record['to']}")
        return "\n".join(relations) if relations else "No related entities found."


# =========================
# Semantical Chunking
# =========================
def split_llm_output_into_chapters(llm_text):
    chapters = []
    current_chapter = []
    lines = llm_text.splitlines()

    for line in lines:
        if re.match(chapter_separator_regex, line):
            if current_chapter:
                chapters.append("\n".join(current_chapter).strip())
            current_chapter = [line]
        else:
            current_chapter.append(line)

    if current_chapter:
        chapters.append("\n".join(current_chapter).strip())

    return chapters


def semantic_chunking(text):
    prompt = f"""
    You received the following text extracted via OCR:

    {text}

    Your task:
    1. Identify headings (short uppercase or bold lines, no period at the end)
    2. Separate paragraphs by heading
    3. Indicate columns with [COLUMN 1], [COLUMN 2] if present
    4. Indicate tables with [TABLE] in markdown format
    """

    response = llm_for_rag.invoke(prompt)
    return response


def read_pdfs(pdf_path):
    if "-ocr" in pdf_path:
        doc_pages = PyMuPDFLoader(str(pdf_path)).load()
    else:
        doc_pages = UnstructuredPDFLoader(str(pdf_path)).load()
    full_text = "\n".join([page.page_content for page in doc_pages])
    return full_text


def smart_split_text(text, max_chunk_size=10_000):
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


# =========================
# Main Function
# =========================
def chat():
    pdf_paths = [
        './Manuals/SOASUITE.pdf',
        './Manuals/SOASUITEHL7.pdf',
        './Manuals/using-integrations-oracle-integration-3.pdf'
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

    for pdf_path in tqdm(pdf_paths, desc=f"📄 Processing PDFs"):
        print(f" {os.path.basename(pdf_path)}")
        if pdf_path in already_indexed_docs:
            print(f"✅ Document already indexed: {pdf_path}")
            continue
        full_text = read_pdfs(pdf_path=pdf_path)

        text_chunks = smart_split_text(full_text, max_chunk_size=10_000)
        overflow_buffer = ""

        for chunk in tqdm(text_chunks, desc=f"📄 Processing text chunks", dynamic_ncols=True, leave=False):
            current_text = overflow_buffer + chunk

            treated_text = semantic_chunking(current_text)

            if hasattr(treated_text, "content"):
                chapters = split_llm_output_into_chapters(treated_text.content)

                last_chapter = chapters[-1] if chapters else ""

                if last_chapter and not last_chapter.strip().endswith((".", "!", "?", "\n\n")):
                    print("📌 Last chapter seems incomplete, saving for the next cycle")
                    overflow_buffer = last_chapter
                    chapters = chapters[:-1]
                else:
                    overflow_buffer = ""

                for chapter_text in chapters:
                    doc = Document(page_content=chapter_text, metadata={"source": pdf_path})
                    new_chunks.append(doc)
                    print(f"✅ New chapter indexed:\n{chapter_text}...\n")

            else:
                print(f"[ERROR] semantic_chunking returned unexpected type: {type(treated_text)}")

        updated_docs.add(str(pdf_path))

    if new_chunks:
        if vectorstore:
            vectorstore.add_documents(new_chunks)
        else:
            vectorstore = FAISS.from_documents(new_chunks, embedding=embeddings)

        vectorstore.save_local(INDEX_PATH)
        save_indexed_docs(already_indexed_docs.union(updated_docs))
        print(f"💾 {len(new_chunks)} chunks added to FAISS index.")

        print("🧠 Building knowledge graph...")
        create_knowledge_graph(new_chunks)

    else:
        print("📁 No new documents to index.")

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 50, "fetch_k": 100})

    template = """ 
        Document context:
        {context}
        
        Graph context:
        {graph_context}
        
        Question:
        {input}
        
        Interpretation rules:
        - You can search for a step-by-step tutorial about a subject
        - You can search a concept description about a subject
        - You can search for a list of components about a subject
    """
    prompt = PromptTemplate.from_template(template)

    def get_context(x):
        query = x.get("input") if isinstance(x, dict) else x
        return retriever.invoke(query)

    chain = (
            RunnableMap({
                "context": RunnableLambda(get_context),
                "graph_context": RunnableLambda(lambda x: query_knowledge_graph(x.get("input") if isinstance(x, dict) else x)),
                "input": lambda x: x.get("input") if isinstance(x, dict) else x
            })
            | prompt
            | llm
            | StrOutputParser()
    )

    print("✅ READY")

    while True:
        query = input("❓ Question (or 'quit' to exit): ")
        if query.lower() == "quit":
            break
        response = chain.invoke(query)
        print("\n📜 RESPONSE:\n")
        print(response)
        print("\n" + "=" * 80 + "\n")


# 🚀 Run
if __name__ == "__main__":
    chat()