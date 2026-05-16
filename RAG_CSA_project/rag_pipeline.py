from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from typing import TypedDict, Any
import os
from getpass import getpass
import warnings

# ----------------------------- handling warnings ---------------------------- #
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore", message=".*allowed_objects.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')
    
# ---------------------------------- API Key setup --------------------------------- #

if not os.environ.get("GROQ_API_KEY"):
    os.environ["GROQ_API_KEY"] = getpass("Enter Groq API key: ")
    print("Key ready.")
    

# ---------------------------------- Config ---------------------------------- #
PDF_DIR     = "./pdfs"          
PERSIST_DIR = "./chroma_db"
COLLECTION  = "support_kb"
TOP_K       = 3

# -------------------- Embedder (shared by ingest + query) ------------------- #
def get_embedder():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

# --------------------------------- Ingestion -------------------------------- #
def ingest(pdf_dir=PDF_DIR):
    all_chunks = []

    for filename in os.listdir(pdf_dir):
        if filename.endswith(".pdf"):
            path   = os.path.join(pdf_dir, filename)

            # 1. Load
            docs   = PyPDFLoader(path).load()

            # 2. Chunk
            chunks = RecursiveCharacterTextSplitter(
                chunk_size=500, chunk_overlap=50
            ).split_documents(docs)

            all_chunks.extend(chunks)
            print(f"Loaded {len(chunks)} chunks from {filename}")

    # 3. Embed + Store
    Chroma.from_documents(
        all_chunks, get_embedder(),
        persist_directory=PERSIST_DIR,
        collection_name=COLLECTION
    )
    print(f"\nTotal {len(all_chunks)} chunks stored in ChromaDB.")

# --------------------------------- Retrival --------------------------------- #
def load_store():
    return Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=get_embedder(),
        collection_name=COLLECTION
    )

def retrieve(query, store):
    return store.similarity_search(query, k=TOP_K)

# -------------------------------- Graph state ------------------------------- #
class State(TypedDict):
    query:     str
    store:     Any
    chunks:    list
    answer:    str
    route:     str
    escalated: bool

# ----------------------------------- Nodes ---------------------------------- #

def retrieve_node(state):
    results = state["store"].similarity_search_with_score(state["query"], k=TOP_K)
    chunks = [doc for doc, score in results if score < 1.4]
    return {"chunks": chunks}

def router_node(state):
    no_context  = len(state["chunks"]) == 0
    wants_human = "human" in state["query"].lower()
    route = "hitl" if (no_context or wants_human) else "llm"
    return {"route": route, "escalated": route == "hitl"}

def llm_node(state):
    llm = ChatGroq(model="llama-3.1-8b-instant")
    context = "\n\n".join([c.page_content for c in state["chunks"]])
    prompt  = f"""You are a helpful customer support assistant.
Use the context below to answer the question in a clear and complete way.
If the context covers the topic, explain it fully in 3-5 sentences.

Context:
{context}

Question: {state['query']}

Answer:"""
    return {"answer": llm.invoke(prompt).content}

def hitl_node(state):
    print(f"\n[HITL] Query: {state['query']}")
    print("[HITL] No automated answer. Agent, please respond:")
    return {"answer": input("Agent > ")}

def output_node(state):
    return {}

# ----------------------------------- Graph ---------------------------------- #
def build_graph():
    g = StateGraph(State)
    g.add_node("retrieve", retrieve_node)
    g.add_node("router",   router_node)    
    g.add_node("llm",      llm_node)
    g.add_node("hitl",     hitl_node)
    g.add_node("output",   output_node)

    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "router")      
    g.add_conditional_edges(
        "router",                          
        lambda s: s["route"],
        {"llm": "llm", "hitl": "hitl"}
    )
    g.add_edge("llm",    "output")
    g.add_edge("hitl",   "output")
    g.add_edge("output", END)
    return g.compile()

# ---------------------------------- cli run --------------------------------- #
def run():
    store = load_store()
    graph = build_graph()

    clear_screen()
    
    print("┌──────────────────────────────────────────┐")
    print("│    🤖 Bot ready. Type 'quit' to exit.    │")
    print("└──────────────────────────────────────────┘\n")
    
    while True:
        query = input("You: ").strip()
        if not query: continue
        if query.lower() == "quit": break

        state = graph.invoke({
            "query": query, "store": store,
            "chunks": [], "answer": "",
            "route": "", "escalated": False
        })

        tag = " [ESCALATED]" if state["escalated"] else ""
        print(f"Bot: {state['answer']}{tag}\n")

if __name__ == "__main__":
    run()
    
# ------------------------------------ END ----------------------------------- #