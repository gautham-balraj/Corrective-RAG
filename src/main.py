from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from state import GraphState
from langgraph.graph import END, StateGraph
from nodes import Nodes
import pprint
import sys

nodes = Nodes()

def doc_loader(url: str):
    loader = WebBaseLoader(url)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=700,
        chunk_overlap=100,
    )
    all_splits = text_splitter.split_documents(docs)
    print(f"Documents loaded and split into chunks - {len(all_splits)}")
    return all_splits

def vector_store(docs):
    embeddings = OllamaEmbeddings(model='nomic-embed-text')
    db_nomic = Chroma.from_documents(docs, embeddings, persist_directory="./new_db")
    db_nomic.persist()
    

def ingest(url: str):
    docs = doc_loader(url)
    vector_store(docs)
    print("vector store created !!!!!!!")
    
def decide_to_generate(state):
    """
    Determines whether to generate an answer or re-generate a question 
    for web search.

    Args:
        state (dict): The current state of the agent, including all keys.

    Returns:
        str: Next node to call
    """

    print("---DECIDE TO GENERATE---")
    state_dict = state["keys"]
    question = state_dict["question"]
    filtered_documents = state_dict["documents"]
    search = state_dict["search_activate"]

    if search == "Yes":
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print("---DECISION: TRANSFORM QUERY and RUN WEB SEARCH---")
        return "transform_query"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"
    
def retrive(state):
    """
    retrive relevant documents
    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents,
        that contains retrieved documents
    """
    print("retrieval tool called")
    state_dict = state["keys"]
    query = state_dict["question"]
    embeddings = OllamaEmbeddings(model='nomic-embed-text')
    retriever = Chroma(persist_directory="./new_db",embedding_function=embeddings).as_retriever()
    documents = retriever.get_relevant_documents(query)
    return {"keys": {"documents": documents, "question": query}}
    
    
    
    
def main(url: str,query: str):    
    
    ingest(url)
    embeddings = OllamaEmbeddings(model='nomic-embed-text')
    Chroma(persist_directory="./new_db",embedding_function=embeddings)
    workflow = StateGraph(GraphState)
    workflow.add_node("initial", nodes.query_optimzer_initial)
    workflow.add_node("retrieval",retrive)
    workflow.add_node("check_relevance", nodes.retrieval_validator)
    workflow.add_node("generate", nodes.generate)
    workflow.add_node("web_search", nodes.web_search)
    workflow.add_node("transform_query", nodes.query_optimzer_web)

    workflow.set_entry_point("initial")
    workflow.add_edge("initial", "retrieval")
    workflow.add_edge("retrieval", "check_relevance")
    workflow.add_conditional_edges(
        "check_relevance",
        decide_to_generate,
        {
            "generate": "generate",
            "transform_query": "transform_query",
        })
    workflow.add_edge('transform_query','web_search')
    workflow.add_edge('web_search','generate')
    workflow.add_edge('generate',END)
    app = workflow.compile()
    
    inputs = {
        "keys": {
            "question": query,
        }
    }
    for output in app.stream(inputs):
        for key, value in output.items():
            # Node
            print(f"Node '{key}':")
            # Optional: print full state at each node
            pprint.pprint(value["keys"], indent=2, width=80, depth=None)
        pprint.pprint("\n---\n")

    # Final generation
    print('------------------final answer-----------------------')
    print(value['keys']['generation'])
    
# if __name__ == "__main__":
#     main(url="https://medium.com/@vidrihmarko/language-processing-units-lpus-a-deep-dive-into-groqs-innovation-74b9f80cb2fd",query="how is LPU is better compared to others?")



if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main.py <URL>")
        sys.exit(1)
    
    url = sys.argv[1]
    query = input("Enter your question: ")
    
    main(url=url, query=query)
