from langchain import hub
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain.output_parsers import PydanticOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.output_parsers import JsonOutputParser
from typing import Dict, TypedDict
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq


load_dotenv()

os.environ["GROQ_API_KEY"] = os.environ.get("GROQ_API_KEY")
os.environ["TAVILY_API_KEY"] = os.environ.get("TAVILY_API_KEY")


class grade(BaseModel):
    """
    binary score for relevance check for the documents
    """

    score: str = Field(description="Relevance score 'yes' or 'no'")


class Nodes:

    def __init__(self):
        self.client = ChatGroq(
            temperature=0,
            model_name="mixtral-8x7b-32768",
            streaming=True,
            callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
        )

        self.tavily_tool = TavilySearchResults()

    def _format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def query_optimzer_initial(self, state):
        """
        Transform the query to produce a better question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates question key with a re-phrased question
        """
        state_dict = state["keys"]
        question = state_dict["question"]
        prompt = PromptTemplate(
            template="""You are generating questions that is well optimized for 
                        retrieval. \n 
            Look at the input and try to reason about the underlying sematic 
            intent / meaning. \n 
            Here is the initial question:
            \n ------- \n
            {question} 
            \n ------- \n
            Provide an improved question without any premable, only respond 
            with the updated question: """,
            input_variables=["question"],
        )
        groq = self.client

        chain = prompt | groq | StrOutputParser()

        response = chain.invoke({"question": question})
        return {"keys": {"question": response}}

    def query_optimzer_web(self, state):
        """
        Transform the query to produce a better question for web search.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates question key with a re-phrased question
        """
        state_dict = state["keys"]
        question = state_dict["question"]
        documents = state_dict["documents"]
        prompt = PromptTemplate(
            template="""You are generating questions that is well optimized for 
                        web search \n 
            Look at the input and try to reason about the underlying sematic 
            intent / meaning. for that it the question will be more efficient in web search to get relevant information \n 
            Here is the initial question:
            \n ------- \n
            {question} 
            \n ------- \n
            Provide an improved question without any premable, only respond 
            with the updated question: """,
            input_variables=["question"],
        )
        llm = self.client
        chain = prompt | llm | StrOutputParser()

        response = chain.invoke({"question": question})
        return {"keys": {"question": response, "documents": documents}}


    def retrieval_validator(self, state):
        """
        Determines whether the retrieved documents are relevant to the question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates documents key with relevant documents
        """
        state_dict = state["keys"]
        documents = state_dict["documents"]
        question = state_dict["question"]
        llm = self.client

        ## output parser
        class grade(BaseModel):
            """
            binary score for relevance check for the documents
            """

            score: str = Field(description="Relevance score 'yes' or 'no'")

        # parser  = PydanticOutputParser(pydantic_object=grade)

        parser = JsonOutputParser(pydantic_object=grade)
        prompt = PromptTemplate(
            template="""You are a grader assessing relevance of a retrieved 
                        document to a user question. \n 
            Here is the retrieved document: \n\n {context} \n\n
            Here is the user question: {question} \n
            If the document contains keywords related to the user question, 
            grade it as relevant. \n
            It does not need to be a stringent test. The goal is to filter out 
            erroneous retrievals. \n
            Give a binary score 'yes' or 'no' score to indicate whether the 
            document is relevant to the question. \n
            Provide the binary score as a JSON,
            DO NOT provide any explanation and use these instructons to format the output  \n: 
            {format_instructions}""",
            input_variables=["query"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )
        chain = prompt | llm | parser
        search_activate = "No"  ## deafult -(search tool not activated)
        filtered_docs = []
        for doc in documents:
            response = chain.invoke(
                {
                    "question": question,
                    "context": doc.page_content,
                    "format_instructions": parser.get_format_instructions(),
                }
            )
            grade = response["score"]
            if grade == "yes":
                print(f"Document is relevant--")
                filtered_docs.append(doc)
            else:
                print(f"Document is not relevant -- ")
                search_activate = "Yes"
                continue
        return {
            "keys": {
                "question": question,
                "documents": filtered_docs,
                "search_activate": search_activate,
            }
        }

    def generate(self, state):
        """
        Generate answer

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, generation,
            that contains LLM generation
        """
        state_dict = state["keys"]
        question = state_dict["question"]
        documents = state_dict["documents"]
        prompt = hub.pull("rlm/rag-prompt")
        llm = self.client

        docs = self._format_docs(documents)

        rag_chain = prompt | llm | StrOutputParser()
        response = rag_chain.invoke({"context": docs, "question": question})
        return {
            "keys": {
                "documents": documents,
                "question": question,
                "generation": response,
            }
        }

    def web_search(self, state):
        """
        Web search based on the re-phrased question using Tavily API.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Web results appended to documents.
        """
        state_dict = state["keys"]
        question = state_dict["question"]
        documents = state_dict["documents"]

        try:
            tool = TavilySearchResults()
            response = tool.invoke({"query": question})
            results = "\n".join([d["content"] for d in response])
            results = Document(page_content=results, metadata={"source": "internet"})
            documents.append(results)
        except Exception as e:
            print(f"Error: {e}")
        return {"keys": {"documents": documents, "question": question}}

    def decide_to_generate(self, state):
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
