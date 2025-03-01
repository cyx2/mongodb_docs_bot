import os

from dotenv import load_dotenv
from fastapi import FastAPI
from langchain import hub
from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from langgraph.graph import START, StateGraph
from pydantic import BaseModel
from pymongo import MongoClient
from typing_extensions import List, TypedDict


class Graph:
    def __init__(self):
        self._initialize_providers()
        self._initialize_mongodb()
        self._construct_graph()

    def _initialize_mongodb(self):
        load_dotenv()
        mongodb_uri = os.getenv("MONGODB_URI")
        mongodb_database = os.getenv("MONGODB_DATABASE")
        prefix = "loader_explorations"

        self.client = MongoClient(mongodb_uri)
        self.collection = self.client[mongodb_database][prefix]

        self.vector_store = MongoDBAtlasVectorSearch(
            embedding=self.embeddings,
            collection=self.collection,
            index_name=prefix,
            relevance_score_fn="cosine",
        )

    def _initialize_providers(self):
        self.llm = init_chat_model(
            "gemini-2.0-flash-001", model_provider="google_vertexai"
        )
        self.embeddings = VertexAIEmbeddings(model="text-embedding-004")

    class State(TypedDict):
        question: str
        context: List[Document]
        answer: str

    def _retrieve(self, state: State):
        retrieved_docs = self.vector_store.similarity_search(state["question"])
        return {"context": retrieved_docs}

    def _generate(self, state: State):
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = self.prompt.invoke(
            {"question": state["question"], "context": docs_content}
        )
        response = self.llm.invoke(messages)
        return {"answer": response.content}

    def _construct_graph(self):
        self.prompt = hub.pull("rlm/rag-prompt")
        graph_builder = StateGraph(self.State).add_sequence(
            [self._retrieve, self._generate]
        )
        graph_builder.add_edge(START, "_retrieve")
        self.graph = graph_builder.compile()

    def serve(self, query: str):
        result = self.graph.invoke({"question": query})

        return {
            "answer": f'Answer: {result["answer"]}\n\n',
            "context": f'Context: {result["context"]}',
        }


app = FastAPI()


@app.get("/")
async def root():
    return {"message": "hello, traveler"}


class Query(BaseModel):
    text: str
    context: bool


@app.post("/api/query")
async def api_query(query: Query):
    graph = Graph()

    result = graph.serve(query.text)

    return result if query.context else {"answer": result["answer"]}
