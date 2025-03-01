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

    class State(TypedDict):
        question: str
        context: List[Document]
        answer: str

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
        print("MongoDB connection opened. Hello!")

    def _initialize_providers(self):
        self.llm = init_chat_model(
            "gemini-2.0-flash-001", model_provider="google_vertexai"
        )
        self.embeddings = VertexAIEmbeddings(model="text-embedding-004")

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
            "answer": result["answer"],
            "context": result["context"],
        }

    def shutdown(self):
        self.client.close()
        print("MongoDB connection closed. Goodbye!")


class Query(BaseModel):
    text: str
    context: bool


def lifespan(app: FastAPI):
    app.state.graph = Graph()

    yield

    app.state.graph.shutdown()


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def root():
    return {"message": "hello, traveler"}


@app.post("/api/query")
async def api_query(query: Query):
    result = app.state.graph.serve(query.text)

    return result if query.context else {"answer": result["answer"]}
