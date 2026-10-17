import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_qdrant import QdrantVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

load_dotenv()

COLLECTION = "langchain_101"
K = 4
FOLDER = "policies"

client = QdrantClient(url="http://localhost:6333")

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    output_dimensionality=1536,
)

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    # temperature=0,
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a company assistant that answers employee questions about the "
            "expense policy and employee handbook. Answer ONLY using the context below. "
            "If the context does not contain the answer, say you don't have that "
            "information in the documents. Be concise and accurate.\n\n"
            "Context:\n{context}",
        ),
        ("human", "{question}"),
    ]
)

vectorstore = QdrantVectorStore(
    client=client,
    collection_name=COLLECTION,
    embedding=embeddings,
)

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": K,
        # metadata filter: only chunks whose metadata "folder" is "policies" are searched
        # "filter": Filter(must=[FieldCondition(key="metadata.folder", match=MatchValue(value=FOLDER))]),
    },
)

# lcel 
chain = prompt | llm | StrOutputParser()


def retrieve(query: str) -> str:
    docs = retriever.invoke(query)

    for d in docs:
        print(f"retrieved: folder={d.metadata.get('folder')} source={d.metadata.get('source')} h2={d.metadata.get('h2')}")

    # joining the context, because k=3, but for llm, only 1 context should be gone na, it concatenates into one
                                  # source added by pdf loader !! 
    return "\n\n---\n\n".join(
        f"[Source: {d.metadata.get('source')}, page {d.metadata.get('page')}]\n{d.page_content}"
        for d in docs
    )


if __name__ == "__main__":
    query = input("Ask a question: ").strip()
    context = retrieve(query)
    # print(f"context: {context}")
    print("------------------------------------------------")
    print("------------------------------------------------")
    try:
        print(chain.invoke({"context": context, "question": query}))
    except Exception as e:
        print(f"Error while invoking chain: {type(e).__name__}: {e}")
