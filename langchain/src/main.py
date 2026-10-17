import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

load_dotenv()

COLLECTION = "langchain_101"
K = 10
FOLDER = "policies"

client = QdrantClient(url="http://localhost:6333")

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    output_dimensionality=1536,
)

llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0,
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

# lcel 
chain = prompt | llm | StrOutputParser()


def retrieve(query: str) -> str:
    vector = embeddings.embed_query(query)
    hits = client.query_points(
        collection_name=COLLECTION,
        query=vector,
        # metadata filter: only chunks whose payload "folder" is "policies" are searched
        # query_filter=Filter(
        #     must=[FieldCondition(key="folder", match=MatchValue(value=FOLDER))]
        # ),
        limit=K,
        with_payload=True,
    ).points

    for h in hits:
        print(f"retrieved: folder={h.payload.get('folder')} source={h.payload.get('source')} score={h.score:.4f}")

    # joining the context, because k=3, but for llm, only 1 context should be gone na, it concatenates into one
                                  # source added by pdf loader !! 
    return "\n\n---\n\n".join(
        f"[Source: {h.payload.get('source')}, page {h.payload.get('page')}]\n{h.payload['page_content']}"
        for h in hits
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
