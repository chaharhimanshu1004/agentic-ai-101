import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from faq_loader import load_faq

load_dotenv()

COLLECTION = "support_faq"
K = 1

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    output_dimensionality=1536,
)

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    google_api_key=os.getenv("GEMINI_API_KEY"),
)

client = QdrantClient(url="http://localhost:6333")

if client.collection_exists(COLLECTION):
    vectorstore = QdrantVectorStore(
        client=client, collection_name=COLLECTION, embedding=embeddings
    )
    print(f"Reusing collection '{COLLECTION}' ({client.count(COLLECTION).count} points)")
else:
    docs = load_faq()
    vectorstore = QdrantVectorStore.from_documents( # langchain doing embeddings + storing in the qdrant
        docs,
        embedding=embeddings,
        url="http://localhost:6333",
        collection_name=COLLECTION,
    )
    print(f"Indexed {len(docs)} rows in new collection '{COLLECTION}'")

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": K})

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a customer support assistant. Answer ONLY using the FAQ entries "
            "below. If they do not contain the answer, say you don't have that "
            "information in the FAQ. Be concise and accurate.\n\n"
            "FAQ:\n{context}",
        ),
        ("human", "{question}"),
    ]
)

chain = prompt | llm | StrOutputParser()


def retrieve(query: str) -> str:
    hits = retriever.invoke(query)
    for d in hits:
        print(f"retrieved: content={d.page_content} category={d.metadata.get('category')} row={d.metadata.get('row')}")
    return "\n\n---\n\n".join(d.page_content for d in hits)


if __name__ == "__main__":
    query = input("Ask a support question: ").strip()
    context = retrieve(query)
    print("------------------------------------------------")
    try:
        print(chain.invoke({"context": context, "question": query}))
    except Exception as e:
        print(f"Error while invoking chain: {type(e).__name__}: {e}")
