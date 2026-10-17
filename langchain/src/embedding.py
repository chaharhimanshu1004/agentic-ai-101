from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
from dotenv import load_dotenv
from chunking import policyChunks, employeeChunks


load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    output_dimensionality=1536,
)

policyVectors = embeddings.embed_documents([c.page_content for c in policyChunks])
employeeVectors = embeddings.embed_documents([c.page_content for c in employeeChunks])

print(f"Generated {len(policyVectors)} policy embeddings, dimension {len(policyVectors[0])}")
print(f"Generated {len(employeeVectors)} employee embeddings, dimension {len(employeeVectors[0])}")
# print(f"policyVectors[0][:5]: {policyVectors[0][:5]}")
# print(f"employeeVectors[0][:5]: {employeeVectors[0][:5]}")