from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from chunking import policyChunks, employeeChunks
from embedding import policyVectors, employeeVectors

COLLECTION = "langchain_hw"
DIMENSIONS = 1536

client = QdrantClient(url="http://localhost:6333")

if not client.collection_exists(COLLECTION):
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=DIMENSIONS, distance=Distance.COSINE),
    )

chunks = policyChunks + employeeChunks
vectors = policyVectors + employeeVectors

points = [
    PointStruct(
        id=i,
        vector=vector,
        payload={"page_content": chunk.page_content, **chunk.metadata},
    )
    for i, (chunk, vector) in enumerate(zip(chunks, vectors))
]

client.upsert(collection_name=COLLECTION, points=points)

print(f"Stored {len(points)} points in '{COLLECTION}' (dimension {DIMENSIONS})")
print(f"Collection point count: {client.count(COLLECTION).count}")
