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

chunks = policyChunks + employeeChunks # arr1 + arr2 = arr3, arr1=n size, arr2 = m size
vectors = policyVectors + employeeVectors # arr1 + arr2 = arr3, so arr3[n] = arr2[0]

points = [
    PointStruct(
        id=i,
        vector=vector,
        payload={"page_content": chunk.page_content, **chunk.metadata},
    )
    for i, (chunk, vector) in enumerate(zip(chunks, vectors)) #zip(chunks, vectors) pairs them via posn, hence i=0 -> chunks[0], vectors[0] we get, enumerate adds the counter in starting 
]

client.upsert(collection_name=COLLECTION, points=points)

print(f"Stored {len(points)} points in '{COLLECTION}' (dimension {DIMENSIONS})")
print(f"Collection point count: {client.count(COLLECTION).count}")
