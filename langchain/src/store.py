import hashlib
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from loaders import load_knowledge
from chunking import chunk_documents
from embedding import embed_chunks

COLLECTION = "langchain_101"
DIMENSIONS = 1536
BATCH_SIZE = 64

client = QdrantClient(url="http://localhost:6333")

if not client.collection_exists(COLLECTION):
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=DIMENSIONS, distance=Distance.COSINE),
    )


def content_id(text: str) -> str:
    # point id = hash of the chunk text (first 128 bits of sha256 as a uuid), so identical
    # content always maps to the same id and can be detected without comparing vectors
    return str(uuid.UUID(hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]))


chunks = chunk_documents(load_knowledge())

# dedupe inside this run too: same text twice -> one point (first occurrence wins)
unique = {}
for chunk in chunks:
    unique.setdefault(content_id(chunk.page_content), chunk)

# ask Qdrant which ids are already stored; only brand-new chunks are embedded and inserted,
# so re-running never rewrites existing points and never pays for embeddings twice
ids = list(unique)
existing = set()
for i in range(0, len(ids), BATCH_SIZE):
    found = client.retrieve(
        collection_name=COLLECTION,
        ids=ids[i : i + BATCH_SIZE],
        with_payload=False,
        with_vectors=False,
    )
    existing.update(str(p.id) for p in found)

new = {pid: chunk for pid, chunk in unique.items() if pid not in existing}
print(f"{len(chunks)} chunks -> {len(unique)} unique, {len(existing)} already stored, {len(new)} new")

if new:
    vectors = embed_chunks(list(new.values()))
    points = [
        PointStruct(
            id=pid,
            vector=vector,
            payload={"page_content": chunk.page_content, **chunk.metadata},
        )
        for (pid, chunk), vector in zip(new.items(), vectors)
    ]
    for i in range(0, len(points), BATCH_SIZE):
        client.upsert(collection_name=COLLECTION, points=points[i : i + BATCH_SIZE])

print(f"Inserted {len(new)} points in '{COLLECTION}' (dimension {DIMENSIONS})")
print(f"Collection point count: {client.count(COLLECTION).count}")