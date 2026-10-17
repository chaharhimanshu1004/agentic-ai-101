import json
from collections import Counter

from langchain_core.documents import Document
from langchain_text_splitters import (
    Language,
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
    RecursiveJsonSplitter,
)

CHUNK_SIZE = 900
CHUNK_OVERLAP = 120

textSplitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    add_start_index=True,
)
codeSplitter = RecursiveCharacterTextSplitter.from_language(
    Language.PYTHON,
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    add_start_index=True,
)
markdownSplitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
    strip_headers=True,
)
jsonSplitter = RecursiveJsonSplitter(max_chunk_size=CHUNK_SIZE)


def chunk_text(doc: Document) -> list[Document]:
    return textSplitter.split_documents([doc])


def chunk_code(doc: Document) -> list[Document]:
    return codeSplitter.split_documents([doc])


def chunk_markdown(doc: Document) -> list[Document]:
    # split on headings first, then by size; prefix every chunk with its heading trail
    chunks = []
    for section in markdownSplitter.split_text(doc.page_content):
        trail = " > ".join(section.metadata[k] for k in ("h1", "h2", "h3") if k in section.metadata)
        section.metadata = {**doc.metadata, **section.metadata}
        for chunk in textSplitter.split_documents([section]):
            if trail:
                chunk.page_content = f"{trail}\n{chunk.page_content}"
            chunks.append(chunk)
    return chunks


def chunk_json(doc: Document) -> list[Document]:
    return jsonSplitter.create_documents(
        texts=[json.loads(doc.page_content)],
        convert_lists=True,
        ensure_ascii=False,
        metadatas=[doc.metadata],
    )


def keep_whole(doc: Document) -> list[Document]:
    # csv rows are already one small record each
    return [doc]


CHUNKERS = {
    "pdf": chunk_text,
    "docx": chunk_text,
    "md": chunk_markdown,
    "html": chunk_markdown,  # html is converted to markdown by the loader
    "json": chunk_json,
    "py": chunk_code,
    "csv": keep_whole,
}


def chunk_documents(docs: list[Document]) -> list[Document]:
    chunks = []
    for doc in docs:
        chunks.extend(CHUNKERS[doc.metadata["doc_type"]](doc)) # just like arr.push_back
    counts = Counter(c.metadata["doc_type"] for c in chunks)
    print(f"Split {len(docs)} documents into {len(chunks)} chunks: {dict(counts)}")
    return chunks
