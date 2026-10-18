from pathlib import Path

from langchain_community.document_loaders import CSVLoader
from langchain_core.documents import Document

FAQ_PATH = Path(__file__).resolve().parent.parent / "knowledge" / "faq" / "support_faq.csv"


def load_faq(path: Path = FAQ_PATH) -> list[Document]:
    # one Document per CSV row; product/category go to metadata so they stay filterable
    docs = CSVLoader(
        str(path),
        encoding="utf-8",
        metadata_columns=["product", "category"], # otherwsie via csv loader, in metadata we only get {source, row}
                        # product, category comes from csv individual row, if not present -> '' stored as empty string
    ).load()
    for doc in docs:
        doc.metadata["source"] = path.name
    print(f"Loaded {len(docs)} FAQ rows from {path.name}")
    return docs
