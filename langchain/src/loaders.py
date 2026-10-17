import base64
import os
from pathlib import Path

import pymupdf
from bs4 import BeautifulSoup
from docx import Document as DocxDocument
from docx.table import Table
from docx.text.paragraph import Paragraph
from dotenv import load_dotenv
from langchain_community.document_loaders import CSVLoader, PyPDFLoader
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

KNOWLEDGE_DIR = Path(__file__).resolve().parent.parent / "knowledge"
OCR_MODEL = "gemini-3.5-flash-lite"
OCR_PROMPT = (
    "Transcribe all the text on this scanned page exactly as written. "
    "Render tables as markdown tables. Output only the transcribed text."
)


def ocr_pdf_page(pdf: pymupdf.Document, page_number: int) -> str:
    # scanned pages have no text layer, so render the page and let Gemini read it
    png = pdf[page_number].get_pixmap(matrix=pymupdf.Matrix(2, 2)).tobytes("png")
    image = base64.b64encode(png).decode()
    message = HumanMessage(
        content=[
            {"type": "text", "text": OCR_PROMPT},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image}"}},
        ]
    )
    ocr = ChatGoogleGenerativeAI(
        model=OCR_MODEL, google_api_key=os.getenv("GEMINI_API_KEY")
    )
    return (ocr | StrOutputParser()).invoke([message])


def load_pdf(path: Path) -> list[Document]:
    pages = PyPDFLoader(str(path)).load()
    with pymupdf.open(path) as pdf:
        for page in pages:
            if not page.page_content.strip():
                page.page_content = ocr_pdf_page(pdf, page.metadata["page"])
                page.metadata["ocr"] = True
    return [p for p in pages if p.page_content.strip()]


def load_docx(path: Path) -> list[Document]:
    doc = DocxDocument(str(path))
    parts = []
    # walk the body in order so paragraphs and tables keep their position
    for child in doc.element.body.iterchildren():
        if child.tag.endswith("}p"):
            parts.append(Paragraph(child, doc).text)
        elif child.tag.endswith("}tbl"):
            for row in Table(child, doc).rows:
                parts.append(" | ".join(cell.text.strip() for cell in row.cells))
    return [Document(page_content="\n".join(p for p in parts if p.strip()))]


def html_to_markdown(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    lines = []
    for el in soup.find_all(["h1", "h2", "h3", "h4", "p", "li", "tr"]):
        if el.name == "tr":
            lines.append(" | ".join(c.get_text(" ", strip=True) for c in el.find_all(["th", "td"])))
        elif el.name == "li":
            lines.append("- " + el.get_text(" ", strip=True))
        elif el.name.startswith("h"):
            lines.append("#" * int(el.name[1]) + " " + el.get_text(" ", strip=True))
        else:
            lines.append(el.get_text(" ", strip=True))
    return "\n\n".join(line for line in lines if line.strip())


def load_html(path: Path) -> list[Document]:
    return [Document(page_content=html_to_markdown(path.read_text(encoding="utf-8")))]


def load_csv(path: Path) -> list[Document]:
    return CSVLoader(str(path), encoding="utf-8").load()


def load_text(path: Path) -> list[Document]:
    return [Document(page_content=path.read_text(encoding="utf-8"))]


LOADERS = {
    ".pdf": load_pdf,
    ".docx": load_docx,
    ".html": load_html,
    ".csv": load_csv,
    ".md": load_text,
    ".json": load_text,
    ".py": load_text,
}


def load_knowledge(root: Path = KNOWLEDGE_DIR) -> list[Document]:
    docs = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name.startswith("."):
            continue
        source = str(path.relative_to(root))
        loader = LOADERS.get(path.suffix.lower())
        if loader is None:
            print(f"Skipping unsupported file: {source}")
            continue
        loaded = loader(path)
        for doc in loaded:
            doc.metadata.update(
                source=source,
                doc_type=path.suffix.lower().lstrip("."),
                folder=path.parent.name,
            )
        docs.extend(loaded) # just like arr.push_back
        print(f"Loaded {len(loaded)} document(s) from {source}")
    return docs
