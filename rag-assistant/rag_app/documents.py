"""Text and PDF ingestion with citation-friendly metadata."""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Iterable

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def load_documents(files: Iterable[tuple[str, bytes]]) -> list[Document]:
    """Read uploaded PDF/TXT bytes and retain file and page citations."""
    documents: list[Document] = []
    for filename, content in files:
        extension = Path(filename).suffix.lower()
        source = Path(filename).name
        if extension == ".txt":
            try:
                text = content.decode("utf-8-sig").strip()
            except UnicodeDecodeError as error:
                raise ValueError(f"{source} must use UTF-8 text encoding") from error
            if text:
                documents.append(
                    Document(page_content=text, metadata={"source": source, "page": 1})
                )
        elif extension == ".pdf":
            try:
                from pypdf import PdfReader

                reader = PdfReader(io.BytesIO(content))
                for page_number, page in enumerate(reader.pages, start=1):
                    text = (page.extract_text() or "").strip()
                    if text:
                        documents.append(
                            Document(
                                page_content=text,
                                metadata={"source": source, "page": page_number},
                            )
                        )
            except Exception as error:
                logger.exception("Could not parse uploaded PDF %s", source)
                raise ValueError(f"Could not read PDF {source}") from error
        else:
            raise ValueError(f"Unsupported file type for {source}; upload PDF or TXT")

    if not documents:
        raise ValueError("No readable text was found in the uploaded documents")
    return documents