"""Stage A: Link Ingestion & Extraction.

``TopicDocumentDownloader`` fetches URLs to disk. ``TopicDocumentIterator`` reads
downloaded files and yields raw records. ``TopicDocumentExtractor`` converts
HTML (jusText) / PDF (pypdf) into clean ``Document`` objects.
"""

from .downloader import DownloadResult, TopicDocumentDownloader
from .extractor import TopicDocumentExtractor, extract_document
from .iterator import TopicDocumentIterator

__all__ = [
    "DownloadResult",
    "TopicDocumentDownloader",
    "TopicDocumentExtractor",
    "TopicDocumentIterator",
    "extract_document",
]
