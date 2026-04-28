"""
PDF Document Parser
Extracts text content from PDF files for RAG indexing
"""

from pypdf import PdfReader
from pathlib import Path
from typing import Optional


class PDFParser:
    """Parse PDF documents and extract text content"""
    
    def __init__(self):
        """Initialize PDF parser"""
        pass
    
    def parse(self, pdf_path: str) -> Optional[str]:
        """
        Extract text from PDF file
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text content or None if parsing fails
        """
        try:
            pdf_file = Path(pdf_path)
            if not pdf_file.exists():
                print(f"PDF file not found: {pdf_path}")
                return None
            
            # Read PDF
            reader = PdfReader(pdf_path)
            
            # Extract text from all pages
            text_content = []
            for page in reader.pages:
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text_content.append(page_text)
                except Exception as e:
                    print(f"Error extracting text from page: {e}")
                    continue
            
            # Combine all pages
            full_text = "\n\n".join(text_content)
            
            if not full_text.strip():
                print(f"No text extracted from PDF: {pdf_path}")
                return None
            
            print(f"Extracted {len(full_text)} characters from {pdf_file.name}")
            return full_text
            
        except Exception as e:
            print(f"Error parsing PDF {pdf_path}: {e}")
            return None
    
    def parse_with_metadata(self, pdf_path: str) -> Optional[dict]:
        """
        Extract text and metadata from PDF file
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dictionary with 'text' and 'metadata' or None if parsing fails
        """
        try:
            pdf_file = Path(pdf_path)
            if not pdf_file.exists():
                print(f"PDF file not found: {pdf_path}")
                return None
            
            # Read PDF
            reader = PdfReader(pdf_file)
            
            # Extract metadata
            metadata = {
                'source': pdf_file.name,
                'path': str(pdf_file.absolute()),
                'num_pages': len(reader.pages),
                'file_type': 'pdf'
            }
            
            # Add PDF metadata if available
            if reader.metadata:
                if reader.metadata.title:
                    metadata['title'] = reader.metadata.title
                if reader.metadata.author:
                    metadata['author'] = reader.metadata.author
                if reader.metadata.creator:
                    metadata['creator'] = reader.metadata.creator
            
            # Extract text from all pages
            text_content = []
            for page in reader.pages:
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text_content.append(page_text)
                except Exception as e:
                    print(f"Error extracting text from page: {e}")
                    continue
            
            # Combine all pages
            full_text = "\n\n".join(text_content)
            
            if not full_text.strip():
                print(f"No text extracted from PDF: {pdf_path}")
                return None
            
            print(f"Extracted {len(full_text)} characters from {pdf_file.name} ({metadata['num_pages']} pages)")
            
            return {
                'text': full_text,
                'metadata': metadata
            }
            
        except Exception as e:
            print(f"Error parsing PDF {pdf_path}: {e}")
            return None
