"""
Document Processing Module
Handles PDF and Image processing with OCR capabilities
"""
import io
import os
from typing import List, Union, Optional
from pathlib import Path

from PIL import Image
import pytesseract
from langchain_community.document_loaders import PyPDFLoader, UnstructuredImageLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from pdf2image import convert_from_bytes


class DocumentProcessor:
    """Process documents and images for RAG pipeline"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize document processor
        
        Args:
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def process_pdf(self, file_path: str) -> List[Document]:
        """
        Process PDF document
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            List of Document objects
        """
        try:
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            
            # Add metadata
            for doc in documents:
                doc.metadata['source_type'] = 'pdf'
                doc.metadata['file_name'] = Path(file_path).name
            
            return documents
        except Exception as e:
            raise Exception(f"Error processing PDF: {str(e)}")
    
    def process_pdf_with_ocr(self, file_bytes: bytes, file_name: str) -> List[Document]:
        """
        Process PDF with OCR for scanned documents
        
        Args:
            file_bytes: PDF file bytes
            file_name: Original file name
            
        Returns:
            List of Document objects
        """
        try:
            # Convert PDF to images
            images = convert_from_bytes(file_bytes)
            
            documents = []
            for page_num, image in enumerate(images, start=1):
                # Extract text using OCR
                text = pytesseract.image_to_string(image)
                
                if text.strip():
                    doc = Document(
                        page_content=text,
                        metadata={
                            'source_type': 'pdf_ocr',
                            'file_name': file_name,
                            'page': page_num
                        }
                    )
                    documents.append(doc)
            
            return documents
        except Exception as e:
            raise Exception(f"Error processing PDF with OCR: {str(e)}")
    
    def process_image(self, file_path: str) -> List[Document]:
        """
        Process image file with OCR
        
        Args:
            file_path: Path to image file
            
        Returns:
            List of Document objects
        """
        try:
            # Load and process image
            image = Image.open(file_path)
            
            # Extract text using OCR
            text = pytesseract.image_to_string(image)
            
            if not text.strip():
                text = f"[Image: {Path(file_path).name}. No text detected.]"
            
            doc = Document(
                page_content=text,
                metadata={
                    'source_type': 'image',
                    'file_name': Path(file_path).name,
                    'image_size': image.size
                }
            )
            
            return [doc]
        except Exception as e:
            raise Exception(f"Error processing image: {str(e)}")
    
    def process_image_bytes(self, file_bytes: bytes, file_name: str) -> List[Document]:
        """
        Process image from bytes
        
        Args:
            file_bytes: Image file bytes
            file_name: Original file name
            
        Returns:
            List of Document objects
        """
        try:
            image = Image.open(io.BytesIO(file_bytes))
            
            # Extract text using OCR
            text = pytesseract.image_to_string(image)
            
            if not text.strip():
                text = f"[Image: {file_name}. No text detected.]"
            
            doc = Document(
                page_content=text,
                metadata={
                    'source_type': 'image',
                    'file_name': file_name,
                    'image_size': image.size
                }
            )
            
            return [doc]
        except Exception as e:
            raise Exception(f"Error processing image bytes: {str(e)}")
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks
        
        Args:
            documents: List of documents to split
            
        Returns:
            List of chunked documents
        """
        try:
            chunks = self.text_splitter.split_documents(documents)
            
            # Add chunk metadata
            for i, chunk in enumerate(chunks):
                chunk.metadata['chunk_id'] = i
            
            return chunks
        except Exception as e:
            raise Exception(f"Error splitting documents: {str(e)}")
    
    def process_document(
        self, 
        file_path: Optional[str] = None, 
        file_bytes: Optional[bytes] = None,
        file_name: Optional[str] = None
    ) -> List[Document]:
        """
        Process any document type
        
        Args:
            file_path: Path to file (if reading from disk)
            file_bytes: File bytes (if reading from upload)
            file_name: Original file name
            
        Returns:
            List of processed and chunked documents
        """
        documents = []
        
        try:
            # Determine file type
            if file_path:
                file_ext = Path(file_path).suffix.lower()
                name = Path(file_path).name
            elif file_name:
                file_ext = Path(file_name).suffix.lower()
                name = file_name
            else:
                raise ValueError("Either file_path or file_name must be provided")
            
            # Process based on file type
            if file_ext == '.pdf':
                if file_path:
                    documents = self.process_pdf(file_path)
                elif file_bytes:
                    documents = self.process_pdf_with_ocr(file_bytes, name)
            elif file_ext in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff']:
                if file_path:
                    documents = self.process_image(file_path)
                elif file_bytes:
                    documents = self.process_image_bytes(file_bytes, name)
            else:
                raise ValueError(f"Unsupported file type: {file_ext}")
            
            # Split documents into chunks
            chunked_documents = self.split_documents(documents)
            
            return chunked_documents
            
        except Exception as e:
            raise Exception(f"Error processing document: {str(e)}")