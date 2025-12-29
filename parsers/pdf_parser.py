"""
PDF Parser - Extracts text from PDFs with basic structure detection
Uses PyMuPDF (fitz) for robust PDF text extraction
"""

import fitz  # PyMuPDF
from pathlib import Path
from typing import Dict, List
import re


class PDFParser:
    """
    Parser für PDF-Dokumente
    Extrahiert Text seitenweise und versucht einfache Struktur zu erkennen
    """
    
    def __init__(self):
        # Regex patterns für Überschriften (heuristic-based)
        self.heading_patterns = [
            r'^[A-ZÄÖÜ][A-ZÄÖÜ\s]{3,}$',  # ALL CAPS (z.B. "KAPITEL 1")
            r'^\d+\.\s+[A-ZÄÖÜ]',          # Nummeriert (z.B. "1. Einführung")
            r'^[A-ZÄÖÜ][a-zäöü\s]{2,}:',   # Titel mit Doppelpunkt
        ]
    
    def parse_document(self, filepath: Path) -> Dict:
        """
        Parst ein PDF-Dokument und extrahiert Text
        
        Args:
            filepath: Pfad zum PDF
            
        Returns:
            dict: Geparste Dokument-Struktur
        """
        doc = fitz.open(str(filepath))
        
        result = {
            'metadata': self._extract_metadata(filepath, doc),
            'chapters': [],
            'paragraphs': []
        }
        
        current_chapter = None
        para_id = 0
        
        # Iteriere durch alle Seiten
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")
            
            # Split in Absätze (doppelte Zeilenumbrüche)
            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
            
            for para_text in paragraphs:
                # Skip sehr kurze Paragraphen (wahrscheinlich Seitenzahlen, etc.)
                if len(para_text) < 20:
                    continue
                
                # Versuche Überschrift zu erkennen
                is_heading = self._is_heading(para_text)
                
                if is_heading:
                    # Erstelle neues Kapitel
                    current_chapter = {
                        'title': para_text,
                        'level': 1,
                        'page': page_num + 1,
                        'sections': []
                    }
                    result['chapters'].append(current_chapter)
                else:
                    # Normaler Paragraph
                    para_obj = {
                        'id': para_id,
                        'text': para_text,
                        'chapter': current_chapter['title'] if current_chapter else None,
                        'section': None,  # PDFs haben selten klare Sections
                        'page': page_num + 1,
                        'word_count': len(para_text.split())
                    }
                    
                    result['paragraphs'].append(para_obj)
                    para_id += 1
        
        doc.close()
        return result
    
    def _is_heading(self, text: str) -> bool:
        """
        Einfache Heuristik zur Überschrift-Erkennung
        
        Args:
            text: Text-Zeile
            
        Returns:
            bool: True wenn vermutlich Überschrift
        """
        # Zu lang für Überschrift
        if len(text) > 100:
            return False
        
        # Zu kurz
        if len(text) < 5:
            return False
        
        # Check patterns
        for pattern in self.heading_patterns:
            if re.match(pattern, text):
                return True
        
        return False
    
    def _extract_metadata(self, filepath: Path, doc: fitz.Document) -> Dict:
        """
        Extrahiert Metadaten aus PDF
        
        Args:
            filepath: Pfad zum Dokument
            doc: PyMuPDF Document Objekt
            
        Returns:
            dict: Metadaten
        """
        filename = filepath.stem
        
        # PDF Metadata (falls vorhanden)
        pdf_meta = doc.metadata
        
        return {
            'filename': filename,
            'filepath': str(filepath),
            'num_pages': len(doc),
            'title': pdf_meta.get('title', filename),
            'author': pdf_meta.get('author', 'Unknown'),
            'subject': pdf_meta.get('subject', ''),
        }
    
    def parse_simple(self, filepath: Path) -> List[Dict]:
        """
        Vereinfachtes Parsing: Nur Paragraphen ohne Struktur-Erkennung
        Schneller und robuster für strukturlose PDFs
        
        Args:
            filepath: Pfad zum PDF
            
        Returns:
            list: Liste von Paragraphen
        """
        doc = fitz.open(str(filepath))
        paragraphs = []
        para_id = 0
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")
            
            # Split in Absätze
            page_paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
            
            for para_text in page_paragraphs:
                if len(para_text) < 20:
                    continue
                
                paragraphs.append({
                    'id': para_id,
                    'text': para_text,
                    'chapter': None,
                    'section': None,
                    'page': page_num + 1,
                    'word_count': len(para_text.split())
                })
                para_id += 1
        
        doc.close()
        return paragraphs
    
    def get_statistics(self, parsed_doc: Dict) -> Dict:
        """
        Gibt Statistiken über das geparste PDF zurück
        
        Args:
            parsed_doc: Geparste Dokument-Struktur
            
        Returns:
            dict: Statistiken
        """
        return {
            'filename': parsed_doc['metadata']['filename'],
            'num_pages': parsed_doc['metadata']['num_pages'],
            'num_chapters': len(parsed_doc['chapters']),
            'num_paragraphs': len(parsed_doc['paragraphs']),
            'avg_words_per_para': sum(p['word_count'] for p in parsed_doc['paragraphs']) / len(parsed_doc['paragraphs']) if parsed_doc['paragraphs'] else 0
        }
