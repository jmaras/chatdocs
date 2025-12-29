"""
Document Manager - Handles document uploads, processing, and management
"""

from pathlib import Path
from typing import List, Dict, Optional
import json
import shutil
from datetime import datetime
import hashlib


class DocumentManager:
    """
    Verwaltet hochgeladene Dokumente und deren Metadaten
    """
    
    def __init__(self, data_dir: Path):
        """
        Args:
            data_dir: Root data directory
        """
        self.data_dir = data_dir
        self.uploads_dir = data_dir / 'uploads'
        self.processed_dir = data_dir / 'processed'
        
        # Ensure directories exist
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Metadata file
        self.metadata_file = data_dir / 'documents_metadata.json'
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict:
        """Lädt Dokument-Metadaten"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {'documents': []}
    
    def _save_metadata(self):
        """Speichert Dokument-Metadaten"""
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
    
    def _generate_doc_id(self, filename: str) -> str:
        """Generiert eindeutige ID für Dokument"""
        timestamp = datetime.now().isoformat()
        content = f"{filename}_{timestamp}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def add_document(self, 
                    file_path: Path, 
                    file_type: str,
                    parsed_data: Dict,
                    chunks: List[Dict]) -> str:
        """
        Fügt ein neues Dokument hinzu
        
        Args:
            file_path: Pfad zum Original-Dokument
            file_type: 'pdf' oder 'docx'
            parsed_data: Geparste Dokument-Struktur
            chunks: Generierte Chunks
            
        Returns:
            str: Dokument-ID
        """
        doc_id = self._generate_doc_id(file_path.name)
        
        # Copy original file
        dest_path = self.uploads_dir / f"{doc_id}_{file_path.name}"
        shutil.copy2(file_path, dest_path)
        
        # Save parsed data
        parsed_file = self.processed_dir / f"{doc_id}_parsed.json"
        with open(parsed_file, 'w', encoding='utf-8') as f:
            json.dump(parsed_data, f, ensure_ascii=False, indent=2)
        
        # Save chunks
        chunks_file = self.processed_dir / f"{doc_id}_chunks.json"
        with open(chunks_file, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        
        # Add metadata
        doc_meta = {
            'doc_id': doc_id,
            'filename': file_path.name,
            'file_type': file_type,
            'upload_date': datetime.now().isoformat(),
            'num_paragraphs': len(parsed_data.get('paragraphs', [])),
            'num_chunks': len(chunks),
            'num_chapters': len(parsed_data.get('chapters', [])),
            'file_path': str(dest_path),
            'parsed_path': str(parsed_file),
            'chunks_path': str(chunks_file)
        }
        
        self.metadata['documents'].append(doc_meta)
        self._save_metadata()
        
        print(f"✅ Added document: {file_path.name} (ID: {doc_id})")
        return doc_id
    
    def get_document(self, doc_id: str) -> Optional[Dict]:
        """Holt Metadaten eines Dokuments"""
        for doc in self.metadata['documents']:
            if doc['doc_id'] == doc_id:
                return doc
        return None
    
    def list_documents(self) -> List[Dict]:
        """Listet alle Dokumente"""
        return self.metadata['documents']
    
    def delete_document(self, doc_id: str) -> bool:
        """
        Löscht ein Dokument
        
        Args:
            doc_id: Dokument-ID
            
        Returns:
            bool: True wenn erfolgreich gelöscht
        """
        doc = self.get_document(doc_id)
        if not doc:
            return False
        
        # Delete files
        try:
            Path(doc['file_path']).unlink(missing_ok=True)
            Path(doc['parsed_path']).unlink(missing_ok=True)
            Path(doc['chunks_path']).unlink(missing_ok=True)
        except Exception as e:
            print(f"⚠️ Error deleting files: {e}")
        
        # Remove from metadata
        self.metadata['documents'] = [
            d for d in self.metadata['documents'] 
            if d['doc_id'] != doc_id
        ]
        self._save_metadata()
        
        print(f"✅ Deleted document: {doc_id}")
        return True
    
    def get_all_chunks(self) -> List[Dict]:
        """
        Holt alle Chunks von allen Dokumenten
        
        Returns:
            Liste aller Chunks
        """
        all_chunks = []
        
        for doc in self.metadata['documents']:
            chunks_path = Path(doc['chunks_path'])
            if chunks_path.exists():
                with open(chunks_path, 'r', encoding='utf-8') as f:
                    chunks = json.load(f)
                    all_chunks.extend(chunks)
        
        return all_chunks
    
    def get_statistics(self) -> Dict:
        """
        Gibt Statistiken über alle Dokumente zurück
        
        Returns:
            dict: Statistiken
        """
        if not self.metadata['documents']:
            return {
                'num_documents': 0,
                'total_chunks': 0,
                'total_paragraphs': 0
            }
        
        return {
            'num_documents': len(self.metadata['documents']),
            'total_chunks': sum(d['num_chunks'] for d in self.metadata['documents']),
            'total_paragraphs': sum(d['num_paragraphs'] for d in self.metadata['documents']),
            'by_type': {
                'pdf': len([d for d in self.metadata['documents'] if d['file_type'] == 'pdf']),
                'docx': len([d for d in self.metadata['documents'] if d['file_type'] == 'docx'])
            }
        }
    
    def clear_all(self):
        """Löscht alle Dokumente (Vorsicht!)"""
        for doc in self.metadata['documents']:
            try:
                Path(doc['file_path']).unlink(missing_ok=True)
                Path(doc['parsed_path']).unlink(missing_ok=True)
                Path(doc['chunks_path']).unlink(missing_ok=True)
            except Exception as e:
                print(f"⚠️ Error deleting {doc['doc_id']}: {e}")
        
        self.metadata = {'documents': []}
        self._save_metadata()
        print("✅ Cleared all documents")
