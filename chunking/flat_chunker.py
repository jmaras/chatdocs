"""
Flat Chunking Strategy: Sliding Window über komplette Dokumente
Token-basiert mit Aggregation über Paragraphen
"""

from typing import List, Dict
import json
from pathlib import Path
from transformers import AutoTokenizer


class FlatChunker:
    """
    Flat Chunking Strategy: Sliding Window über komplette Dokumente
    
    Verwendet TOKEN-basiertes Chunking mit Sliding Window.
    Aggregiert ALLE Paragraphen eines Dokuments und erstellt dann
    überlappende Chunks mit fester Größe.
    
    Dies ermöglicht:
    - Konsistente Chunk-Größe (~400-500 tokens)
    - Kontext über Paragraph-Grenzen hinweg
    - Vergleichbarkeit mit Hierarchical Chunking
    """
    
    def __init__(self, 
                 chunk_size: int = 512,
                 chunk_overlap: int = 50,
                 model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        """
        Args:
            chunk_size: Max. Anzahl TOKENS pro Chunk
            chunk_overlap: Anzahl überlappender TOKENS zwischen Chunks
            model_name: Name des Embedding-Modells (für Tokenizer)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        
        # Tokenizer laden
        print(f"   Loading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"   ✓ Tokenizer loaded")
    
    def _char_based_split(self, text: str, tokens: List[int]) -> List[str]:
        """
        Teilt Text basierend auf Token-Positionen, erhält aber Original-Text
        
        Args:
            text: Original-Text
            tokens: Token-IDs
            
        Returns:
            Liste von Text-Chunks die den Token-Grenzen entsprechen
        """
        # Tokenize mit return_offsets_mapping um Positionen zu bekommen
        encoding = self.tokenizer(
            text, 
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=False
        )
        
        offsets = encoding['offset_mapping']
        chunk_texts = []
        
        step_size = self.chunk_size - self.chunk_overlap
        
        for i in range(0, len(tokens), step_size):
            start_idx = i
            end_idx = min(i + self.chunk_size, len(tokens))
            
            # Skip zu kleine Chunks
            if end_idx - start_idx < 50:
                continue
            
            # Hole Char-Positionen für diese Token-Range
            if start_idx < len(offsets) and end_idx <= len(offsets):
                char_start = offsets[start_idx][0]
                char_end = offsets[end_idx - 1][1]
                
                # Extrahiere Text direkt aus Original
                chunk_text = text[char_start:char_end]
                chunk_texts.append(chunk_text)
        
        return chunk_texts
    
    def chunk_documents(self, parsed_docs: List[Dict]) -> List[Dict]:
        """
        Erstellt Flat Chunks aus geparsten Dokumenten
        
        Strategy:
        1. Sammle ALLE Paragraphen eines Dokuments
        2. Tokenize kompletten Text
        3. Sliding Window über alles (chunk_size mit overlap)
        4. Extrahiere Text-Chunks DIREKT aus Original (erhält Umlaute/Groß)
        
        Args:
            parsed_docs: Liste von geparsten Dokumenten (aus docx_parser.py)
            
        Returns:
            Liste von Chunk-Dictionaries mit Text und Metadata
        """
        all_chunks = []
        chunk_id = 0
        
        for doc in parsed_docs:
            doc_filename = doc['metadata']['filename']
            paragraphs = doc['paragraphs']
            
            print(f"  Processing {doc_filename}: {len(paragraphs)} paragraphs")
            
            # ✨ SAMMLE ALLE Paragraphen zu einem String
            # (mit Leerzeichen zwischen Paragraphen)
            all_text = ' '.join([p['text'] for p in paragraphs if p['text'].strip()])
            
            if not all_text.strip():
                print(f"    ⚠️  No text found in {doc_filename}")
                continue
            
            # Tokenize kompletten Dokument-Text (nur für Counting)
            tokens = self.tokenizer.encode(all_text, add_special_tokens=False)
            
            print(f"    Total tokens: {len(tokens):,}")
            
            # Verwende char-based splitting um Original-Text zu erhalten
            step_size = self.chunk_size - self.chunk_overlap
            num_chunks_expected = max(1, (len(tokens) - self.chunk_overlap) // step_size)
            
            print(f"    Creating ~{num_chunks_expected} chunks (step={step_size})...")
            
            # Hole Text-Chunks die Umlaute/Großbuchstaben erhalten
            chunk_texts = self._char_based_split(all_text, tokens)
            
            for chunk_text in chunk_texts:
                # Zähle Tokens für diesen Chunk
                chunk_tokens = self.tokenizer.encode(chunk_text, add_special_tokens=False)
                
                # Optional: Finde welches Kapitel dieser Chunk überlappt
                chapter = None
                section = None
                
                # Einfache Heuristik: Nimm erstes Wort und suche in Paragraphen
                first_words = chunk_text.split()[:5]
                if first_words:
                    search_text = ' '.join(first_words)
                    for para in paragraphs:
                        if search_text in para['text']:
                            chapter = para.get('chapter')
                            section = para.get('section')
                            break
                
                chunk = {
                    'id': chunk_id,
                    'text': chunk_text,
                    'metadata': {
                        'doc_filename': doc_filename,
                        'chapter': chapter,
                        'section': section,
                        'chunking_strategy': 'flat',
                        'token_count': len(chunk_tokens),
                        'is_split': False  # Bei Flat ist immer alles "gesplittet"
                    }
                }
                
                all_chunks.append(chunk)
                chunk_id += 1
        
        print(f"\n  ✓ Created {len(all_chunks)} flat chunks")
        return all_chunks
    
    def save_chunks(self, chunks: List[Dict], output_file: str):
        """
        Speichert Chunks als JSON
        
        Args:
            chunks: Liste von Chunks
            output_file: Pfad zur Output-Datei
        """
        output_path = Path(output_file)
        
        # Erstelle Verzeichnis falls nötig
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(chunks, f, ensure_ascii=False, indent=2)
            
            print(f"  ✅ Saved {len(chunks)} flat chunks")
            
        except Exception as e:
            print(f"  ❌ Error saving chunks: {e}")
            raise
    
    def get_statistics(self, chunks: List[Dict]) -> Dict:
        """
        Berechnet Statistiken über die Chunks
        
        Returns:
            Dict mit Statistiken
        """
        if not chunks:
            return {}
        
        token_counts = [c['metadata']['token_count'] for c in chunks]
        
        return {
            'num_chunks': len(chunks),
            'avg_tokens': sum(token_counts) / len(token_counts),
            'min_tokens': min(token_counts),
            'max_tokens': max(token_counts),
            'num_split_chunks': 0  # Bei Flat nicht relevant (alles ist sliding window)
        }