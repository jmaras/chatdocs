"""
Dynamic RAG System - Supports incremental document addition
"""

from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
import pickle
import json

try:
    import faiss
    from sentence_transformers import SentenceTransformer
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    HAS_DEPS = True
except ImportError as e:
    print(f"⚠️ Missing dependencies: {e}")
    HAS_DEPS = False


class DynamicRAG:
    """
    RAG System mit dynamischem Index
    Unterstützt inkrementelles Hinzufügen von Dokumenten
    """
    
    def __init__(self, 
                 embedding_model: str = 'sentence-transformers/all-MiniLM-L6-v2',
                 llm_model: str = 'microsoft/Phi-3-mini-4k-instruct',
                 index_dir: Optional[Path] = None):
        """
        Args:
            embedding_model: Sentence-Transformer Modell
            llm_model: LLM für Answer Generation
            index_dir: Verzeichnis für FAISS Index
        """
        if not HAS_DEPS:
            raise RuntimeError("Required dependencies not installed!")
        
        self.embedding_model_name = embedding_model
        self.llm_model_name = llm_model
        
        # Index directory
        if index_dir is None:
            index_dir = Path(__file__).parent.parent / 'data' / 'index'
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        self.index_file = self.index_dir / 'dynamic_index.faiss'
        self.chunks_file = self.index_dir / 'dynamic_chunks.pkl'
        
        # Models (lazy loading)
        self.embedding_model = None
        self.llm_model = None
        self.llm_tokenizer = None
        
        # FAISS Index
        self.index = None
        self.chunks = []
        
        print(f"🤖 Initialized Dynamic RAG System")
        print(f"   Index Dir: {self.index_dir}")
    
    def load_embedding_model(self):
        """Lädt Embedding-Modell"""
        if self.embedding_model is None:
            print(f"📦 Loading embedding model...")
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            print(f"   ✅ Loaded: {self.embedding_model_name}")
    
    def load_llm(self):
        """Lädt LLM (optimiert für Windows mit bitsandbytes-windows)"""
        if self.llm_model is None:
            print(f"🤖 Loading LLM (this may take a minute)...")
            
            self.llm_tokenizer = AutoTokenizer.from_pretrained(self.llm_model_name)
            
            # Try with 4-bit quantization (should work with bitsandbytes-windows)
            try:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16
                )
                
                self.llm_model = AutoModelForCausalLM.from_pretrained(
                    self.llm_model_name,
                    quantization_config=quantization_config,
                    device_map="auto",
                    torch_dtype=torch.float16
                )
                print(f"   ✅ LLM loaded with 4-bit quantization")
                    
            except Exception as e:
                print(f"   ⚠️  4-bit loading failed: {e}")
                print(f"   ⚠️  Loading on CPU (this will be slower)")
                self.llm_model = AutoModelForCausalLM.from_pretrained(
                    self.llm_model_name,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True
                )
                print(f"   ✅ LLM loaded (CPU mode)")
    
    def load_index(self):
        """Lädt existierenden Index oder erstellt neuen"""
        self.load_embedding_model()
        
        if self.index_file.exists() and self.chunks_file.exists():
            print(f"📖 Loading existing index...")
            self.index = faiss.read_index(str(self.index_file))
            
            with open(self.chunks_file, 'rb') as f:
                self.chunks = pickle.load(f)
            
            print(f"   ✅ Loaded index with {len(self.chunks)} chunks")
        else:
            print(f"📖 Creating new index...")
            dimension = 384  # all-MiniLM-L6-v2 dimension
            self.index = faiss.IndexFlatIP(dimension)
            self.chunks = []
            print(f"   ✅ New index created")
    
    def add_chunks(self, new_chunks: List[Dict]):
        """
        Fügt neue Chunks zum Index hinzu (inkrementell)
        
        Args:
            new_chunks: Liste neuer Chunks
        """
        if not new_chunks:
            print("⚠️ No chunks to add")
            return
        
        if self.index is None:
            self.load_index()
        
        print(f"\n📊 Adding {len(new_chunks)} new chunks to index...")
        
        # Re-ID chunks (kontinuierliche IDs)
        start_id = len(self.chunks)
        for i, chunk in enumerate(new_chunks):
            chunk['id'] = start_id + i
        
        # Generate embeddings
        texts = [c['text'] for c in new_chunks]
        embeddings = self.embedding_model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add to index
        self.index.add(embeddings)
        self.chunks.extend(new_chunks)
        
        print(f"   ✅ Index now has {len(self.chunks)} chunks")
        
        # Save
        self._save_index()
    
    def _save_index(self):
        """Speichert Index und Chunks"""
        faiss.write_index(self.index, str(self.index_file))
        
        with open(self.chunks_file, 'wb') as f:
            pickle.dump(self.chunks, f)
        
        print(f"   💾 Saved index to {self.index_file.name}")
    
    def rebuild_index(self, all_chunks: List[Dict]):
        """
        Baut Index komplett neu auf (alle Chunks)
        
        Args:
            all_chunks: Alle Chunks (frisch)
        """
        print(f"\n🔄 Rebuilding index from scratch...")
        
        if not all_chunks:
            print("⚠️ No chunks provided")
            return
        
        self.load_embedding_model()
        
        # Re-ID
        for i, chunk in enumerate(all_chunks):
            chunk['id'] = i
        
        # Create new index
        dimension = 384
        self.index = faiss.IndexFlatIP(dimension)
        self.chunks = all_chunks
        
        # Generate embeddings
        texts = [c['text'] for c in all_chunks]
        print(f"   Generating embeddings for {len(texts)} chunks...")
        
        embeddings = self.embedding_model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        
        print(f"   ✅ Index rebuilt with {len(self.chunks)} chunks")
        
        # Save
        self._save_index()
    
    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        """
        Retrieval
        
        Args:
            query: Suchquery
            k: Anzahl zu retrievender Chunks
            
        Returns:
            Liste von Chunks mit Scores
        """
        if self.index is None or self.index.ntotal == 0:
            return []
        
        # Query embedding
        query_emb = self.embedding_model.encode([query])
        faiss.normalize_L2(query_emb)
        
        # Search
        n_results = min(k, self.index.ntotal)
        similarities, indices = self.index.search(query_emb, n_results)
        
        # Parse results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx == -1 or idx >= len(self.chunks):
                continue
            
            chunk = self.chunks[idx]
            results.append({
                'id': str(chunk['id']),
                'text': chunk['text'],
                'metadata': chunk['metadata'],
                'score': float(similarities[0][i])
            })
        
        return results
    
    def generate_answer(self, query: str, context_chunks: List[Dict]) -> Dict:
        """Generiert Antwort mit LLM"""
        self.load_llm()
        
        if not context_chunks:
            return {
                'answer': 'Keine relevanten Informationen gefunden. Bitte lade Dokumente hoch.',
                'model': self.llm_model_name,
                'tokens': 0
            }
        
        # Build context
        context_parts = []
        for i, chunk in enumerate(context_chunks, 1):
            context_parts.append(f"[Quelle {i}]\n{chunk['text']}")
        
        context = "\n\n".join(context_parts)
        
        # Build prompt
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Du bist ein hilfreicher Assistent für Dokumenten-basierte Fragen.

Beantworte Fragen basierend auf den bereitgestellten Dokumenten.
Antworte klar, präzise und verständlich auf Deutsch.
Wenn die Antwort nicht in den Dokumenten steht, sage das ehrlich.
<|eot_id|><|start_header_id|>user<|end_header_id|>

Basierend auf folgenden Dokumenten:

{context}

Beantworte die Frage: {query}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        
        # Generate
        inputs = self.llm_tokenizer(prompt, return_tensors="pt").to(self.llm_model.device)
        
        with torch.no_grad():
            outputs = self.llm_model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=0.3,
                do_sample=True,
                pad_token_id=self.llm_tokenizer.eos_token_id
            )
        
        # Decode only the NEW tokens (not the prompt)
        # This is the key fix - only decode what was generated
        new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        answer = self.llm_tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        
        # Simple cleanup
        if not answer or len(answer) < 10:
            # Fallback: decode everything and try to extract
            full_response = self.llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
            if "assistant" in full_response:
                answer = full_response.split("assistant")[-1].strip()
            else:
                answer = full_response
        
        return {
            'answer': answer,
            'model': self.llm_model_name,
            'tokens': len(outputs[0])
        }
    
    def answer_question(self, query: str, k: int = 5) -> Dict:
        """End-to-End: Retrieve + Generate"""
        # Retrieve
        retrieved = self.retrieve(query, k)
        
        # Generate
        result = self.generate_answer(query, retrieved)
        
        # Combine
        result['query'] = query
        result['k'] = k
        result['retrieved_chunks'] = retrieved
        
        return result
    
    def get_statistics(self) -> Dict:
        """Gibt Statistiken über den Index zurück"""
        return {
            'total_chunks': len(self.chunks),
            'index_size': self.index.ntotal if self.index else 0,
            'embedding_model': self.embedding_model_name,
            'llm_model': self.llm_model_name
        }