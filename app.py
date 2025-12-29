"""
Flask Web Interface - Document Manager + RAG Chat
3 Tabs: Chat, Upload, Library
"""

from flask import Flask, render_template, request, jsonify, session
from werkzeug.utils import secure_filename
from pathlib import Path
import secrets
from datetime import datetime
import sys
import os

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from parsers.docx_parser import DocxParser
from parsers.pdf_parser import PDFParser
from chunking.flat_chunker import FlatChunker
from core.document_manager import DocumentManager
from core.rag_system import DynamicRAG

app = Flask(__name__)
app.secret_key = secrets.token_hex(32)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max

# Initialize components
data_dir = project_root / 'data'
doc_manager = DocumentManager(data_dir)
rag_system = DynamicRAG(index_dir=data_dir / 'index')

# Parsers and Chunker (lazy init)
docx_parser = None
pdf_parser = None
chunker = None

ALLOWED_EXTENSIONS = {'pdf', 'docx', 'doc'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_parsers_and_chunker():
    """Lazy initialization of parsers"""
    global docx_parser, pdf_parser, chunker
    if docx_parser is None:
        docx_parser = DocxParser()
    if pdf_parser is None:
        pdf_parser = PDFParser()
    if chunker is None:
        chunker = FlatChunker()
    return docx_parser, pdf_parser, chunker


@app.route('/')
def index():
    """Main interface with 3 tabs"""
    return render_template('index.html')


@app.route('/api/chat', methods=['POST'])
def chat():
    """
    Handle chat request
    
    Expected JSON:
    {
        "query": "User question",
        "k": 5
    }
    """
    try:
        data = request.json
        query = data.get('query', '').strip()
        k = int(data.get('k', 5))
        
        if not query:
            return jsonify({'error': 'Query cannot be empty'}), 400
        
        if not (1 <= k <= 10):
            return jsonify({'error': 'k must be between 1 and 10'}), 400
        
        # Load index if needed
        if rag_system.index is None:
            rag_system.load_index()
        
        # Check if index has chunks
        if rag_system.index.ntotal == 0:
            return jsonify({
                'success': True,
                'answer': 'Noch keine Dokumente hochgeladen. Bitte lade zuerst Dokumente im Upload-Tab hoch.',
                'chunks': [],
                'metadata': {'k': k, 'model': rag_system.llm_model_name, 'tokens': 0}
            })
        
        print(f"\n💬 Query: {query}")
        print(f"   k: {k}")
        
        result = rag_system.answer_question(query, k=k)
        
        # Format response
        response = {
            'success': True,
            'answer': result['answer'],
            'chunks': [
                {
                    'text': chunk['text'],
                    'score': round(chunk['score'], 3),
                    'metadata': {
                        'doc': chunk['metadata'].get('doc_filename', 'Unknown'),
                        'chapter': chunk['metadata'].get('chapter', ''),
                        'page': chunk['metadata'].get('page', '')
                    }
                }
                for chunk in result['retrieved_chunks']
            ],
            'metadata': {
                'k': result['k'],
                'model': result['model'],
                'tokens': result['tokens']
            }
        }
        
        print(f"✅ Answer generated ({result['tokens']} tokens)\n")
        
        return jsonify(response)
    
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/upload', methods=['POST'])
def upload_document():
    """
    Handle document upload
    
    Expected: multipart/form-data with 'file'
    """
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Only PDF and DOCX allowed.'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        upload_path = data_dir / 'uploads' / 'temp' / filename
        upload_path.parent.mkdir(parents=True, exist_ok=True)
        file.save(str(upload_path))
        
        print(f"\n📤 Processing upload: {filename}")
        
        # Get parsers
        docx_p, pdf_p, chunk = get_parsers_and_chunker()
        
        # Determine file type
        file_ext = filename.rsplit('.', 1)[1].lower()
        
        # Parse document
        print(f"   📖 Parsing...")
        if file_ext in ['docx', 'doc']:
            parsed = docx_p.parse_document(upload_path)
            file_type = 'docx'
        else:  # pdf
            parsed = pdf_p.parse_document(upload_path)
            file_type = 'pdf'
        
        print(f"   ✅ Parsed: {len(parsed['paragraphs'])} paragraphs")
        
        # Chunk document
        print(f"   ✂️ Chunking...")
        chunks = chunk.chunk_documents([parsed])
        print(f"   ✅ Created: {len(chunks)} chunks")
        
        # Add to document manager
        doc_id = doc_manager.add_document(
            upload_path,
            file_type,
            parsed,
            chunks
        )
        
        # Add to RAG index
        print(f"   🔢 Adding to index...")
        rag_system.add_chunks(chunks)
        
        # Clean up temp file
        upload_path.unlink()
        
        print(f"✅ Upload complete: {filename} (ID: {doc_id})\n")
        
        return jsonify({
            'success': True,
            'doc_id': doc_id,
            'filename': filename,
            'file_type': file_type,
            'num_paragraphs': len(parsed['paragraphs']),
            'num_chunks': len(chunks)
        })
    
    except Exception as e:
        print(f"❌ Upload error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/documents', methods=['GET'])
def list_documents():
    """List all documents"""
    try:
        docs = doc_manager.list_documents()
        stats = doc_manager.get_statistics()
        
        return jsonify({
            'success': True,
            'documents': docs,
            'statistics': stats
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/documents/<doc_id>', methods=['DELETE'])
def delete_document(doc_id):
    """Delete a document"""
    try:
        success = doc_manager.delete_document(doc_id)
        
        if success:
            # Rebuild index without this document
            print(f"\n🔄 Rebuilding index after deletion...")
            all_chunks = doc_manager.get_all_chunks()
            rag_system.rebuild_index(all_chunks)
            
            return jsonify({'success': True})
        else:
            return jsonify({'error': 'Document not found'}), 404
    
    except Exception as e:
        print(f"❌ Delete error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/reindex', methods=['POST'])
def reindex_all():
    """Rebuild index from all documents"""
    try:
        print(f"\n🔄 Reindexing all documents...")
        all_chunks = doc_manager.get_all_chunks()
        rag_system.rebuild_index(all_chunks)
        
        stats = rag_system.get_statistics()
        
        return jsonify({
            'success': True,
            'message': 'Index rebuilt successfully',
            'statistics': stats
        })
    except Exception as e:
        print(f"❌ Reindex error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/statistics', methods=['GET'])
def get_statistics():
    """Get system statistics"""
    try:
        doc_stats = doc_manager.get_statistics()
        rag_stats = rag_system.get_statistics()
        
        return jsonify({
            'success': True,
            'documents': doc_stats,
            'index': rag_stats
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("\n" + "="*70)
    print("💬 CHATDOCS")
    print("="*70)
    print("\n📝 Features:")
    print("   ✅ Upload PDF & DOCX documents")
    print("   ✅ Automatic parsing & chunking")
    print("   ✅ Incremental index updates")
    print("   ✅ Chat with your documents")
    print("   ✅ Document library management")
    
    # Load existing index
    print("\n🔧 Initializing...")
    try:
        rag_system.load_index()
        stats = rag_system.get_statistics()
        if stats['total_chunks'] > 0:
            print(f"   ✅ Index loaded: {stats['total_chunks']} chunks")
        else:
            print(f"   📭 Index empty - upload documents to get started!")
    except Exception as e:
        print(f"   📭 No existing index - will create on first upload")
    
    print("\n🌐 Starting Flask server...\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
