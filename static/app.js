// RAG Document Manager - Frontend Logic

// ===== TAB MANAGEMENT =====
const tabBtns = document.querySelectorAll('.tab-btn');
const tabPanes = document.querySelectorAll('.tab-pane');

tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
        const tabName = btn.dataset.tab;
        
        // Update active states
        tabBtns.forEach(b => b.classList.remove('active'));
        tabPanes.forEach(p => p.classList.remove('active'));
        
        btn.classList.add('active');
        document.getElementById(`${tabName}-tab`).classList.add('active');
        
        // Load library data when switching to library tab
        if (tabName === 'library') {
            loadDocuments();
        }
    });
});

// ===== CHAT TAB =====
const chatContainer = document.getElementById('chat-container');
const userInput = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');
const sendText = document.getElementById('send-text');
const sendLoader = document.getElementById('send-loader');
const kInput = document.getElementById('k-input');
const showChunksCheckbox = document.getElementById('show-chunks');

let isProcessing = false;

sendBtn.addEventListener('click', handleSend);
userInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSend();
    }
});

async function handleSend() {
    const query = userInput.value.trim();
    
    if (!query || isProcessing) return;
    
    isProcessing = true;
    updateSendButton(true);
    
    addMessage(query, 'user');
    userInput.value = '';
    
    const k = parseInt(kInput.value);
    const showChunks = showChunksCheckbox.checked;
    
    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query, k })
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Server error');
        }
        
        const data = await response.json();
        
        addMessage(data.answer, 'assistant', {
            chunks: data.chunks,  // Always send chunks
            metadata: data.metadata
        });
        
    } catch (error) {
        console.error('Error:', error);
        addMessage(
            `❌ Error: ${error.message}. Please try again.`,
            'assistant'
        );
    } finally {
        isProcessing = false;
        updateSendButton(false);
        userInput.focus();
    }
}

function addMessage(text, type, extras = {}) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}-message`;
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    
    const textP = document.createElement('p');
    textP.textContent = text;
    contentDiv.appendChild(textP);
    
    if (extras.chunks && extras.chunks.length > 0) {
        // Add toggle button for chunks
        const toggleBtn = document.createElement('button');
        toggleBtn.textContent = `Show ${extras.chunks.length} Retrieved Chunks`;
        toggleBtn.style.marginTop = '0.75rem';
        toggleBtn.style.padding = '0.5rem 1rem';
        toggleBtn.style.fontSize = '0.85rem';
        toggleBtn.style.cursor = 'pointer';
        toggleBtn.style.background = 'var(--bg-tertiary)';
        toggleBtn.style.border = '1px solid var(--border)';
        toggleBtn.style.borderRadius = '6px';
        toggleBtn.style.color = 'var(--text-secondary)';
        
        const chunksContainer = createChunksDisplay(extras.chunks);
        chunksContainer.style.display = 'none';
        
        toggleBtn.addEventListener('click', () => {
            if (chunksContainer.style.display === 'none') {
                chunksContainer.style.display = 'block';
                toggleBtn.textContent = 'Hide Retrieved Chunks';
            } else {
                chunksContainer.style.display = 'none';
                toggleBtn.textContent = `Show ${extras.chunks.length} Retrieved Chunks`;
            }
        });
        
        contentDiv.appendChild(toggleBtn);
        contentDiv.appendChild(chunksContainer);
    }
    
    if (extras.metadata) {
        contentDiv.appendChild(createMetadataDisplay(extras.metadata));
    }
    
    messageDiv.appendChild(contentDiv);
    chatContainer.appendChild(messageDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

function createChunksDisplay(chunks) {
    const container = document.createElement('div');
    container.className = 'chunks-container';
    
    const header = document.createElement('div');
    header.textContent = `📚 Retrieved Chunks (${chunks.length})`;
    header.style.fontWeight = '600';
    header.style.marginBottom = '0.75rem';
    container.appendChild(header);
    
    chunks.forEach((chunk, i) => {
        const chunkDiv = document.createElement('div');
        chunkDiv.className = 'chunk';
        
        const chunkHeader = document.createElement('div');
        chunkHeader.className = 'chunk-header';
        
        const chunkMeta = document.createElement('div');
        chunkMeta.textContent = `${i + 1}. ${chunk.metadata.doc}`;
        if (chunk.metadata.page) {
            chunkMeta.textContent += ` (Seite ${chunk.metadata.page})`;
        }
        
        const chunkScore = document.createElement('div');
        chunkScore.className = 'chunk-score';
        chunkScore.textContent = chunk.score.toFixed(3);
        
        chunkHeader.appendChild(chunkMeta);
        chunkHeader.appendChild(chunkScore);
        
        const chunkText = document.createElement('div');
        chunkText.textContent = chunk.text.length > 200 
            ? chunk.text.substring(0, 200) + '...'
            : chunk.text;
        chunkText.style.color = 'var(--text-secondary)';
        
        chunkDiv.appendChild(chunkHeader);
        chunkDiv.appendChild(chunkText);
        container.appendChild(chunkDiv);
    });
    
    return container;
}

function createMetadataDisplay(metadata) {
    const metaDiv = document.createElement('div');
    metaDiv.className = 'metadata';
    metaDiv.textContent = `k: ${metadata.k} | Tokens: ${metadata.tokens}`;
    return metaDiv;
}

function updateSendButton(loading) {
    if (loading) {
        sendText.style.display = 'none';
        sendLoader.style.display = 'inline-block';
        sendBtn.disabled = true;
    } else {
        sendText.style.display = 'inline';
        sendLoader.style.display = 'none';
        sendBtn.disabled = false;
    }
}

// Auto-resize textarea
userInput.addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = Math.min(this.scrollHeight, 120) + 'px';
});

// ===== UPLOAD TAB =====
const uploadArea = document.getElementById('upload-area');
const fileInput = document.getElementById('file-input');
const uploadProgress = document.getElementById('upload-progress');
const progressFill = document.getElementById('progress-fill');
const progressText = document.getElementById('progress-text');
const uploadResults = document.getElementById('upload-results');

uploadArea.addEventListener('click', () => fileInput.click());

fileInput.addEventListener('change', (e) => {
    handleFiles(e.target.files);
});

// Drag & Drop
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('drag-over');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('drag-over');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('drag-over');
    handleFiles(e.dataTransfer.files);
});

async function handleFiles(files) {
    uploadResults.innerHTML = '';
    
    for (let file of files) {
        await uploadFile(file);
    }
}

async function uploadFile(file) {
    const formData = new FormData();
    formData.append('file', file);
    
    // Show progress
    uploadProgress.style.display = 'block';
    progressText.textContent = `Processing ${file.name}...`;
    progressFill.style.width = '0%';
    
    // Simulate progress (real progress would need server-sent events)
    let progress = 0;
    const progressInterval = setInterval(() => {
        progress += 5;
        if (progress <= 90) {
            progressFill.style.width = progress + '%';
        }
    }, 200);
    
    try {
        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });
        
        clearInterval(progressInterval);
        progressFill.style.width = '100%';
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Upload failed');
        }
        
        const data = await response.json();
        
        // Show success
        const resultDiv = document.createElement('div');
        resultDiv.className = 'upload-result';
        resultDiv.innerHTML = `
            <strong>✅ ${data.filename}</strong><br>
            <small>
                ${data.file_type.toUpperCase()} | 
                ${data.num_paragraphs} Paragraphen | 
                ${data.num_chunks} Chunks
            </small>
        `;
        uploadResults.appendChild(resultDiv);
        
        setTimeout(() => {
            uploadProgress.style.display = 'none';
        }, 1000);
        
    } catch (error) {
        clearInterval(progressInterval);
        console.error('Upload error:', error);
        
        const resultDiv = document.createElement('div');
        resultDiv.className = 'upload-result error';
        resultDiv.innerHTML = `
            <strong>❌ ${file.name}</strong><br>
            <small>Error: ${error.message}</small>
        `;
        uploadResults.appendChild(resultDiv);
        
        uploadProgress.style.display = 'none';
    }
}

// ===== LIBRARY TAB =====
const documentsList = document.getElementById('documents-list');
const libraryStats = document.getElementById('library-stats');
const reindexBtn = document.getElementById('reindex-btn');

reindexBtn.addEventListener('click', async () => {
    if (!confirm('Rebuild index? This may take a few minutes.')) return;
    
    reindexBtn.disabled = true;
    reindexBtn.textContent = '⏳ Re-indexing...';
    
    try {
        const response = await fetch('/api/reindex', { method: 'POST' });
        const data = await response.json();
        
        if (data.success) {
            alert('✅ Index successfully rebuilt!');
            loadDocuments();
        }
    } catch (error) {
        alert('❌ Error during re-indexing: ' + error.message);
    } finally {
        reindexBtn.disabled = false;
        reindexBtn.textContent = '🔄 Re-index All';
    }
});

async function loadDocuments() {
    documentsList.innerHTML = '<p class="loading-text">Loading documents...</p>';
    
    try {
        const response = await fetch('/api/documents');
        const data = await response.json();
        
        if (!data.success) {
            throw new Error('Failed to load documents');
        }
        
        // Update stats
        const stats = data.statistics;
        libraryStats.innerHTML = `
            📊 ${stats.num_documents} Documents | 
            ${stats.total_chunks} Chunks | 
            ${stats.by_type.pdf} PDFs, ${stats.by_type.docx} DOCX
        `;
        
        // Display documents
        if (data.documents.length === 0) {
            documentsList.innerHTML = '<p class="loading-text">No documents uploaded yet.</p>';
            return;
        }
        
        documentsList.innerHTML = '';
        
        data.documents.forEach(doc => {
            const docCard = document.createElement('div');
            docCard.className = 'doc-card';
            
            const uploadDate = new Date(doc.upload_date).toLocaleDateString('en-US');
            
            docCard.innerHTML = `
                <div class="doc-header">
                    <div>
                        <div class="doc-title">📄 ${doc.filename}</div>
                    </div>
                    <div class="doc-type">${doc.file_type}</div>
                </div>
                <div class="doc-stats">
                    <span>📝 ${doc.num_paragraphs} Paragraphs</span>
                    <span>✂️ ${doc.num_chunks} Chunks</span>
                    ${doc.num_chapters > 0 ? `<span>📚 ${doc.num_chapters} Chapters</span>` : ''}
                    <span>📅 ${uploadDate}</span>
                </div>
                <div class="doc-actions">
                    <button class="btn-delete" onclick="deleteDocument('${doc.doc_id}', '${doc.filename}')">
                        🗑️ Delete
                    </button>
                </div>
            `;
            
            documentsList.appendChild(docCard);
        });
        
    } catch (error) {
        console.error('Error loading documents:', error);
        documentsList.innerHTML = '<p class="loading-text">❌ Error loading documents</p>';
    }
}

async function deleteDocument(docId, filename) {
    if (!confirm(`Really delete document "${filename}"?`)) return;
    
    try {
        const response = await fetch(`/api/documents/${docId}`, {
            method: 'DELETE'
        });
        
        if (!response.ok) {
            throw new Error('Delete failed');
        }
        
        // Reload documents
        loadDocuments();
        
    } catch (error) {
        alert('❌ Error deleting: ' + error.message);
    }
}

// Initial load
document.addEventListener('DOMContentLoaded', () => {
    // Load statistics on startup
    fetch('/api/statistics')
        .then(r => r.json())
        .then(data => {
            if (data.success && data.index.total_chunks > 0) {
                console.log('📊 System ready:', data);
            }
        })
        .catch(e => console.log('No existing index'));
});