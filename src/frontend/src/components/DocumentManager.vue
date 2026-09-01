<script setup lang="ts">
import { ref, onMounted } from "vue";

interface Document {
  id: string;
  title: string;
  created_at: string;
}

const title = ref("");
const content = ref("");
const documents = ref<Document[]>([]);
const loading = ref(false);
const error = ref("");
const success = ref("");
const fileInput = ref<HTMLInputElement | null>(null);
const selectedDocId = ref<string | null>(null);
const selectedDocContent = ref("");

onMounted(async () => {
  await loadDocuments();
});

async function handleFileUpload(event: Event) {
  const target = event.target as HTMLInputElement;
  const file = target.files?.[0];
  if (!file) return;

  loading.value = true;
  error.value = "";
  success.value = "";

  try {
    const text = await file.text();
    title.value = file.name.replace(/\.[^/.]+$/, ""); // Remove file extension
    content.value = text;
    success.value = `File loaded: ${file.name}`;
    setTimeout(() => (success.value = ""), 2000);
  } catch (e) {
    error.value = `Failed to read file: ${e}`;
  } finally {
    loading.value = false;
    // Reset file input
    if (fileInput.value) fileInput.value.value = "";
  }
}

async function loadDocuments() {
  try {
    const res = await fetch("/api/rag/documents");
    documents.value = await res.json();
  } catch (e) {
    error.value = "Failed to load documents";
  }
}

async function addDocument() {
  if (!title.value.trim() || !content.value.trim()) {
    error.value = "Title and content required";
    return;
  }

  loading.value = true;
  error.value = "";
  success.value = "";

  try {
    const res = await fetch("/api/rag/documents/add", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        title: title.value,
        content: content.value,
      }),
    });

    if (res.ok) {
      success.value = "Document added successfully!";
      title.value = "";
      content.value = "";
      await loadDocuments();
      setTimeout(() => (success.value = ""), 3000);
    } else {
      const err = await res.json();
      error.value = err.detail || "Failed to add document";
    }
  } catch (e) {
    error.value = `Error: ${e}`;
  } finally {
    loading.value = false;
  }
}

async function deleteDocument(id: string) {
  if (!confirm("Delete this document?")) return;

  try {
    await fetch(`/api/rag/documents/${id}`, { method: "DELETE" });
    await loadDocuments();
  } catch (e) {
    error.value = `Failed to delete: ${e}`;
  }
}

async function viewDocument(doc: Document) {
  try {
    const res = await fetch(`/api/rag/documents/${doc.id}`);
    const data = await res.json();
    selectedDocContent.value = data.content;
    selectedDocId.value = doc.id;
  } catch (e) {
    error.value = `Failed to load document: ${e}`;
  }
}

function closeDocumentView() {
  selectedDocId.value = null;
  selectedDocContent.value = "";
}
</script>

<template>
  <div class="document-manager">
    <div class="manager-card">
      <h2>📄 Manage Documents</h2>
      
      <div class="add-section">
        <h3>Add Document</h3>
        
        <div class="upload-area">
          <label class="file-label">
            <input
              ref="fileInput"
              type="file"
              @change="handleFileUpload"
              :disabled="loading"
              accept=".txt,.md,.pdf,.json,.csv"
              class="file-input"
            />
            <span class="file-button">📁 Choose File</span>
          </label>
          <span class="divider">OR</span>
          <span class="paste-text">Paste content below</span>
        </div>

        <input
          v-model="title"
          type="text"
          placeholder="Document title"
          :disabled="loading"
          class="input-field"
        />
        <textarea
          v-model="content"
          placeholder="Document content..."
          :disabled="loading"
          rows="4"
          class="input-field"
        />
        
        <div v-if="error" class="alert error">{{ error }}</div>
        <div v-if="success" class="alert success">{{ success }}</div>
        
        <button
          @click="addDocument"
          :disabled="loading || !title.trim() || !content.trim()"
          class="button primary"
        >
          {{ loading ? "Adding..." : "Add Document" }}
        </button>
      </div>

      <div class="documents-section">
        <h3>Stored Documents ({{ documents.length }})</h3>
        <div v-if="documents.length === 0" class="empty">
          No documents yet. Add one to use with RAG!
        </div>
        <div v-else class="documents-list">
          <div v-for="doc in documents" :key="doc.id" class="document-item" @click="viewDocument(doc)">
            <div class="doc-info">
              <div class="doc-title">{{ doc.title }}</div>
              <div class="doc-date">{{ new Date(doc.created_at).toLocaleDateString() }}</div>
            </div>
            <button
              @click.stop="deleteDocument(doc.id)"
              class="button delete"
              title="Delete document"
            >
              🗑️
            </button>
          </div>
        </div>
      </div>

      <!-- Document View Modal -->
      <div v-if="selectedDocId" class="modal-overlay" @click="closeDocumentView">
        <div class="modal-content" @click.stop>
          <div class="modal-header">
            <h3>Document Content</h3>
            <button class="close-button" @click="closeDocumentView">✕</button>
          </div>
          <div class="modal-body">
            <pre class="document-content">{{ selectedDocContent }}</pre>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.document-manager {
  padding: 1rem;
  max-width: 600px;
  margin: 0 auto;
}

.manager-card {
  background: #1a1a1a;
  border: 1px solid #333;
  border-radius: 0.75rem;
  padding: 1.5rem;
  color: #e0e0e0;
}

h2 {
  margin-top: 0;
  margin-bottom: 1.5rem;
  font-size: 1.5rem;
}

h3 {
  margin-top: 0;
  margin-bottom: 1rem;
  font-size: 1.1rem;
  color: #b0b0b0;
}

.add-section {
  margin-bottom: 2rem;
  padding-bottom: 1.5rem;
  border-bottom: 1px solid #333;
}

.upload-area {
  display: flex;
  align-items: center;
  gap: 1rem;
  margin-bottom: 1rem;
  padding: 1rem;
  background: #2a2a2a;
  border: 1px dashed #444;
  border-radius: 0.5rem;
}

.file-label {
  cursor: pointer;
}

.file-input {
  display: none;
}

.file-button {
  display: inline-block;
  padding: 0.6rem 1rem;
  background: #0066cc;
  color: white;
  border-radius: 0.5rem;
  font-weight: 500;
  transition: background 0.2s;
  white-space: nowrap;
}

.file-label:hover .file-button {
  background: #0052a3;
}

.file-input:disabled ~ .file-button {
  opacity: 0.5;
  cursor: not-allowed;
}

.divider {
  color: #666;
  font-size: 0.9rem;
}

.paste-text {
  color: #999;
  font-size: 0.9rem;
}

.input-field {
  width: 100%;
  padding: 0.6rem;
  margin-bottom: 0.75rem;
  background: #2a2a2a;
  border: 1px solid #444;
  border-radius: 0.5rem;
  color: #e0e0e0;
  font-size: 0.95rem;
  font-family: inherit;
}

.input-field:focus {
  outline: none;
  border-color: #0066cc;
  box-shadow: 0 0 0 2px rgba(0, 102, 204, 0.1);
}

.input-field:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.alert {
  padding: 0.75rem;
  margin-bottom: 1rem;
  border-radius: 0.5rem;
  font-size: 0.9rem;
}

.alert.error {
  background: rgba(255, 68, 68, 0.1);
  border: 1px solid rgba(255, 68, 68, 0.3);
  color: #ff6b6b;
}

.alert.success {
  background: rgba(68, 255, 68, 0.1);
  border: 1px solid rgba(68, 255, 68, 0.3);
  color: #6bff6b;
}

.button {
  padding: 0.6rem 1rem;
  border: none;
  border-radius: 0.5rem;
  cursor: pointer;
  font-size: 0.95rem;
  transition: all 0.2s;
}

.button.primary {
  width: 100%;
  background: #0066cc;
  color: white;
  font-weight: 500;
}

.button.primary:hover:not(:disabled) {
  background: #0052a3;
}

.button.primary:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.button.delete {
  background: transparent;
  color: #ff6b6b;
  padding: 0.4rem;
  font-size: 1rem;
}

.button.delete:hover {
  background: rgba(255, 68, 68, 0.1);
}

.documents-section {
  margin-top: 1rem;
}

.empty {
  padding: 1rem;
  text-align: center;
  color: #666;
  font-style: italic;
}

.documents-list {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.document-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0.75rem;
  background: #2a2a2a;
  border: 1px solid #333;
  border-radius: 0.5rem;
  transition: border-color 0.2s;
  cursor: pointer;
}

.document-item:hover {
  border-color: #0066cc;
  background: #333333;
}

.doc-info {
  flex: 1;
  min-width: 0;
}

.doc-title {
  font-weight: 500;
  color: #e0e0e0;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.doc-date {
  font-size: 0.8rem;
  color: #666;
  margin-top: 0.25rem;
}

.modal-overlay {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.7);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
  padding: 1rem;
}

.modal-content {
  background: #1a1a1a;
  border: 1px solid #333;
  border-radius: 0.75rem;
  max-width: 800px;
  width: 100%;
  max-height: 80vh;
  display: flex;
  flex-direction: column;
}

.modal-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 1rem;
  border-bottom: 1px solid #333;
}

.modal-header h3 {
  margin: 0;
}

.close-button {
  background: transparent;
  border: none;
  color: #999;
  font-size: 1.5rem;
  cursor: pointer;
  padding: 0;
  width: 2rem;
  height: 2rem;
  display: flex;
  align-items: center;
  justify-content: center;
}

.close-button:hover {
  color: #e0e0e0;
}

.modal-body {
  flex: 1;
  overflow-y: auto;
  padding: 1rem;
}

.document-content {
  background: #2a2a2a;
  border: 1px solid #333;
  border-radius: 0.5rem;
  padding: 1rem;
  color: #b0b0b0;
  font-size: 0.9rem;
  line-height: 1.6;
  white-space: pre-wrap;
  word-break: break-word;
  margin: 0;
  max-height: 100%;
  overflow-y: auto;
}
</style>
