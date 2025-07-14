<template>
  <div class="admin-container">
    <!-- Header Section -->
    <div class="">
      <h1 class="text-foreground">Import Note to Database</h1>
    </div>

    <p class="text-foreground mb-4">Add new information to the database</p>

    <!-- PDF Upload Section -->
    <div class="upload-section">
      <div class="bg-surface rounded-lg pa-4">
        <h2 class="text-foreground mb-4">Upload PDF</h2>

        <!-- File Upload Area -->
        <div class="file-input-container">
          <input type="file" ref="fileInput" accept=".pdf" @change="handleFileSelect" :disabled="isUploading"
            class="file-input" id="pdf-file-input" />
          <label for="pdf-file-input" class="file-label" :class="{ disabled: isUploading }">
            <span class="file-text">
              {{ selectedFileName || 'Select PDF file' }}
            </span>
          </label>
        </div>

        <!-- File Info Display -->
        <div v-if="fileInfo" class="pa-4">
          <div class="info-item">
            <strong>📄 File:</strong> {{ fileInfo.name }}
          </div>
          <div class="info-item">
            <strong>📊 Size:</strong> {{ fileInfo.sizeMB }}MB
          </div>
          <div class="info-item">
            <strong>🔧 Type:</strong> {{ fileInfo.type }}
          </div>
        </div>

        <!-- Upload Progress -->
        <div v-if="isUploading" class="progress-container">
          <div class="progress-bar">
            <div class="progress-fill" :style="{ width: uploadProgress + '%' }"></div>
          </div>
          <span class="progress-text">{{ uploadProgress }}%</span>
        </div>

        <!-- Success/Error Messages -->
        <div v-if="uploadMessage && !isUploading" class="message-container" :class="{ success: uploadSuccess, error: !uploadSuccess }">
          <!-- Success Message with Details -->
          <div v-if="uploadSuccess && uploadDetails" class="success-details">
            <div class="message-title">✅ Upload Successful!</div>
            <div class="divider"></div>
            <div class="upload-details">
              <div class="detail-item">
                <strong>📄 File:</strong> {{ uploadDetails.filename }}
              </div>
              <div class="detail-item">
                <strong>📊 Size:</strong> {{ uploadDetails.file_size_mb }}MB
              </div>
              <div class="detail-item">
                <strong>📚 Documents processed:</strong> {{ uploadDetails.documents_processed }}
              </div>
              <div class="detail-item">
                <strong>🔢 Chunks created:</strong> {{ uploadDetails.chunks_created }}
              </div>
              <div class="detail-item">
                <strong>🤖 Embedding model:</strong> {{ uploadDetails.embedding_model }}
              </div>
              <div v-if="uploadDetails.vector_store_stats" class="detail-item">
                <strong>💾 Vector store:</strong> Updated successfully
              </div>
            </div>
          </div>

          <!-- Simple Success Message -->
          <div v-else-if="uploadSuccess" class="simple-message">
            {{ uploadMessage }}
          </div>

          <!-- Error Message -->
          <div v-else class="error-details">
            <div class="message-title">❌ Upload Failed</div>
            <div class="error-text">{{ uploadMessage }}</div>
            <div class="tips">
              <strong>💡 Tips:</strong>
              <ul>
                <li>Make sure the file is a valid PDF</li>
                <li>Check that the file size is under 10MB</li>
                <li>Ensure the backend server is running on localhost:8000</li>
              </ul>
            </div>
          </div>
        </div>

        <!-- Upload Button -->
        <div class="d-flex justify-end ga-2">
          <button v-if="selectedFile" @click="clearSelection" :disabled="isUploading" class="pa-4 bg-secondary text-white rounded-lg cursor-pointer">
            🗑️ Clear
          </button>
          <button @click="uploadPDF" :disabled="!selectedFile || isUploading" class="pa-4 bg-secondary text-white rounded-lg cursor-pointer"
            :class="{ loading: isUploading }">
            <span v-if="isUploading">🔄 Processing...</span>
            <span v-else>Upload PDF</span>
          </button>
        </div>
      </div>
    </div>

    <!-- Database Management Section -->
    <div class="d-flex justify-space-between bg-surface mt-4 rounded-lg pa-4 align-center">
      <h2 class="text-foreground">Danger Zone</h2>
      <div class="d-flex justify-end ga-2">
        <button @click="showClearConfirmation" :disabled="isClearingDatabase" class="pa-4 bg-error text-white rounded-lg"
          :class="{ loading: isClearingDatabase }">
          <span v-if="isClearingDatabase">🔄 Clearing...</span>
          <span v-else>Clear Database</span>
        </button>
      </div>

      <!-- Clear Database Progress -->
      <div v-if="isClearingDatabase" class="progress-container">
        <div class="progress-bar">
          <div class="progress-fill danger" :style="{ width: clearProgress + '%' }"></div>
        </div>
        <span class="progress-text">{{ clearProgress }}%</span>
      </div>

      <!-- Clear Database Messages -->
    </div>

    <div v-if="clearMessage" class="message-container mt-4" :class="{ success: clearSuccess, error: !clearSuccess }">
        <div v-if="clearSuccess" class="simple-message">
          <div class="message-title">✅ Database Cleared Successfully!</div>
          <div>{{ clearMessage }}</div>
        </div>

        <div v-else class="error-details">
          <div class="message-title">❌ Clear Database Failed</div>
          <div class="error-text">{{ clearMessage }}</div>
        </div>
      </div>

    <div v-if="showConfirmModal" class="modal-overlay" @click="cancelClear">
      <div class="rounded-lg bg-surface text-white pa-2 d-flex flex-column justify-center align-center" @click.stop>

        <!-- <div class="">Are you sure you want to clear the entire PDF database?</div> -->
        <div class="text-white pa-10 border bg-warning rounded-xl border-warning ma-4">
          <strong>⚠️ Warning:</strong> This will permanently delete:
          <ul>
            <li>All uploaded PDF files</li>
            <li>All generated embeddings/vectors</li>
            <li>All processed document chunks</li>
          </ul>
          <p><strong>This action cannot be undone!</strong></p>
        </div>

        <div class="pa-4 d-flex justify-end ga-2 w-100">
          <button @click="confirmClear" class="pa-4 bg-error text-white rounded-lg">Yes, Clear Database</button>
          <button @click="cancelClear" class="pa-4 bg-secondary text-white rounded-lg">Cancel</button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from "vue";

// Reactive variables
const selectedFile = ref(null);
const fileInput = ref(null);
const isUploading = ref(false);
const uploadProgress = ref(0);
const uploadMessage = ref('');
const uploadSuccess = ref(false);
const uploadDetails = ref(null);

// Database management state
const isClearingDatabase = ref(false);
const clearProgress = ref(0);
const clearMessage = ref('');
const clearSuccess = ref(false);
const showConfirmModal = ref(false);

// Computed properties
const selectedFileName = computed(() => {
  return selectedFile.value ? selectedFile.value.name : null;
});

const fileInfo = computed(() => {
  if (!selectedFile.value) return null;

  const file = selectedFile.value;
  return {
    name: file.name,
    type: file.type,
    size: file.size,
    sizeMB: (file.size / (1024 * 1024)).toFixed(2)
  };
});

// Handle file selection
const handleFileSelect = (event) => {
  const file = event.target.files[0];

  console.log('🔍 File selected:', file);

  // Reset previous messages and details
  uploadMessage.value = '';
  uploadSuccess.value = false;
  uploadDetails.value = null;

  if (!file) {
    selectedFile.value = null;
    return;
  }

  console.log('📄 File details:', {
    name: file.name,
    type: file.type,
    size: file.size,
    sizeMB: (file.size / (1024 * 1024)).toFixed(2)
  });

  // Validate file type
  if (!file.name.toLowerCase().endsWith('.pdf')) {
    uploadMessage.value = 'Please select a valid PDF file';
    uploadSuccess.value = false;
    selectedFile.value = null;
    fileInput.value.value = '';
    return;
  }

  // Validate file size (10MB limit to match backend)
  if (file.size > 10 * 1024 * 1024) {
    uploadMessage.value = `File too large. Maximum size is 10MB, your file is ${(file.size / (1024 * 1024)).toFixed(2)}MB`;
    uploadSuccess.value = false;
    selectedFile.value = null;
    fileInput.value.value = '';
    return;
  }

  // Validate file is not empty
  if (file.size === 0) {
    uploadMessage.value = 'Selected file is empty. Please choose a valid PDF file';
    uploadSuccess.value = false;
    selectedFile.value = null;
    fileInput.value.value = '';
    return;
  }

  // File is valid, store it
  selectedFile.value = file;
  console.log('✅ File validation passed');
};

// Clear file selection
const clearSelection = () => {
  selectedFile.value = null;
  fileInput.value.value = '';
  uploadMessage.value = '';
  uploadSuccess.value = false;
  uploadDetails.value = null;
};

// Upload PDF function
const uploadPDF = async () => {
  console.log('🚀 Starting PDF upload...');
  console.log('📋 selectedFile.value:', selectedFile.value);

  if (!selectedFile.value) {
    uploadMessage.value = 'Please select a PDF file first';
    uploadSuccess.value = false;
    return;
  }

  const file = selectedFile.value;
  console.log('📄 File to upload:', {
    name: file.name,
    type: file.type,
    size: file.size,
    sizeMB: (file.size / (1024 * 1024)).toFixed(2)
  });

  // Create FormData object
  const formData = new FormData();
  formData.append('file', file);

  // Debug FormData contents
  console.log('📦 FormData contents:');
  for (let [key, value] of formData.entries()) {
    console.log(`   ${key}:`, value);
    console.log(`   ${key} instanceof File:`, value instanceof File);
  }

  try {
    isUploading.value = true;
    uploadProgress.value = 0;
    uploadMessage.value = 'Uploading PDF...';
    uploadDetails.value = null;

    // Simulate upload progress
    const progressInterval = setInterval(() => {
      if (uploadProgress.value < 90) {
        uploadProgress.value += 10;
      }
    }, 300);

    console.log('🌐 Making API call to: http://localhost:8000/pdf/upload');

    // Make the API call
    const response = await fetch('http://localhost:8000/pdf/upload', {
      method: 'POST',
      body: formData,
    });

    clearInterval(progressInterval);
    uploadProgress.value = 100;

    console.log('📡 Response status:', response.status);
    console.log('📡 Response ok:', response.ok);

    if (response.ok) {
      const result = await response.json();
      console.log('✅ Upload successful:', result);

      // Store detailed results
      uploadDetails.value = result;
      uploadMessage.value = 'PDF uploaded successfully!';
      uploadSuccess.value = true;

      // Clear the file input after successful upload
      setTimeout(() => {
        clearSelection();
        uploadProgress.value = 0;
        // Keep success message visible for longer
        setTimeout(() => {
          uploadMessage.value = '';
          uploadDetails.value = null;
        }, 5000);
      }, 2000);

    } else {
      // Handle error responses
      const errorText = await response.text();
      console.log('❌ Error response:', errorText);

      let errorData;
      try {
        errorData = JSON.parse(errorText);
      } catch (e) {
        errorData = { detail: errorText };
      }

      // Extract error message (FastAPI returns errors in 'detail' field)
      const errorMessage = errorData.detail || errorData.message || `HTTP error! status: ${response.status}`;

      throw new Error(errorMessage);
    }

  } catch (error) {
    console.error('❌ Upload error:', error);

    // Handle specific error types
    if (error.message.includes('Failed to fetch')) {
      uploadMessage.value = 'Connection failed. Please check if the backend server is running on localhost:8000';
    } else if (error.message.includes('NetworkError')) {
      uploadMessage.value = 'Network error. Please check your internet connection and try again.';
    } else {
      uploadMessage.value = `Upload failed: ${error.message}`;
    }

    uploadSuccess.value = false;
    uploadProgress.value = 0;
    uploadDetails.value = null;
  } finally {
    isUploading.value = false;
  }
};

// Database Management Functions
const showClearConfirmation = () => {
  showConfirmModal.value = true;
};

const cancelClear = () => {
  showConfirmModal.value = false;
};

const confirmClear = async () => {
  showConfirmModal.value = false;
  isClearingDatabase.value = true;
  clearProgress.value = 0;
  clearMessage.value = 'Clearing database...';
  clearSuccess.value = false;

  // Start progress simulation
  const progressInterval = setInterval(() => {
    if (clearProgress.value < 90) {
      clearProgress.value += 10;
    }
  }, 200);

  try {
    console.log('🗑️ Starting database clear operation...');

    const response = await fetch('http://localhost:8000/pdf/clear', {
      method: 'DELETE',
      headers: {
        'Content-Type': 'application/json',
      },
    });

    clearInterval(progressInterval);
    clearProgress.value = 100;

    console.log('📡 Clear response status:', response.status);

    if (response.ok) {
      const result = await response.json();
      console.log('✅ Database cleared successfully:', result);

      // Display success message with details if available
      if (result.message) {
        clearMessage.value = result.message;
      } else {
        clearMessage.value = 'Database cleared successfully! All PDF files and vectors have been removed.';
      }

      clearSuccess.value = true;

      // Clear the success message after a delay
      setTimeout(() => {
        clearMessage.value = '';
        clearProgress.value = 0;
      }, 5000);

    } else {
      const errorText = await response.text();
      console.log('❌ Error clearing database:', errorText);

      let errorData;
      try {
        errorData = JSON.parse(errorText);
      } catch (e) {
        errorData = { detail: errorText };
      }

      const errorMessage = errorData.detail || errorData.message || `HTTP error! status: ${response.status}`;
      throw new Error(errorMessage);
    }
  } catch (error) {
    clearInterval(progressInterval);
    console.error('❌ Clear database error:', error);

    if (error.message.includes('Failed to fetch')) {
      clearMessage.value = 'Connection failed. Please check if the backend server is running on localhost:8000';
    } else if (error.message.includes('NetworkError')) {
      clearMessage.value = 'Network error. Please check your internet connection and try again.';
    } else {
      clearMessage.value = `Failed to clear database: ${error.message}`;
    }

    clearSuccess.value = false;
    clearProgress.value = 0;
  } finally {
    isClearingDatabase.value = false;
  }
};

</script>

<style scoped>
.admin-container {
  max-width: 800px;
  margin: 0 auto;
  padding: 20px;
  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

.header-section {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
}

.page-title {
  font-size: 2rem;
  font-weight: 600;
  color: #333;
  margin: 0;
}

.view-btn {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 12px 20px;
  background: #f5f5f5;
  border: 1px solid #ddd;
  border-radius: 8px;
  cursor: pointer;
  transition: all 0.3s ease;
}

.view-btn:hover {
  background: #e8e8e8;
  border-color: #bbb;
}

.description {
  color: #666;
  margin-bottom: 30px;
  font-size: 1.1rem;
}

.upload-section {
  width: 100%;
}

.upload-card {
  background: #fff;
  border: 2px solid #e0e0e0;
  border-radius: 12px;
  padding: 30px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

.section-title {
  font-size: 1.5rem;
  font-weight: 600;
  color: #333;
  margin-bottom: 20px;
}

.file-input-container {
  position: relative;
  margin-bottom: 20px;
}

.file-input {
  display: none;
}

.file-label {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 20px;
  border: 2px dashed #ccc;
  border-radius: 8px;
  cursor: pointer;
  transition: all 0.3s ease;
  background: #fafafa;
}

.file-label:hover {
  border-color: #007bff;
  background: #f0f8ff;
}

.file-label.disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.file-icon {
  font-size: 1.5rem;
}

.file-text {
  font-size: 1rem;
  color: #555;
}

.file-info {
  background: #f8f9fa;
  border: 1px solid #dee2e6;
  border-radius: 8px;
  padding: 15px;
  margin-bottom: 20px;
}

.info-item {
  display: flex;
  justify-content: space-between;
  margin-bottom: 8px;
  font-size: 0.9rem;
}

.info-item:last-child {
  margin-bottom: 0;
}

.progress-container {
  display: flex;
  align-items: center;
  gap: 15px;
  margin-bottom: 20px;
}

.progress-bar {
  flex: 1;
  height: 8px;
  background: #e9ecef;
  border-radius: 4px;
  overflow: hidden;
}

.progress-fill {
  height: 100%;
  background: linear-gradient(90deg, #007bff, #0056b3);
  transition: width 0.3s ease;
}

.progress-text {
  font-size: 0.9rem;
  font-weight: 600;
  color: #007bff;
}

.message-container {
  padding: 20px;
  border-radius: 8px;
  margin-bottom: 20px;
}

.message-container.success {
  background: #d4edda;
  border: 1px solid #c3e6cb;
  color: #155724;
}

.message-container.error {
  background: #f8d7da;
  border: 1px solid #f5c6cb;
  color: #721c24;
}

.message-title {
  font-size: 1.2rem;
  font-weight: 600;
  margin-bottom: 10px;
}

.divider {
  height: 1px;
  background: #ccc;
  margin: 10px 0;
}

.upload-details {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.detail-item {
  display: flex;
  justify-content: space-between;
  padding: 8px 12px;
  background: rgba(255, 255, 255, 0.3);
  border-radius: 4px;
  font-size: 0.9rem;
}

.simple-message {
  font-size: 1rem;
}

.error-details {
  font-size: 0.95rem;
}

.error-text {
  margin-bottom: 15px;
}

.tips ul {
  margin: 8px 0 0 20px;
  padding: 0;
}

.tips li {
  margin: 4px 0;
  font-size: 0.85rem;
}

.button-container {
  display: flex;
  gap: 15px;
  justify-content: flex-start;
}

.upload-btn,
.clear-btn {
  padding: 12px 24px;
  border: none;
  border-radius: 8px;
  font-size: 1rem;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.3s ease;
  min-width: 140px;
}

.upload-btn {
  background: #007bff;
  color: white;
}

.upload-btn:hover:not(:disabled) {
  background: #0056b3;
}

.upload-btn:disabled {
  background: #6c757d;
  cursor: not-allowed;
}

.upload-btn.loading {
  background: #17a2b8;
}

.clear-btn {
  background: #6c757d;
  color: white;
}

.clear-btn:hover:not(:disabled) {
  background: #5a6268;
}

.clear-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

/* Database Management Styles */
.database-section {
  margin-top: 40px;
}

.database-card {
  background: #fff;
  border: 2px solid #e0e0e0;
  border-radius: 12px;
  padding: 30px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

.database-actions {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
}

.action-info h3 {
  font-size: 1.2rem;
  font-weight: 600;
  color: #333;
  margin: 0 0 5px 0;
}

.action-info p {
  font-size: 0.9rem;
  color: #666;
  margin: 0 0 15px 0;
}

.action-buttons {
  display: flex;
  gap: 15px;
}

.danger-btn,
.confirm-danger-btn,
.cancel-btn {
  padding: 12px 24px;
  border: none;
  border-radius: 8px;
  font-size: 1rem;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.3s ease;
  min-width: 140px;
}

.danger-btn {
  background: #dc3545;
  color: white;
}

.danger-btn:hover:not(:disabled) {
  background: #c82333;
}

.confirm-danger-btn {
  background: #dc3545;
  color: white;
}

.confirm-danger-btn:hover:not(:disabled) {
  background: #c82333;
}

.cancel-btn {
  background: #6c757d;
  color: white;
}

.cancel-btn:hover:not(:disabled) {
  background: #5a6268;
}

.progress-bar.danger .progress-fill {
  background: linear-gradient(90deg, #dc3545, #c82333);
}

/* Modal Styles */
.modal-overlay {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: rgba(0, 0, 0, 0.5);
  display: flex;
  justify-content: center;
  align-items: center;
  z-index: 1000;
}

.modal-content {
  background: #fff;
  border-radius: 10px;
  box-shadow: 0 5px 15px rgba(0, 0, 0, 0.3);
  width: 90%;
  max-width: 500px;
  max-height: 90%;
  display: flex;
  flex-direction: column;
  overflow-y: auto;
}

.modal-header {
  padding: 20px;
  border-bottom: 1px solid #eee;
  text-align: center;
}

.modal-header h3 {
  font-size: 1.8rem;
  font-weight: 600;
  color: #dc3545;
  /* Red for warning */
  margin: 0;
}

.modal-body {
  padding: 20px;
  font-size: 1rem;
  color: #333;
  line-height: 1.6;
}

.warning-box {
  background: #fff3cd;
  border: 1px solid #ffeeba;
  border-radius: 8px;
  padding: 15px;
  margin-top: 15px;
  margin-bottom: 20px;
}

.warning-box strong {
  color: #856404;
  font-weight: 600;
}

.warning-box ul {
  margin: 10px 0 0 20px;
  padding: 0;
}

.warning-box li {
  margin: 5px 0;
  font-size: 0.9rem;
}

.modal-footer {
  padding: 15px 20px;
  border-top: 1px solid #eee;
  display: flex;
  justify-content: flex-end;
  gap: 15px;
}

.cancel-btn,
.confirm-danger-btn {
  padding: 10px 20px;
  border: none;
  border-radius: 6px;
  font-size: 0.9rem;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.3s ease;
}

.cancel-btn {
  background: #6c757d;
  color: white;
}

.cancel-btn:hover:not(:disabled) {
  background: #5a6268;
}

.confirm-danger-btn {
  background: #dc3545;
  color: white;
}

.confirm-danger-btn:hover:not(:disabled) {
  background: #c82333;
}

.confirm-danger-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

/* Responsive design */
@media (max-width: 768px) {
  .header-section {
    flex-direction: column;
    gap: 15px;
    text-align: center;
  }

  .page-title {
    font-size: 1.5rem;
  }

  .upload-card {
    padding: 20px;
  }

  .database-card {
    padding: 20px;
  }

  .database-actions {
    flex-direction: column;
    gap: 15px;
  }

  .action-buttons {
    flex-direction: column;
    width: 100%;
  }

  .button-container {
    flex-direction: column;
  }

  .upload-btn,
  .clear-btn {
    width: 100%;
  }
}
</style>
