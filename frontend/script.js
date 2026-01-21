// Global variables
let selectedFile = null;

// DOM elements
const uploadArea = document.getElementById('uploadArea');
const previewSection = document.getElementById('previewSection');
const resultsSection = document.getElementById('resultsSection');
const loadingSection = document.getElementById('loadingSection');
const errorSection = document.getElementById('errorSection');
const imageInput = document.getElementById('imageInput');
const previewImage = document.getElementById('previewImage');
const analyzeBtn = document.getElementById('analyzeBtn');

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    setupEventListeners();
});

// Setup event listeners
function setupEventListeners() {
    // Drag and drop functionality
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleDrop);

    // File input change
    imageInput.addEventListener('change', handleFileSelect);
}

// Drag and drop handlers
function handleDragOver(e) {
    e.preventDefault();
    uploadArea.classList.add('drag-over');
}

function handleDragLeave(e) {
    e.preventDefault();
    uploadArea.classList.remove('drag-over');
}

function handleDrop(e) {
    e.preventDefault();
    uploadArea.classList.remove('drag-over');

    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
}

function handleFileSelect(e) {
    const files = e.target.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
}

function handleFile(file) {
    if (!file.type.startsWith('image/')) {
        showError('Please select a valid image file.');
        return;
    }

    selectedFile = file;
    displayPreview(file);
}

// Display image preview
function displayPreview(file) {
    const reader = new FileReader();
    reader.onload = function(e) {
        previewImage.src = e.target.result;
        uploadArea.style.display = 'none';
        previewSection.style.display = 'block';
        previewSection.classList.add('fade-in');
    };
    reader.readAsDataURL(file);
}

// Clear the selected image
function clearImage() {
    selectedFile = null;
    previewImage.src = '';
    previewSection.style.display = 'none';
    uploadArea.style.display = 'block';
    hideResults();
    hideError();
}

// Analyze the image
async function analyzeImage() {
    if (!selectedFile) {
        showError('Please select an image first.');
        return;
    }

    // Show loading state
    analyzeBtn.disabled = true;
    analyzeBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing...';
    showLoading();

    try {
        const formData = new FormData();
        formData.append('image', selectedFile);

        const response = await fetch('http://localhost:5000/predict', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        if (!response.ok) {
            throw new Error(result.error || 'Analysis failed');
        }

        displayResults(result);
    } catch (error) {
        console.error('Error:', error);
        showError(error.message || 'Failed to analyze image. Please try again.');
    } finally {
        // Reset button state
        analyzeBtn.disabled = false;
        analyzeBtn.innerHTML = '<i class="fas fa-search"></i> Analyze Image';
        hideLoading();
    }
}

// Display analysis results
function displayResults(result) {
    // Update status badge
    const statusBadge = document.getElementById('statusBadge');
    statusBadge.className = 'status-badge';
    statusBadge.classList.add(result.status);

    let statusText = '';
    switch (result.status) {
        case 'known':
            statusText = 'Known Species';
            break;
        case 'unknown':
            statusText = 'Unknown Species';
            break;
        case 'empty':
            statusText = 'No Animals Detected';
            break;
    }
    statusBadge.textContent = statusText;

    // Update result image
    const resultImage = document.getElementById('resultImage');
    resultImage.src = `http://localhost:5000${result.image_url}`;

    // Update details
    document.getElementById('speciesValue').textContent =
        result.species === 'none' ? 'None detected' : result.species;

    document.getElementById('confidenceValue').textContent =
        result.confidence ? `${(result.confidence * 100).toFixed(1)}%` : 'N/A';

    document.getElementById('noveltyValue').textContent =
        result.novelty_score ? result.novelty_score.toFixed(5) : 'N/A';

    // Show results section
    hideLoading();
    hideError();
    resultsSection.style.display = 'block';
    resultsSection.classList.add('fade-in');

    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// Show loading state
function showLoading() {
    loadingSection.style.display = 'block';
    loadingSection.classList.add('fade-in');
    resultsSection.style.display = 'none';
    errorSection.style.display = 'none';
}

// Hide loading state
function hideLoading() {
    loadingSection.style.display = 'none';
}

// Show error message
function showError(message) {
    document.getElementById('errorMessage').textContent = message;
    errorSection.style.display = 'block';
    errorSection.classList.add('fade-in');
    resultsSection.style.display = 'none';
    loadingSection.style.display = 'none';
}

// Hide error message
function hideError() {
    errorSection.style.display = 'none';
}

// Hide results
function hideResults() {
    resultsSection.style.display = 'none';
}

// Utility function to format numbers
function formatNumber(num, decimals = 2) {
    return num ? num.toFixed(decimals) : 'N/A';
}

// Add drag-over class for visual feedback
const style = document.createElement('style');
style.textContent = `
    .drag-over {
        background: rgba(76, 175, 80, 0.1) !important;
        border-color: #4CAF50 !important;
        transform: scale(1.02);
    }
`;
document.head.appendChild(style);