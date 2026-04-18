/**
 * PolypVision AI — Frontend Logic
 * Handles file upload, drag-and-drop, inference calls, and result display.
 */

// ── DOM Elements ─────────────────────────────────────────────
const uploadZone     = document.getElementById('upload-zone');
const fileInput      = document.getElementById('file-input');
const uploadCard     = document.getElementById('upload-card');
const filePreview    = document.getElementById('file-preview');
const fileName       = document.getElementById('file-name');
const fileSize       = document.getElementById('file-size');
const fileTypeIcon   = document.getElementById('file-type-icon');
const analyzeBtn     = document.getElementById('analyze-btn');
const clearBtn       = document.getElementById('clear-btn');
const loadingOverlay = document.getElementById('loading-overlay');
const loadingHint    = document.getElementById('loading-hint');
const resultsSection = document.getElementById('results-section');
const imageResult    = document.getElementById('image-result');
const videoResult    = document.getElementById('video-result');
const resultImage    = document.getElementById('result-image');
const resultVideo    = document.getElementById('result-video');
const downloadBtn    = document.getElementById('download-btn');
const newUploadBtn   = document.getElementById('new-upload-btn');
const confSlider     = document.getElementById('conf-slider');
const confValue      = document.getElementById('conf-value');
const statusDot      = document.getElementById('status-dot');
const statDetections = document.getElementById('stat-detections');
const statTime       = document.getElementById('stat-time');
const statConf       = document.getElementById('stat-conf');

// ── State ────────────────────────────────────────────────────
let selectedFile = null;

// ── Helpers ──────────────────────────────────────────────────

function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / 1048576).toFixed(1) + ' MB';
}

function isImage(file) {
    return file.type.startsWith('image/');
}

function isVideo(file) {
    return file.type.startsWith('video/');
}

function showToast(message, type = 'info', duration = 4000) {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;

    const icons = { success: '✅', error: '❌', info: 'ℹ️' };
    toast.innerHTML = `<span>${icons[type] || 'ℹ️'}</span><span>${message}</span>`;

    container.appendChild(toast);

    setTimeout(() => {
        toast.style.animation = 'toastOut 0.4s ease forwards';
        setTimeout(() => toast.remove(), 400);
    }, duration);
}

function getConfThreshold() {
    return parseInt(confSlider.value) / 100;
}

// ── Confidence Slider ────────────────────────────────────────
confSlider.addEventListener('input', () => {
    const val = (parseInt(confSlider.value) / 100).toFixed(2);
    confValue.textContent = val;
});

// ── Server Health Check ──────────────────────────────────────
async function checkHealth() {
    try {
        const res = await fetch('/health');
        if (res.ok) {
            statusDot.className = 'status-dot online';
            statusDot.querySelector('.status-text').textContent = 'Online';
        } else {
            throw new Error();
        }
    } catch {
        statusDot.className = 'status-dot offline';
        statusDot.querySelector('.status-text').textContent = 'Offline';
    }
}

checkHealth();
setInterval(checkHealth, 15000);

// ── Drag & Drop ──────────────────────────────────────────────

uploadZone.addEventListener('click', () => fileInput.click());

uploadZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadZone.classList.add('drag-over');
});

uploadZone.addEventListener('dragleave', () => {
    uploadZone.classList.remove('drag-over');
});

uploadZone.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadZone.classList.remove('drag-over');
    const files = e.dataTransfer.files;
    if (files.length > 0) handleFileSelection(files[0]);
});

fileInput.addEventListener('change', () => {
    if (fileInput.files.length > 0) handleFileSelection(fileInput.files[0]);
});

// ── File Selection ───────────────────────────────────────────

function handleFileSelection(file) {
    // Validate file type
    const validTypes = [
        'image/jpeg', 'image/png',
        'video/mp4', 'video/avi', 'video/quicktime', 'video/x-msvideo'
    ];

    if (!validTypes.includes(file.type)) {
        showToast('Unsupported file type. Please upload JPG, PNG, MP4, or AVI.', 'error');
        return;
    }

    selectedFile = file;

    // Update preview
    fileName.textContent = file.name;
    fileSize.textContent = formatFileSize(file.size);
    fileTypeIcon.textContent = isImage(file) ? '🖼️' : '🎬';

    // Show preview, hide upload zone
    uploadZone.style.display = 'none';
    filePreview.style.display = 'flex';

    // Hide any previous results
    resultsSection.style.display = 'none';
}

// ── Clear Selection ──────────────────────────────────────────

clearBtn.addEventListener('click', resetUpload);

function resetUpload() {
    selectedFile = null;
    fileInput.value = '';
    uploadZone.style.display = 'flex';
    filePreview.style.display = 'none';
    resultsSection.style.display = 'none';
    imageResult.style.display = 'none';
    videoResult.style.display = 'none';
}

// ── New Upload Button ────────────────────────────────────────

newUploadBtn.addEventListener('click', resetUpload);

// ── Analyze Button ───────────────────────────────────────────

analyzeBtn.addEventListener('click', async () => {
    if (!selectedFile) {
        showToast('No file selected.', 'error');
        return;
    }

    const conf = getConfThreshold();

    if (isImage(selectedFile)) {
        await analyzeImage(selectedFile, conf);
    } else if (isVideo(selectedFile)) {
        await analyzeVideo(selectedFile, conf);
    }
});

// ── Image Analysis ───────────────────────────────────────────

async function analyzeImage(file, confThresh) {
    showLoading('Analyzing image with YOLOv11...');

    try {
        const formData = new FormData();
        formData.append('file', file);

        const res = await fetch(`/predict?conf_thresh=${confThresh}`, {
            method: 'POST',
            body: formData,
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.error || 'Prediction failed');
        }

        // Read metadata from headers
        const detections = res.headers.get('X-Detections') || '0';
        const processingTime = res.headers.get('X-Processing-Time') || '0ms';

        // Read image blob
        const blob = await res.blob();
        const imageUrl = URL.createObjectURL(blob);

        // Display results
        hideLoading();
        showImageResult(imageUrl, detections, processingTime, confThresh);

        // Set download link
        downloadBtn.href = imageUrl;
        downloadBtn.download = `polypvision_${file.name}`;

        showToast(`Analysis complete! Found ${detections} polyp(s).`, 'success');

    } catch (err) {
        hideLoading();
        showToast(`Error: ${err.message}`, 'error');
    }
}

// ── Video Analysis ───────────────────────────────────────────

async function analyzeVideo(file, confThresh) {
    showLoading('Processing video frame by frame...');
    loadingHint.textContent = 'This may take a few minutes for longer videos';

    try {
        const formData = new FormData();
        formData.append('file', file);

        const res = await fetch(`/predict-video?conf_thresh=${confThresh}`, {
            method: 'POST',
            body: formData,
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.error || 'Video processing failed');
        }

        // Read metadata from headers
        const detections = res.headers.get('X-Detections') || '0';
        const frames = res.headers.get('X-Frames') || '0';
        const processingTime = res.headers.get('X-Processing-Time') || '0ms';

        // Read video blob
        const blob = await res.blob();
        const videoUrl = URL.createObjectURL(blob);

        // Display results
        hideLoading();
        showVideoResult(videoUrl, detections, processingTime, confThresh, frames);

        // Set download link
        downloadBtn.href = videoUrl;
        downloadBtn.download = `polypvision_${file.name}`;

        showToast(`Video processed! ${frames} frames analyzed, ${detections} total detections.`, 'success');

    } catch (err) {
        hideLoading();
        showToast(`Error: ${err.message}`, 'error');
    }
}

// ── Display Helpers ──────────────────────────────────────────

function showLoading(hint) {
    loadingHint.textContent = hint;
    loadingOverlay.style.display = 'flex';
}

function hideLoading() {
    loadingOverlay.style.display = 'none';
}

function showImageResult(imageUrl, detections, time, conf) {
    resultImage.src = imageUrl;
    imageResult.style.display = 'block';
    videoResult.style.display = 'none';

    updateStats(detections, time, conf);
    resultsSection.style.display = 'block';

    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function showVideoResult(videoUrl, detections, time, conf, frames) {
    resultVideo.src = videoUrl;
    videoResult.style.display = 'block';
    imageResult.style.display = 'none';

    updateStats(`${detections} (${frames} frames)`, time, conf);
    resultsSection.style.display = 'block';

    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function updateStats(detections, time, conf) {
    statDetections.textContent = detections;
    statTime.textContent = time;
    statConf.textContent = parseFloat(conf).toFixed(2);
}
