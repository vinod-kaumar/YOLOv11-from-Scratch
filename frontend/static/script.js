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
const reportBtn      = document.getElementById('report-btn');
const explainBtn     = document.getElementById('explain-btn');
const newUploadBtn   = document.getElementById('new-upload-btn');
const liveWebcamBtn  = document.getElementById('live-webcam-btn');
const webcamCard     = document.getElementById('webcam-card');
const webcamVideo    = document.getElementById('webcam-video');
const webcamOverlay  = document.getElementById('webcam-overlay');
const stopWebcamBtn  = document.getElementById('stop-webcam-btn');
const confSlider     = document.getElementById('conf-slider');
const confValue      = document.getElementById('conf-value');
const statusDot      = document.getElementById('status-dot');
const statDetections = document.getElementById('stat-detections');
const statTime       = document.getElementById('stat-time');
const statConf       = document.getElementById('stat-conf');

// ── State ────────────────────────────────────────────────────
let selectedFile = null;
let lastDetections = null;
let lastProcessingTime = null;
let isLiveMode = false;
let socket = null;
let webcamStream = null;

// ── Helpers ──────────────────────────────────────────────────

function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / 1048576).toFixed(1) + ' MB';
}

function isImage(file) {
    return file && file.type.startsWith('image/');
}

function isVideo(file) {
    return file && file.type.startsWith('video/');
}

function showToast(message, type = 'info', duration = 4000) {
    const container = document.getElementById('toast-container');
    if (!container) return;
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
    const validTypes = [
        'image/jpeg', 'image/png',
        'video/mp4', 'video/avi', 'video/quicktime', 'video/x-msvideo'
    ];

    if (!validTypes.includes(file.type)) {
        showToast('Unsupported file type. Please upload JPG, PNG, MP4, or AVI.', 'error');
        return;
    }

    selectedFile = file;
    fileName.textContent = file.name;
    fileSize.textContent = formatFileSize(file.size);
    fileTypeIcon.textContent = isImage(file) ? '🖼️' : '🎬';

    uploadZone.style.display = 'none';
    filePreview.style.display = 'flex';
    resultsSection.style.display = 'none';
    webcamCard.style.display = 'none';
    stopLiveMode();
}

// ── Reset ────────────────────────────────────────────────────

function resetUpload() {
    selectedFile = null;
    fileInput.value = '';
    uploadZone.style.display = 'flex';
    filePreview.style.display = 'none';
    resultsSection.style.display = 'none';
    imageResult.style.display = 'none';
    videoResult.style.display = 'none';
    webcamCard.style.display = 'none';
    stopLiveMode();
}

clearBtn.addEventListener('click', resetUpload);
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

        const detectionsCount = res.headers.get('X-Detections') || '0';
        const processingTime = res.headers.get('X-Processing-Time') || '0ms';
        const detectionsJson = res.headers.get('X-Detections-JSON');

        lastDetections = detectionsJson ? JSON.parse(detectionsJson) : [];
        lastProcessingTime = processingTime;

        const blob = await res.blob();
        const imageUrl = URL.createObjectURL(blob);

        hideLoading();
        showImageResult(imageUrl, detectionsCount, processingTime, confThresh);

        downloadBtn.href = imageUrl;
        downloadBtn.download = `polypvision_${file.name}`;
        
        explainBtn.style.display = 'flex';
        reportBtn.style.display = 'flex';

        showToast(`Analysis complete! Found ${detectionsCount} polyp(s).`, 'success');

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

        const detections = res.headers.get('X-Detections') || '0';
        const frames = res.headers.get('X-Frames') || '0';
        const processingTime = res.headers.get('X-Processing-Time') || '0ms';

        const blob = await res.blob();
        const videoUrl = URL.createObjectURL(blob);

        hideLoading();
        showVideoResult(videoUrl, detections, processingTime, confThresh, frames);

        downloadBtn.href = videoUrl;
        downloadBtn.download = `polypvision_${file.name}`;
        
        explainBtn.style.display = 'none';
        reportBtn.style.display = 'none';

        showToast(`Video processed! ${frames} frames analyzed.`, 'success');

    } catch (err) {
        hideLoading();
        showToast(`Error: ${err.message}`, 'error');
    }
}

// ── Report Generation ────────────────────────────────────────

reportBtn.addEventListener('click', async () => {
    if (!lastDetections) {
        showToast('Please analyze an image first to generate a report.', 'info');
        return;
    }
    
    try {
        const params = new URLSearchParams({
            detections_json: JSON.stringify(lastDetections),
            processing_time: lastProcessingTime
        });
        
        window.open(`/generate-report?${params.toString()}`, '_blank');
        showToast('Generating clinical report...', 'success');
    } catch (err) {
        showToast('Failed to generate report.', 'error');
    }
});

// ── Explainability (Grad-CAM) ────────────────────────────────

explainBtn.addEventListener('click', async () => {
    if (!selectedFile || !isImage(selectedFile)) {
        showToast('Explainability is only available for images.', 'info');
        return;
    }
    
    if (!lastDetections) {
        showToast('Please analyze the image first.', 'info');
        return;
    }

    showLoading('Generating Grad-CAM heatmap...');
    
    try {
        const formData = new FormData();
        formData.append('file', selectedFile);
        
        const res = await fetch('/explain', {
            method: 'POST',
            body: formData
        });
        
        if (!res.ok) throw new Error('Explainability failed');
        
        const blob = await res.blob();
        const imageUrl = URL.createObjectURL(blob);
        
        resultImage.src = imageUrl;
        showToast('Heatmap generated! Red areas show model focus.', 'info');
    } catch (err) {
        showToast('Failed to generate heatmap.', 'error');
    } finally {
        hideLoading();
    }
});

// ── Live Webcam Mode (WebSocket) ─────────────────────────────

liveWebcamBtn.addEventListener('click', startLiveMode);
stopWebcamBtn.addEventListener('click', stopLiveMode);

async function startLiveMode() {
    console.log("[Live] Starting Live Mode...");
    try {
        const constraints = { 
            video: { 
                width: { ideal: 640 }, 
                height: { ideal: 640 },
                facingMode: "environment" 
            } 
        };
        webcamStream = await navigator.mediaDevices.getUserMedia(constraints);
        webcamVideo.srcObject = webcamStream;
        
        webcamVideo.onloadedmetadata = () => {
            console.log(`[Live] Video metadata loaded: ${webcamVideo.videoWidth}x${webcamVideo.videoHeight}`);
            webcamVideo.play();
        };

        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws`;
        console.log(`[Live] Connecting to WebSocket: ${wsUrl}`);
        
        socket = new WebSocket(wsUrl);
        
        socket.onopen = () => {
            console.log("[Live] WebSocket connected!");
            isLiveMode = true;
            webcamCard.style.display = 'block';
            uploadCard.style.display = 'none';
            resultsSection.style.display = 'none';
            showToast('Live Mode Started', 'success');
            processLiveFrame();
        };
        
        socket.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                drawWebcamDetections(data.detections);
            } catch (e) {
                console.error("[Live] Error parsing message:", e);
            }
        };
        
        socket.onerror = (err) => {
            console.error("[Live] WebSocket Error:", err);
            showToast('Connection failed. Is the server running?', 'error');
        };

        socket.onclose = (e) => {
            console.log(`[Live] WebSocket closed: ${e.code} ${e.reason}`);
            if (isLiveMode) stopLiveMode();
        };
        
    } catch (err) {
        console.error("[Live] Camera Error:", err);
        showToast(`Camera Error: ${err.message}. Please allow camera access.`, 'error');
    }
}

function stopLiveMode() {
    isLiveMode = false;
    if (webcamStream) {
        webcamStream.getTracks().forEach(track => track.stop());
        webcamStream = null;
    }
    if (socket) {
        socket.close();
        socket = null;
    }
    webcamCard.style.display = 'none';
    uploadCard.style.display = 'block';
}

async function processLiveFrame() {
    if (!isLiveMode) return;
    
    if (webcamVideo.readyState < 2 || webcamVideo.videoWidth === 0) {
        setTimeout(processLiveFrame, 200);
        return;
    }
    
    const canvas = document.createElement('canvas');
    canvas.width = webcamVideo.videoWidth;
    canvas.height = webcamVideo.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(webcamVideo, 0, 0);
    
    canvas.toBlob((blob) => {
        if (socket && socket.readyState === WebSocket.OPEN) {
            socket.send(blob);
        }
        setTimeout(processLiveFrame, 100);
    }, 'image/jpeg', 0.7);
}

function drawWebcamDetections(detections) {
    const ctx = webcamOverlay.getContext('2d');
    
    const displayWidth = webcamVideo.clientWidth;
    const displayHeight = webcamVideo.clientHeight;
    
    if (webcamOverlay.width !== displayWidth || webcamOverlay.height !== displayHeight) {
        webcamOverlay.width = displayWidth;
        webcamOverlay.height = displayHeight;
    }
    
    ctx.clearRect(0, 0, webcamOverlay.width, webcamOverlay.height);
    
    const scaleX = displayWidth / webcamVideo.videoWidth;
    const scaleY = displayHeight / webcamVideo.videoHeight;
    
    ctx.strokeStyle = '#00c864';
    ctx.lineWidth = 3;
    ctx.font = 'bold 16px Inter';
    ctx.fillStyle = '#00c864';
    
    detections.forEach(det => {
        const x1 = det.x1 * scaleX;
        const y1 = det.y1 * scaleY;
        const x2 = det.x2 * scaleX;
        const y2 = det.y2 * scaleY;
        const w = x2 - x1;
        const h = y2 - y1;
        
        ctx.strokeRect(x1, y1, w, h);
        ctx.fillText(`Polyp ${Math.round(det.score * 100)}%`, x1, y1 > 20 ? y1 - 5 : y1 + 15);
    });
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
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function showVideoResult(videoUrl, detections, time, conf, frames) {
    resultVideo.src = videoUrl;
    videoResult.style.display = 'block';
    imageResult.style.display = 'none';
    updateStats(`${detections} (${frames} frames)`, time, conf);
    resultsSection.style.display = 'block';
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function updateStats(detections, time, conf) {
    statDetections.textContent = detections;
    statTime.textContent = time;
    statConf.textContent = parseFloat(conf).toFixed(2);
}
