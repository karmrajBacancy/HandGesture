/**
 * Hand Gesture Camera - Real-time hand detection and drawing
 * Uses MediaPipe Hand Landmarker for detection
 */

// Hand landmark connections (which points connect to form the hand skeleton)
const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],       // Thumb
  [0, 5], [5, 6], [6, 7], [7, 8],       // Index finger
  [0, 9], [9, 10], [10, 11], [11, 12],  // Middle finger
  [0, 13], [13, 14], [14, 15], [15, 16], // Ring finger
  [0, 17], [17, 18], [18, 19], [19, 20], // Pinky
  [5, 9], [9, 13], [13, 17]              // Palm
];

let handLandmarker = null;
let video = null;
let canvas = null;
let ctx = null;
let isRunning = false;
let lastVideoTime = -1;
let animationId = null;

// Whiteboard - uses full viewport, resized on init and resize
let whiteboard = null;
let whiteboardCtx = null;
let cursorCanvas = null;
let cursorCtx = null;
const DRAW_THRESHOLD = 0.04;  // Tighter pinch for more precise drawing
let lastDrawX = null;
let lastDrawY = null;
let isDrawing = false;
// Smoothing for steadier lines
let smoothedX = null;
let smoothedY = null;
const SMOOTHING = 0.35;  // Lower = smoother but more lag

// DOM elements
const startBtn = document.getElementById('startBtn');
const statusEl = document.getElementById('status');
const clearBtn = document.getElementById('clearBtn');

/**
 * Initialize the Hand Landmarker model
 */
async function initHandLandmarker() {
  try {
    const { HandLandmarker, FilesetResolver } = await import(
      'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs'
    );

    const vision = await FilesetResolver.forVisionTasks(
      'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm'
    );

    handLandmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
        delegate: 'GPU'
      },
      runningMode: 'VIDEO',
      numHands: 2,
      minHandDetectionConfidence: 0.8,
      minHandPresenceConfidence: 0.8,
      minTrackingConfidence: 0.8
    });

    statusEl.textContent = 'Model loaded! Click Start Camera';
    statusEl.className = 'status ready';
    startBtn.disabled = false;
  } catch (err) {
    console.error('Failed to load hand landmarker:', err);
    statusEl.textContent = 'Failed to load model. Check console.';
    statusEl.style.color = '#ff6b6b';
  }
}

/**
 * Draw connections between hand landmarks
 */
function drawConnectors(landmarks, color = '#00ff88', lineWidth = 2) {
  for (const [start, end] of HAND_CONNECTIONS) {
    const startPoint = landmarks[start];
    const endPoint = landmarks[end];
    if (startPoint && endPoint) {
      ctx.beginPath();
      ctx.moveTo(startPoint.x * canvas.width, startPoint.y * canvas.height);
      ctx.lineTo(endPoint.x * canvas.width, endPoint.y * canvas.height);
      ctx.strokeStyle = color;
      ctx.lineWidth = lineWidth;
      ctx.stroke();
    }
  }
}

/**
 * Calculate distance between two landmarks (normalized 0-1)
 */
function landmarkDistance(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y, (a.z || 0) - (b.z || 0));
}

/**
 * Map hand landmark (0-1) to whiteboard coordinates
 * Mirror X so drawing feels natural (matches camera preview)
 */
function landmarkToWhiteboard(landmark) {
  if (!whiteboard) return null;
  const x = (1 - landmark.x) * whiteboard.width;
  const y = landmark.y * whiteboard.height;
  return { x, y };
}

/**
 * Draw on whiteboard from hand gesture (index finger tip when thumb+index pinched)
 */
function updateWhiteboardDraw(landmarks) {
  if (!whiteboardCtx || !whiteboard) return;

  const indexTip = landmarks[8];
  const thumbTip = landmarks[4];
  if (!indexTip || !thumbTip) {
    isDrawing = false;
    lastDrawX = null;
    lastDrawY = null;
    return;
  }

  const pinchDist = landmarkDistance(thumbTip, indexTip);
  const shouldDraw = pinchDist < DRAW_THRESHOLD;
  const pos = landmarkToWhiteboard(indexTip);

  // Smooth position for steadier lines
  if (smoothedX === null) smoothedX = pos.x;
  if (smoothedY === null) smoothedY = pos.y;
  smoothedX = smoothedX + (pos.x - smoothedX) * (1 - SMOOTHING);
  smoothedY = smoothedY + (pos.y - smoothedY) * (1 - SMOOTHING);
  const drawX = smoothedX;
  const drawY = smoothedY;

  if (shouldDraw) {
    if (lastDrawX !== null && lastDrawY !== null && isDrawing) {
      whiteboardCtx.beginPath();
      whiteboardCtx.moveTo(lastDrawX, lastDrawY);
      whiteboardCtx.lineTo(drawX, drawY);
      whiteboardCtx.strokeStyle = '#1a1a2e';
      whiteboardCtx.lineWidth = 2;
      whiteboardCtx.lineCap = 'round';
      whiteboardCtx.lineJoin = 'round';
      whiteboardCtx.stroke();
    }
    lastDrawX = drawX;
    lastDrawY = drawY;
    isDrawing = true;
  } else {
    lastDrawX = drawX;
    lastDrawY = drawY;
    isDrawing = false;
  }
}

/**
 * Draw pen cursor on whiteboard overlay
 */
function drawPenCursor(x, y, drawing) {
  if (!cursorCtx || !cursorCanvas) return;
  cursorCtx.clearRect(0, 0, cursorCanvas.width, cursorCanvas.height);
  if (x === null || y === null) return;

  const size = 14;
  cursorCtx.save();
  cursorCtx.translate(x, y);
  cursorCtx.rotate(-0.35);  // Slight tilt like a pen

  // Pen body (rounded rect)
  cursorCtx.fillStyle = drawing ? '#1a1a2e' : '#00d9ff';
  cursorCtx.strokeStyle = drawing ? '#00ff88' : '#0088aa';
  cursorCtx.lineWidth = 2;
  cursorCtx.beginPath();
  const r = 4;
  cursorCtx.moveTo(-size/2 + r, -size*1.2);
  cursorCtx.lineTo(size/2 - r, -size*1.2);
  cursorCtx.quadraticCurveTo(size/2, -size*1.2, size/2, -size*1.2 + r);
  cursorCtx.lineTo(size/2, size*0.6 - r);
  cursorCtx.quadraticCurveTo(size/2, size*0.6, size/2 - r, size*0.6);
  cursorCtx.lineTo(-size/2 + r, size*0.6);
  cursorCtx.quadraticCurveTo(-size/2, size*0.6, -size/2, size*0.6 - r);
  cursorCtx.lineTo(-size/2, -size*1.2 + r);
  cursorCtx.quadraticCurveTo(-size/2, -size*1.2, -size/2 + r, -size*1.2);
  cursorCtx.closePath();
  cursorCtx.fill();
  cursorCtx.stroke();

  // Pen tip (circle)
  cursorCtx.fillStyle = drawing ? '#1a1a2e' : '#00d9ff';
  cursorCtx.beginPath();
  cursorCtx.arc(0, size*0.9, size/2.5, 0, Math.PI * 2);
  cursorCtx.fill();
  cursorCtx.strokeStyle = '#fff';
  cursorCtx.lineWidth = 1.5;
  cursorCtx.stroke();

  cursorCtx.restore();
}

/**
 * Draw landmark points
 */
function drawLandmarks(landmarks, color = '#00d9ff', radius = 4) {
  for (const landmark of landmarks) {
    ctx.beginPath();
    ctx.arc(
      landmark.x * canvas.width,
      landmark.y * canvas.height,
      radius,
      0,
      2 * Math.PI
    );
    ctx.fillStyle = color;
    ctx.fill();
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 1;
    ctx.stroke();
  }
}

/**
 * Process video frame and draw hand gestures
 */
function detectAndDraw() {
  if (!isRunning || !handLandmarker || !video.videoWidth) {
    return;
  }

  // Resize canvas to match video
  if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
  }

  const startTimeMs = performance.now();
  let results = null;

  if (lastVideoTime !== video.currentTime) {
    lastVideoTime = video.currentTime;
    results = handLandmarker.detectForVideo(video, startTimeMs);
  }

  // Clear canvas
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Draw detected hands and update whiteboard
  if (results && results.landmarks && results.landmarks.length > 0) {
    const colors = ['#00ff88', '#00d9ff'];
    results.landmarks.forEach((landmarks, i) => {
      if (i === 0) updateWhiteboardDraw(landmarks);  // Use first hand for whiteboard
      const color = colors[i % colors.length];
      drawConnectors(landmarks, color, 3);
      drawLandmarks(landmarks, color, 5);
    });
    drawPenCursor(smoothedX, smoothedY, isDrawing);
  } else {
    lastDrawX = null;
    lastDrawY = null;
    smoothedX = null;
    smoothedY = null;
    isDrawing = false;
    drawPenCursor(null, null, false);
  }

  animationId = requestAnimationFrame(detectAndDraw);
}

/**
 * Start camera and begin detection
 */
async function startCamera() {
  if (isRunning) {
    isRunning = false;
    cancelAnimationFrame(animationId);
    if (video.srcObject) {
      video.srcObject.getTracks().forEach(track => track.stop());
    }
    lastDrawX = null;
    lastDrawY = null;
    smoothedX = null;
    smoothedY = null;
    isDrawing = false;
    startBtn.textContent = 'Start Camera';
    return;
  }

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 1920, height: 1080, facingMode: 'user' }
    });

    video = document.getElementById('video');
    canvas = document.getElementById('canvas');
    ctx = canvas.getContext('2d');

    video.srcObject = stream;
    video.onloadedmetadata = () => {
      video.play();
      isRunning = true;
      startBtn.textContent = 'Stop Camera';
      lastVideoTime = -1;
      detectAndDraw();
    };
  } catch (err) {
    console.error('Camera access denied:', err);
    statusEl.textContent = 'Camera access denied. Please allow camera permission.';
    statusEl.style.color = '#ff6b6b';
  }
}

/**
 * Resize whiteboard to fill its container (full screen)
 */
function resizeWhiteboard() {
  whiteboard = document.getElementById('whiteboard');
  cursorCanvas = document.getElementById('cursorCanvas');
  if (!whiteboard) return;
  const rect = whiteboard.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const w = Math.floor(rect.width * dpr);
  const h = Math.floor(rect.height * dpr);
  if (whiteboard.width !== w || whiteboard.height !== h) {
    whiteboard.width = w;
    whiteboard.height = h;
    whiteboardCtx = whiteboard.getContext('2d');
    whiteboardCtx.fillStyle = '#ffffff';
    whiteboardCtx.fillRect(0, 0, whiteboard.width, whiteboard.height);
  }
  if (cursorCanvas) {
    cursorCanvas.width = whiteboard.width;
    cursorCanvas.height = whiteboard.height;
    cursorCtx = cursorCanvas.getContext('2d');
  }
}

/**
 * Initialize whiteboard canvas
 */
function initWhiteboard() {
  whiteboard = document.getElementById('whiteboard');
  cursorCanvas = document.getElementById('cursorCanvas');
  if (!whiteboard) return;
  resizeWhiteboard();
  whiteboardCtx = whiteboard.getContext('2d');
  whiteboardCtx.fillStyle = '#ffffff';
  whiteboardCtx.fillRect(0, 0, whiteboard.width, whiteboard.height);
  if (cursorCanvas) {
    cursorCtx = cursorCanvas.getContext('2d');
  }
  window.addEventListener('resize', resizeWhiteboard);
}

/**
 * Clear whiteboard
 */
function clearWhiteboard() {
  if (!whiteboardCtx || !whiteboard) return;
  whiteboardCtx.fillStyle = '#ffffff';
  whiteboardCtx.fillRect(0, 0, whiteboard.width, whiteboard.height);
  lastDrawX = null;
  lastDrawY = null;
  smoothedX = null;
  smoothedY = null;
  isDrawing = false;
}

// Event listeners
startBtn.addEventListener('click', startCamera);
clearBtn.addEventListener('click', clearWhiteboard);

// Initialize whiteboard on load (before camera starts)
initWhiteboard();
initHandLandmarker();
