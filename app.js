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
let faceLandmarker = null;
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
const DRAW_THRESHOLD = 0.055;  // Thumb + index pinch to draw
let lastDrawX = null;
let lastDrawY = null;
let isDrawing = false;
let smoothedX = null;
let smoothedY = null;
const SMOOTHING_FAST = 0.5;    // When moving fast
const SMOOTHING_SLOW = 0.25;   // Stronger smoothing when slow - reduces fuzzy dots
const GAP_INTERPOLATE = 10;    // Add points when gap exceeds this (px)
const MIN_POINT_DIST = 4;      // Ignore points closer than this - filters jitter
let drawReleaseFrames = 0;     // Hysteresis: stay drawing for a few frames after pinch release

// Stroke history for 3D conversion
let strokeHistory = [];
let currentStroke = [];

// Smile detection
const SMILE_THRESHOLD = 0.5;
const SMILE_COOLDOWN_MS = 1500;
let lastSmilePhotoTime = 0;

// DOM elements
const startBtn = document.getElementById('startBtn');
const statusEl = document.getElementById('status');
const clearBtn = document.getElementById('clearBtn');

/**
 * Initialize the Hand Landmarker model
 */
async function initHandLandmarker() {
  try {
    const { HandLandmarker, FaceLandmarker, FilesetResolver } = await import(
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
      minHandDetectionConfidence: 0.75,
      minHandPresenceConfidence: 0.75,
      minTrackingConfidence: 0.75
    });

    const faceOpts = {
      baseOptions: {
        modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
        delegate: 'GPU'
      },
      runningMode: 'VIDEO',
      numFaces: 1,
      outputFaceBlendshapes: true
    };
    try {
      faceLandmarker = await FaceLandmarker.createFromOptions(vision, faceOpts);
    } catch (faceErr) {
      console.warn('Face Landmarker GPU failed, trying CPU:', faceErr);
      faceOpts.baseOptions.delegate = 'CPU';
      faceLandmarker = await FaceLandmarker.createFromOptions(vision, faceOpts);
    }

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
 * Mirror X so drawing feels natural, preserve aspect ratio with video
 */
function landmarkToWhiteboard(landmark) {
  if (!whiteboard) return null;
  const vw = (video && video.videoWidth) || 1920;
  const vh = (video && video.videoHeight) || 1080;
  const ww = whiteboard.width;
  const wh = whiteboard.height;
  const videoAspect = vw / vh;
  const boardAspect = ww / wh;
  let drawW = ww, drawH = wh, offsetX = 0, offsetY = 0;
  if (videoAspect > boardAspect) {
    drawH = ww / videoAspect;
    offsetY = (wh - drawH) / 2;
  } else {
    drawW = wh * videoAspect;
    offsetX = (ww - drawW) / 2;
  }
  const x = offsetX + (1 - landmark.x) * drawW;
  const y = offsetY + landmark.y * drawH;
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
  const pinchActive = pinchDist < DRAW_THRESHOLD;
  const pos = landmarkToWhiteboard(indexTip);
  if (!pos) return;

  // Stronger smoothing to reduce fuzzy dots
  if (smoothedX === null) smoothedX = pos.x;
  if (smoothedY === null) smoothedY = pos.y;
  const moveDist = Math.hypot(pos.x - smoothedX, pos.y - smoothedY);
  const smoothing = moveDist > 10 ? SMOOTHING_FAST : SMOOTHING_SLOW;
  smoothedX = smoothedX + (pos.x - smoothedX) * (1 - smoothing);
  smoothedY = smoothedY + (pos.y - smoothedY) * (1 - smoothing);
  const drawX = smoothedX;
  const drawY = smoothedY;

  // Hysteresis: stay in draw mode for 3 frames after pinch release (prevents losing drawer)
  if (pinchActive) {
    drawReleaseFrames = 0;
  } else {
    drawReleaseFrames++;
  }
  const shouldDraw = pinchActive || (isDrawing && drawReleaseFrames < 3);

  if (shouldDraw) {
    if (lastDrawX !== null && lastDrawY !== null && isDrawing) {
      const gap = Math.hypot(drawX - lastDrawX, drawY - lastDrawY);
      if (gap >= MIN_POINT_DIST) {
        const segments = Math.max(1, Math.ceil(gap / GAP_INTERPOLATE));
        for (let i = 1; i <= segments; i++) {
          const t = i / segments;
          const x = lastDrawX + (drawX - lastDrawX) * t;
          const y = lastDrawY + (drawY - lastDrawY) * t;
          whiteboardCtx.beginPath();
          whiteboardCtx.moveTo(lastDrawX, lastDrawY);
          whiteboardCtx.lineTo(x, y);
          whiteboardCtx.strokeStyle = '#1a1a2e';
          whiteboardCtx.lineWidth = 2;
          whiteboardCtx.lineCap = 'round';
          whiteboardCtx.lineJoin = 'round';
          whiteboardCtx.stroke();
          currentStroke.push({ x, y });
          lastDrawX = x;
          lastDrawY = y;
        }
      }
    } else {
      currentStroke = [{ x: lastDrawX ?? drawX, y: lastDrawY ?? drawY }, { x: drawX, y: drawY }];
      lastDrawX = drawX;
      lastDrawY = drawY;
    }
    isDrawing = true;
  } else {
    if (currentStroke.length > 1) {
      strokeHistory.push([...currentStroke]);
    }
    currentStroke = [];
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
 * Capture photo (camera + hand overlay) and download
 */
function capturePhoto() {
  if (!video || !video.videoWidth || !canvas) return;
  const tempCanvas = document.createElement('canvas');
  tempCanvas.width = video.videoWidth;
  tempCanvas.height = video.videoHeight;
  const tempCtx = tempCanvas.getContext('2d');
  tempCtx.translate(tempCanvas.width, 0);
  tempCtx.scale(-1, 1);
  tempCtx.drawImage(video, 0, 0);
  tempCtx.drawImage(canvas, 0, 0);
  tempCtx.setTransform(1, 0, 0, 1, 0, 0);
  const link = document.createElement('a');
  link.download = `smile_${Date.now()}.jpg`;
  link.href = tempCanvas.toDataURL('image/jpeg', 0.92);
  link.click();
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
  let faceResults = null;

  if (lastVideoTime !== video.currentTime) {
    lastVideoTime = video.currentTime;
    results = handLandmarker.detectForVideo(video, startTimeMs);
    if (faceLandmarker) {
      try {
        faceResults = faceLandmarker.detectForVideo(video, startTimeMs);
      } catch (e) {
        console.warn('Face detection error:', e);
      }
    }
  }

  // Smile detection - take photo when smiling (blendshapes or landmark fallback)
  let isSmiling = false;
  function doSmileCapture() {
    const now = performance.now();
    if (now - lastSmilePhotoTime > SMILE_COOLDOWN_MS) {
      lastSmilePhotoTime = now;
      capturePhoto();
      const el = document.getElementById('smileIndicator');
      if (el) { el.textContent = '😊 Photo!'; setTimeout(() => { el.textContent = ''; }, 1500); }
    }
  }
  if (faceResults?.faceBlendshapes?.length > 0) {
    const blendshapes = faceResults.faceBlendshapes[0];
    const smileCat = blendshapes.categories?.find(c => c.categoryName === 'mouthSmile');
    if (smileCat && smileCat.score >= SMILE_THRESHOLD) {
      isSmiling = true;
      doSmileCapture();
    }
  } else if (faceResults?.faceLandmarks?.length > 0) {
    const lm = faceResults.faceLandmarks[0];
    if (lm.length >= 292) {
      const left = lm[61], right = lm[291];
      const mouthW = Math.abs(right.x - left.x);
      if (mouthW > 0.12) {
        isSmiling = true;
        doSmileCapture();
      }
    }
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
    drawReleaseFrames = 0;
    drawPenCursor(null, null, false);
  }

  // Draw face landmarks when detected (visual feedback)
  const mx = (x) => (1 - x) * canvas.width;
  const my = (y) => y * canvas.height;
  const FACE_OVAL = [[10,338],[338,297],[297,332],[332,284],[284,251],[251,389],[389,356],[356,454],[454,323],[323,361],[361,288],[288,397],[397,365],[365,379],[379,378],[378,400],[400,377],[377,152],[152,148],[148,176],[176,149],[149,150],[150,136],[136,172],[172,58],[58,132],[132,93],[93,234],[234,127],[127,162],[162,21],[21,54],[54,103],[103,67],[67,109],[109,10]];
  const LIPS = [[61,146],[146,91],[91,181],[181,84],[84,17],[17,314],[314,405],[405,321],[321,375],[375,291],[61,185],[185,40],[40,39],[39,37],[37,0],[0,267],[267,269],[269,270],[270,409],[409,291],[78,95],[95,88],[88,178],[178,87],[87,14],[14,317],[317,402],[402,318],[318,324],[324,308],[78,191],[191,80],[80,81],[81,82],[82,13],[13,312],[312,311],[311,310],[310,415],[415,308]];
  if (faceResults?.faceLandmarks?.length > 0 && ctx) {
    const lm = faceResults.faceLandmarks[0];
    // Face oval
    ctx.strokeStyle = isSmiling ? 'rgba(0, 255, 136, 0.9)' : 'rgba(255, 200, 0, 0.8)';
    ctx.lineWidth = isSmiling ? 3 : 2;
    ctx.beginPath();
    const first = lm[FACE_OVAL[0][0]];
    if (first) ctx.moveTo(mx(first.x), my(first.y));
    for (const [, b] of FACE_OVAL) {
      const p = lm[b];
      if (p) ctx.lineTo(mx(p.x), my(p.y));
    }
    ctx.closePath();
    ctx.stroke();
    // Lips - highlight when smiling
    ctx.strokeStyle = isSmiling ? '#00ff88' : 'rgba(255, 150, 100, 0.6)';
    ctx.lineWidth = isSmiling ? 3 : 1.5;
    ctx.beginPath();
    for (const [a, b] of LIPS) {
      const pa = lm[a], pb = lm[b];
      if (pa && pb) {
        ctx.moveTo(mx(pa.x), my(pa.y));
        ctx.lineTo(mx(pb.x), my(pb.y));
      }
    }
    ctx.stroke();
  }
  // Smile indicator (live when smiling, "Photo!" overrides for 1.5s after capture)
  const smileEl = document.getElementById('smileIndicator');
  if (smileEl && (performance.now() - lastSmilePhotoTime > SMILE_COOLDOWN_MS)) {
    smileEl.textContent = isSmiling ? '😊 Smile!' : '';
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
    drawReleaseFrames = 0;
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
    strokeHistory = [];
    currentStroke = [];
    drawReleaseFrames = 0;
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
  drawReleaseFrames = 0;
  strokeHistory = [];
  currentStroke = [];
}

/**
 * Convert drawing to 3D and show in modal
 */
async function convertTo3D() {
  const strokes = [...strokeHistory];
  if (currentStroke.length > 1) strokes.push([...currentStroke]);
  if (strokes.length === 0) {
    alert('Draw something first, then click Convert to 3D');
    return;
  }

  const THREE = await import('https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js');
  const { OrbitControls } = await import('https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/controls/OrbitControls.js');
  const { Scene, PerspectiveCamera, WebGLRenderer, Mesh, MeshPhongMaterial, TubeGeometry, Vector3, AmbientLight, DirectionalLight, Color, Curve, Group, Box3 } = THREE;

  const modal = document.createElement('div');
  modal.id = 'view3dModal';
  modal.innerHTML = `
    <div class="view3d-overlay">
      <div class="view3d-header">
        <span>3D View — Drag to rotate • Scroll to zoom</span>
        <button id="close3dBtn">Close</button>
      </div>
      <div class="view3d-controls">
        <label>Scale <input type="range" id="scale3d" min="0.5" max="3" step="0.1" value="1"></label>
        <label>Rotate X <input type="range" id="rotX3d" min="0" max="360" step="5" value="0"><span>°</span></label>
        <label>Rotate Y <input type="range" id="rotY3d" min="0" max="360" step="5" value="0"><span>°</span></label>
        <label>Tube <input type="range" id="tube3d" min="0.5" max="3" step="0.1" value="1"></label>
      </div>
      <canvas id="view3dCanvas"></canvas>
    </div>
  `;
  modal.style.cssText = 'position:fixed;inset:0;z-index:1000;background:rgba(0,0,0,0.9);display:flex;align-items:center;justify-content:center;';
  document.body.appendChild(modal);

  const overlay = modal.querySelector('.view3d-overlay');
  overlay.style.cssText = 'width:100%;height:100%;position:relative;display:flex;flex-direction:column;';
  const header = modal.querySelector('.view3d-header');
  header.style.cssText = 'padding:1rem;display:flex;justify-content:space-between;align-items:center;background:#fff;color:#1a1a2e;border-bottom:1px solid #eee;';
  const controlsDiv = modal.querySelector('.view3d-controls');
  controlsDiv.style.cssText = 'padding:0.5rem 1rem;display:flex;flex-wrap:wrap;gap:1rem;align-items:center;background:#f8f8f8;color:#1a1a2e;font-size:0.8rem;border-bottom:1px solid #eee;';
  controlsDiv.querySelectorAll('label').forEach(l => {
    l.style.display = 'flex';
    l.style.alignItems = 'center';
    l.style.gap = '0.5rem';
    l.querySelector('input').style.width = '80px';
  });
  const canvas3d = modal.querySelector('#view3dCanvas');
  canvas3d.style.cssText = 'flex:1;width:100%;min-height:300px;';

  const scene = new Scene();
  scene.background = new Color(0xffffff);
  const camera = new PerspectiveCamera(50, canvas3d.clientWidth / canvas3d.clientHeight, 0.01, 1000);
  const renderer = new WebGLRenderer({ canvas: canvas3d, antialias: true });
  renderer.setSize(canvas3d.clientWidth, canvas3d.clientHeight);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.05;
  controls.enableZoom = true;
  controls.minDistance = 0.5;
  controls.maxDistance = 20;

  scene.add(new AmbientLight(0xffffff, 0.9));
  scene.add(new DirectionalLight(0xffffff, 0.8));
  scene.add(new DirectionalLight(0xcccccc, 0.4));

  const strokeGroup = new Group();
  scene.add(strokeGroup);

  const cx = whiteboard.width / 2;
  const cy = whiteboard.height / 2;
  const baseScale = 0.012;
  const baseTubeRadius = 0.025;
  const meshes = [];

  class SimpleCurve extends Curve {
    constructor(points) {
      super();
      this.points = points;
    }
    getPoint(t) {
      const pts = this.points;
      const i = (pts.length - 1) * t;
      const i0 = Math.min(Math.floor(i), pts.length - 2);
      const i1 = i0 + 1;
      const frac = i - i0;
      return new Vector3(
        pts[i0].x + (pts[i1].x - pts[i0].x) * frac,
        pts[i0].y + (pts[i1].y - pts[i0].y) * frac,
        pts[i0].z + (pts[i1].z - pts[i0].z) * frac
      );
    }
  }

  strokes.forEach(stroke => {
    const unique = stroke.filter((p, i) => i === 0 || (p.x !== stroke[i-1].x || p.y !== stroke[i-1].y));
    if (unique.length < 2) return;
    const totalLen = unique.reduce((a, p, i) => i ? a + Math.hypot(p.x - unique[i-1].x, p.y - unique[i-1].y) : 0, 0);
    if (totalLen < 15) return;
    const points = unique.map(p => new Vector3(
      (p.x - cx) * baseScale,
      -(p.y - cy) * baseScale,
      0
    ));
    try {
      const curve = new SimpleCurve(points);
      const geometry = new TubeGeometry(curve, 24, baseTubeRadius, 8, false);
      const material = new MeshPhongMaterial({
        color: 0x1a1a2e,
        shininess: 60,
        specular: 0x333333
      });
      const mesh = new Mesh(geometry, material);
      mesh.userData = { curve, baseRadius: baseTubeRadius };
      strokeGroup.add(mesh);
      meshes.push(mesh);
    } catch (e) {
      console.warn('Skip stroke:', e);
    }
  });

  const box = new Box3().setFromObject(strokeGroup);
  const center = box.getCenter(new Vector3());
  const size = box.getSize(new Vector3());
  const maxDim = Math.max(size.x, size.y, size.z, 0.5);
  const camDist = Math.max(maxDim * 2.5, 2);
  camera.position.set(center.x + camDist * 0.5, center.y + camDist * 0.3, center.z + camDist);
  camera.lookAt(center);
  controls.target.copy(center);

  function updateFinetune() {
    const scaleVal = parseFloat(document.getElementById('scale3d')?.value ?? 1);
    const rotX = (Math.PI / 180) * parseFloat(document.getElementById('rotX3d')?.value ?? 0);
    const rotY = (Math.PI / 180) * parseFloat(document.getElementById('rotY3d')?.value ?? 0);
    const tubeVal = parseFloat(document.getElementById('tube3d')?.value ?? 1);
    strokeGroup.scale.setScalar(scaleVal);
    strokeGroup.rotation.x = rotX;
    strokeGroup.rotation.y = rotY;
    meshes.forEach(mesh => {
      const { curve: path, baseRadius } = mesh.userData;
      if (path) {
        mesh.geometry.dispose();
        mesh.geometry = new TubeGeometry(path, 24, baseRadius * tubeVal, 8, false);
      }
    });
  }

  ['scale3d', 'rotX3d', 'rotY3d', 'tube3d'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.addEventListener('input', updateFinetune);
  });

  function animate() {
    if (!document.getElementById('view3dModal')) return;
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
  }
  animate();

  function onResize() {
    camera.aspect = canvas3d.clientWidth / canvas3d.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(canvas3d.clientWidth, canvas3d.clientHeight);
  }
  window.addEventListener('resize', onResize);

  modal.querySelector('#close3dBtn').style.cssText = 'padding:0.5rem 1rem;cursor:pointer;background:#1a1a2e;color:#fff;border:none;border-radius:6px;font-weight:600;';
  modal.querySelector('#close3dBtn').onclick = () => {
    window.removeEventListener('resize', onResize);
    renderer.dispose();
    modal.remove();
  };
}

// Event listeners
startBtn.addEventListener('click', startCamera);
clearBtn.addEventListener('click', clearWhiteboard);
document.getElementById('convert3dBtn').addEventListener('click', convertTo3D);

// Initialize whiteboard on load (before camera starts)
initWhiteboard();
initHandLandmarker();
