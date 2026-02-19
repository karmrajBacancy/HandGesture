# Hand Gesture Camera

A simple web app that detects your hand in real-time using your camera and draws the gesture skeleton on screen.

## Features

- **Real-time hand detection** using Google MediaPipe Hand Landmarker
- **21-point hand skeleton** — fingers, palm, and joints are tracked
- **Dual hand support** — detects up to 2 hands (green and cyan)
- **No installation** — runs entirely in the browser

## How to Run

The app must be served over HTTP (not opened as a file) for camera access and CORS to work.

### Option 1: Python (if installed)

```bash
cd handgesture
python3 -m http.server 8000
```

Then open **http://localhost:8000** in Chrome or Safari.

### Option 2: Node.js (npx)

```bash
cd handgesture
npx serve .
```

Then open the URL shown in the terminal (usually http://localhost:3000).

### Option 3: VS Code Live Server

Right-click `index.html` → "Open with Live Server"

## Usage

1. Wait for "Model loaded! Click Start Camera"
2. Click **Start Camera** and allow camera access when prompted
3. Hold your hand in front of the camera — the skeleton will be drawn in real-time
4. **Snap your fingers** (thumb + middle finger together) to trigger an action — a green flash and counter will confirm detection
5. Click **Stop Camera** when done

### Custom snap action

Add your own code to run when a snap is detected. In the browser console or in a `<script>` before `app.js`:

```javascript
window.onSnap = () => {
  console.log('Snap detected!');
  // Your action: toggle UI, play sound, send request, etc.
};
```

## Requirements

- Modern browser (Chrome or Safari recommended)
- Webcam
- HTTPS or localhost (required for `getUserMedia`)

## Tech Stack

- MediaPipe Tasks Vision (Hand Landmarker)
- Vanilla JavaScript
- HTML5 Canvas
