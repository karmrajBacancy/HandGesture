# Hand Gesture Camera

A web app that detects your hand in real-time and lets you draw on a whiteboard using hand gestures — no mouse or touch required.

## Features

- **Real-time hand detection** using Google MediaPipe Hand Landmarker
- **Gesture drawing** — pinch thumb + index finger to draw on the whiteboard
- **50-50 layout** — camera and whiteboard side by side
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

## How to Use

### 1. Start the app

1. Wait for **"Model loaded! Click Start Camera"**
2. Click **Start Camera**
3. Allow camera access when your browser asks for permission

### 2. Draw on the whiteboard

- **Pinch** your thumb and index finger together (like holding a pen) to draw
- **Release** the pinch to move your hand without drawing
- Point your index finger where you want to draw — the camera tracks it and maps it to the whiteboard

### 3. Clear the whiteboard

- Click **Clear Whiteboard** to erase everything and start over

### 4. Stop the camera

- Click **Stop Camera** when you're done

## Tips

- **Lighting** — Good lighting helps hand detection work better
- **Distance** — Keep your hand about 1–2 feet from the camera
- **Pinch tightly** — A firm pinch (thumb + index close together) gives cleaner lines
- **One hand** — The whiteboard uses the first detected hand for drawing

## Requirements

- Modern browser (Chrome or Safari recommended)
- Webcam
- HTTPS or localhost (required for `getUserMedia`)

## Tech Stack

- MediaPipe Tasks Vision (Hand Landmarker)
- Vanilla JavaScript
- HTML5 Canvas
