# Pose Saber – AI Coding Assistant Prompt

## Project Summary
Pose Saber is a webcam-based rhythm game inspired by Beat Saber. It uses real-time pose estimation from a front-facing webcam to track wrist movements, allowing players to slice incoming blocks using hand gestures.

The project is implemented in Python using:
- [MediaPipe](https://google.github.io/mediapipe/) for real-time body pose tracking.
- [OpenCV](https://opencv.org/) for webcam input.
- [Pygame](https://www.pygame.org/) for game rendering and logic.

The game currently features:
- Real-time tracking of left and right wrists with smoothing and clamping.
- Directional slash detection (left and right).
- Falling notes requiring correct direction and position for a successful hit.
- Scoring system with Pygame UI.
- Multi-threaded pose tracking with gesture history buffers.

---

## How to Continue

This project uses Pygame for game rendering and MediaPipe (with OpenCV) for pose tracking from a front-facing webcam. When continuing development, follow these best practices derived from Pygame and MediaPipe communities:

### 🧠 General Development Principles
- Maintain modular architecture: input → processing → game logic → rendering.
- Target smooth 30 FPS+ gameplay.
- Keep pose detection logic decoupled from game rendering.
- Use configuration flags/constants for tunable values.

### 🎮 Pygame Best Practices
- Use `pygame.time.Clock.tick(FPS)` to cap frame rate.
- Optimize rendering: avoid redundant drawing and cache static surfaces.
- Leverage `pygame.sprite.Sprite` and `Group` for reusable game objects.
- Keep visual feedback snappy and non-blocking (e.g., hits, glows).

### 🤖 MediaPipe Pose Best Practices
- Use `selfie_mode=True` and mirror the webcam image (`cv2.flip(frame, 1)`).
- Clamp normalized coordinates to screen bounds.
- Smooth noisy keypoints with exponential smoothing (`alpha = 0.6–0.8`).
- Use history buffers (e.g., `deque`) for gesture detection like slashes or grabs.
- Run pose estimation on a separate thread to avoid blocking the game loop.
- Fall back to last-known good position if detection fails temporarily.

### 🧩 Integration Tips
- Ensure all in-game coordinates are mapped consistently from pose space.
- For new gestures (e.g., closed fist), use landmark distances to classify.
- Build in flexibility: allow some blocks to require direction and others to not.
- All rendering logic should remain frame-based and non-blocking.
- Minimize per-frame allocations to maintain steady frame rates.

### 🧱 Design Constraints
- Must run smoothly on a mid-range laptop (GTX 1650 Ti or equivalent).
- All inference is on-device — no cloud APIs or external ML services.
- Only use Python standard libraries, Pygame, MediaPipe, OpenCV, and NumPy.
- Code should remain portable and readable for future extension.

---

## 🔧 Next Tasks for Codex / GPT

1. 🖐️ **Closed Wrist Detection**  
   Use MediaPipe landmarks to detect a closed wrist (e.g., when the user is holding a sword). You may use relative distances between fingers and palm base to determine if the hand is closed.

2. 🗡️ **Draw a Sword Line**  
   When the wrist is closed, draw a line extending from the wrist to represent a sword. The sword should have a fixed length. The sword's direction should be estimated based on the pose of the wrist.

3. 🚀 **Block Perspective Zoom**  
   Change block behavior to move toward the player from the horizon (depth perspective), instead of falling from the top. Blocks should appear to grow in size as they approach the player.

4. 🔄 **Optional Direction Blocks**  
   Some blocks should not require a directional slash. They can be destroyed by any contact with the sword. Others should retain direction-based rules. Use a flag in the `Note` class to distinguish them.

5. 🧮 **Score Counter Enhancements**  
   Display the score on the top-left of the screen. Award **higher points** for directional blocks than for any-contact blocks (e.g., 10 vs 5).

