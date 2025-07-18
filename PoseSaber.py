import cv2
import mediapipe as mp
import pygame
import threading
import numpy as np
import time
import random
from collections import deque

# =====================
# Pose Detection Setup
# =====================
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
cap = cv2.VideoCapture(0)

# Smoothed wrist positions
left_wrist = [0, 0]
right_wrist = [0, 0]
prev_left = [0, 0]
prev_right = [0, 0]
alpha = 0.7  # smoothing factor

# History for direction detection
history_length = 5
left_history = deque(maxlen=history_length)
right_history = deque(maxlen=history_length)

# =====================
# Pygame Setup
# =====================
pygame.init()
WIDTH, HEIGHT = 640, 480
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Pose Saber")
clock = pygame.time.Clock()

font = pygame.font.SysFont(None, 40)
score = 0

class Note:
    def __init__(self):
        self.x = random.randint(100, WIDTH - 100)
        self.y = -50
        self.speed = 5
        self.hit = False
        self.required_direction = random.choice(['left', 'right'])  # For slash direction

    def update(self):
        self.y += self.speed

    def draw(self):
        color = (0, 255, 255) if not self.hit else (128, 128, 128)
        pygame.draw.rect(screen, color, (self.x, self.y, 50, 50))
        arrow = "<" if self.required_direction == "left" else ">"
        arrow_text = font.render(arrow, True, (255, 0, 0))
        screen.blit(arrow_text, (self.x + 15, self.y + 10))

    def check_hit(self, wrist, direction):
        wx, wy = wrist
        if self.hit:
            return False
        if self.x < wx < self.x + 50 and self.y < wy < self.y + 50:
            if direction == self.required_direction:
                self.hit = True
                return True
        return False

notes = [Note()]

# =====================
# Webcam Thread
# =====================
def webcam_thread():
    global left_wrist, right_wrist, prev_left, prev_right
    while True:
        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)  # selfie mode
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if results.pose_landmarks:
            l_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
            r_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]

            # Normalize and clip
            lx = int(min(max(l_wrist.x, 0), 1) * WIDTH)
            ly = int(min(max(l_wrist.y, 0), 1) * HEIGHT)
            rx = int(min(max(r_wrist.x, 0), 1) * WIDTH)
            ry = int(min(max(r_wrist.y, 0), 1) * HEIGHT)

            # Smooth
            left_wrist[0] = int(alpha * lx + (1 - alpha) * prev_left[0])
            left_wrist[1] = int(alpha * ly + (1 - alpha) * prev_left[1])
            right_wrist[0] = int(alpha * rx + (1 - alpha) * prev_right[0])
            right_wrist[1] = int(alpha * ry + (1 - alpha) * prev_right[1])

            prev_left = left_wrist[:]
            prev_right = right_wrist[:]

            # Update history
            left_history.append(left_wrist[0])
            right_history.append(right_wrist[0])

        # Optional webcam preview
        cv2.imshow("Webcam", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

threading.Thread(target=webcam_thread, daemon=True).start()

# =====================
# Game Loop
# =====================
running = True
spawn_timer = 0

def get_direction(history):
    if len(history) < 2:
        return None
    delta = history[-1] - history[0]
    if abs(delta) < 30:
        return None
    return 'right' if delta > 0 else 'left'

while running:
    screen.fill((0, 0, 0))
    clock.tick(30)
    spawn_timer += 1

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Spawn notes
    if spawn_timer > 40:
        notes.append(Note())
        spawn_timer = 0

    # Detect slash direction
    left_dir = get_direction(left_history)
    right_dir = get_direction(right_history)

    # Update and draw notes
    for note in notes:
        note.update()
        if not note.hit and note.check_hit(left_wrist, left_dir):
            score += 1
        if not note.hit and note.check_hit(right_wrist, right_dir):
            score += 1
        note.draw()

    # Draw cursors
    pygame.draw.circle(screen, (0, 255, 0), (int(left_wrist[0]), int(left_wrist[1])), 15)
    pygame.draw.circle(screen, (255, 0, 0), (int(right_wrist[0]), int(right_wrist[1])), 15)

    # Draw score
    score_text = font.render(f"Score: {score}", True, (255, 255, 255))
    screen.blit(score_text, (10, 10))

    pygame.display.flip()

pygame.quit()
cap.release()
cv2.destroyAllWindows()
