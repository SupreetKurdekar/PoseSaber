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
left_elbow = [0, 0]
right_elbow = [0, 0]
prev_left = [0, 0]
prev_right = [0, 0]
prev_left_elbow = [0, 0]
prev_right_elbow = [0, 0]

left_closed = False
right_closed = False
alpha = 0.7  # smoothing factor

# History for direction detection
history_length = 5
left_history = deque(maxlen=history_length)
right_history = deque(maxlen=history_length)

# Wrist closed detection
def is_hand_closed(results, left=True):
    if left:
        wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
        index = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_INDEX]
        thumb = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_THUMB]
        elbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
    else:
        wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
        index = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_INDEX]
        thumb = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_THUMB]
        elbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]

    d_index = np.hypot(wrist.x - index.x, wrist.y - index.y)
    d_thumb = np.hypot(wrist.x - thumb.x, wrist.y - thumb.y)
    arm_len = np.hypot(wrist.x - elbow.x, wrist.y - elbow.y)
    return (d_index + d_thumb) / 2 < 0.25 * arm_len

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
SWORD_LENGTH = 80
POINT_DIRECTIONAL = 10
POINT_ANY = 5

class Note:
    def __init__(self):
        self.x = random.randint(100, WIDTH - 100)
        self.z = 0.0
        self.speed = 0.02
        self.size = 50
        self.hit = False
        self.require_direction = random.choice([True, False])
        self.required_direction = random.choice(['left', 'right'])

    def update(self):
        self.z += self.speed
        scale = 0.3 + 0.7 * self.z
        self.size = int(50 * scale)
        self.y = int(100 + (HEIGHT - 150) * self.z)

    def draw(self):
        color = (0, 255, 255) if not self.hit else (128, 128, 128)
        rect = pygame.Rect(self.x - self.size // 2, self.y - self.size // 2,
                           self.size, self.size)
        pygame.draw.rect(screen, color, rect)
        if self.require_direction:
            arrow = "<" if self.required_direction == "left" else ">"
            arrow_text = font.render(arrow, True, (255, 0, 0))
            screen.blit(arrow_text, (self.x - arrow_text.get_width() // 2,
                                     self.y - arrow_text.get_height() // 2))

    def check_hit(self, wrist, direction):
        wx, wy = wrist
        half = self.size / 2
        if self.hit:
            return False
        if (self.x - half < wx < self.x + half and
                self.y - half < wy < self.y + half):
            if not self.require_direction or direction == self.required_direction:
                self.hit = True
                return True
        return False

notes = [Note()]

# =====================
# Webcam Thread
# =====================
def webcam_thread():
    global left_wrist, right_wrist, left_elbow, right_elbow
    global prev_left, prev_right, prev_left_elbow, prev_right_elbow
    global left_closed, right_closed
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
            l_elbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
            r_elbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]

            # Normalize and clip
            lx = int(min(max(l_wrist.x, 0), 1) * WIDTH)
            ly = int(min(max(l_wrist.y, 0), 1) * HEIGHT)
            rx = int(min(max(r_wrist.x, 0), 1) * WIDTH)
            ry = int(min(max(r_wrist.y, 0), 1) * HEIGHT)
            lex = int(min(max(l_elbow.x, 0), 1) * WIDTH)
            ley = int(min(max(l_elbow.y, 0), 1) * HEIGHT)
            rex = int(min(max(r_elbow.x, 0), 1) * WIDTH)
            rey = int(min(max(r_elbow.y, 0), 1) * HEIGHT)

            # Smooth
            left_wrist[0] = int(alpha * lx + (1 - alpha) * prev_left[0])
            left_wrist[1] = int(alpha * ly + (1 - alpha) * prev_left[1])
            right_wrist[0] = int(alpha * rx + (1 - alpha) * prev_right[0])
            right_wrist[1] = int(alpha * ry + (1 - alpha) * prev_right[1])
            left_elbow[0] = int(alpha * lex + (1 - alpha) * prev_left_elbow[0])
            left_elbow[1] = int(alpha * ley + (1 - alpha) * prev_left_elbow[1])
            right_elbow[0] = int(alpha * rex + (1 - alpha) * prev_right_elbow[0])
            right_elbow[1] = int(alpha * rey + (1 - alpha) * prev_right_elbow[1])

            prev_left = left_wrist[:]
            prev_right = right_wrist[:]
            prev_left_elbow = left_elbow[:]
            prev_right_elbow = right_elbow[:]

            left_closed = is_hand_closed(results, True)
            right_closed = is_hand_closed(results, False)

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

def draw_sword(wrist, elbow, closed, color):
    if not closed:
        return
    dx = wrist[0] - elbow[0]
    dy = wrist[1] - elbow[1]
    length = max(np.hypot(dx, dy), 1)
    nx, ny = dx / length, dy / length
    end_pos = (int(wrist[0] + nx * SWORD_LENGTH),
               int(wrist[1] + ny * SWORD_LENGTH))
    pygame.draw.line(screen, color, (int(wrist[0]), int(wrist[1])), end_pos, 4)

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
            score += POINT_DIRECTIONAL if note.require_direction else POINT_ANY
        if not note.hit and note.check_hit(right_wrist, right_dir):
            score += POINT_DIRECTIONAL if note.require_direction else POINT_ANY
        note.draw()

    # Draw cursors and swords
    pygame.draw.circle(screen, (0, 255, 0), (int(left_wrist[0]), int(left_wrist[1])), 15)
    pygame.draw.circle(screen, (255, 0, 0), (int(right_wrist[0]), int(right_wrist[1])), 15)
    draw_sword(left_wrist, left_elbow, left_closed, (0, 255, 0))
    draw_sword(right_wrist, right_elbow, right_closed, (255, 0, 0))

    # Draw score
    score_text = font.render(f"Score: {score}", True, (255, 255, 255))
    screen.blit(score_text, (10, 10))

    pygame.display.flip()

pygame.quit()
cap.release()
cv2.destroyAllWindows()
