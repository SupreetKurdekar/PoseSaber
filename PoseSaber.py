import cv2
import mediapipe as mp
import pygame
import threading
import numpy as np
import random

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

# Last punch direction vectors
left_punch_dir = None
right_punch_dir = None

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
HORIZON_Y = HEIGHT // 2

font = pygame.font.SysFont(None, 40)
score = 0
POINT = 5

def create_glove(color=(255, 0, 0)):
    surf = pygame.Surface((40, 40), pygame.SRCALPHA)
    pygame.draw.circle(surf, color, (20, 15), 15)
    pygame.draw.rect(surf, color, pygame.Rect(5, 15, 30, 18))
    return surf

left_glove = create_glove()
right_glove = pygame.transform.flip(left_glove, True, False)

class Note:
    def __init__(self):
        self.x = random.randint(100, WIDTH - 100)
        self.z = 0.0
        self.speed = 0.02
        self.size = 50
        self.hit = False
        self.vx = 0
        self.vy = 0
        self.y = HORIZON_Y

    def update(self):
        if not self.hit:
            self.z += self.speed
            scale = 0.3 + 0.7 * self.z
            self.size = int(50 * scale)
            self.y = int(HORIZON_Y + (HEIGHT - HORIZON_Y - 50) * self.z)
        else:
            self.x += self.vx
            self.y += self.vy
            self.vx *= 0.95
            self.vy *= 0.95

    def draw(self):
        color = (0, 255, 255) if not self.hit else (128, 128, 128)
        rect = pygame.Rect(self.x - self.size // 2, self.y - self.size // 2,
                           self.size, self.size)
        pygame.draw.rect(screen, color, rect)

    def check_hit(self, wrist, punch_dir):
        wx, wy = wrist
        half = self.size / 2
        if self.hit or punch_dir is None:
            return False
        if (self.x - half < wx < self.x + half and
                self.y - half < wy < self.y + half):
            self.hit = True
            self.vx = punch_dir[0] * 20
            self.vy = punch_dir[1] * 20
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
    global left_punch_dir, right_punch_dir
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

            # Punch detection
            l_vec = np.array(left_wrist) - np.array(left_elbow)
            r_vec = np.array(right_wrist) - np.array(right_elbow)
            l_norm = l_vec / (np.linalg.norm(l_vec) + 1e-5)
            r_norm = r_vec / (np.linalg.norm(r_vec) + 1e-5)
            l_vel = np.array(left_wrist) - np.array(prev_left)
            r_vel = np.array(right_wrist) - np.array(prev_right)
            l_speed = np.dot(l_vel, l_norm)
            r_speed = np.dot(r_vel, r_norm)
            left_punch_dir = l_norm if left_closed and l_speed > 20 else None
            right_punch_dir = r_norm if right_closed and r_speed > 20 else None

            prev_left = left_wrist[:]
            prev_right = right_wrist[:]
            prev_left_elbow = left_elbow[:]
            prev_right_elbow = right_elbow[:]

            left_closed = is_hand_closed(results, True)
            right_closed = is_hand_closed(results, False)

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

    # Update and draw notes
    for note in notes:
        note.update()
        if not note.hit and note.check_hit(left_wrist, left_punch_dir):
            score += POINT
        if not note.hit and note.check_hit(right_wrist, right_punch_dir):
            score += POINT
        note.draw()

    # Draw boxing gloves
    screen.blit(left_glove, (int(left_wrist[0]) - left_glove.get_width() // 2,
                             int(left_wrist[1]) - left_glove.get_height() // 2))
    screen.blit(right_glove, (int(right_wrist[0]) - right_glove.get_width() // 2,
                              int(right_wrist[1]) - right_glove.get_height() // 2))

    # Draw score
    score_text = font.render(f"Score: {score}", True, (255, 255, 255))
    screen.blit(score_text, (10, 10))

    pygame.display.flip()

pygame.quit()
cap.release()
cv2.destroyAllWindows()
