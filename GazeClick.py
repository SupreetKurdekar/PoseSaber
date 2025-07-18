import cv2
import mediapipe as mp

# Initialize MediaPipe Face Mesh with iris tracking
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# Drawing specifications for landmarks
mp_drawing = mp.solutions.drawing_utils
landmark_style = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
connection_style = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

# Toggle for displaying eye tracking overlay
show_overlay = False

print("Press Ctrl+Q to toggle eye tracking overlay. Press ESC to exit.")

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if show_overlay and results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=landmark_style,
                connection_drawing_spec=connection_style,
            )
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=landmark_style,
                connection_drawing_spec=connection_style,
            )

    cv2.imshow("Gaze Tracker", frame)
    key = cv2.waitKey(1) & 0xFF

    # Ctrl+Q key code is 17 on most systems
    if key == 17:
        show_overlay = not show_overlay
    elif key == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()
