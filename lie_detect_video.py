import cv2
import mediapipe as mp
import numpy as np
from fer import FER

# Initialize MediaPipe Face Mesh and Hands
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize FER for emotion detection
detector = FER()

# Indices for the eyes, smile, and face oval
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
SMILE = [61, 185, 40, 39, 37, 0, 267, 269, 270]
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

# Function to calculate the eye aspect ratio (EAR)
def calculate_ear(landmarks, eye_indices):
    eye_points = np.array([landmarks[i] for i in eye_indices])
    A = np.linalg.norm(eye_points[1] - eye_points[5])
    B = np.linalg.norm(eye_points[2] - eye_points[4])
    C = np.linalg.norm(eye_points[0] - eye_points[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Function to check if hand is covering face
def is_hand_covering_face(hand_landmarks, face_oval, frame_shape):
    if hand_landmarks is None:
        return False
    
    face_oval_np = np.array(face_oval, dtype=np.int32)
    for lm in hand_landmarks:
        if cv2.pointPolygonTest(face_oval_np, (int(lm.x * frame_shape[1]), int(lm.y * frame_shape[0])), False) >= 0:
            return True
    return False

# Function to check gaze change
def is_gaze_changed(previous_gaze, current_gaze, threshold=5):
    if previous_gaze is None:
        return False, current_gaze
    gaze_diff = np.linalg.norm(np.array(previous_gaze) - np.array(current_gaze))
    return gaze_diff > threshold, current_gaze

# Function to check emotion
def emotion_change(frame, last_emotion, emotion_count):
    result = detector.detect_emotions(frame)
    current_emotion = last_emotion 

    for face in result:
        emotions = face['emotions']
        current_emotion = max(emotions, key=emotions.get)
        
    if last_emotion is None or last_emotion == current_emotion:
        emotion_diff = False
    else:
        emotion_diff = True
        emotion_count += 0.5
        
    return emotion_diff, current_emotion, emotion_count

# Start video capture
cap = cv2.VideoCapture('/home/dheeraj/Deep Learning/Lie_Detector_Project/Dj_Lie.mp4')

frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
fps = cap.get(cv2.CAP_PROP_FPS) 
duration = frame_count / fps


blink_count = 0
eye_closed_frames = 0
eye_open_threshold = 0.25
eye_closed_frames_threshold = 3
gaze_truth = False
hand_truth = False

previous_gaze_left = None
previous_gaze_right = None

current_emotion = None

hand_count = 0
gaze_count = 0
emotion_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_face = face_mesh.process(frame_rgb)
    results_hands = hands.process(frame_rgb)

    landmarks = None
    face_oval = []
    hand_covering_face = False  # Initialize hand covering face variable

    if results_face.multi_face_landmarks:
        for face_landmarks in results_face.multi_face_landmarks:
            landmarks = [(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark]
            landmarks = np.array([(int(lm[0] * frame.shape[1]), int(lm[1] * frame.shape[0]), lm[2]) for lm in landmarks])
            face_oval = [(int(landmarks[i][0]), int(landmarks[i][1])) for i in FACE_OVAL]

            # Blinks check
            left_ear = calculate_ear(landmarks, LEFT_EYE)
            right_ear = calculate_ear(landmarks, RIGHT_EYE)
            avg_ear = (left_ear + right_ear) / 2.0

            if avg_ear < eye_open_threshold:
                eye_closed_frames += 1
            else:
                if eye_closed_frames >= eye_closed_frames_threshold:
                    blink_count += 1
                eye_closed_frames = 0

            # Gaze check
            current_gaze_left = np.mean(landmarks[LEFT_EYE], axis=0)
            current_gaze_right = np.mean(landmarks[RIGHT_EYE], axis=0)
            gaze_changed_left, previous_gaze_left = is_gaze_changed(previous_gaze_left, current_gaze_left)
            gaze_changed_right, previous_gaze_right = is_gaze_changed(previous_gaze_right, current_gaze_right)

            gaze_changed = gaze_changed_left or gaze_changed_right

            if gaze_changed:
                gaze_truth = True
            
            if gaze_truth and not gaze_changed:
                gaze_count += 1
                gaze_truth = False

            cv2.putText(frame, f'Blinks: {blink_count}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f'Gaze Changed: {gaze_changed}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Hands check
    if results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:
            if face_oval and is_hand_covering_face(hand_landmarks.landmark, face_oval, frame.shape):
                hand_covering_face = True
    
    if hand_covering_face:
        hand_truth = True
    
    if hand_truth and not hand_covering_face:
        hand_count += 1
        hand_truth = False

    # Emotion check
    emotion_diff, current_emotion, emotion_count = emotion_change(frame, current_emotion, emotion_count)

    cv2.putText(frame, f'Hand Covering Face: {hand_covering_face}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f'Mood: {current_emotion}', (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow('Lie Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()

# Detect lying based on conditions (adjust thresholds and logic as needed)
lies_detected = False
blink_lie = False
hand_lie = False
gaze_lie = False
emotion_lie = False
lie_ratio = 0

# average less than 20 blinks per minute
if (blink_count * 60 / duration) > 20:
    blink_lie = True
    lies_detected = True
    lie_ratio += 25

# adjusted based on observation
if (hand_count * 60 / duration) > 20:
    hand_lie = True
    lies_detected = True
    lie_ratio += 25

# adjusted based on observation
if (gaze_count * 60 / duration) >= 60:
    gaze_lie = True
    lies_detected = True
    lie_ratio += 25

# adjusted based on observation
if (int(emotion_count) * 60 / duration) >= 60:
    emotion_lie = True
    lies_detected = True
    lie_ratio += 25

if lies_detected:
    print("The person is lying")
    print(f'Confidence : {lie_ratio}%')
else:
    print("The person is telling the truth")

details = input('Do you want the details of this observation? (y/n)')

if details.lower() == 'y':
    print('*' * 100)
    print('Duration : ', duration)
    print('Number of times blinked : ', blink_count)
    print('Number of times gaze changed : ', gaze_count)
    print('Number of times hands covered face : ', hand_count)
    print('Number of times emotions changed : ', int(emotion_count))

    if lies_detected:
        print('*' * 100)
        print('Reason(s) : ')
        if blink_lie:
            print('Blinking too much')
        if hand_lie:
            print('Nervously touching face')
        if gaze_lie:
            print('Couldn\'t concentrate on a single point ')
        if emotion_lie:
            print('Emotions were unstable')
