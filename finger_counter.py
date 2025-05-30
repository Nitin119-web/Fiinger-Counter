import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize MediaPipe Hands with max_num_hands=2
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.8, min_tracking_confidence=0.8)

FINGER_TIPS = [4, 8, 12, 16, 20]

def count_fingers(hand_landmarks, hand_label):
    fingers = 0

    # Thumb: logic depends on whether it's right or left hand
    if hand_label == "Right":
        # Right hand thumb: tip x less than preceding joint x means thumb is open
        if hand_landmarks.landmark[FINGER_TIPS[0]].x < hand_landmarks.landmark[FINGER_TIPS[0] - 1].x:
            fingers += 1
    else:  # Left hand
        # Left hand thumb: tip x greater than preceding joint x means thumb is open
        if hand_landmarks.landmark[FINGER_TIPS[0]].x > hand_landmarks.landmark[FINGER_TIPS[0] - 1].x:
            fingers += 1

    # Other fingers: tip y less than pip y means finger is open
    for tip_id in FINGER_TIPS[1:]:
        if hand_landmarks.landmark[tip_id].y < hand_landmarks.landmark[tip_id - 2].y:
            fingers += 1

    return fingers

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    total_fingers = 0

    if result.multi_hand_landmarks and result.multi_handedness:
        for hand_landmarks, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
            hand_label = handedness.classification[0].label  # 'Left' or 'Right'
            total_fingers += count_fingers(hand_landmarks, hand_label)
            mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.putText(img, f'Total Fingers: {total_fingers}', (10, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

    cv2.imshow("Finger Counter", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
