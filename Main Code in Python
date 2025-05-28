import cv2
import mediapipe as mp
import numpy as np
import math

# Initialize MediaPipe solutions separately to avoid SSL issues
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh


# Drawing specifications for different body parts
def get_pose_drawing_spec():
    return mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=3, circle_radius=4)


def get_hand_drawing_spec():
    return mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=2, circle_radius=3)


def get_face_drawing_spec():
    return mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=1, circle_radius=1)


def get_connection_spec(color, thickness=2):
    return mp_drawing.DrawingSpec(color=color, thickness=thickness)


# Function to calculate angle between three points
def calculate_angle(a, b, c):
    """Calculate angle between three points (a-b-c)"""
    try:
        a = np.array([a.x, a.y]) if hasattr(a, 'x') else np.array(a)
        b = np.array([b.x, b.y]) if hasattr(b, 'x') else np.array(b)
        c = np.array([c.x, c.y]) if hasattr(c, 'x') else np.array(c)

        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)

        if angle > 180.0:
            angle = 360 - angle

        return angle
    except:
        return 0


def calculate_distance(p1, p2):
    """Calculate distance between two points"""
    try:
        return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)
    except:
        return 0


def analyze_finger_positions(hand_landmarks):
    """Analyze individual finger positions and angles"""
    if not hand_landmarks:
        return {}

    landmarks = hand_landmarks.landmark
    finger_data = {}

    # Finger tip and joint indices
    finger_tips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky tips
    finger_pips = [3, 6, 10, 14, 18]  # PIP joints
    finger_mcps = [2, 5, 9, 13, 17]  # MCP joints
    finger_names = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']

    for i, (tip, pip, mcp, name) in enumerate(zip(finger_tips, finger_pips, finger_mcps, finger_names)):
        if i == 0:  # Thumb has different joint structure
            angle = calculate_angle(landmarks[1], landmarks[2], landmarks[4])
        else:
            angle = calculate_angle(landmarks[mcp], landmarks[pip], landmarks[tip])

        # Determine if finger is extended or bent
        extended = angle > 160 if i == 0 else angle > 140
        finger_data[name] = {
            'angle': angle,
            'extended': extended
        }

    return finger_data


def analyze_comprehensive_pose(pose_landmarks, left_hand_landmarks, right_hand_landmarks, face_landmarks, image):
    """Comprehensive analysis of all body parts"""
    info_y = 25
    line_height = 22
    left_col = 10
    right_col = 320

    # Header
    cv2.putText(image, "=== COMPREHENSIVE BODY ANALYSIS ===", (left_col, info_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    info_y += line_height + 5

    # Pose analysis
    if pose_landmarks:
        landmarks = pose_landmarks.landmark

        # Major joint angles
        try:
            # Arms
            left_shoulder_angle = calculate_angle(landmarks[11], landmarks[13], landmarks[15])
            right_shoulder_angle = calculate_angle(landmarks[12], landmarks[14], landmarks[16])
            left_elbow_angle = calculate_angle(landmarks[11], landmarks[13], landmarks[15])
            right_elbow_angle = calculate_angle(landmarks[12], landmarks[14], landmarks[16])

            # Legs
            left_hip_angle = calculate_angle(landmarks[11], landmarks[23], landmarks[25])
            right_hip_angle = calculate_angle(landmarks[12], landmarks[24], landmarks[26])
            left_knee_angle = calculate_angle(landmarks[23], landmarks[25], landmarks[27])
            right_knee_angle = calculate_angle(landmarks[24], landmarks[26], landmarks[28])

            # Display joint angles
            cv2.putText(image, "JOINT ANGLES:", (left_col, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            info_y += line_height

            cv2.putText(image, f"L Elbow: {int(left_elbow_angle):3d}°", (left_col, info_y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (0, 255, 0), 1)
            cv2.putText(image, f"R Elbow: {int(right_elbow_angle):3d}°", (left_col + 120, info_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            info_y += line_height

            cv2.putText(image, f"L Knee:  {int(left_knee_angle):3d}°", (left_col, info_y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (0, 255, 0), 1)
            cv2.putText(image, f"R Knee:  {int(right_knee_angle):3d}°", (left_col + 120, info_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            info_y += line_height

            cv2.putText(image, f"L Hip:   {int(left_hip_angle):3d}°", (left_col, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (0, 255, 0), 1)
            cv2.putText(image, f"R Hip:   {int(right_hip_angle):3d}°", (left_col + 120, info_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            info_y += line_height + 5
        except:
            cv2.putText(image, "Joint analysis unavailable", (left_col, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (0, 255, 0), 1)
            info_y += line_height + 5

    # Hand analysis
    cv2.putText(image, "HAND TRACKING:", (left_col, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
    info_y += line_height

    # Left hand
    if left_hand_landmarks:
        finger_data = analyze_finger_positions(left_hand_landmarks)
        cv2.putText(image, "Left Hand: DETECTED", (left_col, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
        info_y += line_height

        extended_fingers = sum(1 for f in finger_data.values() if f['extended'])
        cv2.putText(image, f"  Extended fingers: {extended_fingers}/5", (left_col, info_y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (255, 0, 255), 1)
        info_y += line_height
    else:
        cv2.putText(image, "Left Hand: NOT DETECTED", (left_col, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (100, 100, 100), 1)
        info_y += line_height * 2

    # Right hand
    if right_hand_landmarks:
        finger_data = analyze_finger_positions(right_hand_landmarks)
        cv2.putText(image, "Right Hand: DETECTED", (left_col, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
        info_y += line_height

        extended_fingers = sum(1 for f in finger_data.values() if f['extended'])
        cv2.putText(image, f"  Extended fingers: {extended_fingers}/5", (left_col, info_y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (255, 0, 255), 1)
        info_y += line_height + 5
    else:
        cv2.putText(image, "Right Hand: NOT DETECTED", (left_col, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (100, 100, 100), 1)
        info_y += line_height + 5

    # Face analysis
    face_landmarks_count = len(face_landmarks.landmark) if face_landmarks else 0
    cv2.putText(image, "FACE TRACKING:", (left_col, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    info_y += line_height

    if face_landmarks_count > 0:
        cv2.putText(image, f"Face: DETECTED ({face_landmarks_count} points)", (left_col, info_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        info_y += line_height
        cv2.putText(image, "  Eyes & mouth tracked", (left_col, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255),
                    1)
    else:
        cv2.putText(image, "Face: NOT DETECTED", (left_col, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)

    info_y += line_height + 10

    # Overall status
    cv2.putText(image, "TRACKING STATUS:", (left_col, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    info_y += line_height

    active_systems = []
    if pose_landmarks: active_systems.append("POSE")
    if left_hand_landmarks or right_hand_landmarks: active_systems.append("HANDS")
    if face_landmarks: active_systems.append("FACE")

    status_text = " + ".join(active_systems) if active_systems else "NONE"
    cv2.putText(image, f"Active: {status_text}", (left_col, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)


def create_digital_model_advanced(pose_landmarks, left_hand_landmarks, right_hand_landmarks, face_landmarks, width=640,
                                  height=480):
    """Create advanced digital model with all body parts"""
    digital_model = np.zeros((height, width, 3), dtype=np.uint8)

    # Draw pose landmarks
    if pose_landmarks:
        mp_drawing.draw_landmarks(
            digital_model,
            pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=4),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2)
        )

    # Draw face landmarks with contours
    if face_landmarks:
        # Draw face mesh contours
        mp_drawing.draw_landmarks(
            digital_model,
            face_landmarks,
            mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=1, circle_radius=1),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=1)
        )

        # Highlight eyes
        mp_drawing.draw_landmarks(
            digital_model,
            face_landmarks,
            mp_face_mesh.FACEMESH_LEFT_EYE,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=1, circle_radius=1),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=1)
        )

        mp_drawing.draw_landmarks(
            digital_model,
            face_landmarks,
            mp_face_mesh.FACEMESH_RIGHT_EYE,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=1, circle_radius=1),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=1)
        )

        # Highlight lips
        mp_drawing.draw_landmarks(
            digital_model,
            face_landmarks,
            mp_face_mesh.FACEMESH_LIPS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1)
        )

    # Draw left hand
    if left_hand_landmarks:
        mp_drawing.draw_landmarks(
            digital_model,
            left_hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=2, circle_radius=3),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=2)
        )

    # Draw right hand
    if right_hand_landmarks:
        mp_drawing.draw_landmarks(
            digital_model,
            right_hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=2, circle_radius=3),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=2)
        )

    # Add subtle digital grid
    grid_color = (15, 15, 15)
    for i in range(0, width, 30):
        cv2.line(digital_model, (i, 0), (i, height), grid_color, 1)
    for i in range(0, height, 30):
        cv2.line(digital_model, (0, i), (width, i), grid_color, 1)

    # Add status information
    status_y = 25
    cv2.putText(digital_model, 'DIGITAL HUMAN MODEL', (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    status_y += 35

    # Component status with color coding
    pose_color = (0, 255, 255) if pose_landmarks else (100, 100, 100)
    face_color = (255, 255, 0) if face_landmarks else (100, 100, 100)
    hand_color = (255, 0, 255) if (left_hand_landmarks or right_hand_landmarks) else (100, 100, 100)

    cv2.putText(digital_model, f'● POSE', (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, pose_color, 1)
    status_y += 25
    cv2.putText(digital_model, f'● FACE', (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, face_color, 1)
    status_y += 25
    cv2.putText(digital_model, f'● HANDS', (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, hand_color, 1)

    # Add landmark counts
    if pose_landmarks or face_landmarks or left_hand_landmarks or right_hand_landmarks:
        total_landmarks = 0
        if pose_landmarks: total_landmarks += len(pose_landmarks.landmark)
        if face_landmarks: total_landmarks += len(face_landmarks.landmark)
        if left_hand_landmarks: total_landmarks += len(left_hand_landmarks.landmark)
        if right_hand_landmarks: total_landmarks += len(right_hand_landmarks.landmark)

        cv2.putText(digital_model, f'Total Points: {total_landmarks}',
                    (width - 180, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    return digital_model


def main():
    cap = cv2.VideoCapture(0)

    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Initialize separate MediaPipe solutions
    with mp_pose.Pose(
            min_detection_confidence=0.8,
            min_tracking_confidence=0.8,
            model_complexity=1
    ) as pose, \
            mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.8,
                min_tracking_confidence=0.8
            ) as hands, \
            mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.8,
                min_tracking_confidence=0.8
            ) as face_mesh:

        print("Starting comprehensive multi-modal tracking...")
        print("Tracking: Pose (33 points) + Hands (42 points) + Face (468 points)")
        print("Press 'q' or ESC to quit")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Failed to read from camera")
                break

            # Prepare frame
            frame = cv2.flip(frame, 1)  # Mirror effect
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False

            # Process each component separately
            pose_results = pose.process(rgb_frame)
            hands_results = hands.process(rgb_frame)
            face_results = face_mesh.process(rgb_frame)

            # Convert back to BGR
            rgb_frame.flags.writeable = True
            tracking_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

            # Create detailed analysis frame
            analysis_frame = tracking_frame.copy()

            # Draw all landmarks on analysis frame
            if pose_results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    analysis_frame,
                    pose_results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=get_pose_drawing_spec(),
                    connection_drawing_spec=get_connection_spec((0, 255, 0))
                )

            if hands_results.multi_hand_landmarks:
                for hand_landmarks in hands_results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        analysis_frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        landmark_drawing_spec=get_hand_drawing_spec(),
                        connection_drawing_spec=get_connection_spec((255, 0, 255))
                    )

            if face_results.multi_face_landmarks:
                for face_landmarks in face_results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        analysis_frame,
                        face_landmarks,
                        mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=get_face_drawing_spec(),
                        connection_drawing_spec=get_connection_spec((0, 255, 255), 1)
                    )

            # Extract individual hand landmarks
            left_hand = None
            right_hand = None
            if hands_results.multi_hand_landmarks and hands_results.multi_handedness:
                for hand_landmarks, handedness in zip(hands_results.multi_hand_landmarks,
                                                      hands_results.multi_handedness):
                    if handedness.classification[0].label == 'Left':
                        left_hand = hand_landmarks
                    else:
                        right_hand = hand_landmarks

            # Extract face landmarks
            face_landmarks = None
            if face_results.multi_face_landmarks:
                face_landmarks = face_results.multi_face_landmarks[0]

            # Add comprehensive analysis overlay
            analyze_comprehensive_pose(
                pose_results.pose_landmarks,
                left_hand,
                right_hand,
                face_landmarks,
                analysis_frame
            )

            # Create digital model
            digital_model = create_digital_model_advanced(
                pose_results.pose_landmarks,
                left_hand,
                right_hand,
                face_landmarks,
                640, 480
            )

            # Display windows
            cv2.namedWindow('Comprehensive Analysis', cv2.WINDOW_NORMAL)
            cv2.namedWindow('Digital Model', cv2.WINDOW_NORMAL)

            # Position windows
            cv2.moveWindow('Comprehensive Analysis', 50, 50)
            cv2.moveWindow('Digital Model', 750, 50)

            cv2.imshow('Comprehensive Analysis', analysis_frame)
            cv2.imshow('Digital Model', digital_model)

            # Exit condition
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC
                break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Comprehensive tracking stopped.")


if __name__ == "__main__":
    main()
