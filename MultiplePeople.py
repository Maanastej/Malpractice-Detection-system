import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe face mesh and drawing
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Define landmarks for left, right eye, and chin
LEFT_EYE = [33, 133]
RIGHT_EYE = [362, 263]
CHIN = 152
NOSE_TIP = 1

# Global variables to store the initial head pose for up to five faces
initial_yaws = [None] * 5  # List to store the initial yaw for up to five faces
look_away_counters = [0] * 5  # Counter for each person
look_away_flags = [False] * 5  # Flag to track ongoing look-away state

# Function to get 3D coordinates of landmarks
def get_3d_landmark(face_landmarks, idx, image_shape):
    h, w, _ = image_shape
    landmark = face_landmarks.landmark[idx]
    return np.array([landmark.x * w, landmark.y * h, landmark.z * w])  # z is scaled by width

# Function to calculate the head pose using nose, chin, and eyes
def get_head_pose(left_eye_3d, right_eye_3d, nose_3d, chin_3d, image_shape):
    model_points = np.array([
        (0.0, 0.0, 0.0),         # Nose tip
        (0.0, -100.0, 0.0),      # Chin
        (-50.0, 50.0, 0.0),      # Left eye
        (50.0, 50.0, 0.0)        # Right eye
    ])

    # 2D image points from face landmarks
    image_points = np.array([
        nose_3d[:2],  # Nose tip
        chin_3d[:2],  # Chin
        left_eye_3d[:2],  # Left eye
        right_eye_3d[:2]  # Right eye
    ], dtype="double")

    # Camera internals (using image shape)
    focal_length = image_shape[1]
    center = (image_shape[1] / 2, image_shape[0] / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")

    # Assuming no lens distortion
    dist_coeffs = np.zeros((4, 1))

    # Solve PnP to get rotation and translation vectors
    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

    return rotation_vector, translation_vector

# Function to convert rotation vector to Euler angles (yaw, pitch, roll)
def rotation_vector_to_euler_angles(rotation_vector):
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    sy = np.sqrt(rotation_matrix[0, 0]*2 + rotation_matrix[1, 0]*2)
    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        y = np.arctan2(-rotation_matrix[2, 0], sy)
        z = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        x = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        y = np.arctan2(-rotation_matrix[2, 0], sy)
        z = 0

    return np.degrees(x), np.degrees(y), np.degrees(z)  # Pitch, yaw, roll

# Function to check if the user is looking away based on initial head pose
def is_looking_away_relative(current_yaw, initial_yaw, gaze_threshold=20):
    yaw_difference = current_yaw - initial_yaw
    return abs(yaw_difference) > gaze_threshold

# Function to check if two gaze lines intersect
def do_gaze_lines_intersect(eye1_center, eye2_center, gaze1_dir, gaze2_dir):
    distance_between_eyes = np.linalg.norm(eye1_center - eye2_center)
    angle_diff = np.abs(np.dot(gaze1_dir, gaze2_dir))
    return distance_between_eyes < 1000 and angle_diff < 0.8  # Adjusted thresholds

# Initialize webcam and face mesh detector
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Set width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Set height

# Thresholds for gaze direction alerts
GAZE_AWAY_THRESHOLD = 20  # Base degrees threshold for yaw angle

with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, max_num_faces=5) as face_mesh:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Flip image and convert to RGB
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = face_mesh.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        eye_centers = []
        gaze_directions = []
        face_rects = []

        # Process faces detected in frame
        if results.multi_face_landmarks:
            for idx, face_landmarks in enumerate(results.multi_face_landmarks):
                if idx >= 5:  # Limit to a maximum of five faces
                    break
                
                # Get 3D coordinates for landmarks
                left_eye_3d = get_3d_landmark(face_landmarks, LEFT_EYE[0], image.shape)
                right_eye_3d = get_3d_landmark(face_landmarks, RIGHT_EYE[0], image.shape)
                nose_3d = get_3d_landmark(face_landmarks, NOSE_TIP, image.shape)
                chin_3d = get_3d_landmark(face_landmarks, CHIN, image.shape)

                # Calculate head pose (rotation vector)
                rotation_vector, translation_vector = get_head_pose(left_eye_3d, right_eye_3d, nose_3d, chin_3d, image.shape)

                # Convert rotation vector to Euler angles (yaw, pitch, roll)
                pitch, yaw, roll = rotation_vector_to_euler_angles(rotation_vector)

                # Capture the initial yaw for each face if it's not set yet
                if initial_yaws[idx] is None:
                    initial_yaws[idx] = yaw
                    print(f"Initial head yaw for person {idx+1}: {initial_yaws[idx]:.2f} degrees")

                # Check if the person is looking away based on yaw difference from initial yaw
                is_looking_away = is_looking_away_relative(yaw, initial_yaws[idx], GAZE_AWAY_THRESHOLD)

                # Only increment look-away counter once per look-away event
                if is_looking_away:
                    if not look_away_flags[idx]:  # If not already flagged as looking away
                        look_away_counters[idx] += 1  # Increment look-away counter
                        look_away_flags[idx] = True   # Set the look-away flag
                        print(f"Alert: Person {idx+1} is looking away! Look-away count: {look_away_counters[idx]}")
                else:
                    look_away_flags[idx] = False  # Reset the flag when the person is not looking away

                # Draw gaze direction line
                gaze_direction = (right_eye_3d[:2] - left_eye_3d[:2]) / np.linalg.norm(right_eye_3d[:2] - left_eye_3d[:2])
                eye_center = np.mean([left_eye_3d[:2], right_eye_3d[:2]], axis=0)
                eye_centers.append(eye_center)
                gaze_directions.append(gaze_direction)
                end_point = (int(eye_center[0] + gaze_direction[0] * 100), int(eye_center[1] + gaze_direction[1] * 100))
                cv2.line(image, tuple(eye_center.astype(int)), end_point, (0, 255, 0), 2)

                # Draw bounding box if the person is looking away
                if is_looking_away:
                    face_x_min = int(min(face_landmarks.landmark[LEFT_EYE[0]].x, face_landmarks.landmark[RIGHT_EYE[0]].x) * image.shape[1]) - 50
                    face_x_max = int(max(face_landmarks.landmark[LEFT_EYE[0]].x, face_landmarks.landmark[RIGHT_EYE[0]].x) * image.shape[1]) + 50
                    face_y_min = int(min(face_landmarks.landmark[LEFT_EYE[0]].y, face_landmarks.landmark[CHIN].y) * image.shape[0]) - 50
                    face_y_max = int(max(face_landmarks.landmark[LEFT_EYE[0]].y, face_landmarks.landmark[CHIN].y) * image.shape[0]) + 50
                    cv2.rectangle(image, (face_x_min, face_y_min), (face_x_max, face_y_max), (0, 0, 255), 2)

                # Draw face mesh
                mp_drawing.draw_landmarks(
                    image, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS,
                    mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                    mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
                )

                # Display the look-away count on the screen for each person
                cv2.putText(image, f'Look-aways: {look_away_counters[idx]}',
                            (50, 130 + idx * 30),  # Adjust the position as needed
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

            # Check for gaze interactions between people
            for i in range(len(eye_centers)):
                for j in range(i + 1, len(eye_centers)):
                    if do_gaze_lines_intersect(eye_centers[i], eye_centers[j], gaze_directions[i], gaze_directions[j]):
                        cv2.putText(image, "Gaze Interaction Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow("Gaze Tracking with Interaction Detection", image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()