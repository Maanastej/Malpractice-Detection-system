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

# Global variables to store the initial head pose
initial_yaws = []  # List to store the initial yaw for multiple faces
looking_away_counters = []  # List to store looking away counters for each face
looking_away_states = []  # List to track if each face is currently looking away

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
    singular = sy < 1e-6  # Singular condition

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
    # Calculate the distance between the centers of the two eyes
    distance_between_eyes = np.linalg.norm(eye1_center - eye2_center)
    
    # Calculate the cosine of the angle between the two gaze direction vectors
    angle_diff = np.abs(np.dot(gaze1_dir, gaze2_dir))  # Cosine of the angle between vectors
    
    # Adjust the distance and angle thresholds for people sitting farther apart
    return distance_between_eyes < 1000 and angle_diff < 0.8  # Adjusted thresholds

# Function to draw the gaze direction line from the eyes
def draw_gaze_line(image, eye_center, direction_vector, length=100, color=(0, 255, 0)):
    end_point = (int(eye_center[0] + direction_vector[0] * length),
                 int(eye_center[1] + direction_vector[1] * length))
    cv2.line(image, tuple(eye_center.astype(int)), end_point, color, 2)

# Initialize webcam and face mesh detector
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Set width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Set height

# Thresholds for gaze direction alerts
GAZE_AWAY_THRESHOLD = 20  # Base degrees threshold for yaw angle

with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, max_num_faces=2) as face_mesh:
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
                if len(initial_yaws) <= idx:
                    initial_yaws.append(yaw)
                    looking_away_counters.append(0)  # Initialize counter for each person
                    looking_away_states.append(False)  # Initialize looking away state
                    print(f"Initial head yaw for person {idx+1}: {initial_yaws[idx]:.2f} degrees")

                # Check if the person is looking away based on yaw difference from initial yaw
                is_looking_away = is_looking_away_relative(yaw, initial_yaws[idx], GAZE_AWAY_THRESHOLD)

                # Calculate gaze direction (vector from left eye to right eye)
                gaze_direction = right_eye_3d[:2] - left_eye_3d[:2]
                gaze_direction = gaze_direction / np.linalg.norm(gaze_direction)  # Normalize vector
                eye_centers.append(np.mean([left_eye_3d[:2], right_eye_3d[:2]], axis=0))
                gaze_directions.append(gaze_direction)

                # Draw gaze direction line
                draw_gaze_line(image, np.mean([left_eye_3d[:2], right_eye_3d[:2]], axis=0), gaze_direction)

                # Draw face landmarks (optional)
                mp_drawing.draw_landmarks(image, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION)

                # Update the counter if the state changes
                if is_looking_away and not looking_away_states[idx]:
                    looking_away_counters[idx] += 1  # Increment the counter if looking away
                    looking_away_states[idx] = True  # Update the state to looking away
                    print(f"Person {idx+1} looking away. Counter: {looking_away_counters[idx]}")
                elif not is_looking_away and looking_away_states[idx]:
                    looking_away_states[idx] = False  # Update the state to looking at

                # Check if the person is looking away
                if is_looking_away:
                    # Draw a red rectangle around the face
                    face_x_min = int(min(face_landmarks.landmark[LEFT_EYE[0]].x, face_landmarks.landmark[RIGHT_EYE[0]].x) * image.shape[1]) - 50
                    face_x_max = int(max(face_landmarks.landmark[LEFT_EYE[0]].x, face_landmarks.landmark[RIGHT_EYE[0]].x) * image.shape[1]) + 50
                    face_y_min = int(min(face_landmarks.landmark[LEFT_EYE[0]].y, face_landmarks.landmark[CHIN].y) * image.shape[0]) - 50
                    face_y_max = int(max(face_landmarks.landmark[LEFT_EYE[0]].y, face_landmarks.landmark[CHIN].y) * image.shape[0]) + 50
                    face_rects.append((face_x_min, face_y_min, face_x_max, face_y_max))

                    cv2.putText(image, f'ALERT: Looking away! (Yaw: {yaw:.2f})', (50, 100 + idx*30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    print(f"Alert: Person {idx+1} is looking away! (Current Yaw: {yaw:.2f}, Initial Yaw: {initial_yaws[idx]:.2f})")
                    cv2.putText(image, f'Counter: {looking_away_counters[idx]}', (50, 150 + idx*30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            # If two faces are detected, check if their gaze lines intersect
            if len(eye_centers) == 2:
                if do_gaze_lines_intersect(eye_centers[0], eye_centers[1], gaze_directions[0], gaze_directions[1]):
                    # Draw an alert message on the image
                    cv2.putText(image, 'ALERT: Gaze directions intersect!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                                1, (0, 0, 255), 2, cv2.LINE_AA)
                    print("Alert: Gaze directions are intersecting!")

                    # Prevent "looking away" alert if gazes are intersecting
                    face_rects.clear()  # Clear face rectangles if gazes intersect

        # Highlight the faces of those looking away
        for (x_min, y_min, x_max, y_max) in face_rects:
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)

        # Show the final image with gaze interaction and looking-away alerts
        cv2.imshow('Gaze and Head Pose Detection', image)
        if cv2.waitKey(5) & 0xFF == 27:  # Press 'ESC' to exit
            break

cap.release()
cv2.destroyAllWindows()