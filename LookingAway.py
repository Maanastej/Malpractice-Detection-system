import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe face detection and face mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Eye landmarks for left and right eyes in the MediaPipe face mesh
LEFT_EYE = [33, 133]  # Landmarks for the left eye
RIGHT_EYE = [362, 263]  # Landmarks for the right eye
CHIN = 152  # Landmark for chin

# Function to calculate the center of an eye
def get_eye_center(face_landmarks, image_shape, eye_landmarks):
    h, w, _ = image_shape
    eye_points = np.array([[face_landmarks.landmark[idx].x * w,
                            face_landmarks.landmark[idx].y * h] for idx in eye_landmarks])
    return np.mean(eye_points, axis=0)

# Function to calculate the chin position
def get_chin_position(face_landmarks, image_shape):
    h, w, _ = image_shape
    chin_point = face_landmarks.landmark[CHIN]
    return np.array([chin_point.x * w, chin_point.y * h])

# Function to check if eye angles indicate looking at another person's paper
def is_looking_away(left_eye_center, right_eye_center, threshold_angle):
    # Calculate the center of the frame (where the camera is)
    center_of_frame_x = 640 // 2  # Assuming frame width is 640
    left_diff = left_eye_center[0] - center_of_frame_x
    right_diff = right_eye_center[0] - center_of_frame_x
    
    # Check if the distance of either eye from the center exceeds the threshold
    if abs(left_diff) > threshold_angle or abs(right_diff) > threshold_angle:
        return True

    return False

# Function to draw the gaze direction line from the eyes
def draw_gaze_line(image, eye_center, direction_vector, length=100, color=(0, 255, 0)):
    end_point = (int(eye_center[0] + direction_vector[0] * length), 
                 int(eye_center[1] + direction_vector[1] * length))
    cv2.line(image, tuple(eye_center.astype(int)), end_point, color, 2)

# Function to check if two gaze lines intersect
def do_gaze_lines_intersect(eye1_center, eye2_center, gaze1_dir, gaze2_dir):
    # Simple intersection detection based on proximity of the gaze lines
    distance_between_eyes = np.linalg.norm(eye1_center - eye2_center)
    angle_diff = np.abs(np.dot(gaze1_dir, gaze2_dir))  # Cosine of angle between vectors
    return distance_between_eyes < 600 and angle_diff < 0.7  # Adjust thresholds as needed

# Initialize webcam capture and MediaPipe face mesh
cap = cv2.VideoCapture(0)

# Set frame size for performance
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Set width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Set height

# Thresholds for gaze direction alerts
GAZE_AWAY_THRESHOLD = 80  # pixels

with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, max_num_faces=2) as face_mesh:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        
        # Flip and convert the image to RGB
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Detect face landmarks
        results = face_mesh.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        eye_centers = []
        gaze_directions = []

        # If faces are detected
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Calculate left and right eye centers
                left_eye_center = get_eye_center(face_landmarks, image.shape, LEFT_EYE)
                right_eye_center = get_eye_center(face_landmarks, image.shape, RIGHT_EYE)
                
                # Average eye center (center of the two eyes)
                avg_eye_center = np.mean([left_eye_center, right_eye_center], axis=0)
                eye_centers.append(avg_eye_center)
                
                # Check if the user is looking away
                if is_looking_away(left_eye_center, right_eye_center, GAZE_AWAY_THRESHOLD):
                    cv2.putText(image, 'ALERT: Looking away!', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 
                                1, (0, 0, 255), 2, cv2.LINE_AA)
                    print("Alert: Looking away detected!")

                # Rough estimate of gaze direction (from left eye to right eye)
                gaze_direction = right_eye_center - left_eye_center
                gaze_direction = gaze_direction / np.linalg.norm(gaze_direction)  # Normalize vector
                gaze_directions.append(gaze_direction)
                
                # Draw gaze direction line
                draw_gaze_line(image, avg_eye_center, gaze_direction)
                
                # Draw face landmarks (optional)
                mp_drawing.draw_landmarks(image, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION)
            
            # If two faces are detected, check if their gaze lines intersect
            if len(eye_centers) == 2:
                if do_gaze_lines_intersect(eye_centers[0], eye_centers[1], gaze_directions[0], gaze_directions[1]):
                    # Draw an alert message on the image
                    cv2.putText(image, 'ALERT: Gaze directions intersect!', (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 
                                1, (0, 0, 255), 2, cv2.LINE_AA)
                    print("Alert: Gaze directions are intersecting!")
        
        # Display the image
        cv2.imshow('Head Movement and Gaze Detection', image)
        if cv2.waitKey(5) & 0xFF == 27:  # Press 'ESC' to exit
            break

cap.release()
cv2.destroyAllWindows()
