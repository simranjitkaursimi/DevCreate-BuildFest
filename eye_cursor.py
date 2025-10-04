import cv2
import numpy as np
import pyautogui
import time

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam. Please check your camera connection.")
    exit()

# Set camera properties
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)  # Enable autofocus
cap.set(cv2.CAP_PROP_BRIGHTNESS, 150)  # Increase brightness for better eye detection

# Get actual camera resolution
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Camera resolution: {frame_width}x{frame_height}")

# Get screen size
screen_width, screen_height = pyautogui.size()
print(f"Screen size: {screen_width}x{screen_height}")

# Load OpenCV's pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Smoothing factor for cursor movement (lower = smoother but more lag)
smoothing = 0.3
prev_x, prev_y = 0, 0

# Cursor movement scaling (adjust these values to change sensitivity)
x_scale = 1.0
y_scale = 1.0

# Eye tracking parameters
pupil_threshold = 30  # Threshold for pupil detection
min_eye_size = 20    # Minimum eye size for detection

def map_coordinates(x, y, frame_width, frame_height):
    """Map coordinates from frame to screen with scaling"""
    # Add padding to prevent cursor from going off-screen
    padding = 50
    screen_x = int(np.interp(x, [padding, frame_width-padding], [0, screen_width])) * x_scale
    screen_y = int(np.interp(y, [padding, frame_height-padding], [0, screen_height])) * y_scale
    
    # Ensure coordinates stay within screen bounds
    screen_x = max(0, min(screen_x, screen_width))
    screen_y = max(0, min(screen_y, screen_height))
    
    return screen_x, screen_y

def detect_pupil(eye_roi):
    """Detect pupil in eye region using thresholding"""
    # Convert to grayscale if not already
    if len(eye_roi.shape) == 3:
        eye_gray = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY)
    else:
        eye_gray = eye_roi
    
    # Apply Gaussian blur to reduce noise
    eye_gray = cv2.GaussianBlur(eye_gray, (5, 5), 0)
    
    # Apply threshold to isolate dark regions (pupil)
    _, thresh = cv2.threshold(eye_gray, pupil_threshold, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the largest contour (likely the pupil)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) > 10:  # Filter out noise
            # Calculate the center of the contour
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                return (cx, cy)
    
    return None

def main():
    global prev_x, prev_y, x_scale, y_scale, pupil_threshold
    
    print("Starting eye tracking...")
    print("Press 'q' to quit")
    print("Press 'r' to reset cursor position")
    print("Press '+' or '-' to adjust sensitivity")
    print("Press 'p' to adjust pupil detection threshold")
    
    # FPS calculation variables
    fps = 0
    frame_count = 0
    start_time = time.time()
    
    # Track the last known good eye position
    last_eye_pos = None
    frames_without_eye = 0
    
    while True:
        success, frame = cap.read()
        if not success:
            print("Error: Failed to capture frame from camera")
            break
            
        # Flip frame horizontally
        frame = cv2.flip(frame, 1)
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Enhance contrast for better eye detection
        gray = cv2.equalizeHist(gray)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        face_detected = False
        eye_detected = False
        eye_center_x = 0
        eye_center_y = 0
        
        for (x, y, w, h) in faces:
            face_detected = True
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Region of interest for eyes (upper half of face)
            roi_gray = gray[y:y+int(h/2), x:x+w]
            roi_color = frame[y:y+int(h/2), x:x+w]
            
            # Detect eyes
            eyes = eye_cascade.detectMultiScale(
                roi_gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(min_eye_size, min_eye_size)
            )
            
            # Calculate eye centers
            eye_centers = []
            for (ex, ey, ew, eh) in eyes:
                # Draw rectangle around eyes
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
                
                # Extract eye region for pupil detection
                eye_roi = roi_color[ey:ey+eh, ex:ex+ew]
                
                # Detect pupil
                pupil_center = detect_pupil(eye_roi)
                
                if pupil_center:
                    # Calculate pupil position relative to frame
                    pupil_x = x + ex + pupil_center[0]
                    pupil_y = y + ey + pupil_center[1]
                    
                    # Draw circle at pupil center
                    cv2.circle(frame, (pupil_x, pupil_y), 3, (0, 0, 255), -1)
                    
                    # Store pupil position
                    eye_centers.append((pupil_x, pupil_y))
                    eye_detected = True
                else:
                    # If pupil not detected, use eye center
                    eye_center_x = x + ex + ew//2
                    eye_center_y = y + ey + eh//2
                    eye_centers.append((eye_center_x, eye_center_y))
                    eye_detected = True
                    
                    # Draw circle at eye center
                    cv2.circle(frame, (eye_center_x, eye_center_y), 5, (0, 0, 255), -1)
            
            # If we detected at least one eye, use it for cursor control
            if eye_centers:
                # Use the first eye detected
                eye_center_x, eye_center_y = eye_centers[0]
                last_eye_pos = (eye_center_x, eye_center_y)
                frames_without_eye = 0
                
                # Map coordinates to screen
                screen_x, screen_y = map_coordinates(eye_center_x, eye_center_y, frame.shape[1], frame.shape[0])
                
                # Apply smoothing
                smooth_x = int(smoothing * prev_x + (1 - smoothing) * screen_x)
                smooth_y = int(smoothing * prev_y + (1 - smoothing) * screen_y)
                
                # Move cursor
                pyautogui.moveTo(smooth_x, smooth_y)
                
                # Update previous coordinates
                prev_x, prev_y = smooth_x, smooth_y
                
                # Draw line from eye to cursor position on screen
                cursor_x = int(smooth_x * frame.shape[1] / screen_width)
                cursor_y = int(smooth_y * frame.shape[0] / screen_height)
                cv2.line(frame, (eye_center_x, eye_center_y), (cursor_x, cursor_y), (255, 0, 0), 2)
                cv2.circle(frame, (cursor_x, cursor_y), 5, (0, 0, 255), -1)
            else:
                frames_without_eye += 1
                
                # If we've lost eye tracking for a few frames but have a last known position, use it
                if last_eye_pos and frames_without_eye < 10:
                    eye_center_x, eye_center_y = last_eye_pos
                    
                    # Map coordinates to screen
                    screen_x, screen_y = map_coordinates(eye_center_x, eye_center_y, frame.shape[1], frame.shape[0])
                    
                    # Apply smoothing
                    smooth_x = int(smoothing * prev_x + (1 - smoothing) * screen_x)
                    smooth_y = int(smoothing * prev_y + (1 - smoothing) * screen_y)
                    
                    # Move cursor
                    pyautogui.moveTo(smooth_x, smooth_y)
                    
                    # Update previous coordinates
                    prev_x, prev_y = smooth_x, smooth_y
                    
                    # Draw line from eye to cursor position on screen
                    cursor_x = int(smooth_x * frame.shape[1] / screen_width)
                    cursor_y = int(smooth_y * frame.shape[0] / screen_height)
                    cv2.line(frame, (eye_center_x, eye_center_y), (cursor_x, cursor_y), (255, 0, 0), 2)
                    cv2.circle(frame, (cursor_x, cursor_y), 5, (0, 0, 255), -1)
        
        # Calculate and display FPS
        frame_count += 1
        if frame_count >= 30:
            end_time = time.time()
            fps = frame_count / (end_time - start_time)
            frame_count = 0
            start_time = time.time()
        
        # Display face detection status
        status_text = "Face Detected" if face_detected else "No Face Detected"
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if face_detected else (0, 0, 255), 2)
        
        # Display eye detection status
        eye_status = "Eye Detected" if eye_detected else "No Eye Detected"
        cv2.putText(frame, eye_status, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if eye_detected else (0, 0, 255), 2)
        
        # Display FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Display sensitivity
        cv2.putText(frame, f"Sensitivity: {x_scale:.1f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Display pupil threshold
        cv2.putText(frame, f"Pupil Threshold: {pupil_threshold}", (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Display the frame
        cv2.imshow('Eye Tracking', frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            # Reset cursor to center of screen
            center_x = screen_width // 2
            center_y = screen_height // 2
            pyautogui.moveTo(center_x, center_y)
            prev_x, prev_y = center_x, center_y
            print("Cursor position reset to center")
        elif key == ord('+'):
            # Increase sensitivity
            x_scale += 0.1
            y_scale += 0.1
            print(f"Sensitivity increased to {x_scale:.1f}")
        elif key == ord('-'):
            # Decrease sensitivity
            x_scale = max(0.1, x_scale - 0.1)
            y_scale = max(0.1, y_scale - 0.1)
            print(f"Sensitivity decreased to {x_scale:.1f}")
        elif key == ord('p'):
            # Adjust pupil detection threshold
            pupil_threshold = (pupil_threshold + 5) % 100
            print(f"Pupil threshold adjusted to {pupil_threshold}")
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    print("Eye tracking stopped")

if __name__ == "__main__":
    main() 