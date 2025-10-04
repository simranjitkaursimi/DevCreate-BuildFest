import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,  # Lowered for better detection
    min_tracking_confidence=0.5    # Lowered for better tracking
)
mp_draw = mp.solutions.drawing_utils

# Get screen size
screen_width, screen_height = pyautogui.size()
print(f"Screen size: {screen_width}x{screen_height}")

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

# Get actual camera resolution
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Camera resolution: {frame_width}x{frame_height}")

# Smoothing factor for cursor movement (lower = smoother but more lag)
smoothing = 0.3
prev_x, prev_y = 0, 0

# Cursor movement scaling (adjust these values to change sensitivity)
x_scale = 1.0
y_scale = 1.0

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

def main():
    global prev_x, prev_y
    
    print("Starting hand tracking...")
    print("Press 'q' to quit")
    print("Press 'r' to reset cursor position")
    print("Press '+' or '-' to adjust sensitivity")
    
    # FPS calculation variables
    fps = 0
    frame_count = 0
    start_time = time.time()
    
    while True:
        success, frame = cap.read()
        if not success:
            print("Error: Failed to capture frame from camera")
            break
            
        # Flip frame horizontally for a later selfie-view display
        frame = cv2.flip(frame, 1)
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame and detect hands
        results = hands.process(rgb_frame)
        
        # Draw hand landmarks and move cursor
        hand_detected = False
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                hand_detected = True
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Get index finger tip coordinates (landmark 8)
                index_finger = hand_landmarks.landmark[8]
                x = int(index_finger.x * frame.shape[1])
                y = int(index_finger.y * frame.shape[0])
                
                # Map coordinates to screen
                screen_x, screen_y = map_coordinates(x, y, frame.shape[1], frame.shape[0])
                
                # Apply smoothing
                smooth_x = int(smoothing * prev_x + (1 - smoothing) * screen_x)
                smooth_y = int(smoothing * prev_y + (1 - smoothing) * screen_y)
                
                # Move cursor
                pyautogui.moveTo(smooth_x, smooth_y)
                
                # Update previous coordinates
                prev_x, prev_y = smooth_x, smooth_y
                
                # Draw circle at index finger tip
                cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)
                
                # Draw line from finger to cursor position on screen
                cursor_x = int(smooth_x * frame.shape[1] / screen_width)
                cursor_y = int(smooth_y * frame.shape[0] / screen_height)
                cv2.line(frame, (x, y), (cursor_x, cursor_y), (255, 0, 0), 2)
                cv2.circle(frame, (cursor_x, cursor_y), 5, (0, 0, 255), -1)
        
        # Calculate and display FPS
        frame_count += 1
        if frame_count >= 30:
            end_time = time.time()
            fps = frame_count / (end_time - start_time)
            frame_count = 0
            start_time = time.time()
        
        # Display hand detection status
        status_text = "Hand Detected" if hand_detected else "No Hand Detected"
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if hand_detected else (0, 0, 255), 2)
        
        # Display FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Display sensitivity
        cv2.putText(frame, f"Sensitivity: {x_scale:.1f}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Display the frame
        cv2.imshow('Hand Tracking', frame)
        
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
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    print("Hand tracking stopped")

if __name__ == "__main__":
    main() 