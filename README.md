# Cursor Control with Hand and Eye Tracking

This project allows you to control your computer's cursor using either hand gestures or eye tracking. It uses computer vision techniques to track your hand movements or eye position and translate them into cursor movements.

## Features

- Hand gesture-based cursor control using index finger tracking
- Eye tracking-based cursor control using OpenCV's face and eye detection
- Smooth cursor movement with position interpolation
- Real-time visualization of tracking
- Adjustable sensitivity controls

## Requirements

- Python 3.7+
- Webcam
- Required Python packages (install using `pip install -r requirements.txt`):
  - opencv-python
  - mediapipe
  - pyautogui
  - numpy

## Installation

1. Clone this repository
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Hand Gesture Control
Run the hand tracking script:
```bash
python hand_cursor.py
```
- Use your index finger to control the cursor
- The cursor will follow your index finger tip
- Press 'q' to quit
- Press 'r' to reset cursor position
- Press '+' or '-' to adjust sensitivity

### Eye Tracking Control
Run the eye tracking script:
```bash
python eye_cursor.py
```
- Position your face in front of the camera
- The cursor will follow your eye movements
- Press 'q' to quit
- Press 'r' to reset cursor position
- Press '+' or '-' to adjust sensitivity

## Controls

Both hand and eye tracking scripts support the following keyboard controls:
- 'q': Quit the application
- 'r': Reset cursor to center of screen
- '+': Increase sensitivity (make cursor move more)
- '-': Decrease sensitivity (make cursor move less)

## Notes

- Ensure good lighting conditions for better tracking
- Keep your hand or face within the camera frame
- The tracking sensitivity can be adjusted in real-time using the '+' and '-' keys
- For eye tracking, maintain a stable head position for better results
- If tracking is not smooth, try adjusting the sensitivity

## Troubleshooting

- If the tracking is not smooth, try adjusting the sensitivity using the '+' and '-' keys
- If the cursor movement is too sensitive or not sensitive enough, modify the sensitivity
- Make sure your webcam is properly connected and accessible
- Ensure good lighting conditions for better detection
- For hand tracking, make sure your index finger is clearly visible and extended
- For eye tracking, position your face directly in front of the camera

## Additional Notes

- For eye tracking, you may need to adjust the sensitivity based on your specific setup
- If the tracking is not smooth, try adjusting the sensitivity
- If the cursor movement is too sensitive or not sensitive enough, modify the sensitivity
- Make sure your webcam is properly connected and accessible
- Ensure good lighting conditions for better detection
- For hand tracking, make sure your index finger is clearly visible and extended
- For eye tracking, position your face directly in front of the camera 