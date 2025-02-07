import cv2
import mediapipe as mp
from mediapipe.tasks import python


model_path = 'model/pose_landmarker_lite.task'


BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode


# Create a pose landmarker instance with the live stream mode:
def print_result(result: PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    print('pose landmarker result: {}'.format(result))


options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)



# Open a connection to the webcam (0 is the default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()  # Read a frame from the webcam

    if not ret:
        print("Error: Could not read frame.")
        break
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)



    cv2.imshow("Webcam Stream", frame)  # Display the frame
    with PoseLandmarker.create_from_options(options) as landmarker:
        result = landmarker.detect_async(mp_image, 5)
    print(result)
    # Press 'q' to exit the video stream
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
