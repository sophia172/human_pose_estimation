import cv2
import os
from logger import logging
camera_index = 0
max_cameras = 5
output_dir = "cameras_data"
os.makedirs(output_dir, exist_ok=True)
for i in range(max_cameras):
    try:
        # Loop through camera indices to test each available camera

        cap = cv2.VideoCapture(camera_index)
        if cap.isOpened():
            logging.info(f"Successfully accessed camera with index {camera_index}")
            while True:
                ret, frame = cap.read()
                if ret:
                    # Display the video feed in the window
                    cv2.imshow("Camera Feed", frame)
                    # Wait for a key press for 1 ms
                    key = cv2.waitKey(1) & 0xFF
                    # If the 'space' key is pressed, save the image
                    if key == ord(' '):
                        # Save the frame as an image in the test_images folder
                        counter = 1
                        file_path = os.path.join(output_dir, f"captured_image_{camera_index}_{counter}.png")

                        while os.path.exists(file_path):
                            counter += 1
                            file_path = os.path.join(output_dir, f"captured_image_{camera_index}_{counter}.png")
                        cv2.imwrite(file_path, frame)
                        print(f"Image saved at {file_path}")
                        # Exit the loop if the 'q' key is pressed
                    if key == ord('q'):
                        height, width = frame.shape[:2]  # Only height and width, ignore channels
                        logging.info(f"Camera {camera_index} has image size of ({width}, {height})")

                        print("Exiting...")
                        break
                else:
                    print(f"Failed to grab frame from camera with index {camera_index}")

        else:
            print(f"Camera with index {camera_index} is not available")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        break
    finally:
        cap.release()  # Release the resource after testing the camera
        cv2.destroyAllWindows()
        logging.info(f"Camera {camera_index} have been tested and windows have been closed.")
        camera_index += 1
