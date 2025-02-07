import cv2
import os
import time
import sounddevice as sd
import soundfile as sf
import asyncio
output_dir = "config"
os.makedirs(output_dir, exist_ok=True)
async def take_image(camera_indexes, output_dir, auto=False, time_delta=5):
    cap = {}
    start_time = time.time()
    data, fs = sf.read("sound/camera-shutter-click.mp3", dtype='float32')

    try:
        # Loop through camera indices to test each available camera

        caps = {i: cv2.VideoCapture(i) for i in camera_indexes}
        if all([cap.isOpened() for cap in caps.values()]):
            print(f"Successfully accessed camera with index {camera_indexes}")
            while True:
                cap_read = {i: caps[i].read() for i in caps}
                if all([ret for ret, frame in cap_read.values()]):

                    # Display the video feed in the window

                    frame = cv2.hconcat([cap_read[i][1] for i in camera_indexes])
                    # If depth and RGB are combined, split the data
                    if frame.shape == (480, 640, 3):  # Modify if your frame dimensions are different
                        # Example: First 2 channels for RGB, third for depth (hypothetical)
                        rgb_image = frame[:, :, :3]  # Assuming the first three channels are RGB
                        depth_image = frame[:, :, 2]  # Assuming depth is encoded in the last channel

                        # Display RGB and Depth (depth visualization may need normalization)
                        cv2.imshow("RGB Image", rgb_image)
                        depth_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                        cv2.imshow("Depth Image", depth_normalized)
                    # cv2.imshow("Camera Feed", frame)

                    # Wait for a key press for 1 ms
                    key = cv2.waitKey(1) & 0xFF
                    # If the 'space' key is pressed, save the image
                    if auto and time.time() - start_time > time_delta:
                        start_time = time.time()
                        save_image(data, fs, camera_indexes, cap_read)

                        # Exit the loop if the 'q' key is pressed
                    if key == ord(" "):
                        save_image(data, fs, camera_indexes, cap_read)
                    if key == ord('q'):
                        print("Exiting...")
                        break
                else:
                    print(f"Failed to grab frame from camera with index {camera_indexes}")

        else:
            print(f"Camera with index {camera_indexes} is not available")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        for cap in caps.values():
            cap.release()  # Release the resource after testing the camera

        cv2.destroyAllWindows()
        print("All cameras have been tested and windows have been closed.")

def save_image(data, fs, camera_indexes, cap_read):
    # Save the frame as an image in the test_images folder
    sd.play(data, fs)
    sd.wait()
    for i in camera_indexes:
        counter = 1
        file_path = os.path.join(output_dir, f"captured_image_{i}_{counter}.png")

        while os.path.exists(file_path):
            counter += 1
            file_path = os.path.join(output_dir, f"captured_image_{i}_{counter}.png")
        cv2.imwrite(file_path, cap_read[i][1])
        print(f"Image saved at {file_path}")

if __name__ == "__main__":
    output_dir = "cam_images"
    os.makedirs(output_dir, exist_ok=True)
    asyncio.run(take_image([0], output_dir))