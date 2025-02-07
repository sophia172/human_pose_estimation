import time

from cv2 import findChessboardCorners, imshow, waitKey, destroyAllWindows, cvtColor, COLOR_BGR2GRAY, imwrite, imread, \
    hconcat, undistort, drawChessboardCorners, calibrateCamera, VideoCapture
import numpy as np
import sounddevice as sd
import soundfile as sf
from logger import logging

checkerboard_size = (8, 6)  #  grid of checkerboard corners
square_size = 28.5  # Size of a square in mm


object_points = np.zeros((np.prod(checkerboard_size), 3), dtype=np.float32)
object_points[:, :2] = np.indices(checkerboard_size).T.reshape(-1, 2)
object_points *= square_size

def detectCorners(img):
    gray = cvtColor(img, COLOR_BGR2GRAY)
    # Find the checkerboard corners
    ret, corners = findChessboardCorners(gray, checkerboard_size, None)
    if ret:
        return (object_points, corners, gray, ret)
    return None

def calibrate(obj_points, img_points, gray):
    # Calibrate the camera using the detected points
    ret, camera_matrix, distortion_coeffs, rvecs, tvecs = calibrateCamera(
                            obj_points,
                            img_points,
                            gray.shape[::-1],
                            None,
                            None)


    # camera_matrix: Intrinsic camera matrix (focal length, principal point)
    #camera_matrix = [[fx, 0, cx]
                    # [0, fy, cy]
                    # [0,  0, 1]]
    # distortion_coeffs: Distortion coefficients (radial and tangential distortion)
    logging.info(f"Camera Matrix: {camera_matrix}")
    logging.info(f"Distortion Coefficients: {distortion_coeffs}")
    return camera_matrix, distortion_coeffs



if __name__ == '__main__':
    camera_idx = 0
    obj_points_list = []
    img_points_list = []
    gray = None
    calibrated = False
    sound_data, fs = sf.read("sound/camera-shutter-click.mp3", dtype='float32')


    try:
        cap = VideoCapture(camera_idx)
        i = 0
        if cap.isOpened():
            while True:
                ret, frame = cap.read()
                i += 1
                img_pairs = detectCorners(frame)
                if img_pairs is not None:
                    obj_points, img_points, gray, chess_ret = detectCorners(frame)
                    obj_points_list.append(obj_points)
                    img_points_list.append(img_points)
                    drawChessboardCorners(frame, checkerboard_size, img_points, chess_ret)

                # Wait for a key press for 1 ms
                key = waitKey(1) & 0xFF
                if key == ord(" ") and len(obj_points_list) > 0:
                    camera_matrix, distortion_coeffs = calibrate(obj_points_list, img_points_list, gray)
                    np.savez(
                        f"config/calibration_matrix_camera_{camera_idx}.npz",
                        camera_matrix=camera_matrix,
                        distortion_coeffs=distortion_coeffs
                    )
                    sd.play(sound_data, fs)
                    sd.wait()
                    calibrated = True


                if calibrated:
                    frame = undistort(frame, camera_matrix, distortion_coeffs)

                imshow('Press Space for Calibration', frame)

                if key == ord('q'):
                    print("Exiting...")
                    break
    except Exception as e:
        print(f"{e}")
    finally:
        # Release the camera and close all OpenCV windows
        cap.release()
        destroyAllWindows()
