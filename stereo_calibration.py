import cv2
import numpy as np
import glob
import os
from numpy.ma.core import left_shift
from logger import logging
# Prepare object points (3D world coordinates of the checkerboard corners)
# For a checkerboard of size (rows, cols) with square size in millimeters
checkerboard_size = (8, 6)  # 9x6 grid of checkerboard corners
square_size = 28.5  # Size of a square in mm
debug = False

def detect_corners_stereo(left_images):
    obj_points = []  # 3D points in real world space
    img_points_left = []  # 2D points in left image plane
    img_points_right = []  # 2D points in right image plane
    gray_left = None
    gray_right = None
    imageSize = None
    # Prepare object points: (0,0,0), (1,0,0), ..., for the checkerboard
    object_points = np.zeros((np.prod(checkerboard_size), 3), dtype=np.float32)
    object_points[:, :2] = np.indices(checkerboard_size).T.reshape(-1, 2)
    object_points *= square_size

    for left_path in left_images:
        right_path = left_path.replace("_1_", "_0_")

        img_left = cv2.imread(left_path)
        img_right = cv2.imread(right_path)
        gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

        # Find the checkerboard corners in both images
        ret_left, corners_left = cv2.findChessboardCorners(gray_left, checkerboard_size, None)
        ret_right, corners_right = cv2.findChessboardCorners(gray_right, checkerboard_size, None)
        logging.info(f" ret and corner for {os.path.basename(left_path)} are \n"
                     f"ret:  {ret_left}, {ret_right}, \n"
                     f"corner: {corners_left}, {corners_right}")
        if ret_left and ret_right:
            obj_points.append(object_points)
            img_points_left.append(corners_left)
            img_points_right.append(corners_right)

            if debug:
                # Draw and display the corners for debugging
                cv2.drawChessboardCorners(img_left, checkerboard_size, corners_left, ret_left)
                cv2.drawChessboardCorners(img_right, checkerboard_size, corners_right, ret_right)
                frame = cv2.hconcat([img_left, img_right])
                cv2.imshow('Left', frame)
                key = cv2.waitKey(10000) & 0xFF
                # If the 'space' key is pressed, save the image and update captured_image
                if key == ord(' '):
                    # Save the frame as an image in the test_images folder
                    continue
                if key == ord('q'):
                    print("Exiting...")
                    break


    cv2.destroyAllWindows()

    return obj_points, img_points_left, img_points_right, gray_left

def stereo_calibrate(obj_points, img_points_left, img_points_right, img_size, camera_matrix_left, dist_coeffs_left, camera_matrix_right, dist_coeffs_right):
    # Stereo calibration
    flags = cv2.CALIB_FIX_INTRINSIC  # Fix intrinsic parameters from individual calibration
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)

    ret, camera_matrix_left, dist_coeffs_left, camera_matrix_right, dist_coeffs_right, R, T, E, F = cv2.stereoCalibrate(
        obj_points, img_points_left, img_points_right,
        camera_matrix_left, dist_coeffs_left,
        camera_matrix_right, dist_coeffs_right,
        img_size, criteria=criteria, flags=flags
    )
    logging.info(f" retval: {ret} \n")
    logging.info(f"camera_matrix_left: {camera_matrix_left} \n")
    logging.info(f"dist_coeffs_left: {dist_coeffs_left} \n")
    logging.info(f"camera_matrix_right: {camera_matrix_right} \n")
    logging.info(f"dist_coeffs_right: {dist_coeffs_right} \n")

    logging.info(f"Rotation Matrix (R):\n{R}")
    logging.info(f"Translation Vector (T):\n{T} and it's magnitude {np.linalg.norm(T)}")
    logging.info(f"E:\n{E}")
    logging.info(f"F:\n{F}")
    return R, T, E, F

def stereo_rectify(camera_matrix_left, dist_coeffs_left, camera_matrix_right, dist_coeffs_right, img_size, R, T):
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        camera_matrix_left, dist_coeffs_left,
        camera_matrix_right, dist_coeffs_right,
        img_size, R, T, alpha=0
    )
    logging.info("Rectification Matrices:")
    logging.info(f"R1:\n{R1}")
    logging.info(f"R2:\n{R2}")
    logging.info(f"Projection Matrices (P1, P2):\n{P1}\n{P2}")
    return R1, R2, P1, P2, Q

if __name__ == '__main__':
    from intrinsic_calibration import calibrate
    import glob
    left_images = glob.glob("config/captured_image_1_*.png")

    obj_points, img_points_left, img_points_right, gray_left = detect_corners_stereo(left_images)
    camera_matrix_left, dist_coeffs_left = calibrate(obj_points, img_points_left, gray_left)
    camera_matrix_right, dist_coeffs_right = calibrate(obj_points, img_points_right, gray_left)
    img_size = (1920, 1080)
    R, T, E, F = stereo_calibrate(obj_points, img_points_left, img_points_right, img_size, camera_matrix_left, dist_coeffs_left,
                     camera_matrix_right, dist_coeffs_right)

    assert camera_matrix_left.shape == (3, 3)
    assert camera_matrix_right.shape == (3, 3)
    assert len(dist_coeffs_left[0]) <= 5  # Typically up to 5 distortion coefficients
    assert len(dist_coeffs_right[0]) <= 5
    assert R.shape == (3, 3)
    assert T.shape == (3, 1)
    assert img_size[0] > 0 and img_size[1] > 0
    R1, R2, P1, P2, Q = stereo_rectify(camera_matrix_left, dist_coeffs_left, camera_matrix_right, dist_coeffs_right, img_size, R, T)
