import cv2
import numpy as np
from logger import logging

def preprocess_images(left_img, right_img, camera_matrix_left, dist_coeffs_left, camera_matrix_right, dist_coeffs_right, R1, R2, P1, P2, img_size):
    # Compute rectification maps
    map1_left, map2_left = cv2.initUndistortRectifyMap(camera_matrix_left, dist_coeffs_left, R1, P1, img_size, cv2.CV_16SC2)
    map1_right, map2_right = cv2.initUndistortRectifyMap(camera_matrix_right, dist_coeffs_right, R2, P2, img_size, cv2.CV_16SC2)
    logging.info(f"initUndistortRectifyMap Finished")
    # Apply rectification
    rectified_left = cv2.remap(left_img, map1_left, map2_left, cv2.INTER_LINEAR)
    rectified_right = cv2.remap(right_img, map1_right, map2_right, cv2.INTER_LINEAR)
    logging.info(f"rectification Finished")
    return rectified_left, rectified_right

def compute_disparity_sgbm(rectified_left, rectified_right):
    # Create StereoSGBM object
    min_disp = 0
    num_disp = 16 * 5  # Must be divisible by 16
    block_size = 5

    stereo_sgbm = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=block_size,
        P1=8 * 3 * block_size**2,  # Smoothness parameter
        P2=32 * 3 * block_size**2,  # Smoothness parameter
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    # Compute disparity map
    disparity = stereo_sgbm.compute(cv2.cvtColor(rectified_left, cv2.COLOR_BGR2GRAY),
                                    cv2.cvtColor(rectified_right, cv2.COLOR_BGR2GRAY))
    logging.info(f"disparity SGBM Finished")
    # Normalize for visualization
    # disparity = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    # disparity = np.uint8(disparity)
    logging.info(f"disparity.shape: {disparity.shape}")
    return disparity

def disparity_to_depth(disparity, Q):
    logging.info(f"Started desparity to depth function")
    # Reproject disparity to 3D space
    points_3d = cv2.reprojectImageTo3D(disparity, Q)
    depth = points_3d[:, :, 2]  # Extract depth (Z-axis)
    logging.info(f"Extract depth (Z-axis) Finished with maximum value as {depth.max()}")
    if depth is None or depth.size == 0:
        raise ValueError("Input depth map is empty or invalid.")
    logging.info(f"Double check depth information: {depth.max(), depth.min()}")
    # Normalize for visualization
    depth_normalized = cv2.normalize(
        src=depth,  # Input array
        dst=None,  # Let OpenCV allocate the output
        alpha=0,  # Minimum value for normalization
        beta=255,  # Maximum value for normalization
        norm_type=cv2.NORM_MINMAX,  # Normalization type
        dtype=cv2.CV_8U  # Ensure the output is in 8-bit format
    )
    logging.info(f"Double check normalised depth information: {depth_normalized.max(), depth_normalized.min()}")

    logging.info(f"disparity_to_depth Finished")

    return depth_normalized, points_3d

def imgP_to_objP(fx, # Focal length in pixels
                 fy,
                 cx, # Principal point x-coordinate
                 cy, # Principal point y-coordinate
                 u_L, # Image point1 in the left camera
                 v_L, # Image point2 in the left camera
                 u_R, # Image point1 in the right camera
                 v_R, # Image point2 in the right camera
                 baseline,  # Baseline distance (in meters)
                disparity):
    # Example for a disparity map
    h, w = disparity.shape

    # Create a meshgrid for pixel coordinates
    u, v = np.meshgrid(np.arange(w), np.arange(h))

    # Compute depth
    Z = (fx * baseline) / (disparity + 1e-6)  # Avoid division by zero

    # Compute X and Y
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy

    # Combine into 3D points
    points_camera = np.stack((X, Y, Z), axis=-1)
if __name__ == '__main__':
    from intrinsic_calibration import calibrate
    from extrinsic_calibration import stereo_calibrate, stereo_rectify, detect_corners_stereo
    import glob
    left_images = glob.glob("config/captured_image_1_*.png")

    obj_points, img_points_left, img_points_right, gray_left = detect_corners_stereo(left_images)

    camera_matrix_left, dist_coeffs_left = calibrate(obj_points, img_points_left, gray_left)
    camera_matrix_right, dist_coeffs_right = calibrate(obj_points, img_points_right, gray_left)
    img_size = (1920, 1080)
    R, T, E, F = stereo_calibrate(obj_points, img_points_left, img_points_right, img_size, camera_matrix_left, dist_coeffs_left,
                     camera_matrix_right, dist_coeffs_right)

    R1, R2, P1, P2, Q = stereo_rectify(camera_matrix_left, dist_coeffs_left, camera_matrix_right, dist_coeffs_right, img_size, R, T)

    cap1 = cv2.VideoCapture(0)
    cap2 = cv2.VideoCapture(2)

    cv2.namedWindow("Camera Feed", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("Depth Map", cv2.WINDOW_AUTOSIZE)
    if cap1.isOpened() and cap2.isOpened():
        print(f"Successfully accessed camera 0 and 1")
        while True:
            ret1, left_img = cap1.read()
            ret2, right_img = cap2.read()
            if ret1 and ret2:
                # Display the video feed in the window
                frame = cv2.hconcat([left_img, right_img])
                cv2.imshow("Camera Feed", frame)


            # Wait for a key press for 1 ms
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                logging.info(f"Start Rectify stereo images")
                # Rectify stereo images (replace with your actual calibration results)
                rectified_left, rectified_right = preprocess_images(left_img, right_img,
                                                                    camera_matrix_left, dist_coeffs_left,
                                                                    camera_matrix_right, dist_coeffs_right,
                                                                    R1, R2, P1, P2, img_size)

                # Compute disparity (SGM or BM)
                disparity = compute_disparity_sgbm(rectified_left, rectified_right)

                # Convert disparity to depth map
                depth_normalized, points_3d = disparity_to_depth(disparity, Q)
                # Display depth map
                cv2.imshow('Depth Map', depth_normalized)

            # If the 'q' key is pressed, exit
            if key == ord('q'):
                logging.info("Exiting...")
                break
    cv2.destroyAllWindows()