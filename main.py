import cv2
import numpy as np
from cv2 import aruco
from tqdm import tqdm

ARUCO_DICT = cv2.aruco.DICT_6X6_250
SQUARES_VERTICALLY = 7
SQUARES_HORIZONTALLY = 5
SQUARE_LENGTH = 0.022
MARKER_LENGTH = 0.014


def calibrate_camera_aruco(video_path):
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    board = aruco.CharucoBoard((SQUARES_HORIZONTALLY, SQUARES_VERTICALLY), SQUARE_LENGTH, MARKER_LENGTH, dictionary)
    params = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(dictionary, params)
    cap = cv2.VideoCapture(video_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    all_charuco_ids = []
    all_charuco_corners = []

    process_bar = tqdm(total=total_frames, desc='frame', position=0)
    while True:

        ret, frame = cap.read()
        process_bar.update(1)
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        marker_corners, marker_ids, _ = detector.detectMarkers(gray)

        if len(marker_corners) > 0:
            ret, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
                marker_corners, marker_ids, gray, board)

            if charuco_corners is not None and charuco_ids is not None and len(charuco_corners) == 24:
                all_charuco_ids.append(charuco_ids)
                all_charuco_corners.append(charuco_corners)

    cap.release()
    ret, mtx, dist, _, _ = aruco.calibrateCameraCharuco(all_charuco_corners, all_charuco_ids, board,
                                                        [width, height], None, None)
    return ret, mtx, dist


def main():
    ret, mtx, dist = calibrate_camera_aruco(r'data\1_calibration.mp4')

    if ret:
        print("Calibration successful!")
        print("Camera matrix:\n", mtx)
        print("Distortion coefficients:\n", dist)
        np.savez('calibration_results.npz', mtx=mtx, dist=dist)
    else:
        print("Calibration failed.")


if __name__ == "__main__":
    main()
