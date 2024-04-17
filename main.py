import cv2
from cv2 import aruco

ARUCO_DICT = cv2.aruco.DICT_6X6_250
SQUARES_VERTICALLY = 7
SQUARES_HORIZONTALLY = 5
SQUARE_LENGTH = 0.03
MARKER_LENGTH = 0.015


def calibrate_camera_aruco(video_path):
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    board = aruco.CharucoBoard((SQUARES_VERTICALLY, SQUARES_HORIZONTALLY), SQUARE_LENGTH, MARKER_LENGTH, dictionary)
    params = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(dictionary, params)
    cap = cv2.VideoCapture(video_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    all_charuco_ids = []
    all_charuco_corners = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        marker_corners, marker_ids, _ = detector.detectMarkers(gray)

        if len(marker_corners) > 0:
            ret, detected_corners, detected_ids = aruco.interpolateCornersCharuco(
                marker_corners, marker_ids, gray, board)

            if detected_corners is not None and detected_ids is not None and len(detected_corners) > 3:
                ret, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
                    marker_corners, marker_ids, gray, board)

                if charuco_corners is not None and charuco_ids is not None:
                    all_charuco_ids.extend(charuco_ids)
                    all_charuco_corners.append(charuco_corners)

    cap.release()
    ret, mtx, dist, _, _ = cv2.aruco.calibrateCameraCharuco(all_charuco_corners, all_charuco_ids, board,
                                                            [width, height], None, None)
    return ret, mtx, dist


def main():
    ret, mtx, dist = calibrate_camera_aruco(r'data\1_calibration.mp4')

    if ret:
        print("Calibration successful!")
        print("Camera matrix:\n", mtx)
        print("Distortion coefficients:\n", dist)
    else:
        print("Calibration failed.")


if __name__ == "__main__":
    main()
