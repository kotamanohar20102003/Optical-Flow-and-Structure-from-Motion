import cv2
import numpy as np

def track_two_frames(frame1_path, frame2_path):
    img1 = cv2.imread(frame1_path)
    img2 = cv2.imread(frame2_path)

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    p0 = cv2.goodFeaturesToTrack(gray1, maxCorners=10, qualityLevel=0.3, minDistance=7)

    p1, st, err = cv2.calcOpticalFlowPyrLK(gray1, gray2, p0, None)

    for i, (new, old) in enumerate(zip(p1, p0)):
        a, b = new.ravel()
        c, d = old.ravel()

        print(f"Point {i}: Old: ({c:.2f},{d:.2f}) -> New: ({a:.2f},{b:.2f})")

        cv2.circle(img2, (int(a), int(b)), 5, (0, 255, 0), -1)

    cv2.imshow("Tracked Points", img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    track_two_frames("data/frame1.jpg", "data/frame2.jpg")