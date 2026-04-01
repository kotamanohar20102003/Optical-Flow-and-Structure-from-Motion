import cv2

def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if count % 30 == 0:
            cv2.imwrite(f"data/frame_{count}.jpg", frame)

        count += 1

    cap.release()


if __name__ == "__main__":
    extract_frames("data/video1.mp4")