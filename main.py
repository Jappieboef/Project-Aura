import cv2
import os

def extract_frames(video_path, output_dir="frames", fps_sample=5):
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(original_fps / fps_sample)

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            cv2.imwrite(f"{output_dir}/frame_{saved_count:05d}.jpg", frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"Saved {saved_count} frames")

if __name__ == "__main__":
    extract_frames("prac.mp4")   # <---- ADD THIS name of video file
