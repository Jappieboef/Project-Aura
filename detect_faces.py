from ultralytics import YOLO
import cv2
import os

def detect_faces(
        video_path="Australia.mp4",
        output_frames="frames",
        output_crops="face_crops",
        model_path = "yolov8m-face-lindevs.pt"
    ):

    print(" Loading YOLO face model...")
    model = YOLO(model_path)

    # Ensure folders exist
    os.makedirs(output_frames, exist_ok=True)
    os.makedirs(output_crops, exist_ok=True)

    print(f"🎥 Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(" ERROR: Could not open the video file.")
        return

    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video reached.")
            break

        # Save raw frame for later
        frame_path = os.path.join(output_frames, f"{frame_index:05d}.jpg")
        cv2.imwrite(frame_path, frame)

        # Run YOLO face detection
        results = model.predict(frame, verbose=False)

        if len(results) == 0:
            print(f"⚠ No detection output for frame {frame_index}")
            frame_index += 1
            continue

        det = results[0]

        if det.boxes is None or len(det.boxes.xyxy) == 0:
            print(f" No faces found in frame {frame_index}")
        else:
            print(f" Found {len(det.boxes.xyxy)} faces in frame {frame_index}")

        # Draw boxes and save crops
        for i, box in enumerate(det.boxes.xyxy):
            x1, y1, x2, y2 = map(int, box)
            crop = frame[y1:y2, x1:x2]

            crop_path = os.path.join(output_crops, f"{frame_index:05d}_{i}.jpg")
            cv2.imwrite(crop_path, crop)

            # Draw bounding box on frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, "face", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Show the current frame
        cv2.imshow("Face Detection Preview", frame)

        # Press Q to quit early
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print(" Exiting early...")
            break

        frame_index += 1

    cap.release()
    cv2.destroyAllWindows()
    print(" Face detection complete.")


if __name__ == "__main__":
    detect_faces()
