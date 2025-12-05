import os
import cv2
import json
from fer import FER

def analyze_emotions(
        frame_dir="frames",
        crop_dir="face_crops",
        output_json="emotion_results.json",
        output_video="emotion_preview.mp4"):

    # Emotion detector
    detector = FER(mtcnn=False)

    # Grab all frames
    frame_files = sorted(os.listdir(frame_dir))

    # Prepare video writer using first frame as reference
    first_frame = cv2.imread(os.path.join(frame_dir, frame_files[0]))
    height, width, _ = first_frame.shape

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_out = cv2.VideoWriter(output_video, fourcc, 30, (width, height))

    # Emotion results dictionary
    results = {}

    # Load crop files so we know which face belongs to which frame
    crop_files = sorted(os.listdir(crop_dir))

    # Build mapping: frame_id → [list of crops belonging to that frame]
    frame_to_crops = {}
    for fname in crop_files:
        # crop filenames look like: 00012_0.jpg
        frame_id = int(fname.split("_")[0])
        frame_to_crops.setdefault(frame_id, []).append(fname)

    # Loop through every frame
    for idx, frame_name in enumerate(frame_files):
        frame_path = os.path.join(frame_dir, frame_name)
        frame = cv2.imread(frame_path)

        if frame is None:
            continue

        frame_results = []

        # Retrieve all crops that belong to this frame
        crops_for_frame = frame_to_crops.get(idx, [])

        for crop_name in crops_for_frame:
            crop_path = os.path.join(crop_dir, crop_name)
            crop_img = cv2.imread(crop_path)

            if crop_img is None:
                continue

            # Run emotion detection
            emotions = detector.detect_emotions(crop_img)

            # Store or fallback
            if emotions:
                emo_dict = emotions[0]["emotions"]
            else:
                emo_dict = {
                    "angry": 0, "disgust": 0, "fear": 0,
                    "happy": 0, "sad": 0, "surprise": 0, "neutral": 1
                }

            # Save to JSON results
            results[crop_name] = emo_dict

            # WRITE emotion label ONTO FRAME
            # Grab the face bounding box from YOLO filename
            # (we need to extract it again—YOLO stage saved coords inside filename)
            # Example name: 00012_0.jpg → no bbox info so we approximate from crop size
            # We'll draw text only.
            emo_label = max(emo_dict, key=emo_dict.get)
            confidence = emo_dict[emo_label]

            # Put text in upper-left of the frame for now
            cv2.putText(
                frame,
                f"{emo_label} ({confidence:.2f})",
                (30, 40 + 30 * len(frame_results)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            frame_results.append((emo_label, confidence))

        # Write frame into output video
        video_out.write(frame)

    # Save JSON file
    with open(output_json, "w") as f:
        json.dump(results, f, indent=4)

    video_out.release()
    print(f"[✓] Emotion JSON saved → {output_json}")
    print(f"[✓] Emotion preview video saved → {output_video}")


if __name__ == "__main__":
    analyze_emotions()
