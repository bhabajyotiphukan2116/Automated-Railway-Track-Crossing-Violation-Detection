import cv2
import numpy as np
import argparse
from ultralytics import YOLO


def point_in_polygon(point, polygon):
    return cv2.pointPolygonTest(polygon, point, False) >= 0


def main(input_video, output_video):

    # Load YOLO model
    model = YOLO("/content/drive/MyDrive/Project/bhaba/yolov8l.pt")  # Use large model for better person detection
    device = 0  # GPU

    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print("Error opening video")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(
        output_video,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height)
    )

    print("Running Railway Track Crossing Detection ...")

    # manually define rectangle adjust to your video)

    track_polygon = np.array([
        [int(0.22 * width), int(0.62 * height)],  # top-left rail
        [int(0.68 * width), int(0.78 * height)],  # top-right rail (reduced)
        [int(0.65 * width), int(0.85 * height)],  # bottom-right rail (pulled back)
        [int(0.20 * width), int(0.67 * height)]   # bottom-left rail (lowered)
    ], dtype=np.int32)

    track_polygon = track_polygon.reshape((-1,1,2))


    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(
            frame,
            conf=0.4,
            imgsz=1280,
            device=device,
            classes=[0],  # person only
            verbose=False
        )

        # Draw divider region
        overlay = frame.copy()
        cv2.fillPoly(overlay, [track_polygon], (255, 0, 0))
        frame = cv2.addWeighted(overlay, 0.15, frame, 0.85, 0)

        if results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()

            for box in boxes:
                x1, y1, x2, y2 = map(int, box)

                person_height = y2 - y1

                # 🔥 Distance filtering
                if person_height < 0.12 * height:
                    continue

                # Compute centroid
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                # Check if inside divider
                if point_in_polygon((cx, cy), track_polygon):
                    color = (0, 0, 255)
                    label = "ANOMALY - CROSSING RAILWAY TRACK"
                else:
                    color = (0, 255, 0)
                    label = "NORMAL"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )


        out.write(frame)

    cap.release()
    out.release()
    print("Saved:", output_video)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    args = parser.parse_args()

    main(args.input, args.output)



