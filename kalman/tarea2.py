"""Person Tracking Using the Kalman Filter."""

import cv2
import numpy as np
import torch
from sort import Sort

model = torch.hub.load("ultralytics/yolov5", "yolov5n", pretrained=True)


def object_tracking():
    """Video loading, resizing, and tracking."""
    cap = cv2.VideoCapture(r"test.mp4")
    mov_tracker = Sort()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_resized = cv2.resize(frame, (640, 360))

        results = model(frame_resized)
        results = results.xyxy[0].numpy()
        people_det = results[results[:, 5] == 0]

        dets = []
        for *xyxy, conf, cls in people_det:
            x1, y1, x2, y2 = map(int, xyxy)
            dets.append([x1, y1, x2, y2, conf])
        dets = np.array(dets)

        trackers = mov_tracker.update(dets)

        for d in trackers:
            x1, y1, x2, y2, track_id = map(int, d[:5])
            cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(
                frame_resized,
                str(track_id),
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_PLAIN,
                2,
                (0, 255, 0),
                thickness=2,
            )

        cv2.imshow("view", frame_resized)

        key = cv2.waitKey(1)
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    object_tracking()
