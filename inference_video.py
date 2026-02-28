import os
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")


def main() -> None:
    root = Path(__file__).resolve().parent
    weights = root / "pre_trained.pt"
    video_path = root / "samples" / "videos" / "vid1.mp4"
    outputs_dir = root / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    output_path = outputs_dir / "vid1_annotated.mp4"

    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    model = YOLO(str(weights))
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    window_name = "YOLO-TS Real-time Inference (press q to quit)"
    target_fps = 24.0
    frame_interval = 1.0 / target_fps
    prev_time = time.time()
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    # output_fps = source_fps if source_fps and source_fps > 0 else target_fps
    output_fps = 10.0
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"), # type: ignore
        output_fps,
        (1280, 720),
    )

    while True:
        loop_start = time.time()
        ok, frame = cap.read()
        if not ok:
            break
        frame_720p = cv2.resize(frame, (1280, 720))

        results = model.predict(frame_720p, conf=0.25, verbose=False)
        annotated = results[0].plot()

        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now
        cv2.putText(
            annotated,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow(window_name, annotated)
        writer.write(annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        elapsed = time.time() - loop_start
        if elapsed < frame_interval:
            time.sleep(frame_interval - elapsed)

    cap.release()
    writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
