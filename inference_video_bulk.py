import os
from pathlib import Path

import cv2
from ultralytics import YOLO

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")


def main() -> None:
    root = Path(__file__).resolve().parent
    models_dir = root / "models"
    videos_dir = root / "samples" / "videos"
    outputs_dir = root / "outputs"

    if not models_dir.exists():
        raise FileNotFoundError(f"Models directory not found: {models_dir}")
    if not videos_dir.exists():
        raise FileNotFoundError(f"Videos directory not found: {videos_dir}")

    # Get all model files (.pt)
    model_files = sorted(models_dir.glob("*.pt"))
    if not model_files:
        raise FileNotFoundError(f"No .pt model files found in {models_dir}")

    # Get all video files (common extensions)
    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    video_files = sorted(
        [f for f in videos_dir.iterdir() if f.suffix.lower() in video_extensions]
    )
    if not video_files:
        raise FileNotFoundError(f"No video files found in {videos_dir}")

    print(f"Found {len(model_files)} models and {len(video_files)} videos")
    print("-" * 60)

    # Run inference for each model on all videos
    for model_path in model_files:
        model_name = model_path.stem  # e.g., "CCTSDB2021_best"
        print(f"\nProcessing model: {model_name}")

        model = YOLO(str(model_path))
        output_folder = outputs_dir / model_name
        output_folder.mkdir(parents=True, exist_ok=True)

        for video_path in video_files:
            video_name = video_path.stem
            output_path = output_folder / f"{video_name}_annotated.mp4"

            print(f"  - Processing {video_path.name}...", end=" ", flush=True)

            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                print(f"ERROR: Could not open video")
                continue

            # Get video properties
            source_fps = cap.get(cv2.CAP_PROP_FPS)
            output_fps = 10.0
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Create VideoWriter with 720p resolution
            writer = cv2.VideoWriter(
                str(output_path),
                cv2.VideoWriter_fourcc(*"mp4v"),  # type: ignore
                output_fps,
                (1280, 720),
            )

            processed_frames = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                # Resize to 720p
                frame_720p = cv2.resize(frame, (1280, 720))

                # Run inference
                results = model.predict(frame_720p, conf=0.25, verbose=False)
                annotated = results[0].plot()

                # Write frame
                writer.write(annotated)
                processed_frames += 1

            cap.release()
            writer.release()

            print(f"{processed_frames} frames → {output_path.name}")

        print(f"  → Saved to: {output_folder}")

    print("\n" + "=" * 60)
    print("Bulk video inference complete!")


if __name__ == "__main__":
    main()
