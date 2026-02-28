import os
from pathlib import Path

from ultralytics import YOLO

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")


def main() -> None:
    root = Path(__file__).resolve().parent
    models_dir = root / "models"
    images_dir = root / "samples" / "images"
    outputs_dir = root / "outputs"

    if not models_dir.exists():
        raise FileNotFoundError(f"Models directory not found: {models_dir}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    # Get all model files (.pt)
    model_files = sorted(models_dir.glob("*.pt"))
    if not model_files:
        raise FileNotFoundError(f"No .pt model files found in {models_dir}")

    # Get all image files (common extensions)
    image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
    image_files = sorted(
        [f for f in images_dir.iterdir() if f.suffix.lower() in image_extensions]
    )
    if not image_files:
        raise FileNotFoundError(f"No image files found in {images_dir}")

    print(f"Found {len(model_files)} models and {len(image_files)} images")
    print("-" * 60)

    # Run inference for each model on all images
    for model_path in model_files:
        model_name = model_path.stem  # e.g., "CCTSDB2021_best"
        print(f"\nProcessing model: {model_name}")

        model = YOLO(str(model_path))
        output_folder = outputs_dir / model_name
        output_folder.mkdir(parents=True, exist_ok=True)

        for image_path in image_files:
            print(f"  - Inferring {image_path.name}...", end=" ")
            results = model.predict(
                source=str(image_path),
                save=True,
                project=str(outputs_dir),
                name=model_name,
                exist_ok=True,
                conf=0.25,
                verbose=False,
            )
            detections = len(results[0].boxes)
            print(f"{detections} detections")

        print(f"  → Saved to: {output_folder}")

    print("\n" + "=" * 60)
    print("Bulk inference complete!")


if __name__ == "__main__":
    main()
