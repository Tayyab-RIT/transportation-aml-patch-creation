from ultralytics import YOLO
import os
from pathlib import Path

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")


def main() -> None:
    root = Path(__file__).resolve().parent
    weights = root / "pre_trained.pt"
    image = root / "ex1.jpg"

    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")
    if not image.exists():
        raise FileNotFoundError(f"Image not found: {image}")

    model = YOLO(str(weights))
    results = model.predict(
        source=str(image),
        save=True,
        project=str(root / "runs"),
        name="infer_pretrained",
        exist_ok=True,
        conf=0.25,
    )

    print(f"Detections: {len(results[0].boxes)}")
    print(f"Saved output under: {root / 'runs' / 'infer_pretrained'}")


if __name__ == "__main__":
    main()
