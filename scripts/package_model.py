from pathlib import Path
import shutil

PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXPERIMENT_ID = "743355619123453207"
RUN_ID = "947054c746e9405d93c5c76017b6683e"
MODEL_ID = "m-f14beb5747844474a66f68d948d02ebc"

MLRUNS_DIR = PROJECT_ROOT / "mlruns"
MODEL_SOURCE = (
    MLRUNS_DIR / EXPERIMENT_ID / "models" / MODEL_ID / "artifacts"
)
METADATA_SOURCE = (
    MLRUNS_DIR / EXPERIMENT_ID / RUN_ID / "artifacts"
)

MODEL_DESTINATION = PROJECT_ROOT / "src" / "serving" / "model"


def main():
    required_model_files = [
        "MLmodel",
        "model.ubj",
        "requirements.txt",
    ]

    for filename in required_model_files:
        if not (MODEL_SOURCE / filename).is_file():
            raise FileNotFoundError(
                f"Missing model file: {MODEL_SOURCE / filename}"
            )

    for filename in ["feature_columns.txt", "preprocessing.pkl"]:
        if not (METADATA_SOURCE / filename).is_file():
            raise FileNotFoundError(
                f"Missing metadata file: {METADATA_SOURCE / filename}"
            )

    if MODEL_DESTINATION.exists():
        shutil.rmtree(MODEL_DESTINATION)

    MODEL_DESTINATION.parent.mkdir(parents=True, exist_ok=True)

    shutil.copytree(MODEL_SOURCE, MODEL_DESTINATION)

    for filename in ["feature_columns.txt", "preprocessing.pkl"]:
        shutil.copy2(
            METADATA_SOURCE / filename,
            MODEL_DESTINATION / filename,
        )

    print("Model packaging completed successfully.")
    print(f"Destination: {MODEL_DESTINATION}")
    print("\nPackaged files:")

    for file in sorted(MODEL_DESTINATION.iterdir()):
        print(f"  {file.name}")


if __name__ == "__main__":
    main()