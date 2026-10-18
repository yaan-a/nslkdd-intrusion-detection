from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_backend_source_files_exist():
    for relative in ["backend/app.py", "backend/inference.py",
                     "backend/schema.json", "backend/examples.json",
                     "backend/requirements.txt", "README.md"]:
        assert (ROOT / relative).exists(), f"missing {relative}"


def test_model_and_data_files_exist():
    assert (ROOT / "model/nslkdd_model.pkl").exists()
    official_test = ROOT / "data/KDDTest+.csv"
    merged = ROOT / "data/NSL-KDD_merged.csv"
    assert official_test.exists() or merged.exists()
