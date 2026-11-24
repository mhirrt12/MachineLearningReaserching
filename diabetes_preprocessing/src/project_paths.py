
from pathlib import Path

# The project root is the parent folder of src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Main project folders
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

# Ensure output folders exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
