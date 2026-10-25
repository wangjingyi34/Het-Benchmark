"""Run the published measured-validation framework."""
import runpy
from pathlib import Path
runpy.run_path(str(Path(__file__).resolve().parent/'experiments/20261025/run.py'),run_name='__main__')
