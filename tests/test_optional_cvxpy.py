"""The optimizer package works without cvxpy unless the joint design is requested. Run: python tests/test_optional_cvxpy.py"""
import subprocess, sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
code = ("import sys; sys.modules['cvxpy'] = None; "          # makes `import cvxpy` raise ImportError
        "from src.utils.config import settings; settings.load_config(); "
        "from src.opt import design, evaluate, loss, beamforming; print('IMPORT_OK')")
r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
ok = "IMPORT_OK" in r.stdout
print(f"[{'PASS' if ok else 'FAIL'}] design / evaluate / loss / beamforming import without cvxpy", "" if ok else r.stderr[-300:])
print("\nALL PASS" if ok else "\nSOME CHECKS FAILED")
