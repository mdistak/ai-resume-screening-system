"""Runs every test script, prints a summary and saves the full log to docs/test_log.txt."""
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TESTS = ["test_parser.py", "test_preprocessing.py", "test_skills.py", "test_matcher.py",
         "test_evaluation.py", "test_visualization.py", "test_database.py",
         "test_step2_features.py", "test_end_to_end.py"]


def run(test):
    result = subprocess.run(
        [sys.executable, str(ROOT / "tests" / test)], cwd=ROOT, capture_output=True,
        text=True, encoding="utf-8", errors="replace",
        env=dict(os.environ, PYTHONIOENCODING="utf-8"))
    output = result.stdout + result.stderr
    passed, failed = output.count("[PASS]"), output.count("[FAIL]")
    ok = result.returncode == 0 and failed == 0
    return ok, passed, failed, output


def main():
    log = [f"Test run: {datetime.now():%Y-%m-%d %H:%M:%S}",
           f"Python: {sys.version.split()[0]}", ""]
    all_ok, total_passed, total_failed = True, 0, 0
    print(f"{'Test script':28} {'Result':7} {'Passed':>6} {'Failed':>6}")
    print("-" * 52)
    for test in TESTS:
        ok, passed, failed, output = run(test)
        all_ok &= ok
        total_passed += passed
        total_failed += failed
        print(f"{test:28} {'PASS' if ok else 'FAIL':7} {passed:>6} {failed:>6}")
        log += ["=" * 70, f"{test} -> {'PASS' if ok else 'FAIL'}", "=" * 70, output, ""]
    print("-" * 52)
    print(f"{'TOTAL':28} {'PASS' if all_ok else 'FAIL':7} {total_passed:>6} {total_failed:>6}")

    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs" / "test_log.txt").write_text("\n".join(log), encoding="utf-8")
    print("\nFull log saved to docs/test_log.txt")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()