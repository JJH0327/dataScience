"""
Phase 1 파이프라인 한 번에 실행
프로젝트 루트에서:  python run_all.py
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS = [
    "src/01_load_and_league_table.py",
    "src/02_power_ranking_elo.py",
    "src/03_win_probability.py",
    "src/04_line_disparity.py",
    "src/05_visualization.py",
]


def main():
    print("=" * 60)
    print("Phase 1 파이프라인: 01 → 02 → 03 → 04 → 05")
    print("=" * 60)
    for i, script in enumerate(SCRIPTS, 1):
        path = ROOT / script
        if not path.exists():
            print(f"오류: {script} 없음")
            return 1
        print(f"\n--- [{i}/5] {script} ---")
        ret = subprocess.run([sys.executable, str(path)], cwd=str(ROOT))
        if ret.returncode != 0:
            print(f"실패: {script} (exit {ret.returncode})")
            return ret.returncode
        print(f"  → OK (exit 0)")
    print("\n" + "=" * 60)
    print("모든 단계 완료. output/ 에서 결과 파일 확인.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
