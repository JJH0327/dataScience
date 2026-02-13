# WHSDSC 2026 — 경로·파일명 설정
# 프로젝트 루트에서 실행한다고 가정 (python run_all.py 또는 python src/01_...py)

from pathlib import Path

# 프로젝트 루트 = 이 파일이 있는 디렉터리
PROJECT_ROOT = Path(__file__).resolve().parent

# 데이터 파일: 루트에 "2) whl_2025.xlsx" 등이 있으면 사용, 없으면 data/ 내 파일 사용
_path_whl = PROJECT_ROOT / "2) whl_2025.xlsx"
if not _path_whl.exists():
    _path_whl = PROJECT_ROOT / "data" / "whl_2025.xlsx"
_path_m = PROJECT_ROOT / "5) WHSDSC_Rnd1_matchups.xlsx"
if not _path_m.exists():
    _path_m = PROJECT_ROOT / "data" / "WHSDSC_Rnd1_matchups.xlsx"

PATH_WHL = _path_whl
PATH_MATCHUPS = _path_m
OUTPUT_DIR = PROJECT_ROOT / "output"

# 컬럼명 정규화: Excel이 공백/대소문자를 다르게 쓸 수 있음
def normalize_columns(df):
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    return df
