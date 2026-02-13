"""
01: 데이터 로드 → 경기 단위 집계 → 리그 테이블 생성
====================================================
이 스크립트가 하는 일:
  1) whl_2025 엑셀을 읽어서
  2) "경기(game_id)" 기준으로 한 경기당 한 행으로 묶고
  3) 리그 순위표(팀별 승/패/승점)를 만든 뒤
  4) game_level.csv, league_table.csv 로 저장합니다.
  → 02번 스크립트가 이 경기 목록을 사용해서 ELO 랭킹을 만듭니다.

실행: 프로젝트 루트에서  python src/01_load_and_league_table.py
"""

# --- 파이썬 기본: 다른 폴더의 파일을 불러오기 위해 경로를 설정합니다 ---
import sys
from pathlib import Path

# __file__ = 지금 이 파이썬 파일(01_load_and_league_table.py)의 경로
# .resolve().parent = 이 파일이 있는 폴더(src), .parent.parent = 그 위 폴더(프로젝트 루트)
ROOT = Path(__file__).resolve().parent.parent
# sys.path 에 루트를 넣으면, "from config import ..." 처럼 config.py 를 찾을 수 있습니다
sys.path.insert(0, str(ROOT))

# --- pandas: 표 형태 데이터(엑셀, CSV)를 다루는 라이브러리 ---
import pandas as pd
# config.py 에서 경로·폴더 이름 등을 가져옵니다 (한 곳에서 관리)
from config import PATH_WHL, OUTPUT_DIR, normalize_columns


def main():
    """
    main 함수: 스크립트가 실행되면 이 함수가 호출됩니다.
    return 0 = 성공, return 1 = 실패(오류 발생) → run_all.py 등에서 종료 코드로 사용
    """

    # OUTPUT_DIR = output/ 폴더. 없으면 만듭니다.
    # mkdir(parents=True) = 부모 폴더까지 필요하면 생성, exist_ok=True = 이미 있으면 에러 안 냄
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # PATH_WHL = whl_2025 엑셀 파일 경로. 없으면 여기서 멈추고 오류 메시지 출력
    if not PATH_WHL.exists():
        print(f"오류: 데이터 파일을 찾을 수 없습니다. {PATH_WHL}")
        return 1

    # --- 1단계: 엑셀 읽기 ---
    print("whl_2025 로드 중...")
    # read_excel: 엑셀 시트를 표(DataFrame)로 읽습니다
    df = pd.read_excel(PATH_WHL)
    # normalize_columns: 컬럼명 공백/대소문자 등을 통일 (데이터에 따라 다를 수 있어서)
    df = normalize_columns(df)

    # 필요한 컬럼이 모두 있는지 확인합니다. 없으면 나중에 에러가 나므로 미리 체크
    # for col in [...] : 리스트에 있는 각 컬럼명을 col 에 넣어가며 반복
    for col in ["game_id", "home_team", "away_team", "home_goals", "away_goals", "went_ot"]:
        if col not in df.columns:
            print(f"오류: 컬럼 '{col}'이 없습니다. 실제 컬럼: {list(df.columns)}")
            return 1

    # --- 2단계: 경기(game_id) 단위로 집계 → "1경기 1행" 만들기 ---
    # 원본 데이터는 "한 경기 안에서도 여러 행"(이벤트/라인별 등)일 수 있습니다.
    # groupby("game_id"): 같은 game_id 를 가진 행들을 묶습니다.
    # .agg(...): 묶인 각 그룹에서 어떤 값을 어떻게 뽑을지 정합니다.
    #   - home_team: "first" = 그룹에서 첫 번째 값 (한 경기면 팀명은 하나)
    #   - home_goals: "sum" = 합계 (여러 행에 나뉜 득점을 더함)
    #   - went_ot: "max" = 최댓값 (한 번이라도 OT 갔으면 1)
    # .reset_index(): groupby 하면 game_id 가 인덱스가 되는데, 다시 일반 컬럼으로 넣습니다
    game_level = df.groupby("game_id").agg(
        home_team=("home_team", "first"),
        away_team=("away_team", "first"),
        home_goals=("home_goals", "sum"),
        away_goals=("away_goals", "sum"),
        went_ot=("went_ot", "max"),
    ).reset_index()

    # len(DataFrame) = 행 개수. 즉 경기 수
    n_games = len(game_level)
    # 홈 팀 컬럼과 원정 팀 컬럼을 합쳐서 "나온 적 있는 팀 이름"만 모은 뒤, 중복 제거(.unique())
    n_teams = len(pd.concat([game_level["home_team"], game_level["away_team"]]).unique())
    print(f"[01 설계] game_id 기준 1경기 1행 집계")
    print(f"  경기 수: {n_games} (기대: 1312)")
    if n_games != 1312:
        print("  경고: 1312가 아니면 데이터를 확인하세요.")
    print(f"  팀 수: {n_teams}")

    # --- 3단계: 승/패 컬럼 추가 (홈 팀 관점) ---
    # home_goals > away_goals 이면 홈 승 → 1, 아니면 0
    # .astype(int): True/False 를 1/0 숫자로 바꿉니다 (나중에 ELO 등에서 계산하기 위해)
    game_level["home_win"] = (game_level["home_goals"] > game_level["away_goals"]).astype(int)

    # --- 4단계: 리그 테이블 (팀별 경기수, 승, 패, 승점) ---
    # 다시 팀 목록: 홈으로 나온 팀 + 원정으로 나온 팀을 합쳐서 유일한 이름만
    teams = pd.concat([game_level["home_team"], game_level["away_team"]]).unique()
    # records = 나중에 DataFrame 으로 만들 "한 팀당 한 줄" 정보를 담는 리스트
    records = []
    # for t in teams: 각 팀 t 에 대해 반복
    for t in teams:
        # as_home = 이 팀이 홈으로 뛴 경기들만 있는 DataFrame
        as_home = game_level[game_level["home_team"] == t]
        # as_away = 이 팀이 원정으로 뛴 경기들만
        as_away = game_level[game_level["away_team"] == t]
        # 홈일 때: home_win 이 1이면 승. .sum() 으로 승 수
        wins_home = as_home["home_win"].sum()
        # 원정일 때: home_win 이 0이면 원정팀 승. (1 - home_win).sum() 이 원정팀 승 수
        wins_away = (1 - as_away["home_win"]).sum()
        wins = int(wins_home + wins_away)
        gp = len(as_home) + len(as_away)  # 총 경기 수
        losses = gp - wins
        points = wins * 2  # 리그 규칙: 승당 2점 (1d 방법론에 명시)
        # dict 한 개 = 한 팀의 기록. records 리스트에 추가
        records.append({"team": t, "games_played": gp, "wins": wins, "losses": losses, "points": points})
    # DataFrame(records) = 리스트 안의 dict 들을 표로 만듦
    # .sort_values("points", ascending=False) = 승점 높은 순 정렬
    # .reset_index(drop=True) = 정렬 후 인덱스를 0,1,2,... 로 다시 매김
    league_table = pd.DataFrame(records).sort_values("points", ascending=False).reset_index(drop=True)

    # --- 5단계: CSV 로 저장 ---
    # to_csv(경로, index=False): DataFrame 을 CSV 파일로 저장. index=False = 행 번호 컬럼은 저장 안 함
    # 02에서 경기 목록이 필요하므로 game_level 도 저장합니다
    game_level.to_csv(OUTPUT_DIR / "game_level.csv", index=False)
    league_table.to_csv(OUTPUT_DIR / "league_table.csv", index=False)
    # Path / "파일명" = 그 폴더 안의 파일 경로
    print(f"저장: game_level.csv ({n_games}행), league_table.csv ({len(league_table)}팀)")
    return 0


# 파이썬 관례: 이 파일을 "직접 실행"했을 때만 main() 을 호출합니다.
# (다른 파일에서 "import 01_load_..." 로 불러올 때는 main 이 자동 실행되지 않음)
if __name__ == "__main__":
    sys.exit(main())  # main() 의 반환값(0 또는 1)을 프로그램 종료 코드로 사용
