"""
03: 16경기 홈팀 승률 예측 (Phase 1a 제출용)
==========================================
이 스크립트가 하는 일:
  1) 02에서 만든 power_ranking.csv(팀별 ELO)와 best_params.json(홈어드밴티지, scale, clamp)을 읽고
  2) 대회에서 주어진 "토너먼트 1라운드 16경기 매치업" 엑셀을 읽어서
  3) 각 경기마다 "홈팀 승리 확률(Win Probability)"을 계산한 뒤
  4) matchup_win_probs.csv 로 저장합니다.
  → Phase 1a 제출물 중 "16경기 승률"에 해당합니다.

입력: output/power_ranking.csv, output/best_params.json, 5) WHSDSC_Rnd1_matchups.xlsx
출력: output/matchup_win_probs.csv
실행: 프로젝트 루트에서  python src/03_win_probability.py
"""

import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from config import PATH_MATCHUPS, OUTPUT_DIR, normalize_columns

# 02를 실행하지 않고 03만 돌릴 때 쓰는 기본값 (보통은 best_params.json 이 있으므로 안 씀)
DEFAULT_HOME_ADV = 70
DEFAULT_SCALE = 300
DEFAULT_CLAMP_LOW, DEFAULT_CLAMP_HIGH = 0.15, 0.85


def win_prob(r_home, r_away, home_adv, scale):
    """
    두 팀의 레이팅으로 "홈팀이 이길 확률"을 계산합니다.
    로지스틱 함수: 1 / (1 + exp(-(홈+어드밴티지 - 원정) / scale))
    차이가 크면 확률이 0 또는 1에 가까워집니다.
    """
    diff = (r_home + home_adv) - r_away
    return 1.0 / (1.0 + np.exp(-diff / scale))


def main():
    # --- 02 결과 파일이 있는지 확인 ---
    rank_path = OUTPUT_DIR / "power_ranking.csv"
    if not rank_path.exists():
        print("오류: 먼저 02_power_ranking_elo.py 를 실행하세요.")
        return 1
    if not PATH_MATCHUPS.exists():
        print(f"오류: 매치업 파일을 찾을 수 없습니다. {PATH_MATCHUPS}")
        return 1

    # --- 팀별 ELO 레이팅을 "팀이름 → 레이팅" 딕셔너리로 로드 ---
    ranking = pd.read_csv(rank_path)
    # zip(ranking["team"], ranking["rating"]) = (팀1, 레이팅1), (팀2, 레이팅2), ...
    # dict(zip(...)) = {팀1: 레이팅1, 팀2: 레이팅2, ...}  → 나중에 team_rating.get(팀이름) 으로 조회
    team_rating = dict(zip(ranking["team"], ranking["rating"]))
    # 매치업에 나왔는데 랭킹에 없는 팀(이상 케이스)을 위한 기본값 = 전체 레이팅의 중앙값
    default_rating = float(ranking["rating"].median())

    # --- 02 크로스체크 결과: best_params.json 에서 home_adv, scale, clamp 읽기 ---
    params_path = OUTPUT_DIR / "best_params.json"
    if params_path.exists():
        # with open(...) as f: 파일을 열고, 블록이 끝나면 자동으로 닫힘
        with open(params_path, "r", encoding="utf-8") as f:
            params = json.load(f)   # JSON 문자열 → 파이썬 dict
        # .get("키", 기본값): 키가 없으면 기본값 반환. 02가 예전 버전이면 clamp 가 없을 수 있음
        home_adv = params.get("home_adv", DEFAULT_HOME_ADV)
        scale = params.get("scale", DEFAULT_SCALE)
        clamp_low = params.get("clamp_low", DEFAULT_CLAMP_LOW)
        clamp_high = params.get("clamp_high", DEFAULT_CLAMP_HIGH)
        print("[03 설계] 02 크로스체크 결과 사용: best_params.json")
        print(f"  home_adv={home_adv}, scale={scale}, clamp=[{clamp_low}, {clamp_high}]")
    else:
        home_adv, scale = DEFAULT_HOME_ADV, DEFAULT_SCALE
        clamp_low, clamp_high = DEFAULT_CLAMP_LOW, DEFAULT_CLAMP_HIGH
        print("[03] best_params 없음 → 기본값 사용")
        print(f"  home_adv={home_adv}, scale={scale}, clamp=[{clamp_low}, {clamp_high}]")

    # --- 매치업 엑셀 읽기 (16경기 대진) ---
    matchups = pd.read_excel(PATH_MATCHUPS)
    matchups = normalize_columns(matchups)
    for c in ["game_id", "home_team", "away_team"]:
        if c not in matchups.columns:
            print(f"오류: 매치업에 컬럼 '{c}'이 없습니다. {list(matchups.columns)}")
            return 1

    # --- 각 경기마다 홈팀 승률 계산 → 리스트에 담기 ---
    probs = []
    for _, row in matchups.iterrows():
        # 홈팀/원정팀 레이팅. 랭킹에 없으면 default_rating 사용
        r_h = team_rating.get(row["home_team"], default_rating)
        r_a = team_rating.get(row["away_team"], default_rating)
        p = win_prob(r_h, r_a, home_adv, scale)
        # 클램핑: 확률을 [clamp_low, clamp_high] 구간으로 자름 (과신 방지. 02에서 Brier로 고른 구간)
        p = max(clamp_low, min(clamp_high, p))  # min(p, clamp_high) 로 위를 막고, max(..., clamp_low) 로 아래를 막음
        # 한 경기 정보를 dict 로 만들어 리스트에 추가
        probs.append({
            "game_id": row["game_id"],
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "home_win_prob": round(p, 4),   # 소수점 넷째 자리까지
        })

    # 리스트를 DataFrame 으로 바꿔서 CSV 저장
    out = pd.DataFrame(probs)
    out.to_csv(OUTPUT_DIR / "matchup_win_probs.csv", index=False)
    print(f"저장: matchup_win_probs.csv ({len(out)}경기)")
    if len(out) > 0:
        first = out.iloc[0]   # 첫 번째 행 (시리즈 객체)
        print(f"  샘플(1행): {first['home_team']} vs {first['away_team']} → home_win_prob={first['home_win_prob']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
