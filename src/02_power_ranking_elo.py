"""
02: ELO 파워 랭킹 (디벨롭: MOV 반영 + 그리드 서치 + 시계열 검증)
===============================================================
이 스크립트가 하는 일:
  1) 01에서 만든 game_level.csv(경기 목록)를 읽고
  2) 시계열로 "앞쪽 경기 = 학습(train)", "뒤쪽 경기 = 검증(val)" 으로 나눈 뒤
  3) 여러 파라미터 조합(K, home_adv, scale, mov_cap, train_ratio, clamp)을 시도해서
     "검증 구간에서 Brier 점수가 가장 좋은 조합"을 고르고 (크로스체크)
  4) 그 조합으로 "전체 경기"에 ELO를 한 번 더 돌려서 최종 팀별 레이팅을 만들고
  5) power_ranking.csv, best_params.json 을 저장합니다.
  → 03에서 이 랭킹과 best_params 를 사용해 16경기 승률을 예측합니다.

입력: output/game_level.csv (01에서 생성)
출력: output/power_ranking.csv, output/best_params.json
실행: 프로젝트 루트에서  python src/02_power_ranking_elo.py
"""

import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np   # 수치 계산 (exp, 배열 등)
import pandas as pd
from config import OUTPUT_DIR

# ========== ELO 관련 상수 (데이터로 추정하지 않고, 관례/공식 값) ==========
INITIAL_RATING = 1500   # 모든 팀의 시작 레이팅. Elo 표준·FIDE·538 등에서 쓰는 값
ELO_SCALE = 400         # Elo 공식에 나오는 400. "레이팅 차이 400 ≈ 기대승률 10:1" 역할
MOV_DIVISOR = 5         # 골 차이로 K를 보정할 때 나누는 수. K_eff = K * (1 + min(골차, mov_cap)/5)

# ========== 그리드 후보: 이 값들 조합을 전부 시도해서 "검증 Brier 최소"인 걸 고릅니다 ==========
TRAIN_RATIO_CANDIDATES = [0.75, 0.8, 0.85]   # 전체의 몇 %를 학습에 쓸지 (나머지는 검증)
K_CANDIDATES = [16, 20, 24, 28, 32]         # ELO가 한 경기 결과에 얼마나 반응할지 (K)
HOME_ADV_CANDIDATES = [50, 70, 90]          # 홈 어드밴티지를 레이팅 몇 점으로 줄지
SCALE_CANDIDATES = [200, 250, 300, 350, 400]  # 레이팅 차이 → 승률 변환할 때 쓰는 스케일
MOV_CAP_CANDIDATES = [3, 5, 7]              # 골 차이 상한 (이걸 넘어가도 K 보정은 여기까지만)
CLAMP_CANDIDATES = [(0.15, 0.85), (0.2, 0.8), (0.1, 0.9)]  # 승률을 이 구간으로 자를 때 후보


def elo_expected(r_home, r_away, home_adv):
    """
    Elo 공식: 홈팀이 이 경기에서 "기대 승률"을 계산합니다.
    r_home, r_away = 팀 레이팅, home_adv = 홈 어드밴티지(레이팅에 더함)
    반환값 = 0~1 사이. 1에 가까우면 홈이 유리하다는 뜻.
    공식: 1 / (1 + 10^((원정 - (홈+어드밴티지)) / 400))
    """
    r_h = r_home + home_adv
    return 1.0 / (1.0 + 10.0 ** ((r_away - r_h) / ELO_SCALE))


def win_prob_from_elo(r_home, r_away, home_adv, scale):
    """
    레이팅 차이를 "승률"로 바꿀 때 로지스틱 함수를 씁니다.
    ELO 업데이트용 기대승률(elo_expected)과는 다른 scale 을 쓸 수 있어서 별도 함수.
    diff = (홈+어드밴티지) - 원정. diff 가 크면 홈 승률이 1에 가까워짐.
    """
    diff = (r_home + home_adv) - r_away
    return 1.0 / (1.0 + np.exp(-diff / scale))


def run_elo_on_games(game_df, K, home_adv, use_mov=True, mov_cap=5):
    """
    경기 DataFrame 을 "시간 순서대로" 돌면서, 한 경기씩 결과를 반영해 팀별 레이팅을 갱신합니다.
    - game_df: 경기 목록 (home_team, away_team, home_goals, away_goals, home_win 등)
    - K: 한 경기당 레이팅이 얼마나 움직일지 (기본 감도)
    - home_adv: 홈 어드밴티지(레이팅 포인트)
    - use_mov: True 면 "골 차이(Margin of Victory)"로 K 를 보정 (이긴 정도가 크면 더 많이 반영)
    - mov_cap: 골 차이 상한 (이걸 넘어가도 보정은 cap 까지만)
    반환: {팀이름: 레이팅} 딕셔너리
    """
    # 모든 팀 이름을 모아서, 처음에는 전부 INITIAL_RATING 으로 시작
    teams = pd.concat([game_df["home_team"], game_df["away_team"]]).unique()
    rating = {t: INITIAL_RATING for t in teams}  # dict comprehension: 팀별 1500

    # iterrows(): DataFrame 의 한 행씩 (인덱스, 행) 을 넘겨줌. _ 는 인덱스는 안 쓴다는 뜻
    for _, row in game_df.iterrows():
        h, a = row["home_team"], row["away_team"]
        r_h, r_a = rating[h], rating[a]
        e_h = elo_expected(r_h, r_a, home_adv)  # 이 경기에서 홈 기대승률
        result_h = row["home_win"]  # 실제 결과 1=홈승, 0=원정승
        goal_diff = abs(row["home_goals"] - row["away_goals"])  # 골 차이

        # MOV 반영: 골 차이가 크면 K 를 더 크게 써서 "큰 승리에 더 반응"
        if use_mov:
            k_eff = K * (1.0 + min(goal_diff, mov_cap) / MOV_DIVISOR)
        else:
            k_eff = K

        # Elo 업데이트 공식: 새 레이팅 = 기존 + K * (실제 - 기대)
        # 홈팀: result_h - e_h. 원정팀: (1 - result_h) - (1 - e_h) = -(result_h - e_h). zero-sum
        rating[h] = r_h + k_eff * (result_h - e_h)
        rating[a] = r_a + k_eff * ((1 - result_h) - (1 - e_h))
    return rating


def brier_score(game_df, rating, home_adv, scale, clamp_low=None, clamp_high=None):
    """
    "예측 승률"과 "실제 결과(1 또는 0)"의 차이를 Brier score 로 계산합니다.
    Brier = 평균((예측 - 실제)^2). 낮을수록 예측이 좋음.
    clamp_low, clamp_high 를 주면 예측 확률을 그 구간으로 자른 뒤(과신 방지) Brier 계산.
    """
    preds, actuals = [], []
    for _, row in game_df.iterrows():
        r_h = rating.get(row["home_team"], INITIAL_RATING)  # .get: 없으면 초기값
        r_a = rating.get(row["away_team"], INITIAL_RATING)
        p = win_prob_from_elo(r_h, r_a, home_adv, scale)
        preds.append(p)
        actuals.append(row["home_win"])
    preds = np.array(preds)
    if clamp_low is not None and clamp_high is not None:
        preds = np.clip(preds, clamp_low, clamp_high)  # 예측을 [clamp_low, clamp_high] 구간으로 자름
    actuals = np.array(actuals, dtype=float)
    return float(np.mean((preds - actuals) ** 2))


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    game_path = OUTPUT_DIR / "game_level.csv"
    if not game_path.exists():
        print("오류: 먼저 01_load_and_league_table.py 를 실행하세요.")
        return 1

    game_level = pd.read_csv(game_path)
    # 경기 순서가 중요하므로 game_id 순으로 정렬
    game_level = game_level.sort_values("game_id").reset_index(drop=True)
    n = len(game_level)
    n_teams = len(pd.concat([game_level["home_team"], game_level["away_team"]]).unique())

    print("[02 설계] 시계열 분할 → train으로 ELO 학습, val로 Brier 계산 → 최소 조합 선택")
    print(f"  전체 경기: {n}, 팀 수: {n_teams}")
    print(f"  그리드 후보: train_ratio={TRAIN_RATIO_CANDIDATES}, mov_cap={MOV_CAP_CANDIDATES}, K={K_CANDIDATES}, home_adv={HOME_ADV_CANDIDATES}, scale={SCALE_CANDIDATES}, clamp={CLAMP_CANDIDATES}")

    # ========== 그리드 서치: 모든 조합을 돌려서 "검증 Brier 최소"인 조합을 찾습니다 ==========
    best_brier = float("inf")   # 처음에는 "무한대"로 두고, 더 작은 Brier 나올 때마다 갱신
    best_params = {"train_ratio": 0.8, "K": 24, "home_adv": 70, "scale": 300, "mov_cap": 5, "clamp_low": 0.15, "clamp_high": 0.85}
    total = len(TRAIN_RATIO_CANDIDATES) * len(MOV_CAP_CANDIDATES) * len(K_CANDIDATES) * len(HOME_ADV_CANDIDATES) * len(SCALE_CANDIDATES) * len(CLAMP_CANDIDATES)
    print(f"  크로스체크: {total} 조합 평가 중...")
    idx = 0

    # 중첩 for: train_ratio → mov_cap → K → home_adv → scale → (clamp_low, clamp_high) 순서로 모든 조합
    for train_ratio in TRAIN_RATIO_CANDIDATES:
        n_train = int(n * train_ratio)   # 앞쪽 경기 개수
        train_games = game_level.iloc[:n_train]   # iloc = 위치로 행 자르기. 앞 n_train 개
        val_games = game_level.iloc[n_train:]    # 나머지 = 검증용
        for mov_cap in MOV_CAP_CANDIDATES:
            for K in K_CANDIDATES:
                for home_adv in HOME_ADV_CANDIDATES:
                    # train 경기만 보고 ELO 레이팅 계산 (이 조합으로)
                    rating = run_elo_on_games(train_games, K, home_adv, use_mov=True, mov_cap=mov_cap)
                    for scale in SCALE_CANDIDATES:
                        for (clamp_low, clamp_high) in CLAMP_CANDIDATES:
                            # 검증 경기에 대해 "이 레이팅+scale+clamp"로 Brier 계산
                            b = brier_score(val_games, rating, home_adv, scale, clamp_low=clamp_low, clamp_high=clamp_high)
                            if b < best_brier:
                                best_brier = b
                                best_params = {
                                    "train_ratio": train_ratio,
                                    "K": K,
                                    "home_adv": home_adv,
                                    "scale": scale,
                                    "mov_cap": mov_cap,
                                    "clamp_low": clamp_low,
                                    "clamp_high": clamp_high,
                                }
                            idx += 1
                            if idx % 500 == 0:
                                print(f"  진행: {idx}/{total}, 현재 best Brier={best_brier:.4f}")

    print("")
    print("[02 크로스체크 결과] validation Brier 최소인 조합:")
    print(f"  train_ratio = {best_params['train_ratio']}  (train {int(n * best_params['train_ratio'])}경기 / val {n - int(n * best_params['train_ratio'])}경기)")
    print(f"  K = {best_params['K']}, home_adv = {best_params['home_adv']}, scale = {best_params['scale']}, mov_cap = {best_params['mov_cap']}")
    print(f"  clamp = [{best_params['clamp_low']}, {best_params['clamp_high']}]")
    print(f"  validation Brier = {best_brier:.4f}")

    # ========== 최종: "전체 경기"에 best 조합(K, home_adv, mov_cap)으로 ELO 한 번 더 돌리기 ==========
    # train_ratio, clamp 는 "어떤 조합이 검증에 좋았는지" 정할 때만 쓴 것이고,
    # 최종 랭킹은 "전체 1312경기"로 한 번만 돌립니다.
    K_final = best_params["K"]
    home_adv_final = best_params["home_adv"]
    mov_cap_final = best_params["mov_cap"]
    teams = pd.concat([game_level["home_team"], game_level["away_team"]]).unique()
    rating = run_elo_on_games(game_level, K_final, home_adv_final, use_mov=True, mov_cap=mov_cap_final)

    # 레이팅을 표로 만들고, 레이팅 높은 순 정렬 → 1등, 2등, ... rank 컬럼 추가
    ranking = (
        pd.DataFrame([{"team": t, "rating": rating[t]} for t in teams])
        .sort_values("rating", ascending=False)
        .reset_index(drop=True)
    )
    ranking["rank"] = range(1, len(ranking) + 1)   # 1, 2, 3, ...
    ranking = ranking[["rank", "team", "rating"]]
    ranking.to_csv(OUTPUT_DIR / "power_ranking.csv", index=False)

    # best_params 에 검증 Brier 값도 넣어서 JSON 으로 저장. 03에서 home_adv, scale, clamp 읽어 씀
    out_params = {**best_params, "brier_score_val": round(best_brier, 4)}  # ** = dict 풀어서 합치기
    with open(OUTPUT_DIR / "best_params.json", "w", encoding="utf-8") as f:
        json.dump(out_params, f, indent=2)
    print(f"저장: power_ranking.csv ({len(ranking)}팀), best_params.json (03에서 home_adv/scale/clamp 사용)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
