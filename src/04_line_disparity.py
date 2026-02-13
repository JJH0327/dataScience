"""
04: 공격 라인 격차 (1라인 vs 2라인) — Phase 1b 제출용
====================================================
이 스크립트가 하는 일:
  1) whl_2025 엑셀에서 "5-on-5" 상황만 골라냅니다 (first_off, second_off = 1라인·2라인).
     → PP(파워플레이)·PK(패널티킬) 등 특수부대 구간은 제외해 "순수 라인 실력"만 봅니다.
  2) 팀별·라인별로 xG(기대득점)를 모으되, "상대 디펜스 강도"로 보정(adj_xg)합니다.
  3) 1라인 성능 / 2라인 성능 의 "격차 비율(disparity_ratio)"을 계산하고
  4) 격차가 큰 순으로 정렬해 "상위 10팀"과 "전체 팀" 두 CSV 를 저장합니다.
  → Phase 1b 제출물: 상위 10팀 리스트.

입력: 2) whl_2025.xlsx
출력: output/line_disparity_top10.csv, output/line_disparity_all.csv
실행: 프로젝트 루트에서  python src/04_line_disparity.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from config import PATH_WHL, OUTPUT_DIR, normalize_columns

# 데이터 컬럼에 들어 있는 값과 맞춰야 함. "5-on-5 에서의 1라인/2라인"만 쓴다는 뜻
LINE_FIRST = "first_off"    # 1라인
LINE_SECOND = "second_off"  # 2라인
TOI_EPS = 1e-6              # Time on Ice 로 나눌 때 0으로 나누기 방지용 아주 작은 수


def main():
    if not PATH_WHL.exists():
        print(f"오류: {PATH_WHL} 을 찾을 수 없습니다.")
        return 1

    print("[04 설계] 5-on-5 first_off/second_off만 사용, 상대 디펜스 xG 보정")
    df = pd.read_excel(PATH_WHL)
    df = normalize_columns(df)
    for c in ["home_team", "home_off_line", "home_xg", "toi", "away_team", "away_def_pairing"]:
        if c not in df.columns:
            print(f"오류: 컬럼 '{c}' 없음. {list(df.columns)}")
            return 1

    # --- 5-on-5 필터: first_off, second_off 인 행만 남깁니다 ---
    # .isin([...]) = 컬럼 값이 리스트 안에 있으면 True. PP/PK 등 다른 값은 False → 제거됨
    mask = df["home_off_line"].isin([LINE_FIRST, LINE_SECOND])
    df55 = df.loc[mask].copy()   # .loc[mask] = True 인 행만 선택. .copy() = 원본과 분리된 복사본

    # --- 리그 전체 5-on-5 xG per 60 (기준값. 보정에 사용) ---
    # TOI 는 초 단위일 수 있어서 3600 으로 나누면 "시간". xG/시간 = per 60 분당
    league_xg = df55["home_xg"].sum()
    league_toi = df55["toi"].sum() / 3600.0 + TOI_EPS   # 0 나누기 방지
    league_xg_per_60 = league_xg / league_toi

    # --- 상대(원정팀 + 수비 페어링)별 "허용한 xG per 60" 계산 ---
    # "이 팀 이 수비조가 얼마나 xG 를 허용했는지" → 나중에 우리 팀 xG 를 보정할 때 씀
    def_allowed = (
        df55.groupby(["away_team", "away_def_pairing"])   # 원정팀 + 수비 조합으로 묶고
        .agg(xg_allowed=("home_xg", "sum"), toi=("toi", "sum"))  # 그 구간에서 홈팀이 낸 xG 합, TOI 합
        .reset_index()
    )
    def_allowed["toi_hr"] = def_allowed["toi"] / 3600.0 + TOI_EPS
    def_allowed["xg_allowed_per_60"] = def_allowed["xg_allowed"] / def_allowed["toi_hr"]
    # 딕셔너리로 만들면 나중에 (away_team, away_def_pairing) 키로 빠르게 조회 가능
    def_lookup = def_allowed.set_index(["away_team", "away_def_pairing"])["xg_allowed_per_60"].to_dict()

    # --- 행별 보정 xG: adj_xg = home_xg * (리그평균 / 상대가 허용한 비율) ---
    # "강한 상대를 만났으면 xG 가 낮게 나올 수 있으니, 상대 수비 강도로 보정해 주자"
    def get_adj_xg(row):
        key = (row["away_team"], row["away_def_pairing"])
        opp_60 = def_lookup.get(key, league_xg_per_60)   # 없으면 리그 평균으로
        if opp_60 <= 0:
            return row["home_xg"]
        return row["home_xg"] * (league_xg_per_60 / opp_60)
    # .apply(get_adj_xg, axis=1): 각 행(row)마다 get_adj_xg(row) 를 호출한 결과를 모아서 새 컬럼으로
    df55["adj_xg"] = df55.apply(get_adj_xg, axis=1)

    # --- 팀·라인별로 보정 xG 와 TOI 를 묶어서 "per 60" 성능 계산 ---
    line_stats = (
        df55.groupby(["home_team", "home_off_line"])
        .agg(adj_xg=("adj_xg", "sum"), toi=("toi", "sum"))
        .reset_index()
    )
    line_stats["toi_hr"] = line_stats["toi"] / 3600.0 + TOI_EPS
    line_stats["xg_per_60"] = line_stats["adj_xg"] / line_stats["toi_hr"]

    # --- 1라인만 / 2라인만 따로 DataFrame 만들고, 팀 기준으로 합치기(merge) ---
    first = line_stats[line_stats["home_off_line"] == LINE_FIRST][["home_team", "xg_per_60"]].rename(
        columns={"xg_per_60": "first_xg60"}
    )
    second = line_stats[line_stats["home_off_line"] == LINE_SECOND][["home_team", "xg_per_60"]].rename(
        columns={"xg_per_60": "second_xg60"}
    )
    # merge(..., on="home_team", how="outer"): 팀 이름으로 좌우를 붙임. outer = 한쪽에만 있어도 행 유지
    # .fillna(TOI_EPS): 2라인 데이터가 없는 팀 등으로 생긴 NaN 을 TOI_EPS 로 채움 (나눗셈 시 0 방지)
    merge = first.merge(second, on="home_team", how="outer").fillna(TOI_EPS)
    # 격차 비율 = 1라인 성능 / 2라인 성능. 클수록 "1라인이 2라인보다 훨씬 낫다"
    merge["disparity_ratio"] = merge["first_xg60"] / (merge["second_xg60"] + TOI_EPS)
    merge = merge.sort_values("disparity_ratio", ascending=False).reset_index(drop=True)
    merge["rank"] = range(1, len(merge) + 1)

    # 상위 10팀 = Phase 1b 제출용. 컬럼명 home_team → team 으로 맞춤
    top10 = merge.head(10)[["rank", "home_team", "disparity_ratio"]].rename(columns={"home_team": "team"})
    merge.rename(columns={"home_team": "team"}, inplace=True)   # inplace=True = merge 자체를 수정
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    top10.to_csv(OUTPUT_DIR / "line_disparity_top10.csv", index=False)
    merge.to_csv(OUTPUT_DIR / "line_disparity_all.csv", index=False)
    print(f"저장: line_disparity_top10.csv (상위 10팀), line_disparity_all.csv ({len(merge)}팀)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
