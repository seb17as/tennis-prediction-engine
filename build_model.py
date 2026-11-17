"""
build_model.py — Tennis Prediction Engine
==========================================
Loads TennisMyLife data, computes Elo + rolling serve/return stats,
trains XGBoost, saves everything for predict.py to use.

Run after download_data.py

Usage:
    python build_model.py
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
OUT_DIR = Path(__file__).parent / "model"


# ═══════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ═══════════════════════════════════════════════════════════════
def load_matches(year_start: int = 2018, year_end: int = 2027) -> pd.DataFrame:
    dfs = []
    for year in range(year_start, year_end):
        f = DATA_DIR / f"{year}.csv"
        if f.exists():
            df = pd.read_csv(f, low_memory=False)
            dfs.append(df)

    matches = pd.concat(dfs, ignore_index=True)
    matches['tourney_date'] = pd.to_datetime(matches['tourney_date'], format='%Y%m%d')
    matches = matches.sort_values(['tourney_date', 'match_num']).reset_index(drop=True)

    n_players = len(set(matches['winner_name']) | set(matches['loser_name']))
    print(f"Loaded {len(matches):,} matches ({year_start}-{year_end-1})")
    print(f"Players: {n_players:,} | Surfaces: {matches['surface'].value_counts().to_dict()}")
    return matches


# ═══════════════════════════════════════════════════════════════
# 2. ELO SYSTEM
# ═══════════════════════════════════════════════════════════════
class Elo:
    def __init__(self, k_base=32, k_min=16, init=1500):
        self.ratings = {}
        self.surface_ratings = {}
        self.n_matches = {}
        self.k_base, self.k_min, self.init = k_base, k_min, init

    def r(self, p, surface=None):
        if surface:
            return self.surface_ratings.get((p, surface), self.init)
        return self.ratings.get(p, self.init)

    def _k(self, p):
        n = self.n_matches.get(p, 0)
        return max(self.k_min, self.k_base / (1 + n / 100))

    def _e(self, ra, rb):
        return 1 / (1 + 10 ** ((rb - ra) / 400))

    def update(self, w, l, surface=None):
        ra, rb = self.r(w), self.r(l)
        ea = self._e(ra, rb)
        ka, kb = self._k(w), self._k(l)

        pre = {'w_elo': ra, 'l_elo': rb}

        self.ratings[w] = ra + ka * (1 - ea)
        self.ratings[l] = rb + kb * (0 - (1 - ea))

        if surface:
            ra_s, rb_s = self.r(w, surface), self.r(l, surface)
            ea_s = self._e(ra_s, rb_s)
            pre['w_selo'] = ra_s
            pre['l_selo'] = rb_s
            self.surface_ratings[(w, surface)] = ra_s + ka * (1 - ea_s)
            self.surface_ratings[(l, surface)] = rb_s + kb * (0 - (1 - ea_s))

        self.n_matches[w] = self.n_matches.get(w, 0) + 1
        self.n_matches[l] = self.n_matches.get(l, 0) + 1
        return pre


def compute_elo(matches: pd.DataFrame):
    elo = Elo()
    records = []
    for _, m in matches.iterrows():
        w, l = m['winner_name'], m['loser_name']
        if pd.isna(w) or pd.isna(l):
            continue
        pre = elo.update(w, l, m['surface'])
        records.append({'date': m['tourney_date'], 'match_num': m['match_num'],
                        'winner': w, 'loser': l, **pre})

    df = pd.DataFrame(records)

    top = sorted(elo.ratings.items(), key=lambda x: -x[1])[:20]
    print(f"\nElo computed for {len(df):,} matches. Top 20:")
    print(f"  {'#':>3} {'Player':<28} {'Elo':>6} {'Hard':>6} {'Clay':>6} {'Grass':>6}")
    for i, (p, r) in enumerate(top, 1):
        h, c, g = elo.r(p, 'Hard'), elo.r(p, 'Clay'), elo.r(p, 'Grass')
        print(f"  {i:>3} {p:<28} {r:>6.0f} {h:>6.0f} {c:>6.0f} {g:>6.0f}")

    return df, elo


# ═══════════════════════════════════════════════════════════════
# 3. MATCH-LEVEL STATS (2 rows per match: winner + loser)
# ═══════════════════════════════════════════════════════════════
def match_stats(matches: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, m in matches.iterrows():
        if pd.isna(m.get('w_svpt')) or pd.isna(m.get('l_svpt')):
            continue
        if m['w_svpt'] == 0 or m['l_svpt'] == 0:
            continue

        def safe(val):
            return int(val) if not pd.isna(val) else 0

        w = {k: safe(m[f'w_{k}']) for k in ['svpt','1stIn','1stWon','2ndWon','SvGms','bpSaved','bpFaced','ace','df']}
        l = {k: safe(m[f'l_{k}']) for k in ['svpt','1stIn','1stWon','2ndWon','SvGms','bpSaved','bpFaced','ace','df']}

        def calc(s, opp):
            sv_won = s['1stWon'] + s['2ndWon']
            opp_sv_won = opp['1stWon'] + opp['2ndWon']
            return {
                'serve_pct': sv_won / s['svpt'] if s['svpt'] > 0 else np.nan,
                '1st_in': s['1stIn'] / s['svpt'] if s['svpt'] > 0 else np.nan,
                '1st_won': s['1stWon'] / s['1stIn'] if s['1stIn'] > 0 else np.nan,
                '2nd_won': s['2ndWon'] / (s['svpt'] - s['1stIn']) if (s['svpt'] - s['1stIn']) > 0 else np.nan,
                'bp_saved': s['bpSaved'] / s['bpFaced'] if s['bpFaced'] > 0 else np.nan,
                'return_pct': (opp['svpt'] - opp_sv_won) / opp['svpt'] if opp['svpt'] > 0 else np.nan,
                'bp_conv': (opp['bpFaced'] - opp['bpSaved']) / opp['bpFaced'] if opp['bpFaced'] > 0 else np.nan,
            }

        base = {'date': m['tourney_date'], 'surface': m['surface'], 'tourney': m['tourney_name']}

        w_stats = calc(w, l)
        rows.append({**base, 'player': m['winner_name'], 'opponent': m['loser_name'],
                      'won': 1, 'rank': m.get('winner_rank', np.nan), **w_stats})

        l_stats = calc(l, w)
        rows.append({**base, 'player': m['loser_name'], 'opponent': m['winner_name'],
                      'won': 0, 'rank': m.get('loser_rank', np.nan), **l_stats})

    df = pd.DataFrame(rows)
    print(f"\nMatch stats: {len(df):,} player-match rows ({df['date'].min().date()} to {df['date'].max().date()})")
    return df


# ═══════════════════════════════════════════════════════════════
# 4. ROLLING FEATURES (pre-match, shift(1) = no leakage)
# ═══════════════════════════════════════════════════════════════
def rolling_features(stats: pd.DataFrame, span: int = 20) -> pd.DataFrame:
    stats = stats.sort_values('date').copy()
    stat_cols = ['serve_pct', '1st_in', '1st_won', '2nd_won', 'return_pct', 'bp_saved', 'bp_conv']

    results = []
    players = stats['player'].unique()
    print(f"\nComputing rolling features for {len(players):,} players...")

    for i, player in enumerate(players):
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(players)}...")

        pdf = stats[stats['player'] == player].copy()

        # EWM rolling (shift(1) = only past data)
        for col in stat_cols:
            pdf[f'r_{col}'] = pdf[col].shift(1).ewm(span=span, min_periods=3).mean()

        # Surface-specific serve & return
        for surf in ['Hard', 'Clay', 'Grass']:
            mask = pdf['surface'] == surf
            for col in ['serve_pct', 'return_pct']:
                pdf.loc[mask, f's_{surf}_{col}'] = (
                    pdf.loc[mask, col].shift(1).ewm(span=15, min_periods=3).mean()
                )

        results.append(pdf)

    df = pd.concat(results, ignore_index=True)
    print(f"Features: {df.shape}")
    return df


# ═══════════════════════════════════════════════════════════════
# 5. BUILD MATCHUP DATASET (one row per match, both players)
# ═══════════════════════════════════════════════════════════════
def build_matchups(matches: pd.DataFrame, features: pd.DataFrame, elo_df: pd.DataFrame) -> pd.DataFrame:
    fcols = [c for c in features.columns if c.startswith(('r_', 's_'))]

    # Winner features
    wf = features[features['won'] == 1][['player', 'date', 'opponent'] + fcols].copy()
    wf.columns = ['winner_name', 'date', 'loser_name'] + [f'w_{c}' for c in fcols]

    # Loser features
    lf = features[features['won'] == 0][['player', 'date', 'opponent'] + fcols].copy()
    lf.columns = ['loser_name', 'date', 'winner_name'] + [f'l_{c}' for c in fcols]

    # Base match info
    mu = matches[['tourney_date', 'surface', 'winner_name', 'loser_name',
                   'winner_rank', 'loser_rank']].copy()
    mu = mu.rename(columns={'tourney_date': 'date'})

    mu = mu.merge(wf, on=['winner_name', 'loser_name', 'date'], how='left')
    mu = mu.merge(lf, on=['winner_name', 'loser_name', 'date'], how='left')

    # Elo
    elo_m = elo_df.rename(columns={'winner': 'winner_name', 'loser': 'loser_name'})
    elo_cols = [c for c in ['date', 'winner_name', 'loser_name', 'w_elo', 'l_elo', 'w_selo', 'l_selo'] if c in elo_m.columns]
    mu = mu.merge(elo_m[elo_cols], on=['date', 'winner_name', 'loser_name'], how='left')

    # Differentials
    for col in ['r_serve_pct', 'r_return_pct', 'r_1st_won', 'r_2nd_won', 'r_1st_in']:
        if f'w_{col}' in mu.columns and f'l_{col}' in mu.columns:
            mu[f'd_{col}'] = mu[f'w_{col}'] - mu[f'l_{col}']

    if 'w_elo' in mu.columns:
        mu['d_elo'] = mu['w_elo'] - mu['l_elo']
    if 'w_selo' in mu.columns:
        mu['d_selo'] = mu['w_selo'] - mu['l_selo']

    mu['d_rank'] = mu['loser_rank'] - mu['winner_rank']
    mu['log_rank'] = np.where(
        (mu['winner_rank'] > 0) & (mu['loser_rank'] > 0),
        np.log(mu['loser_rank'] / mu['winner_rank']), np.nan
    )

    print(f"\nMatchups: {mu.shape}")
    cov = mu[['d_elo', 'd_r_serve_pct', 'd_r_return_pct']].notna().mean() * 100
    print(f"Feature coverage: elo={cov.iloc[0]:.0f}% serve={cov.iloc[1]:.0f}% return={cov.iloc[2]:.0f}%")
    return mu


# ═══════════════════════════════════════════════════════════════
# 6. TRAIN MODEL
# ═══════════════════════════════════════════════════════════════
def train(matchups: pd.DataFrame):
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import accuracy_score, log_loss
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    try:
        from xgboost import XGBClassifier
    except ImportError:
        import subprocess
        subprocess.run(['pip', 'install', 'xgboost', '--break-system-packages', '-q'])
        from xgboost import XGBClassifier

    # Filter to matches with key features, 2020+
    key = ['w_r_serve_pct', 'l_r_serve_pct', 'w_r_return_pct', 'l_r_return_pct']
    df = matchups.dropna(subset=key)
    df = df[df['date'] >= '2020-01-01'].copy()
    print(f"\nTraining on {len(df):,} matches (2020+)")

    # Randomize A/B to avoid winner bias
    np.random.seed(42)
    swap = np.random.random(len(df)) > 0.5

    feature_cols = []
    rows = []
    for i, (_, m) in enumerate(df.iterrows()):
        a_pre, b_pre = ('w_', 'l_') if swap[i] else ('l_', 'w_')
        target = 1 if swap[i] else 0

        row = {'target': target, 'date': m['date']}

        # Rolling stats
        for stat in ['r_serve_pct', 'r_return_pct', 'r_1st_in', 'r_1st_won', 'r_2nd_won', 'r_bp_saved', 'r_bp_conv']:
            a_val = m.get(f'{a_pre}{stat}', np.nan)
            b_val = m.get(f'{b_pre}{stat}', np.nan)
            row[f'a_{stat}'] = a_val
            row[f'b_{stat}'] = b_val
            if not (pd.isna(a_val) or pd.isna(b_val)):
                row[f'd_{stat}'] = a_val - b_val
            else:
                row[f'd_{stat}'] = np.nan

        # Surface stats
        for stat in [c for c in df.columns if c.startswith('w_s_') or c.startswith('l_s_')]:
            base = stat[2:]  # remove w_ or l_
            if stat.startswith('w_'):
                row[f'a_{base}'] = m.get(f'{a_pre}{base}', np.nan)
                row[f'b_{base}'] = m.get(f'{b_pre}{base}', np.nan)

        # Elo
        a_elo_col = 'w_elo' if swap[i] else 'l_elo'
        b_elo_col = 'l_elo' if swap[i] else 'w_elo'
        a_selo_col = 'w_selo' if swap[i] else 'l_selo'
        b_selo_col = 'l_selo' if swap[i] else 'w_selo'

        row['a_elo'] = m.get(a_elo_col, np.nan)
        row['b_elo'] = m.get(b_elo_col, np.nan)
        if not (pd.isna(row['a_elo']) or pd.isna(row['b_elo'])):
            row['d_elo'] = row['a_elo'] - row['b_elo']
        else:
            row['d_elo'] = np.nan

        row['a_selo'] = m.get(a_selo_col, np.nan)
        row['b_selo'] = m.get(b_selo_col, np.nan)
        if not (pd.isna(row['a_selo']) or pd.isna(row['b_selo'])):
            row['d_selo'] = row['a_selo'] - row['b_selo']
        else:
            row['d_selo'] = np.nan

        # Rank
        a_rank = m.get('winner_rank' if swap[i] else 'loser_rank', np.nan)
        b_rank = m.get('loser_rank' if swap[i] else 'winner_rank', np.nan)
        row['d_rank'] = b_rank - a_rank if not (pd.isna(a_rank) or pd.isna(b_rank)) else np.nan
        row['log_rank'] = np.log(b_rank / a_rank) if (not pd.isna(a_rank) and not pd.isna(b_rank) and a_rank > 0 and b_rank > 0) else np.nan

        rows.append(row)

    fdf = pd.DataFrame(rows)

    # Select features (exclude leaky ones)
    feat_cols = [c for c in fdf.columns if c not in ['target', 'date']]
    null_pct = fdf[feat_cols].isnull().mean()
    feat_cols = null_pct[null_pct < 0.3].index.tolist()
    print(f"Features: {len(feat_cols)}")

    X = fdf[feat_cols].fillna(fdf[feat_cols].median())
    y = fdf['target'].values

    # Temporal CV
    tscv = TimeSeriesSplit(n_splits=5)

    # Logistic Regression
    scaler = StandardScaler()
    Xs = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    lr_acc = []
    for tr, te in tscv.split(Xs):
        lr = LogisticRegression(max_iter=1000)
        lr.fit(Xs.iloc[tr], y[tr])
        lr_acc.append(accuracy_score(y[te], lr.predict(Xs.iloc[te])))
    print(f"\nLogistic Regression: {np.mean(lr_acc):.4f} ± {np.std(lr_acc):.4f}")

    # XGBoost
    xgb_acc, xgb_ll = [], []
    for tr, te in tscv.split(X):
        xgb = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                            subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                            eval_metric='logloss', random_state=42, verbosity=0)
        xgb.fit(X.iloc[tr], y[tr])
        pred = xgb.predict(X.iloc[te])
        prob = xgb.predict_proba(X.iloc[te])[:, 1]
        xgb_acc.append(accuracy_score(y[te], pred))
        xgb_ll.append(log_loss(y[te], prob))
    print(f"XGBoost:             {np.mean(xgb_acc):.4f} ± {np.std(xgb_acc):.4f} (log_loss: {np.mean(xgb_ll):.4f})")

    # Elo baseline
    if 'd_elo' in X.columns:
        elo_pred = (X['d_elo'] > 0).astype(int).values
        _, te = list(tscv.split(X))[-1]
        print(f"Elo baseline:        {accuracy_score(y[te], elo_pred[te]):.4f} (last fold)")

    # Train final model on all data
    print("\nTraining final model...")
    final = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                          eval_metric='logloss', random_state=42, verbosity=0)
    final.fit(X, y)

    # Feature importance
    imp = pd.Series(final.feature_importances_, index=feat_cols).sort_values(ascending=False)
    print("\nTop 10 features:")
    for f, s in imp.head(10).items():
        print(f"  {f:<40} {s:.4f}")

    return final, feat_cols, X.median().to_dict()


# ═══════════════════════════════════════════════════════════════
# 7. MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("  TENNIS PREDICTION ENGINE — BUILD")
    print("  Data: TennisMyLife (stats.tennismylife.org)")
    print("=" * 60)

    OUT_DIR.mkdir(exist_ok=True)

    # Load
    matches = load_matches()

    # Elo
    elo_df, elo = compute_elo(matches)

    # Stats
    stats = match_stats(matches)

    # Rolling features
    features = rolling_features(stats)

    # Matchups
    matchups = build_matchups(matches, features, elo_df)

    # Train
    model, feat_cols, medians = train(matchups)

    # Save everything
    features.to_csv(OUT_DIR / 'player_features.csv', index=False)
    with open(OUT_DIR / 'elo.pkl', 'wb') as f:
        pickle.dump(elo, f)
    with open(OUT_DIR / 'model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open(OUT_DIR / 'feat_cols.pkl', 'wb') as f:
        pickle.dump(feat_cols, f)
    with open(OUT_DIR / 'medians.pkl', 'wb') as f:
        pickle.dump(medians, f)

    print(f"\n✅ All saved to {OUT_DIR}/")
    print(f"   Next: python predict.py \"Player A\" \"Player B\"")
