"""
project_models.py — CAP6619 Advanced Topics in ML Project
============================================================
Deep Neural Networks for ATP Match Outcome Prediction

Extends the TennisPicks build_model.py pipeline with:
  - Logistic Regression (baseline)
  - XGBoost (baseline)  
  - MLP (PyTorch, deep learning)
  - Full evaluation: accuracy, log loss, AUC, Brier score
  - Calibration plots, confusion matrices, ROI curves
  - Feature importance analysis

Usage:
    1. Place this file in the same directory as build_model.py
    2. Make sure you already ran download_data.py (data/ folder exists)
    3. python project_models.py

Output:
    figures/  — all plots for the report
    results/  — CSV tables with metrics
"""

import pandas as pd
import numpy as np
import pickle
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

# ── Import the existing pipeline ──────────────────────────────
from build_model import load_matches, compute_elo, match_stats, rolling_features, build_matchups

# ── ML imports ──────────────────────────────────────────────────
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (accuracy_score, log_loss, roc_auc_score,
                             brier_score_loss, confusion_matrix, roc_curve,
                             classification_report)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import calibration_curve

try:
    from xgboost import XGBClassifier
except ImportError:
    import subprocess
    subprocess.run(['pip', 'install', 'xgboost', '-q'])
    from xgboost import XGBClassifier

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Directories ─────────────────────────────────────────────────
BASE = Path(__file__).parent
FIG_DIR = BASE / "figures"
RES_DIR = BASE / "results"
FIG_DIR.mkdir(exist_ok=True)
RES_DIR.mkdir(exist_ok=True)


# ═══════════════════════════════════════════════════════════════
# 1. PREPARE DATA (reuses the pipeline)
# ═══════════════════════════════════════════════════════════════
def prepare_data():
    """Run your existing pipeline and return X, y with temporal split."""
    print("=" * 65)
    print("  STEP 1: DATA PIPELINE (TennisMyLife)")
    print("=" * 65)

    matches = load_matches()
    elo_df, elo = compute_elo(matches)
    stats = match_stats(matches)
    features = rolling_features(stats)
    matchups = build_matchups(matches, features, elo_df)

    # ── Build training dataset with A/B randomization ───────────
    key = ['w_r_serve_pct', 'l_r_serve_pct', 'w_r_return_pct', 'l_r_return_pct']
    df = matchups.dropna(subset=key)
    df = df[df['date'] >= '2020-01-01'].copy()
    print(f"\nUsable matches (2020+): {len(df):,}")

    np.random.seed(42)
    swap = np.random.random(len(df)) > 0.5

    rows = []
    for i, (_, m) in enumerate(df.iterrows()):
        a_pre, b_pre = ('w_', 'l_') if swap[i] else ('l_', 'w_')
        target = 1 if swap[i] else 0

        row = {'target': target, 'date': m['date'], 'surface': m['surface']}

        # Rolling stats
        for stat in ['r_serve_pct', 'r_return_pct', 'r_1st_in', 'r_1st_won',
                      'r_2nd_won', 'r_bp_saved', 'r_bp_conv']:
            a_val = m.get(f'{a_pre}{stat}', np.nan)
            b_val = m.get(f'{b_pre}{stat}', np.nan)
            row[f'a_{stat}'] = a_val
            row[f'b_{stat}'] = b_val
            row[f'd_{stat}'] = (a_val - b_val) if not (pd.isna(a_val) or pd.isna(b_val)) else np.nan

        # Surface stats
        for stat in [c for c in df.columns if c.startswith('w_s_')]:
            base = stat[2:]
            row[f'a_{base}'] = m.get(f'{a_pre}{base}', np.nan)
            row[f'b_{base}'] = m.get(f'{b_pre}{base}', np.nan)

        # Elo
        row['a_elo'] = m.get('w_elo' if swap[i] else 'l_elo', np.nan)
        row['b_elo'] = m.get('l_elo' if swap[i] else 'w_elo', np.nan)
        row['d_elo'] = (row['a_elo'] - row['b_elo']) if not (pd.isna(row['a_elo']) or pd.isna(row['b_elo'])) else np.nan

        row['a_selo'] = m.get('w_selo' if swap[i] else 'l_selo', np.nan)
        row['b_selo'] = m.get('l_selo' if swap[i] else 'w_selo', np.nan)
        row['d_selo'] = (row['a_selo'] - row['b_selo']) if not (pd.isna(row['a_selo']) or pd.isna(row['b_selo'])) else np.nan

        # Rank
        a_rank = m.get('winner_rank' if swap[i] else 'loser_rank', np.nan)
        b_rank = m.get('loser_rank' if swap[i] else 'winner_rank', np.nan)
        row['d_rank'] = (b_rank - a_rank) if not (pd.isna(a_rank) or pd.isna(b_rank)) else np.nan
        row['log_rank'] = np.log(b_rank / a_rank) if (not pd.isna(a_rank) and not pd.isna(b_rank) and a_rank > 0 and b_rank > 0) else np.nan

        rows.append(row)

    fdf = pd.DataFrame(rows)

    # Feature selection
    feat_cols = [c for c in fdf.columns if c not in ['target', 'date', 'surface']]
    null_pct = fdf[feat_cols].isnull().mean()
    feat_cols = null_pct[null_pct < 0.3].index.tolist()

    X = fdf[feat_cols].fillna(fdf[feat_cols].median())
    y = fdf['target'].values
    dates = fdf['date'].values
    surfaces = fdf['surface'].values

    print(f"Final dataset: {X.shape[0]:,} matches, {X.shape[1]} features")
    print(f"Features: {feat_cols}")

    return X, y, feat_cols, dates, surfaces


# ═══════════════════════════════════════════════════════════════
# 2. MLP MODEL (PyTorch)
# ═══════════════════════════════════════════════════════════════
class TennisMLP(nn.Module):
    """
    Multi-Layer Perceptron for match outcome prediction.
    Architecture: Input → 128 → 64 → 32 → 1
    With BatchNorm and Dropout for regularization.
    """
    def __init__(self, input_dim, hidden_dims=[128, 64, 32], dropout=0.3):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def train_mlp(X_train, y_train, X_val, y_val, input_dim,
              hidden_dims=[128, 64, 32], dropout=0.3,
              lr=0.001, epochs=200, batch_size=256, patience=20):
    """Train MLP with early stopping."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Convert to tensors
    X_tr = torch.FloatTensor(X_train).to(device)
    y_tr = torch.FloatTensor(y_train).to(device)
    X_v = torch.FloatTensor(X_val).to(device)
    y_v = torch.FloatTensor(y_val).to(device)

    train_ds = TensorDataset(X_tr, y_tr)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model = TennisMLP(input_dim, hidden_dims, dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    best_loss = float('inf')
    best_state = None
    wait = 0

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            out = model(xb).squeeze()
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_out = model(X_v).squeeze()
            val_loss = criterion(val_out, y_v).item()

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    return model, device


def predict_mlp(model, X, device):
    """Get probabilities from trained MLP."""
    model.eval()
    with torch.no_grad():
        X_t = torch.FloatTensor(X).to(device)
        logits = model(X_t).squeeze()
        probs = torch.sigmoid(logits).cpu().numpy()
    return probs


# ═══════════════════════════════════════════════════════════════
# 3. EVALUATION
# ═══════════════════════════════════════════════════════════════
def evaluate_all(X, y, feat_cols, dates):
    """Train and evaluate all models with temporal CV."""
    print("\n" + "=" * 65)
    print("  STEP 2: MODEL TRAINING & EVALUATION")
    print("=" * 65)

    tscv = TimeSeriesSplit(n_splits=5)
    scaler = StandardScaler()

    results = {name: {'acc': [], 'logloss': [], 'auc': [], 'brier': []}
               for name in ['Logistic Regression', 'XGBoost', 'MLP']}

    # Store predictions from last fold for plots
    last_preds = {}

    for fold, (tr, te) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[tr].values, X.iloc[te].values
        y_train, y_test = y[tr], y[te]

        # Scale for LR and MLP
        scaler.fit(X_train)
        X_tr_s = scaler.transform(X_train)
        X_te_s = scaler.transform(X_test)

        # ── Logistic Regression ─────────────────────────────────
        lr = LogisticRegression(max_iter=1000, C=1.0)
        lr.fit(X_tr_s, y_train)
        lr_prob = lr.predict_proba(X_te_s)[:, 1]
        lr_pred = lr.predict(X_te_s)

        results['Logistic Regression']['acc'].append(accuracy_score(y_test, lr_pred))
        results['Logistic Regression']['logloss'].append(log_loss(y_test, lr_prob))
        results['Logistic Regression']['auc'].append(roc_auc_score(y_test, lr_prob))
        results['Logistic Regression']['brier'].append(brier_score_loss(y_test, lr_prob))

        # ── XGBoost ─────────────────────────────────────────────
        xgb = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                            subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                            eval_metric='logloss', random_state=42, verbosity=0)
        xgb.fit(X_train, y_train)
        xgb_prob = xgb.predict_proba(X_test)[:, 1]
        xgb_pred = xgb.predict(X_test)

        results['XGBoost']['acc'].append(accuracy_score(y_test, xgb_pred))
        results['XGBoost']['logloss'].append(log_loss(y_test, xgb_prob))
        results['XGBoost']['auc'].append(roc_auc_score(y_test, xgb_prob))
        results['XGBoost']['brier'].append(brier_score_loss(y_test, xgb_prob))

        # ── MLP ─────────────────────────────────────────────────
        mlp, device = train_mlp(X_tr_s, y_train, X_te_s, y_test,
                                input_dim=X_tr_s.shape[1],
                                hidden_dims=[128, 64, 32],
                                dropout=0.3, lr=0.001,
                                epochs=200, batch_size=256, patience=20)
        mlp_prob = predict_mlp(mlp, X_te_s, device)
        mlp_pred = (mlp_prob > 0.5).astype(int)

        results['MLP']['acc'].append(accuracy_score(y_test, mlp_pred))
        results['MLP']['logloss'].append(log_loss(y_test, mlp_prob))
        results['MLP']['auc'].append(roc_auc_score(y_test, mlp_prob))
        results['MLP']['brier'].append(brier_score_loss(y_test, mlp_prob))

        print(f"  Fold {fold+1}: LR={results['Logistic Regression']['acc'][-1]:.4f}  "
              f"XGB={results['XGBoost']['acc'][-1]:.4f}  "
              f"MLP={results['MLP']['acc'][-1]:.4f}")

        # Save last fold predictions for plots
        if fold == tscv.get_n_splits() - 1:
            last_preds = {
                'y_test': y_test,
                'lr_prob': lr_prob, 'lr_pred': lr_pred,
                'xgb_prob': xgb_prob, 'xgb_pred': xgb_pred,
                'mlp_prob': mlp_prob, 'mlp_pred': mlp_pred,
            }

    # ── Elo baseline ────────────────────────────────────────────
    if 'd_elo' in feat_cols:
        elo_idx = feat_cols.index('d_elo')
        _, te = list(tscv.split(X))[-1]
        elo_prob = 1 / (1 + 10 ** (-X.iloc[te].values[:, elo_idx] / 400))
        elo_pred = (elo_prob > 0.5).astype(int)
        elo_acc = accuracy_score(y[te], elo_pred)
        elo_ll = log_loss(y[te], elo_prob)
        elo_auc = roc_auc_score(y[te], elo_prob)
        elo_brier = brier_score_loss(y[te], elo_prob)
        last_preds['elo_prob'] = elo_prob
        last_preds['elo_pred'] = elo_pred
    else:
        elo_acc = elo_ll = elo_auc = elo_brier = np.nan

    # ── Print results ───────────────────────────────────────────
    print("\n" + "─" * 65)
    print(f"  {'Model':<25} {'Accuracy':>10} {'Log Loss':>10} {'AUC':>10} {'Brier':>10}")
    print("─" * 65)
    for name, m in results.items():
        print(f"  {name:<25} {np.mean(m['acc']):>10.4f} {np.mean(m['logloss']):>10.4f} "
              f"{np.mean(m['auc']):>10.4f} {np.mean(m['brier']):>10.4f}")
    print(f"  {'Elo Baseline':<25} {elo_acc:>10.4f} {elo_ll:>10.4f} {elo_auc:>10.4f} {elo_brier:>10.4f}")
    print("─" * 65)

    # ── Save results table ──────────────────────────────────────
    rows = []
    for name, m in results.items():
        rows.append({
            'Model': name,
            'Accuracy': f"{np.mean(m['acc']):.4f} ± {np.std(m['acc']):.4f}",
            'Log Loss': f"{np.mean(m['logloss']):.4f} ± {np.std(m['logloss']):.4f}",
            'AUC-ROC': f"{np.mean(m['auc']):.4f} ± {np.std(m['auc']):.4f}",
            'Brier Score': f"{np.mean(m['brier']):.4f} ± {np.std(m['brier']):.4f}",
        })
    rows.append({
        'Model': 'Elo Baseline',
        'Accuracy': f"{elo_acc:.4f}",
        'Log Loss': f"{elo_ll:.4f}",
        'AUC-ROC': f"{elo_auc:.4f}",
        'Brier Score': f"{elo_brier:.4f}",
    })
    pd.DataFrame(rows).to_csv(RES_DIR / 'model_comparison.csv', index=False)

    return results, last_preds, xgb, feat_cols


# ═══════════════════════════════════════════════════════════════
# 4. GENERATE ALL FIGURES
# ═══════════════════════════════════════════════════════════════
def plot_all(results, last_preds, xgb_model, feat_cols, X):
    """Generate all figures for the report."""
    print("\n" + "=" * 65)
    print("  STEP 3: GENERATING FIGURES")
    print("=" * 65)

    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 13,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
        'figure.dpi': 150,
    })

    # ── Figure 1: Model Comparison Bar Chart ────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    metrics = ['acc', 'logloss', 'auc', 'brier']
    titles = ['Accuracy ↑', 'Log Loss ↓', 'AUC-ROC ↑', 'Brier Score ↓']
    colors = ['#2196F3', '#4CAF50', '#FF5722']
    model_names = list(results.keys())

    for ax, metric, title in zip(axes, metrics, titles):
        means = [np.mean(results[m][metric]) for m in model_names]
        stds = [np.std(results[m][metric]) for m in model_names]
        bars = ax.bar(range(len(model_names)), means, yerr=stds,
                      color=colors, capsize=5, edgecolor='black', linewidth=0.5)
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(['LR', 'XGB', 'MLP'], fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        # Add value labels
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{mean:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig1_model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ fig1_model_comparison.png")

    # ── Figure 2: Calibration Plot ──────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 7))
    y_test = last_preds['y_test']

    for name, prob, color, marker in [
        ('Logistic Regression', last_preds['lr_prob'], '#2196F3', 'o'),
        ('XGBoost', last_preds['xgb_prob'], '#4CAF50', 's'),
        ('MLP', last_preds['mlp_prob'], '#FF5722', '^'),
        ('Elo Baseline', last_preds.get('elo_prob'), '#9C27B0', 'D'),
    ]:
        if prob is None:
            continue
        frac_pos, mean_pred = calibration_curve(y_test, prob, n_bins=10, strategy='uniform')
        ax.plot(mean_pred, frac_pos, marker=marker, label=name, color=color,
                linewidth=2, markersize=8)

    ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=1.5)
    ax.set_xlabel('Predicted Probability', fontsize=13)
    ax.set_ylabel('Observed Frequency', fontsize=13)
    ax.set_title('Calibration Plot (Reliability Diagram)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    plt.savefig(FIG_DIR / 'fig2_calibration.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ fig2_calibration.png")

    # ── Figure 3: ROC Curves ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 7))

    for name, prob, color in [
        ('Logistic Regression', last_preds['lr_prob'], '#2196F3'),
        ('XGBoost', last_preds['xgb_prob'], '#4CAF50'),
        ('MLP', last_preds['mlp_prob'], '#FF5722'),
        ('Elo Baseline', last_preds.get('elo_prob'), '#9C27B0'),
    ]:
        if prob is None:
            continue
        fpr, tpr, _ = roc_curve(y_test, prob)
        auc = roc_auc_score(y_test, prob)
        ax.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})', color=color, linewidth=2)

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
    ax.set_xlabel('False Positive Rate', fontsize=13)
    ax.set_ylabel('True Positive Rate', fontsize=13)
    ax.set_title('ROC Curves', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(alpha=0.3)
    plt.savefig(FIG_DIR / 'fig3_roc_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ fig3_roc_curves.png")

    # ── Figure 4: Confusion Matrices ────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, (name, pred) in zip(axes, [
        ('Logistic Regression', last_preds['lr_pred']),
        ('XGBoost', last_preds['xgb_pred']),
        ('MLP', last_preds['mlp_pred']),
    ]):
        cm = confusion_matrix(y_test, pred)
        im = ax.imshow(cm, cmap='Blues', interpolation='nearest')

        # Labels
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f'{cm[i, j]}', ha='center', va='center',
                        fontsize=16, fontweight='bold',
                        color='white' if cm[i, j] > cm.max()/2 else 'black')

        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('Actual', fontsize=12)
        ax.set_title(name, fontsize=13, fontweight='bold')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Player B', 'Player A'], fontsize=11)
        ax.set_yticklabels(['Player B', 'Player A'], fontsize=11)
        fig.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig4_confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ fig4_confusion_matrices.png")

    # ── Figure 5: Feature Importance (XGBoost) ──────────────────
    imp = pd.Series(xgb_model.feature_importances_, index=feat_cols).sort_values(ascending=True)
    top15 = imp.tail(15)

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.barh(range(len(top15)), top15.values, color='#2196F3', edgecolor='black', linewidth=0.5)
    ax.set_yticks(range(len(top15)))

    # Clean up feature names for display
    clean_names = []
    for name in top15.index:
        name = name.replace('d_', 'Δ ').replace('r_', '').replace('a_', 'P1 ').replace('b_', 'P2 ')
        name = name.replace('_', ' ').replace('serve pct', 'Serve%').replace('return pct', 'Return%')
        name = name.replace('1st won', '1st Serve Won%').replace('2nd won', '2nd Serve Won%')
        name = name.replace('1st in', '1st Serve In%').replace('bp saved', 'BP Saved%')
        name = name.replace('bp conv', 'BP Conv%').replace('elo', 'Elo').replace('selo', 'Surface Elo')
        name = name.replace('log rank', 'Log Rank Ratio').replace('Δ rank', 'Δ Rank')
        clean_names.append(name)

    ax.set_yticklabels(clean_names, fontsize=11)
    ax.set_xlabel('Feature Importance', fontsize=13)
    ax.set_title('Top 15 Features (XGBoost)', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig5_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ fig5_feature_importance.png")

    # ── Figure 6: Accuracy by Confidence Bucket ─────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = [(0.50, 0.55), (0.55, 0.60), (0.60, 0.65), (0.65, 0.70),
            (0.70, 0.80), (0.80, 1.00)]
    bin_labels = ['50-55%', '55-60%', '60-65%', '65-70%', '70-80%', '80%+']

    for name, prob, color in [
        ('XGBoost', last_preds['xgb_prob'], '#4CAF50'),
        ('MLP', last_preds['mlp_prob'], '#FF5722'),
    ]:
        accs = []
        counts = []
        for lo, hi in bins:
            mask = (prob >= lo) & (prob < hi) | (prob >= (1-hi)) & (prob < (1-lo))
            if mask.sum() > 5:
                # Map probs to "confidence" (max of prob, 1-prob)
                conf = np.maximum(prob[mask], 1 - prob[mask])
                correct = ((prob[mask] > 0.5) == y_test[mask]).mean()
                accs.append(correct)
                counts.append(mask.sum())
            else:
                accs.append(np.nan)
                counts.append(0)

        ax.plot(range(len(bins)), accs, marker='o', label=name, color=color,
                linewidth=2, markersize=8)

    ax.set_xticks(range(len(bins)))
    ax.set_xticklabels(bin_labels, fontsize=11)
    ax.set_xlabel('Confidence Level', fontsize=13)
    ax.set_ylabel('Accuracy', fontsize=13)
    ax.set_title('Accuracy by Model Confidence', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_ylim(0.4, 1.0)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig6_accuracy_by_confidence.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ fig6_accuracy_by_confidence.png")

    # ── Figure 7: Cross-Validation Folds ────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    folds = range(1, 6)
    for name, color, marker in [
        ('Logistic Regression', '#2196F3', 'o'),
        ('XGBoost', '#4CAF50', 's'),
        ('MLP', '#FF5722', '^'),
    ]:
        ax.plot(folds, results[name]['acc'], marker=marker, label=name,
                color=color, linewidth=2, markersize=8)

    ax.set_xlabel('Fold', fontsize=13)
    ax.set_ylabel('Accuracy', fontsize=13)
    ax.set_title('Accuracy Across CV Folds (Temporal Split)', fontsize=14, fontweight='bold')
    ax.set_xticks(folds)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig7_cv_folds.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ fig7_cv_folds.png")

    print(f"\n✅ All figures saved to {FIG_DIR}/")


# ═══════════════════════════════════════════════════════════════
# 5. BETTING SIMULATION (simplified for report)
# ═══════════════════════════════════════════════════════════════
def betting_simulation(last_preds):
    """
    Simplified betting simulation using model confidence as proxy.
    For the full backtest with real odds, use backtest_with_odds.py.
    """
    print("\n" + "=" * 65)
    print("  STEP 4: BETTING SIMULATION")
    print("=" * 65)

    y = last_preds['y_test']

    for name, prob in [('XGBoost', last_preds['xgb_prob']),
                        ('MLP', last_preds['mlp_prob'])]:
        confidence = np.maximum(prob, 1 - prob)
        predicted = (prob > 0.5).astype(int)
        correct = predicted == y

        print(f"\n  {name}:")
        print(f"  {'Confidence':<15} {'Picks':>6} {'Correct':>8} {'Accuracy':>10}")
        print(f"  {'─'*45}")

        for label, lo, hi in [('All', 0.50, 1.0), ('55%+', 0.55, 1.0),
                                ('60%+', 0.60, 1.0), ('65%+', 0.65, 1.0),
                                ('70%+', 0.70, 1.0)]:
            mask = (confidence >= lo) & (confidence <= hi)
            n = mask.sum()
            if n > 0:
                acc = correct[mask].mean()
                print(f"  {label:<15} {n:>6} {correct[mask].sum():>8} {acc:>10.1%}")

    print("\n  Note: For ROI with real odds, run backtest_with_odds.py separately.")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "═" * 65)
    print("  CAP6619 — Deep Neural Networks for ATP Match Prediction")
    print("  Data: TennisMyLife (stats.tennismylife.org)")
    print("═" * 65)

    # Step 1: Data
    X, y, feat_cols, dates, surfaces = prepare_data()

    # Step 2: Train & Evaluate
    results, last_preds, xgb_model, _ = evaluate_all(X, y, feat_cols, dates)

    # Step 3: Figures
    plot_all(results, last_preds, xgb_model, feat_cols, X)

    # Step 4: Betting simulation
    betting_simulation(last_preds)

    print("\n" + "═" * 65)
    print("  ✅ PROJECT COMPLETE")
    print(f"  Figures: {FIG_DIR}/")
    print(f"  Results: {RES_DIR}/")
    print("═" * 65)
