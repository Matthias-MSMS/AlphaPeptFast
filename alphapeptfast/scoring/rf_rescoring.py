"""Random-Forest rescoring of DDA PSMs.

Semi-supervised RF rescoring: from an initial-search target/decoy PSM table,
extract a feature matrix, iteratively train RandomForest on top-targets vs
all-decoys, then re-rank by RF probability and compute q-values.

This module is shared across DDA primary-search engines so that the same
rescoring pipeline applies to any AlphaX engine output (AlphaPeptLookup,
AlphaPeptTag, etc) without per-engine duplication.

Scope: DDA only.
DIA rescoring uses a different feature set (XIC quality, fragment elution
correlation, RT-prediction agreement, ion-mobility delta) and lives separately.
For DIA, see the AlphaDIA scoring module rather than this one.

Typical use
-----------
>>> from alphapeptfast.scoring.rf_rescoring import rf_rescore
>>> # psm_df: required cols hyperscore, delta_score, matched_peaks, matched_b,
>>> # matched_y, sequence, charge, is_decoy, plus an initial ranking score column.
>>> rescored = rf_rescore(psm_df, score_col='discriminant', fdr_threshold=0.01)
>>> # rescored has new 'rf_score' and 'rf_q_value' columns

Features extracted
------------------
Always (required input columns must exist):
- hyperscore, delta_score, matched_peaks, matched_b, matched_y, charge
- log(1 + hyperscore), log(1 + matched_peaks)
- ion_balance: min(b,y)/(b+y)
- peptide_length
- coverage = (b + y) / (2 * (length-1))
- missed_cleavages (count of K/R not at C-term)
- score_per_peak = hyperscore / max(b+y, 1)
- delta_score_ratio = delta_score / max(hyperscore, 1)

If present (engine-specific, used as features when available):
- ppm_error / abs_ppm_error (mass error in ppm vs theoretical)
- abs_delta_mass (modification mass shift, for tiered/dependent search)
- precursor_mz, rt
- index_score, index_count (fragment-index hit statistics)
- loc_confidence, residue_match, delta_m_known (modification-localization features)
- Any column starting with 'dl_' (deep-learning features pre-computed externally)
"""
from __future__ import annotations
import time
from typing import Iterable
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score


REQUIRED_COLS = (
    'hyperscore', 'delta_score', 'matched_peaks', 'matched_b', 'matched_y',
    'sequence', 'charge', 'is_decoy',
)

# Optional columns that, if present, become RF features. Engine-specific.
OPTIONAL_FEATURE_COLS = (
    'ppm_error', 'abs_ppm_error', 'abs_delta_mass',
    'precursor_mz', 'rt',
    'index_score', 'index_count',
    'loc_confidence', 'residue_match', 'delta_m_known',
    # Sage-style fragment-quality features (added 2026-05-07).
    # Strong discriminators between real-but-dim signal and lucky decoy
    # matches: a real peptide produces a long contiguous b/y ladder and
    # accounts for a large fraction of the spectrum's matched intensity;
    # decoys typically scatter matches and account for less intensity.
    'longest_b', 'longest_y',
    'longest_b_pct', 'longest_y_pct',
    'matched_intensity', 'fraction_matched_intensity',
    # Multi-charge fragment counts (added 2026-05-07): z=1 vs z=2 b/y matches.
    # When precursor charge ≥ 3, real peptides often produce z=2 b/y ions;
    # the right peptide gets more z=2 matches than wrong peptides at the
    # same precursor mass. LDA learns to weight these per-charge counts.
    'matched_b_z1', 'matched_y_z1', 'matched_b_z2', 'matched_y_z2',
    # Multiplicative (Sage / X!Tandem) hyperscore — exposed alongside the
    # additive form so LDA gets a second look at the b/y intensity balance.
    'hyperscore_mul',
)


def compute_dda_features(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Extract DDA RF features. Returns (X, feature_names)."""
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f'compute_dda_features: missing required columns {missing}')

    feats: dict[str, np.ndarray] = {}

    feats['hyperscore'] = df['hyperscore'].values.astype(float)
    feats['delta_score'] = df['delta_score'].values.astype(float)
    feats['matched_peaks'] = df['matched_peaks'].values.astype(float)
    feats['matched_b'] = df['matched_b'].values.astype(float)
    feats['matched_y'] = df['matched_y'].values.astype(float)
    feats['charge'] = df['charge'].values.astype(float)

    feats['log_hyperscore'] = np.log1p(df['hyperscore'].values)
    feats['log_matched'] = np.log1p(df['matched_peaks'].values)

    total_ions = df['matched_b'].values + df['matched_y'].values
    feats['ion_balance'] = np.where(
        total_ions > 0,
        np.minimum(df['matched_b'].values, df['matched_y'].values) / np.maximum(total_ions, 1),
        0.0,
    )

    pep_len = df['sequence'].str.len().values.astype(float)
    feats['peptide_length'] = pep_len
    feats['coverage'] = np.where(pep_len > 1, total_ions / (2.0 * (pep_len - 1)), 0.0)

    feats['missed_cleavages'] = df['sequence'].apply(
        lambda s: sum(1 for i, c in enumerate(s[:-1]) if c in 'KR')
    ).values.astype(float)

    feats['score_per_peak'] = np.where(
        total_ions > 0, df['hyperscore'].values / np.maximum(total_ions, 1), 0.0
    )

    feats['delta_score_ratio'] = np.where(
        df['hyperscore'].values > 0,
        df['delta_score'].values / np.maximum(df['hyperscore'].values, 1.0),
        0.0,
    )

    for col in OPTIONAL_FEATURE_COLS:
        if col in df.columns:
            vals = df[col].values.astype(float)
            vals = np.where(np.isfinite(vals), vals, 0.0)
            feats[col] = vals

    for col in df.columns:
        if col.startswith('dl_'):
            vals = df[col].values.astype(float)
            vals = np.where(np.isfinite(vals), vals, 0.0)
            feats[col] = vals

    X = np.column_stack(list(feats.values()))
    return X, list(feats.keys())


def compute_target_decoy_qvalues(scores: np.ndarray, is_decoy: np.ndarray) -> np.ndarray:
    """Standard target-decoy q-values, monotonic from worst to best score."""
    order = np.argsort(-scores)
    is_decoy_sorted = is_decoy[order]
    cum_targets = np.cumsum(~is_decoy_sorted)
    cum_decoys = np.cumsum(is_decoy_sorted)
    fdr = cum_decoys / np.maximum(cum_targets, 1)
    q = np.empty_like(fdr)
    min_fdr = 1.0
    for i in range(len(fdr) - 1, -1, -1):
        if fdr[i] < min_fdr:
            min_fdr = fdr[i]
        q[i] = min_fdr
    out = np.empty_like(q)
    out[order] = q
    return out


def rf_rescore(
    df: pd.DataFrame,
    score_col: str = 'discriminant',
    n_iter: int = 3,
    n_trees: int = 200,
    fdr_train: float = 0.01,
    max_depth: int = 10,
    min_samples_leaf: int = 20,
    tier_col: str | None = None,
    random_state: int = 42,
    verbose: bool = True,
) -> pd.DataFrame:
    """Semi-supervised RF rescoring of a DDA PSM dataframe.

    Returns a copy of df with added columns: 'rf_score', 'rf_q_value',
    and (if tier_col provided) per-tier q-values via the same column.

    Parameters
    ----------
    df : pd.DataFrame
        PSM table with required columns (see REQUIRED_COLS) and at least
        an initial ranking score in `score_col`. Should include both
        target and decoy PSMs.
    score_col : str
        Column name with the initial ranking score (e.g., 'discriminant',
        'hyperscore', or any per-PSM score). Used for the first iteration's
        positive-set selection at FDR <= fdr_train.
    n_iter : int
        Semi-supervised iterations. Each iteration retrains RF on positives
        selected at the previous iteration's RF score.
    n_trees, max_depth, min_samples_leaf : int
        sklearn RandomForestClassifier hyperparameters.
    fdr_train : float
        FDR threshold for selecting positive training samples each iteration.
    tier_col : str | None
        If given (e.g., 'tier'), positive selection and q-value computation
        are done per tier. Otherwise global.
    random_state : int
        Random seed; iteration i uses random_state + i.
    verbose : bool
        Print progress, AUC, and top features per iteration.
    """
    t0 = time.time()
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f'rf_rescore: missing required columns {missing}')
    if score_col not in df.columns:
        raise ValueError(f'rf_rescore: missing score_col={score_col}')

    df = df.copy().reset_index(drop=True)

    if verbose:
        print(f'rf_rescore: {len(df):,} PSMs ({(~df["is_decoy"]).sum():,} targets, '
              f'{df["is_decoy"].sum():,} decoys)')

    X, feature_names = compute_dda_features(df)
    if verbose:
        print(f'  {len(feature_names)} features')

    is_decoy = df['is_decoy'].values
    is_target = ~is_decoy
    scores = df[score_col].values.astype(float).copy()

    if tier_col is not None and tier_col in df.columns:
        tier_values = df[tier_col].values
        tiers: Iterable = pd.unique(df[tier_col].dropna())
    else:
        tier_values = np.array(['_global'] * len(df))
        tiers = ['_global']

    for it in range(n_iter):
        if verbose:
            print(f'  iteration {it + 1}/{n_iter}')

        positive_mask = np.zeros(len(df), dtype=bool)
        for tier in tiers:
            tier_mask = tier_values == tier
            tier_targets = tier_mask & is_target
            tier_decoys = tier_mask & is_decoy
            if tier_targets.sum() < 10 or tier_decoys.sum() < 10:
                continue
            tier_idx = np.where(tier_mask)[0]
            tier_scores = scores[tier_idx]
            tier_is_decoy = is_decoy[tier_idx]
            order = np.argsort(-tier_scores)
            cum_t = np.cumsum(~tier_is_decoy[order])
            cum_d = np.cumsum(tier_is_decoy[order])
            fdr = cum_d / np.maximum(cum_t, 1)
            passing = fdr <= fdr_train
            if passing.any():
                cutoff = int(np.max(np.where(passing)[0])) + 1
                selected = tier_idx[order[:cutoff]]
                selected_targets = selected[~is_decoy[selected]]
                positive_mask[selected_targets] = True

        n_pos = int(positive_mask.sum())
        n_neg = int(is_decoy.sum())
        if verbose:
            print(f'    train: {n_pos:,} positives + {n_neg:,} decoys')
        if n_pos < 100:
            if verbose:
                print('    too few positives; skipping iteration')
            continue

        train_mask = positive_mask | is_decoy
        X_train = X[train_mask]
        y_train = positive_mask[train_mask].astype(int)

        rf = RandomForestClassifier(
            n_estimators=n_trees,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            n_jobs=-1,
            random_state=random_state + it,
            class_weight='balanced',
        )
        rf.fit(X_train, y_train)
        scores = rf.predict_proba(X)[:, 1]

        if verbose:
            try:
                auc = roc_auc_score(y_train, rf.predict_proba(X_train)[:, 1])
                print(f'    train AUC: {auc:.4f}')
            except Exception:
                pass
            top_imp = sorted(zip(feature_names, rf.feature_importances_), key=lambda x: -x[1])[:6]
            for name, importance in top_imp:
                print(f'      {name}: {importance:.3f}')

    df['rf_score'] = scores

    q = np.full(len(df), 1.0)
    for tier in tiers:
        m = tier_values == tier
        if m.sum() == 0:
            continue
        q[m] = compute_target_decoy_qvalues(scores[m], is_decoy[m])
    df['rf_q_value'] = q

    if verbose:
        dt = time.time() - t0
        n_at_1 = int(((df['rf_q_value'] <= 0.01) & ~df['is_decoy']).sum())
        n_at_5 = int(((df['rf_q_value'] <= 0.05) & ~df['is_decoy']).sum())
        print(f'  done in {dt:.1f}s: {n_at_1:,} target PSMs at 1% RF FDR, {n_at_5:,} at 5%')

    return df


def lda_rescore(
    df: pd.DataFrame,
    score_col: str = 'discriminant',
    n_iter: int = 3,
    fdr_train: float = 0.01,
    n_folds: int = 3,
    tier_col: str | None = None,
    random_state: int = 42,
    verbose: bool = True,
) -> pd.DataFrame:
    """Percolator-style LDA rescoring with k-fold cross-validated predictions.

    Linear Discriminant Analysis (LDA) is the classifier behind Sage's,
    Percolator's, and MS-GF+'s rescoring layer. It is linear, doesn't memorise
    the training set the way overfit RandomForest does, and produces a single
    discriminant score per PSM via signed distance to the LDA hyperplane.

    Cross-validated prediction (Percolator's standard trick): the data is
    split into k folds; each fold's predictions come from a model trained
    on the OTHER folds. This eliminates the "predict on training data"
    bias that drives RF q-values to zero artificially.

    Returns a copy of df with added 'lda_score' and 'lda_q_value' columns.

    Parameters
    ----------
    df : pd.DataFrame
        PSM table; must contain REQUIRED_COLS plus score_col.
    score_col : str
        Initial ranking score for the first iteration's positive selection.
    n_iter : int
        Semi-supervised iterations (default 3).
    fdr_train : float
        FDR cutoff for selecting positive training samples (default 1%).
    n_folds : int
        Cross-validation folds for prediction (default 3).
    tier_col : str | None
        Optional tier column for tier-specific positive selection + q-values.
    random_state : int
        Random seed.
    verbose : bool
        Print progress + LDA coefficients per iteration.
    """
    t0 = time.time()
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f'lda_rescore: missing required columns {missing}')
    if score_col not in df.columns:
        raise ValueError(f'lda_rescore: missing score_col={score_col}')

    df = df.copy().reset_index(drop=True)

    if verbose:
        print(f'lda_rescore: {len(df):,} PSMs ({(~df["is_decoy"]).sum():,} targets, '
              f'{df["is_decoy"].sum():,} decoys)')

    X, feature_names = compute_dda_features(df)
    if verbose:
        print(f'  {len(feature_names)} features, {n_folds}-fold CV, {n_iter} iterations')

    # Standardise features so LDA's regularisation behaves consistently.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    is_decoy = df['is_decoy'].values
    is_target = ~is_decoy
    scores = df[score_col].values.astype(float).copy()

    if tier_col is not None and tier_col in df.columns:
        tier_values = df[tier_col].values
        tiers: Iterable = pd.unique(df[tier_col].dropna())
    else:
        tier_values = np.array(['_global'] * len(df))
        tiers = ['_global']

    for it in range(n_iter):
        if verbose:
            print(f'  iteration {it + 1}/{n_iter}')

        # Pick positives per tier at fdr_train using current scores.
        positive_mask = np.zeros(len(df), dtype=bool)
        for tier in tiers:
            tier_mask = tier_values == tier
            tier_targets = tier_mask & is_target
            tier_decoys = tier_mask & is_decoy
            if tier_targets.sum() < 10 or tier_decoys.sum() < 10:
                continue
            tier_idx = np.where(tier_mask)[0]
            tier_scores = scores[tier_idx]
            tier_is_decoy = is_decoy[tier_idx]
            order = np.argsort(-tier_scores)
            cum_t = np.cumsum(~tier_is_decoy[order])
            cum_d = np.cumsum(tier_is_decoy[order])
            fdr = cum_d / np.maximum(cum_t, 1)
            passing = fdr <= fdr_train
            if passing.any():
                cutoff = int(np.max(np.where(passing)[0])) + 1
                selected = tier_idx[order[:cutoff]]
                selected_targets = selected[~is_decoy[selected]]
                positive_mask[selected_targets] = True

        n_pos = int(positive_mask.sum())
        n_neg = int(is_decoy.sum())
        if verbose:
            print(f'    train: {n_pos:,} positives + {n_neg:,} decoys')
        if n_pos < 100:
            if verbose:
                print('    too few positives; skipping iteration')
            continue

        # Train + predict via k-fold CV. Restrict to the train mask
        # (positives + decoys); ambiguous targets predict via the trained
        # models on the full data.
        train_mask = positive_mask | is_decoy
        idx_train = np.where(train_mask)[0]
        y_all = positive_mask.astype(int)

        # CV-predicted scores for training PSMs (so q-values aren't biased).
        cv_scores = np.zeros(len(df), dtype=float)
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state + it)
        # Iterate over training PSMs only; each LDA model is trained on
        # train-PSMs in the OTHER folds, then predicts the held-out fold's
        # train-PSMs AND the entire ambiguous-target population.
        # (For ambiguous targets, average across the n_folds models.)
        ambiguous_idx = np.where(~train_mask)[0]
        amb_pred_sum = np.zeros(len(ambiguous_idx), dtype=float)
        last_lda = None
        for fold_i, (tr_local_idx, te_local_idx) in enumerate(kf.split(idx_train)):
            tr_idx = idx_train[tr_local_idx]
            te_idx = idx_train[te_local_idx]
            X_tr = X_scaled[tr_idx]
            y_tr = y_all[tr_idx]
            if y_tr.sum() < 10 or (y_tr == 0).sum() < 10:
                continue
            lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
            lda.fit(X_tr, y_tr)
            cv_scores[te_idx] = lda.decision_function(X_scaled[te_idx])
            if len(ambiguous_idx) > 0:
                amb_pred_sum += lda.decision_function(X_scaled[ambiguous_idx])
            last_lda = lda
        if len(ambiguous_idx) > 0 and last_lda is not None:
            cv_scores[ambiguous_idx] = amb_pred_sum / n_folds
        scores = cv_scores

        if verbose and last_lda is not None:
            try:
                auc = roc_auc_score(y_all[idx_train], cv_scores[idx_train])
                print(f'    CV-AUC (held-out): {auc:.4f}')
            except Exception:
                pass
            coefs = last_lda.coef_[0]
            top = sorted(zip(feature_names, np.abs(coefs)), key=lambda x: -x[1])[:6]
            for name, w in top:
                print(f'      {name}: |coef|={w:.3f}')

    df['lda_score'] = scores

    q = np.full(len(df), 1.0)
    for tier in tiers:
        m = tier_values == tier
        if m.sum() == 0:
            continue
        q[m] = compute_target_decoy_qvalues(scores[m], is_decoy[m])
    df['lda_q_value'] = q

    if verbose:
        dt = time.time() - t0
        n_at_1 = int(((df['lda_q_value'] <= 0.01) & ~df['is_decoy']).sum())
        n_at_5 = int(((df['lda_q_value'] <= 0.05) & ~df['is_decoy']).sum())
        print(f'  done in {dt:.1f}s: {n_at_1:,} target PSMs at 1% LDA FDR, {n_at_5:,} at 5%')

    return df
