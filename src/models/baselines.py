"""Baseline predictive models for FIF8A World Cup 2026 (Phase 4).

Simple, robust models:
- EmpiricalFavouriteModel (Model 0): W/D/L frequencies binned on a strength gap.
- multinomial logistic regression pipeline (Model 1).
- PoissonScoreModel (Model 2): two Poisson GLMs -> scoreline matrix -> W/D/L.
- HistGradientBoosting classifier pipeline (Model 3).

The class order for all W/D/L probabilities is fixed: ['home_win','draw','away_win'].
"""
import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.ensemble import HistGradientBoostingClassifier

CLASSES = ["home_win", "draw", "away_win"]

NUMERIC_FEATURES = [
    "elo_diff", "fifa_points_diff", "fifa_rank_diff",
    "ppm_10_diff", "gd_10_diff", "ppm_5_diff", "win_rate_10_diff",
    "gf_10_diff", "ga_10_diff", "days_since_2000",
    "overperf_elo_10_diff",  # Hypothesis 3: recent over/under-performance vs Elo
]
CATEGORICAL_FEATURES = ["tournament_category"]
BINARY_FEATURES = [
    "neutral", "is_world_cup", "is_friendly",
    "is_world_cup_qualifier", "is_continental_championship",
    # Hypothesis 1: per-context home-advantage interactions
    "nonneutral", "home_adv_friendly", "home_adv_world_cup_qualifier",
    "home_adv_continental", "home_adv_world_cup", "neutral_world_cup_context",
]

# Poisson goal models: per-side attack/defence proxies (asymmetric on purpose).
POISSON_HOME_FEATURES = ["home_elo", "away_elo", "elo_diff", "home_gf_10", "away_ga_10",
                         "home_ppm_10", "neutral", "is_friendly"]
POISSON_AWAY_FEATURES = ["home_elo", "away_elo", "elo_diff", "away_gf_10", "home_ga_10",
                         "away_ppm_10", "neutral", "is_friendly"]


def _num_cat_preprocessor():
    num = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    cat = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                    ("oh", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([
        ("num", num, NUMERIC_FEATURES),
        ("bin", "passthrough", BINARY_FEATURES),
        ("cat", cat, CATEGORICAL_FEATURES),
    ])


def make_logit_pipeline():
    return Pipeline([
        ("pre", _num_cat_preprocessor()),
        ("clf", LogisticRegression(max_iter=2000, C=1.0)),
    ])


def make_hgb_pipeline():
    # HGB handles NaN natively; still one-hot the single categorical for simplicity.
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"),
         CATEGORICAL_FEATURES),
    ], remainder="passthrough")
    return Pipeline([
        ("pre", pre),
        ("clf", HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05,
                                                max_iter=300, l2_regularization=1.0)),
    ])


def proba_in_class_order(pipeline, X):
    """Return predicted proba as an (n,3) array in CLASSES order."""
    classes = list(pipeline.named_steps["clf"].classes_)
    p = pipeline.predict_proba(X)
    idx = [classes.index(c) for c in CLASSES]
    return p[:, idx]


class EmpiricalFavouriteModel:
    """Model 0: empirical W/D/L frequencies binned on a strength gap (elo or FIFA)."""

    def __init__(self, gap_col="elo_diff", n_bins=12):
        self.gap_col = gap_col
        self.n_bins = n_bins
        self.bins_ = None
        self.freqs_ = None
        self.global_ = None

    def fit(self, df):
        gap = pd.to_numeric(df[self.gap_col], errors="coerce")
        y = df["result_label"].values
        self.global_ = np.array([(y == c).mean() for c in CLASSES])
        valid = gap.notna()
        gv, yv = gap[valid], y[valid]
        self.bins_ = np.unique(np.quantile(gv, np.linspace(0, 1, self.n_bins + 1)))
        codes = np.digitize(gv, self.bins_[1:-1])
        self.freqs_ = {}
        for b in np.unique(codes):
            sub = yv[codes == b]
            self.freqs_[b] = np.array([(sub == c).mean() for c in CLASSES])
        return self

    def predict_proba(self, df):
        gap = pd.to_numeric(df[self.gap_col], errors="coerce")
        out = np.tile(self.global_, (len(df), 1))
        valid = gap.notna().values
        codes = np.digitize(gap[valid], self.bins_[1:-1])
        for i, b in zip(np.where(valid)[0], codes):
            out[i] = self.freqs_.get(b, self.global_)
        return out


class PoissonScoreModel:
    """Model 2: two Poisson GLMs (home/away goals) -> scoreline matrix."""

    def __init__(self, max_goals=10):
        self.max_goals = max_goals
        self.home_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("glm", PoissonRegressor(alpha=1e-3, max_iter=500)),
        ])
        self.away_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("glm", PoissonRegressor(alpha=1e-3, max_iter=500)),
        ])

    def fit(self, df):
        self.home_pipe.fit(df[POISSON_HOME_FEATURES], df["home_goals"])
        self.away_pipe.fit(df[POISSON_AWAY_FEATURES], df["away_goals"])
        return self

    def predict_lambdas(self, df):
        lh = np.clip(self.home_pipe.predict(df[POISSON_HOME_FEATURES]), 1e-3, 12)
        la = np.clip(self.away_pipe.predict(df[POISSON_AWAY_FEATURES]), 1e-3, 12)
        return lh, la

    def score_matrix(self, lam_home, lam_away):
        """Independent-Poisson scoreline matrix (max_goals+1) x (max_goals+1)."""
        k = np.arange(self.max_goals + 1)
        ph = poisson.pmf(k, lam_home)
        pa = poisson.pmf(k, lam_away)
        # renormalize the truncated tail so it sums to 1
        ph = ph / ph.sum()
        pa = pa / pa.sum()
        return np.outer(ph, pa)  # [home_goals, away_goals]

    def predict_proba(self, df):
        lh, la = self.predict_lambdas(df)
        out = np.zeros((len(df), 3))
        for i in range(len(df)):
            M = self.score_matrix(lh[i], la[i])
            out[i] = wdl_from_matrix(M)
        return out


def wdl_from_matrix(M):
    """[home_win, draw, away_win] from a scoreline matrix [home_goals, away_goals]."""
    home = np.tril(M, -1).sum()   # home_goals > away_goals
    draw = np.trace(M)
    away = np.triu(M, 1).sum()
    return np.array([home, draw, away])
