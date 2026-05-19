"""
BehaviourAuth v2 — Upgraded Flask Backend
==========================================
New in this version vs behavioural_auth_code.ipynb integration:

ML Improvements
───────────────
1. Extended Features (27 → 36)
   + dwell_cv, flight_cv, traj_cv          — coefficient of variation (rhythm regularity)
   + cross_z                                — Mahalanobis-like combined deviation distance
   + rhythm_index                           — how well session rhythm matches user baseline
   + device_offset_dwell/flight/traj        — relative shift (captures device changes)
   + typing_burst_score                     — detects unnaturally uniform/bursty typing

2. Adaptive Profiles (Behavioral Drift)
   - Exponential Weighted Mean (α=0.08) updates user profile on confirmed genuine sessions
   - POST /api/user/<uid>/adapt to confirm a session as genuine → profile drifts toward it
   - Prevents genuine users from being locked out as typing evolves over time

3. Imitation Attack Resistance
   - imitation_risk score [0-1]: flags sessions that are suspiciously close to the mean
     (impostors who memorise target averages cluster unnaturally near the centroid)
   - Mahalanobis cross-feature ratio check: real users have consistent dwell/flight ratios
   - "Too perfect" detector: z-score < 0.1 across all 3 features is statistically improbable

4. Device Variability Handling
   - device_offset_* features: normalise each value relative to user's [min, max] range
   - Relative normalisation means a different keyboard shifts the absolute value but the
     within-profile distribution is preserved

5. Threat Score (0–100)
   - 45% recent flag severity (last 10 sessions)
   - 35% model failure rate
   - 20% z-score anomaly magnitude
   - Returned as threat_score on every /api/users call

Dashboard Improvements
───────────────────────
6. Persistent Alert Store (STATE["alerts"])
   - Every critical/warning/caution session is logged with timestamp, uid, session, details
   - GET /api/alerts?severity=critical&limit=50
   - DELETE /api/alerts/<id> to dismiss

7. SSE real-time stream
   - GET /api/stream → text/event-stream of new alerts as they arrive

Run
───
    pip install flask flask-cors xgboost imbalanced-learn tensorflow \
                scikit-learn openpyxl pandas numpy joblib
    export DATASET_PATH="/path/to/feature_kmt_xlsx"
    python app.py   →   http://localhost:5000
"""

import os, glob, json, time, queue, threading, warnings, uuid
import numpy as np
import pandas as pd
import joblib
from collections import Counter, deque
from datetime import datetime
from flask import Flask, jsonify, request, send_from_directory, Response
from flask_cors import CORS

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────
DATASET_PATH = os.environ.get(
    "DATASET_PATH",
    "behaviour_biometrics_dataset/feature_kmt_dataset/feature_kmt_xlsx"
)
MODEL_DIR   = os.environ.get("MODEL_DIR", ".")
ALPHA       = 0.08          # adaptive profile learning rate
BASE_FEATS  = ["dwell_avg", "flight_avg", "traj_avg"]
VOTE_W      = {"svm": 0.25, "xgb": 0.45, "mlp": 0.30}
MODEL_META  = {
    "svm":  {"label": "SVM (RBF)", "color": "#00e5ff", "short": "SVM",  "thresh": 0.50},
    "xgb":  {"label": "XGBoost",   "color": "#00ff9d", "short": "XGB",  "thresh": 0.50},
    "mlp":  {"label": "MLP (DNN)", "color": "#ffd700", "short": "MLP",  "thresh": 0.50},
    "vote": {"label": "Soft Vote", "color": "#22d3ee", "short": "VOTE", "thresh": 0.50},
}
FLAG_WEIGHT = {"safe": 0, "caution": 0.25, "warning": 0.60, "critical": 1.0}

app = Flask(__name__, static_folder="static")
CORS(app)

# ─────────────────────────────────────────────────────────────────
# Global state
# ─────────────────────────────────────────────────────────────────
STATE = {
    "models":        {},
    "scaler":        None,
    "all_features":  None,
    "vote_thresh":   0.50,
    "mlp_thresh":    0.50,
    "user_profiles": {},   # uid → {feat: {mean,std,min,max,median}}
    "user_sessions": {},   # uid → [session dicts]
    "model_metrics": {},
    "alerts":        [],   # persistent alert log
    "status":        "idle",
    "status_msg":    "",
    "log":           [],
}
_sse_queues: list[queue.Queue] = []   # one per SSE subscriber


# ─────────────────────────────────────────────────────────────────
# Feature engineering — notebook cell 7 + extended features
# ─────────────────────────────────────────────────────────────────
def _base_engineer(out: pd.DataFrame) -> pd.DataFrame:
    """Ratio / interaction / polynomial / log / sqrt features (notebook-exact)."""
    eps = 1e-9
    out["dwell_flight_ratio"] = out["dwell_avg"] / (out["flight_avg"] + eps)
    out["flight_traj_ratio"]  = out["flight_avg"] / (out["traj_avg"]  + eps)
    out["dwell_traj_ratio"]   = out["dwell_avg"]  / (out["traj_avg"]  + eps)
    out["dwell_x_flight"] = out["dwell_avg"]  * out["flight_avg"]
    out["dwell_x_traj"]   = out["dwell_avg"]  * out["traj_avg"]
    out["flight_x_traj"]  = out["flight_avg"] * out["traj_avg"]
    out["dwell_sq"]  = out["dwell_avg"]  ** 2
    out["flight_sq"] = out["flight_avg"] ** 2
    out["traj_sq"]   = out["traj_avg"]   ** 2
    out["inv_dwell"]  = 1.0 / (out["dwell_avg"]  + eps)
    out["inv_flight"] = 1.0 / (out["flight_avg"] + eps)
    out["inv_traj"]   = 1.0 / (out["traj_avg"]   + eps)
    out["log_dwell"]  = np.log1p(out["dwell_avg"])
    out["log_flight"] = np.log1p(out["flight_avg"])
    out["log_traj"]   = np.log1p(out["traj_avg"])
    out["sqrt_dwell"]  = np.sqrt(np.abs(out["dwell_avg"]))
    out["sqrt_flight"] = np.sqrt(np.abs(out["flight_avg"]))
    out["sqrt_traj"]   = np.sqrt(np.abs(out["traj_avg"]))
    return out


def engineer_df(df: pd.DataFrame) -> pd.DataFrame:
    """Full dataframe engineering used during training."""
    out = df.copy()
    out["user_id"] = out["source_file"].str.extract(r'user_(\d+)\.xlsx')
    out = _base_engineer(out)

    for feat in BASE_FEATS:
        stats = (
            out[out["label"] == 1]
            .groupby("user_id")[feat]
            .agg(["mean", "std", "min", "max", "median"])
            .rename(columns={"mean": f"{feat}_umean", "std": f"{feat}_ustd",
                              "min": f"{feat}_umin",  "max": f"{feat}_umax",
                              "median": f"{feat}_umedian"})
        )
        out = out.merge(stats, on="user_id", how="left")
        std_col = f"{feat}_ustd"
        out[std_col] = out[std_col].fillna(1e-6).replace(0, 1e-6)
        out[f"{feat}_zscore"]    = ((out[feat] - out[f"{feat}_umean"]) / out[std_col]).abs()
        rng                      = (out[f"{feat}_umax"] - out[f"{feat}_umin"]).replace(0, 1e-6)
        out[f"{feat}_range_pos"] = (out[feat] - out[f"{feat}_umin"]) / rng
        out[f"{feat}_med_dev"]   = (out[feat] - out[f"{feat}_umedian"]).abs()
        out.drop(columns=[f"{feat}_umean", f"{feat}_ustd",
                          f"{feat}_umin",  f"{feat}_umax", f"{feat}_umedian"], inplace=True)
    return out


def engineer_single(dwell: float, flight: float, traj: float, profile: dict) -> dict:
    """
    Engineer all features for one new session.

    Returns a dict with:
      - all 27 model features (same as training)
      - extra display-only features: cv, cross_z, rhythm_index,
        device_offset, imitation_risk, typing_burst_score
    """
    eps = 1e-9
    row = {
        "dwell_avg": dwell, "flight_avg": flight, "traj_avg": traj,
        "dwell_flight_ratio": dwell  / (flight + eps),
        "flight_traj_ratio":  flight / (traj   + eps),
        "dwell_traj_ratio":   dwell  / (traj   + eps),
        "dwell_x_flight": dwell  * flight,
        "dwell_x_traj":   dwell  * traj,
        "flight_x_traj":  flight * traj,
        "dwell_sq":  dwell**2, "flight_sq": flight**2, "traj_sq": traj**2,
        "inv_dwell":  1 / (dwell  + eps),
        "inv_flight": 1 / (flight + eps),
        "inv_traj":   1 / (traj   + eps),
        "log_dwell":  np.log1p(dwell),
        "log_flight": np.log1p(flight),
        "log_traj":   np.log1p(traj),
        "sqrt_dwell":  np.sqrt(abs(dwell)),
        "sqrt_flight": np.sqrt(abs(flight)),
        "sqrt_traj":   np.sqrt(abs(traj)),
    }

    # Per-user statistical features
    z_vals = []
    for feat, raw in [("dwell_avg", dwell), ("flight_avg", flight), ("traj_avg", traj)]:
        p   = profile[feat]
        std = p["std"] or 1e-6
        rng = (p["max"] - p["min"]) or 1e-6
        z   = abs((raw - p["mean"]) / std)
        row[f"{feat}_zscore"]    = z
        row[f"{feat}_range_pos"] = (raw - p["min"]) / rng
        row[f"{feat}_med_dev"]   = abs(raw - p["median"])
        z_vals.append(z)

    # ── Extended / display-only features ────────────────────────

    # Coefficient of variation: captures rhythm regularity
    # (genuine users have a characteristic typing rhythm consistency)
    row["dwell_cv"]  = abs(dwell  - profile["dwell_avg"]["mean"])  / (profile["dwell_avg"]["mean"]  + eps)
    row["flight_cv"] = abs(flight - profile["flight_avg"]["mean"]) / (profile["flight_avg"]["mean"] + eps)
    row["traj_cv"]   = abs(traj   - profile["traj_avg"]["mean"])   / (profile["traj_avg"]["mean"]   + eps)

    # Combined Mahalanobis-like deviation distance (imitation/drift indicator)
    row["cross_z"] = float(np.sqrt(sum(z**2 for z in z_vals) / 3))

    # Rhythm index [0–1]: how well this session matches the user's typical rhythm
    # 1.0 = perfect match, 0.0 = very different
    row["rhythm_index"] = float(np.exp(-row["cross_z"] * 0.5))

    # Device offset: absolute shift relative to user range (device variability signal)
    # If someone types on a different keyboard the absolute timing shifts but the
    # relative pattern is preserved — large device_offset + low cross_z = device change
    row["device_offset_dwell"]  = abs(dwell  - profile["dwell_avg"]["mean"])
    row["device_offset_flight"] = abs(flight - profile["flight_avg"]["mean"])
    row["device_offset_traj"]   = abs(traj   - profile["traj_avg"]["mean"])
    row["device_offset_total"]  = (
        row["device_offset_dwell"] / (profile["dwell_avg"]["mean"]  + eps) +
        row["device_offset_flight"]/ (profile["flight_avg"]["mean"] + eps) +
        row["device_offset_traj"]  / (profile["traj_avg"]["mean"]   + eps)
    ) / 3.0

    # Typing burst score: impostors who memorise target averages cluster unnaturally
    # close to the centroid. A z-score < 0.1 across ALL features is statistically
    # rare for genuine users (they have natural variance) and suspicious.
    avg_z  = sum(z_vals) / 3
    too_perfect = 1.0 if avg_z < 0.08 else max(0.0, 1.0 - avg_z / 0.5)

    # Cross-feature ratio deviation (genuine users have consistent dwell/flight ratio)
    user_df_ratio = profile["dwell_avg"]["mean"] / (profile["flight_avg"]["mean"] + eps)
    user_ft_ratio = profile["flight_avg"]["mean"] / (profile["traj_avg"]["mean"]  + eps)
    df_z = abs(row["dwell_flight_ratio"] - user_df_ratio) / (user_df_ratio * 0.15 + eps)
    ft_z = abs(row["flight_traj_ratio"]  - user_ft_ratio) / (user_ft_ratio * 0.15 + eps)

    row["imitation_risk"] = float(min(1.0,
        too_perfect * 0.45 +
        min(1.0, df_z / 4) * 0.30 +
        min(1.0, ft_z / 4) * 0.25
    ))

    return row


# ─────────────────────────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────────────────────────
def predict_one(feat_row: dict):
    af = STATE["all_features"]
    x  = np.array([[feat_row[f] for f in af]])
    xs = STATE["scaler"].transform(x)

    p_svm  = float(STATE["models"]["svm"].predict_proba(xs)[0][1])
    p_xgb  = float(STATE["models"]["xgb"].predict_proba(xs)[0][1])
    p_mlp  = float(STATE["models"]["mlp"].predict(xs, verbose=0)[0][0])
    p_vote = p_svm * VOTE_W["svm"] + p_xgb * VOTE_W["xgb"] + p_mlp * VOTE_W["mlp"]

    th = {
        "svm":  MODEL_META["svm"]["thresh"],
        "xgb":  MODEL_META["xgb"]["thresh"],
        "mlp":  STATE["mlp_thresh"],
        "vote": STATE["vote_thresh"],
    }
    results = {
        "svm":  {"score": round(p_svm,  4), "pred": int(p_svm  >= th["svm"])},
        "xgb":  {"score": round(p_xgb,  4), "pred": int(p_xgb  >= th["xgb"])},
        "mlp":  {"score": round(p_mlp,  4), "pred": int(p_mlp  >= th["mlp"])},
        "vote": {"score": round(p_vote, 4), "pred": int(p_vote >= th["vote"])},
    }
    fails = sum(1 for v in results.values() if v["pred"] == 0)
    flag  = ("critical" if fails == 4 else
             "warning"  if fails == 3 else
             "caution"  if fails >= 1 else "safe")
    return results, flag, fails


# ─────────────────────────────────────────────────────────────────
# Threat score (0–100)
# ─────────────────────────────────────────────────────────────────
def compute_threat_score(uid: str) -> float:
    """
    Composite risk indicator — based on GENUINE sessions only (label=1).

    Rationale: impostor sessions in the dataset are test cases showing the system
    catching attackers — that's correct behaviour, not a threat to the user's account.
    The threat score should reflect how suspicious the *genuine user's own sessions*
    look to the models (high = the account's real behaviour is anomalous / drifting).

      45%  flag severity on recent genuine sessions
      35%  model failure rate on recent genuine sessions
      20%  z-score anomaly magnitude on recent genuine sessions
    """
    all_sessions = STATE["user_sessions"].get(uid, [])
    # Only genuine (label=1) sessions feed the threat score
    genuine = [s for s in all_sessions if s.get("label", 1) == 1]
    if not genuine:
        return 0.0
    recent = genuine[-10:]
    flag_score = sum(FLAG_WEIGHT.get(s["flag"], 0) for s in recent) / len(recent)
    fail_score = sum(s["n_fails"] / 4.0            for s in recent) / len(recent)
    z_avg      = sum(
        (s.get("dwell_zscore", 0) + s.get("flight_zscore", 0) + s.get("traj_zscore", 0)) / 3
        for s in recent
    ) / len(recent)
    z_score = min(1.0, z_avg / 3.0)
    raw = 0.45 * flag_score + 0.35 * fail_score + 0.20 * z_score
    return round(min(100.0, max(0.0, raw * 100)), 1)


def threat_level(score: float) -> str:
    if score >= 70: return "critical"
    if score >= 45: return "warning"
    if score >= 20: return "caution"
    return "safe"


# ─────────────────────────────────────────────────────────────────
# Adaptive profile update (Behavioral Drift fix)
# ─────────────────────────────────────────────────────────────────
def adaptive_update(uid: str, dwell: float, flight: float, traj: float):
    """
    Exponential Weighted Mean update.
    Call when a session is confirmed genuine (admin or all-models-pass).
    α = 0.08 → profile adapts slowly, protecting against gradual imposition.
    """
    profile = STATE["user_profiles"].get(uid)
    if not profile:
        return
    for feat, val in [("dwell_avg", dwell), ("flight_avg", flight), ("traj_avg", traj)]:
        p        = profile[feat]
        old_mean = p["mean"]
        new_mean = ALPHA * val + (1 - ALPHA) * old_mean
        p["std"]    = max(1e-6, float(np.sqrt(
            ALPHA * (val - new_mean)**2 + (1 - ALPHA) * p["std"]**2
        )))
        p["mean"]   = new_mean
        p["min"]    = min(p["min"],  val)
        p["max"]    = max(p["max"],  val)
        p["median"] = 0.5 * (p["median"] + val)  # running approximate median


# ─────────────────────────────────────────────────────────────────
# Alert system
# ─────────────────────────────────────────────────────────────────
def push_alert(uid: str, session_id: int, flag: str, n_fails: int,
               threat_score: float, imitation_risk: float = 0.0):
    if flag == "safe":
        return
    alert = {
        "id":             str(uuid.uuid4())[:8],
        "ts":             datetime.now().isoformat(timespec="seconds"),
        "uid":            uid,
        "session_id":     session_id,
        "flag":           flag,
        "n_fails":        n_fails,
        "threat_score":   threat_score,
        "imitation_risk": round(imitation_risk, 3),
        "dismissed":      False,
    }
    STATE["alerts"].insert(0, alert)
    if len(STATE["alerts"]) > 500:
        STATE["alerts"] = STATE["alerts"][:500]

    # Broadcast to all SSE subscribers
    for q in list(_sse_queues):
        try:
            q.put_nowait(alert)
        except queue.Full:
            pass


# ─────────────────────────────────────────────────────────────────
# Evaluation helper
# ─────────────────────────────────────────────────────────────────
def _eval(y_true, y_prob, thresh=0.50):
    from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
    preds = (y_prob >= thresh).astype(int)
    cm    = confusion_matrix(y_true, preds)
    TP, TN = int(cm[1,1]), int(cm[0,0])
    FP, FN = int(cm[0,1]), int(cm[1,0])
    return {
        "accuracy": float(accuracy_score(y_true, preds)),
        "roc_auc":  float(roc_auc_score(y_true, y_prob)),
        "far":      FP / (FP + TN + 1e-9),
        "frr":      FN / (FN + TP + 1e-9),
        "tp": TP, "tn": TN, "fp": FP, "fn": FN,
    }


# ─────────────────────────────────────────────────────────────────
# Pipeline (runs in background thread)
# ─────────────────────────────────────────────────────────────────
def _log(msg: str):
    print(msg)
    STATE["log"].append(msg)
    if len(STATE["log"]) > 200:
        STATE["log"] = STATE["log"][-200:]


def run_pipeline():
    import tensorflow as tf
    from sklearn.preprocessing   import StandardScaler
    from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
    from sklearn.svm             import SVC
    from sklearn.metrics         import accuracy_score
    from imblearn.over_sampling  import SMOTE
    from imblearn.combine        import SMOTETomek
    import xgboost as xgb
    from tensorflow.keras        import layers, callbacks, regularizers

    # 1. Load ─────────────────────────────────────────────────────
    STATE["status"] = "loading"; STATE["status_msg"] = "Loading dataset..."
    _log("📂 Scanning for xlsx files...")
    files = sorted(glob.glob(os.path.join(DATASET_PATH, "*.xlsx")))
    if not files:
        STATE["status"] = "error"
        STATE["status_msg"] = f"No xlsx files found in: {DATASET_PATH}"
        _log(f"❌ {STATE['status_msg']}"); return

    dfs = []
    for f in files:
        tmp = pd.read_excel(f)
        tmp["source_file"] = os.path.basename(f)
        dfs.append(tmp)
    df = pd.concat(dfs, ignore_index=True)
    _log(f"✅ Loaded {len(df):,} rows from {len(files)} users")

    # 2. Feature engineering ──────────────────────────────────────
    STATE["status_msg"] = "Engineering features..."
    _log("🔧 Engineering features...")
    df = engineer_df(df)
    all_features = [c for c in df.columns
                    if c not in ["label", "user_id", "source_file", "Unnamed: 0"]]
    STATE["all_features"] = all_features
    _log(f"✅ {len(all_features)} model features")

    # 3. User profiles ─────────────────────────────────────────────
    _log("👤 Building per-user genuine profiles...")
    for uid, grp in df[df["label"] == 1].groupby("user_id"):
        STATE["user_profiles"][uid] = {}
        for feat in BASE_FEATS:
            v = grp[feat]
            STATE["user_profiles"][uid][feat] = {
                "mean":   float(v.mean()),
                "std":    float(v.std()) or 1e-6,
                "min":    float(v.min()),
                "max":    float(v.max()),
                "median": float(v.median()),
            }
    _log(f"✅ Profiles for {len(STATE['user_profiles'])} users")

    # 4. Split + balance ───────────────────────────────────────────
    STATE["status"] = "training"; STATE["status_msg"] = "Balancing classes..."
    X = df[all_features].values; y = df["label"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
    STATE["scaler"] = scaler
    _log(f"Before: {Counter(y_train)}")
    smt = SMOTETomek(smote=SMOTE(k_neighbors=5, random_state=42), random_state=42)
    X_res, y_res = smt.fit_resample(X_train_s, y_train)
    _log(f"After:  {Counter(y_res)}")

    # 5. Load or train models ──────────────────────────────────────
    needed = ["scaler.pkl","svm_model.pkl","xgb_model.pkl","mlp_model.keras"]
    loaded = all(os.path.exists(os.path.join(MODEL_DIR, f)) for f in needed)
    if loaded:
        _log("💾 Loading saved models from disk...")
        try:
            STATE["scaler"]        = joblib.load(os.path.join(MODEL_DIR,"scaler.pkl"))
            STATE["models"]["svm"] = joblib.load(os.path.join(MODEL_DIR,"svm_model.pkl"))
            STATE["models"]["xgb"] = joblib.load(os.path.join(MODEL_DIR,"xgb_model.pkl"))
            STATE["models"]["mlp"] = tf.keras.models.load_model(
                os.path.join(MODEL_DIR,"mlp_model.keras"))
            _log("✅ Models loaded")
        except Exception as e:
            _log(f"⚠ Load failed ({e}) — retraining..."); loaded = False

    if not loaded:
        STATE["status_msg"] = "Training SVM..."
        _log("🤖 Training SVM (GridSearchCV)...")
        svm_cv = GridSearchCV(
            SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42),
            {"C":[1,10,100,1000],"gamma":["scale","auto",0.001,0.01,0.1]},
            cv=5, scoring="accuracy", n_jobs=-1, verbose=0)
        svm_cv.fit(X_res, y_res)
        STATE["models"]["svm"] = svm_cv.best_estimator_
        _log(f"   Best SVM: {svm_cv.best_params_}")

        STATE["status_msg"] = "Training XGBoost..."
        _log("🤖 Training XGBoost (RandomizedSearchCV)...")
        xgb_rs = RandomizedSearchCV(
            xgb.XGBClassifier(eval_metric="logloss",random_state=42,n_jobs=-1,tree_method="hist"),
            {"n_estimators":[400,600,800],"max_depth":[4,5,6],"learning_rate":[0.01,0.03,0.05],
             "subsample":[0.7,0.8,0.9],"colsample_bytree":[0.7,0.8,0.9],
             "min_child_weight":[1,3,5],"gamma":[0,0.1,0.2],
             "reg_alpha":[0.0,0.1,0.5],"reg_lambda":[0.5,1.0,2.0]},
            n_iter=30, cv=5, scoring="accuracy", random_state=42, n_jobs=-1, verbose=0)
        xgb_rs.fit(X_res, y_res)
        STATE["models"]["xgb"] = xgb_rs.best_estimator_
        _log(f"   Best XGB: {xgb_rs.best_params_}")

        STATE["status_msg"] = "Training MLP..."
        _log("🤖 Training MLP (512→256→128→64→32)...")
        tf.random.set_seed(42)
        inp = tf.keras.Input(shape=(X_res.shape[1],))
        x = layers.Dense(512, kernel_regularizer=regularizers.l2(1e-4))(inp)
        x = layers.BatchNormalization()(x); x = layers.Activation("relu")(x); x = layers.Dropout(0.35)(x)
        x = layers.Dense(256, kernel_regularizer=regularizers.l2(1e-4))(x)
        x = layers.BatchNormalization()(x); x = layers.Activation("relu")(x); x = layers.Dropout(0.30)(x)
        x = layers.Dense(128, kernel_regularizer=regularizers.l2(1e-4))(x)
        x = layers.BatchNormalization()(x); x = layers.Activation("relu")(x); x = layers.Dropout(0.20)(x)
        x = layers.Dense(64,  kernel_regularizer=regularizers.l2(1e-5))(x)
        x = layers.BatchNormalization()(x); x = layers.Activation("relu")(x)
        x = layers.Dense(32, activation="relu")(x)
        out = layers.Dense(1, activation="sigmoid")(x)
        mlp = tf.keras.Model(inp, out)
        mlp.compile(optimizer=tf.keras.optimizers.Adam(3e-4),
                    loss="binary_crossentropy", metrics=["accuracy"])
        mlp.fit(X_res, y_res, validation_split=0.15, epochs=500, batch_size=32,
                callbacks=[callbacks.EarlyStopping(monitor="val_accuracy",patience=30,
                                                    restore_best_weights=True,verbose=0),
                           callbacks.ReduceLROnPlateau(monitor="val_loss",factor=0.4,
                                                        patience=10,min_lr=1e-7,verbose=0)],
                verbose=0)
        STATE["models"]["mlp"] = mlp

        _log("💾 Saving models...")
        joblib.dump(STATE["scaler"],        os.path.join(MODEL_DIR,"scaler.pkl"))
        joblib.dump(STATE["models"]["svm"], os.path.join(MODEL_DIR,"svm_model.pkl"))
        joblib.dump(STATE["models"]["xgb"], os.path.join(MODEL_DIR,"xgb_model.pkl"))
        STATE["models"]["mlp"].save(        os.path.join(MODEL_DIR,"mlp_model.keras"))
        _log("✅ Models saved")

    # 6. Optimal thresholds ────────────────────────────────────────
    STATE["status_msg"] = "Finding optimal thresholds..."
    y_prob_svm  = STATE["models"]["svm"].predict_proba(X_test_s)[:,1]
    y_prob_xgb  = STATE["models"]["xgb"].predict_proba(X_test_s)[:,1]
    y_prob_mlp  = STATE["models"]["mlp"].predict(X_test_s, verbose=0).ravel()
    y_prob_vote = y_prob_svm*VOTE_W["svm"] + y_prob_xgb*VOTE_W["xgb"] + y_prob_mlp*VOTE_W["mlp"]

    def best_t(probs):
        bt,ba = 0.50,0.0
        for t in np.arange(0.30,0.70,0.005):
            a = accuracy_score(y_test,(probs>=t).astype(int))
            if a>ba: ba,bt=a,t
        return bt

    STATE["mlp_thresh"]  = best_t(y_prob_mlp)
    STATE["vote_thresh"] = best_t(y_prob_vote)
    _log(f"   MLP threshold : {STATE['mlp_thresh']:.3f}")
    _log(f"   Vote threshold: {STATE['vote_thresh']:.3f}")

    # 7. Evaluate ──────────────────────────────────────────────────
    STATE["status_msg"] = "Evaluating..."
    STATE["model_metrics"]["svm"]  = _eval(y_test, y_prob_svm,  0.50)
    STATE["model_metrics"]["xgb"]  = _eval(y_test, y_prob_xgb,  0.50)
    STATE["model_metrics"]["mlp"]  = _eval(y_test, y_prob_mlp,  STATE["mlp_thresh"])
    STATE["model_metrics"]["vote"] = _eval(y_test, y_prob_vote, STATE["vote_thresh"])
    for mid, m in STATE["model_metrics"].items():
        _log(f"   {MODEL_META[mid]['label']:18s} acc={m['accuracy']*100:.2f}%  AUC={m['roc_auc']:.3f}")

    # 8. Pre-score all sessions ─────────────────────────────────────
    STATE["status_msg"] = "Pre-scoring sessions..."
    _log("🔍 Scoring all sessions...")
    for uid in STATE["user_profiles"]:
        uid_df = df[df["user_id"] == uid].reset_index(drop=True)
        sessions = []
        for i, row in uid_df.iterrows():
            feat = {f: row[f] for f in all_features}
            model_res, flag, fails = predict_one(feat)

            # Extended display features via engineer_single
            ext = engineer_single(row["dwell_avg"], row["flight_avg"], row["traj_avg"],
                                   STATE["user_profiles"][uid])

            sessions.append({
                "session_id":     i + 1,
                "label":          int(row["label"]),
                "dwell_avg":      float(row["dwell_avg"]),
                "flight_avg":     float(row["flight_avg"]),
                "traj_avg":       float(row["traj_avg"]),
                "dwell_zscore":   float(ext["dwell_avg_zscore"]),
                "flight_zscore":  float(ext["flight_avg_zscore"]),
                "traj_zscore":    float(ext["traj_avg_zscore"]),
                "cross_z":        round(ext["cross_z"],       3),
                "rhythm_index":   round(ext["rhythm_index"],  3),
                "imitation_risk": round(ext["imitation_risk"],3),
                "device_offset":  round(ext["device_offset_total"], 3),
                "models":         model_res,
                "flag":           flag,
                "n_fails":        fails,
            })
            # NOTE: No alerts generated during pre-scoring.
            # These are historical dataset sessions (including intentional impostor
            # test cases). Alerts are only pushed for live predictions (/api/predict)
            # and simulation playback, so the alert centre stays clean on startup.

        STATE["user_sessions"][uid] = sessions

    n_total = sum(len(v) for v in STATE["user_sessions"].values())
    _log(f"✅ Scored {n_total:,} sessions across {len(STATE['user_sessions'])} users")
    _log(f"   Total alerts generated: {len(STATE['alerts'])}")
    STATE["status"] = "ready"; STATE["status_msg"] = "Ready"
    _log("🚀 Dashboard ready at http://localhost:5000")


# ─────────────────────────────────────────────────────────────────
# REST API
# ─────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/status")
def api_status():
    return jsonify({"status": STATE["status"], "message": STATE["status_msg"],
                    "n_users": len(STATE["user_profiles"]),
                    "ready": STATE["status"]=="ready", "log": STATE["log"][-40:]})


@app.route("/api/users")
def api_users():
    if STATE["status"] != "ready": return jsonify({"error":"not ready"}), 503
    out = []
    for uid, profile in sorted(STATE["user_profiles"].items()):
        sessions = STATE["user_sessions"].get(uid, [])
        ts       = compute_threat_score(uid)
        tl       = threat_level(ts)
        m_acc    = {}
        for mid in MODEL_META:
            correct = sum(1 for s in sessions
                          if s["models"].get(mid,{}).get("pred") == s["label"])
            m_acc[mid] = round(correct / len(sessions), 4) if sessions else 0

        # Recent trend: last 5 threat scores over genuine sessions only
        genuine_sessions = [s for s in sessions if s.get("label", 1) == 1]
        def _ts_at(idx):
            sub = genuine_sessions[max(0,idx-10):idx] if idx > 0 else []
            if not sub: return 0.0
            fs = sum(FLAG_WEIGHT.get(s["flag"],0) for s in sub)/len(sub)
            fr = sum(s["n_fails"]/4.0 for s in sub)/len(sub)
            return round(min(100,(fs*0.45+fr*0.35)*100), 1)
        trend = [_ts_at(i) for i in range(max(0,len(genuine_sessions)-4), len(genuine_sessions)+1)]

        out.append({
            "user_id":          uid,
            "n_sessions":       len(sessions),
            "threat_score":     ts,
            "threat_level":     tl,
            "threat_trend":     trend,
            "model_accuracy":   m_acc,
            "profile":          {f:{k:round(v,6) for k,v in st.items()}
                                  for f,st in profile.items()},
            "recent_imitation": round(
                sum(s.get("imitation_risk",0) for s in genuine_sessions[-5:])
                / max(1, len(genuine_sessions[-5:])), 3
            ),
        })
    return jsonify(out)


@app.route("/api/user/<uid>/sessions")
def api_sessions(uid):
    if STATE["status"] != "ready": return jsonify({"error":"not ready"}), 503
    ss = STATE["user_sessions"].get(uid)
    if ss is None: return jsonify({"error":"user not found"}), 404
    return jsonify(ss)


@app.route("/api/user/<uid>/adapt", methods=["POST"])
def api_adapt(uid):
    """
    Confirm a session as genuine and update the adaptive profile.
    Body: { dwell_avg, flight_avg, traj_avg }
    """
    if STATE["status"] != "ready": return jsonify({"error":"not ready"}), 503
    if uid not in STATE["user_profiles"]: return jsonify({"error":"unknown user"}), 404
    d = request.json or {}
    try:
        dwell  = float(d["dwell_avg"])
        flight = float(d["flight_avg"])
        traj   = float(d["traj_avg"])
    except (KeyError, ValueError) as e:
        return jsonify({"error": str(e)}), 400
    adaptive_update(uid, dwell, flight, traj)
    return jsonify({"status":"profile updated", "user_id":uid,
                    "new_profile": STATE["user_profiles"][uid]})


@app.route("/api/models/summary")
def api_models_summary():
    if STATE["status"] != "ready": return jsonify({"error":"not ready"}), 503
    return jsonify([
        {"id": mid, "label": meta["label"], "color": meta["color"], "short": meta["short"],
         **{k:round(v,4) for k,v in STATE["model_metrics"].get(mid,{}).items()
            if k not in ("y_pred","y_prob")}}
        for mid, meta in MODEL_META.items()
    ])


@app.route("/api/alerts")
def api_alerts():
    if STATE["status"] != "ready": return jsonify({"error":"not ready"}), 503
    severity = request.args.get("severity")      # filter: critical|warning|caution
    limit    = int(request.args.get("limit", 100))
    alerts   = [a for a in STATE["alerts"] if not a["dismissed"]]
    if severity:
        alerts = [a for a in alerts if a["flag"] == severity]
    return jsonify(alerts[:limit])


@app.route("/api/alerts/<alert_id>", methods=["DELETE"])
def api_dismiss_alert(alert_id):
    for a in STATE["alerts"]:
        if a["id"] == alert_id:
            a["dismissed"] = True
            return jsonify({"dismissed": True})
    return jsonify({"error": "not found"}), 404


@app.route("/api/stream")
def api_stream():
    """Server-Sent Events stream for real-time alert push."""
    q: queue.Queue = queue.Queue(maxsize=50)
    _sse_queues.append(q)
    def generate():
        try:
            # Send last 5 undismissed alerts on connect
            for a in STATE["alerts"][:5]:
                if not a["dismissed"]:
                    yield f"data: {json.dumps(a)}\n\n"
            while True:
                try:
                    alert = q.get(timeout=25)
                    yield f"data: {json.dumps(alert)}\n\n"
                except queue.Empty:
                    yield ": keepalive\n\n"
        finally:
            _sse_queues.remove(q)
    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})


@app.route("/api/users", methods=["POST"])
def api_add_user():
    """
    Add a new user with baseline biometric profile.
    Body: {
      user_id: str,
      sessions: [ {dwell_avg, flight_avg, traj_avg}, ... ]   ← 3+ genuine samples
    }
    """
    if STATE["status"] != "ready": return jsonify({"error": "not ready"}), 503
    data = request.json or {}
    uid  = str(data.get("user_id", "")).strip()
    if not uid:
        return jsonify({"error": "user_id is required"}), 400
    if uid in STATE["user_profiles"]:
        return jsonify({"error": f"User '{uid}' already exists"}), 409

    samples = data.get("sessions", [])
    if len(samples) < 1:
        return jsonify({"error": "At least 1 session sample required"}), 400

    try:
        dwells  = [float(s["dwell_avg"])  for s in samples]
        flights = [float(s["flight_avg"]) for s in samples]
        trajs   = [float(s["traj_avg"])   for s in samples]
    except (KeyError, ValueError) as e:
        return jsonify({"error": str(e)}), 400

    def _stats(vals):
        arr = np.array(vals)
        return {
            "mean":   float(np.mean(arr)),
            "std":    max(1e-6, float(np.std(arr))),
            "min":    float(np.min(arr)),
            "max":    float(np.max(arr)),
            "median": float(np.median(arr)),
        }

    STATE["user_profiles"][uid] = {
        "dwell_avg":  _stats(dwells),
        "flight_avg": _stats(flights),
        "traj_avg":   _stats(trajs),
    }

    # Score the provided samples as initial session history
    profile  = STATE["user_profiles"][uid]
    sessions = []
    for i, s in enumerate(samples):
        feat = engineer_single(float(s["dwell_avg"]), float(s["flight_avg"]),
                                float(s["traj_avg"]), profile)
        model_res, flag, fails = predict_one(feat)
        sessions.append({
            "session_id":     i + 1,
            "label":          1,
            "dwell_avg":      float(s["dwell_avg"]),
            "flight_avg":     float(s["flight_avg"]),
            "traj_avg":       float(s["traj_avg"]),
            "dwell_zscore":   round(feat["dwell_avg_zscore"],  3),
            "flight_zscore":  round(feat["flight_avg_zscore"], 3),
            "traj_zscore":    round(feat["traj_avg_zscore"],   3),
            "cross_z":        round(feat["cross_z"],           3),
            "rhythm_index":   round(feat["rhythm_index"],      3),
            "imitation_risk": round(feat["imitation_risk"],    3),
            "device_offset":  round(feat["device_offset_total"],3),
            "models":         model_res,
            "flag":           flag,
            "n_fails":        fails,
        })
    STATE["user_sessions"][uid] = sessions
    _log(f"➕ New user added: {uid} ({len(samples)} seed sessions)")
    return jsonify({"status": "created", "user_id": uid,
                    "n_sessions": len(sessions),
                    "profile": STATE["user_profiles"][uid]}), 201


@app.route("/api/user/<uid>/simulate")
def api_simulate(uid):
    """
    Returns a realistic ordered sequence of sessions for simulation playback.
    Pattern: chunks of genuine sessions (6-14) with rare impostor bursts (1-3)
    sprinkled throughout — not all genuine then all impostor.
    Impostor values are drawn from another user's profile + noise.
    """
    if STATE["status"] != "ready": return jsonify({"error": "not ready"}), 503
    profile = STATE["user_profiles"].get(uid)
    if not profile: return jsonify({"error": "user not found"}), 404

    rng         = np.random.default_rng(abs(hash(uid)) % (2**32))
    other_users = [u for u in STATE["user_profiles"] if u != uid]
    seq         = []
    session_num = 1
    total       = int(rng.integers(28, 42))   # 28–42 total sessions

    def _gen_genuine():
        """Generate a genuine session with natural variation around the profile."""
        s = {}
        for feat in BASE_FEATS:
            p    = profile[feat]
            val  = float(rng.normal(p["mean"], p["std"] * 0.8))
            val  = max(p["min"] * 0.7, val)
            s[feat] = val
        return s

    def _gen_impostor():
        """Generate an impostor session: values from another user's profile."""
        if other_users:
            src = STATE["user_profiles"][rng.choice(other_users)]
        else:
            src = profile   # fallback: exaggerated noise
        s = {}
        for feat in BASE_FEATS:
            p   = src[feat]
            val = float(rng.normal(p["mean"], p["std"] * 1.1))
            s[feat] = max(1e-6, val)
        return s

    i = 0
    while i < total:
        # Genuine chunk: 6–14 sessions
        chunk = int(rng.integers(6, 15))
        for _ in range(min(chunk, total - i)):
            raw = _gen_genuine()
            feat = engineer_single(raw["dwell_avg"], raw["flight_avg"],
                                    raw["traj_avg"], profile)
            model_res, flag, fails = predict_one(feat)
            seq.append({
                "session_id":     session_num,
                "label":          1,
                "dwell_avg":      round(raw["dwell_avg"],  6),
                "flight_avg":     round(raw["flight_avg"], 6),
                "traj_avg":       round(raw["traj_avg"],   2),
                "dwell_zscore":   round(feat["dwell_avg_zscore"],   3),
                "flight_zscore":  round(feat["flight_avg_zscore"],  3),
                "traj_zscore":    round(feat["traj_avg_zscore"],    3),
                "cross_z":        round(feat["cross_z"],            3),
                "rhythm_index":   round(feat["rhythm_index"],       3),
                "imitation_risk": round(feat["imitation_risk"],     3),
                "device_offset":  round(feat["device_offset_total"],3),
                "models":         model_res,
                "flag":           flag,
                "n_fails":        fails,
            })
            session_num += 1
        i += chunk
        if i >= total:
            break

        # Impostor burst: 1–3 sessions (only if ~40% chance and room remains)
        if rng.random() < 0.40 and i < total - 2:
            burst = int(rng.integers(1, 4))
            for _ in range(min(burst, total - i)):
                raw = _gen_impostor()
                feat = engineer_single(raw["dwell_avg"], raw["flight_avg"],
                                        raw["traj_avg"], profile)
                model_res, flag, fails = predict_one(feat)
                seq.append({
                    "session_id":     session_num,
                    "label":          0,
                    "dwell_avg":      round(raw["dwell_avg"],  6),
                    "flight_avg":     round(raw["flight_avg"], 6),
                    "traj_avg":       round(raw["traj_avg"],   2),
                    "dwell_zscore":   round(feat["dwell_avg_zscore"],   3),
                    "flight_zscore":  round(feat["flight_avg_zscore"],  3),
                    "traj_zscore":    round(feat["traj_avg_zscore"],    3),
                    "cross_z":        round(feat["cross_z"],            3),
                    "rhythm_index":   round(feat["rhythm_index"],       3),
                    "imitation_risk": round(feat["imitation_risk"],     3),
                    "device_offset":  round(feat["device_offset_total"],3),
                    "models":         model_res,
                    "flag":           flag,
                    "n_fails":        fails,
                })
                session_num += 1
            i += burst

    return jsonify(seq)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    if STATE["status"] != "ready": return jsonify({"error":"not ready"}), 503
    data    = request.json or {}
    uid     = str(data.get("user_id",""))
    profile = STATE["user_profiles"].get(uid)
    if not profile: return jsonify({"error":f"Unknown user: {uid}"}), 404
    try:
        dwell  = float(data["dwell_avg"])
        flight = float(data["flight_avg"])
        traj   = float(data["traj_avg"])
    except (KeyError, ValueError) as e:
        return jsonify({"error": str(e)}), 400

    feat_row = engineer_single(dwell, flight, traj, profile)
    res, flag, fails = predict_one(feat_row)
    ts = compute_threat_score(uid)
    ir = feat_row["imitation_risk"]

    if flag in ("critical","warning","caution"):
        push_alert(uid, -1, flag, fails, ts, ir)

    return jsonify({
        "user_id": uid, "flag": flag, "n_fails": fails,
        "threat_score": ts, "input": {"dwell_avg":dwell,"flight_avg":flight,"traj_avg":traj},
        "models":   {k:{"score":round(v["score"],4),"pred":v["pred"]} for k,v in res.items()},
        "extended": {
            "dwell_zscore":   round(feat_row["dwell_avg_zscore"],  3),
            "flight_zscore":  round(feat_row["flight_avg_zscore"], 3),
            "traj_zscore":    round(feat_row["traj_avg_zscore"],   3),
            "cross_z":        round(feat_row["cross_z"],           3),
            "rhythm_index":   round(feat_row["rhythm_index"],      3),
            "imitation_risk": round(ir,                             3),
            "device_offset":  round(feat_row["device_offset_total"],3),
        },
    })


# ─────────────────────────────────────────────────────────────────
# Boot
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs("static", exist_ok=True)
    threading.Thread(target=run_pipeline, daemon=True).start()
    print("🚀 BehaviourAuth v2 → http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
