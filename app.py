import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
interactions = pd.read_csv("interactions.csv")

for df in [train, test, interactions]:
    if "service_date" in df.columns:
        df["service_date"] = pd.to_datetime(df["service_date"], format="%d-%m-%Y")
    if "interaction_date" in df.columns:
        df["interaction_date"] = pd.to_datetime(df["interaction_date"], format="%d-%m-%Y")


def add_time_features(df):
    df["year"] = df["service_date"].dt.year
    df["month"] = df["service_date"].dt.month
    df["day"] = df["service_date"].dt.day
    df["dayofweek"] = df["service_date"].dt.dayofweek
    df["dayofyear"] = df["service_date"].dt.dayofyear
    df["week"] = df["service_date"].dt.isocalendar().week
    df["quarter"] = df["service_date"].dt.quarter
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["is_month_start"] = df["service_date"].dt.is_month_start.astype(int)
    df["is_month_end"] = df["service_date"].dt.is_month_end.astype(int)
    df["is_quarter_start"] = df["service_date"].dt.is_quarter_start.astype(int)
    df["is_quarter_end"] = df["service_date"].dt.is_quarter_end.astype(int)
    return df


def build_interaction_features(df, offset=15):
    base = df[df["days_before_service"] == offset]

    agg = (
        base.groupby(["service_date", "origin_hub_id", "destination_hub_id"])
        .agg({
            "cumulative_commitments": ["last", "mean"],
            "cumulative_interest_signals": ["last", "mean"]
        })
        .reset_index()
    )

    agg.columns = [
        "service_date", "origin_hub_id", "destination_hub_id",
        "commit_15d", "commit_15d_mean",
        "interest_15d", "interest_15d_mean"
    ]

    recent = df[
        (df["days_before_service"] >= offset) &
        (df["days_before_service"] <= offset + 7)
    ]

    trend = (
        recent.groupby(["service_date", "origin_hub_id", "destination_hub_id"])
        .agg({
            "cumulative_commitments": ["min", "max", "std"],
            "cumulative_interest_signals": ["min", "max", "std"]
        })
        .reset_index()
    )

    trend.columns = [
        "service_date", "origin_hub_id", "destination_hub_id",
        "commit_min", "commit_max", "commit_std",
        "interest_min", "interest_max", "interest_std"
    ]

    feats = agg.merge(trend, on=["service_date", "origin_hub_id", "destination_hub_id"], how="left")

    feats["commit_growth"] = (feats["commit_max"] - feats["commit_min"]) / (feats["commit_min"] + 1)
    feats["interest_growth"] = (feats["interest_max"] - feats["interest_min"]) / (feats["interest_min"] + 1)
    feats["commit_interest_ratio"] = feats["commit_15d"] / (feats["interest_15d"] + 1)

    return feats


def hub_route_features(df):
    origin = df.groupby("origin_hub_id").agg({
        "cumulative_commitments": ["mean", "std", "max"],
        "cumulative_interest_signals": ["mean", "std", "max"]
    }).reset_index()

    origin.columns = [
        "origin_hub_id",
        "origin_commit_mean", "origin_commit_std", "origin_commit_max",
        "origin_interest_mean", "origin_interest_std", "origin_interest_max"
    ]

    dest = df.groupby("destination_hub_id").agg({
        "cumulative_commitments": ["mean", "std", "max"],
        "cumulative_interest_signals": ["mean", "std", "max"]
    }).reset_index()

    dest.columns = [
        "destination_hub_id",
        "dest_commit_mean", "dest_commit_std", "dest_commit_max",
        "dest_interest_mean", "dest_interest_std", "dest_interest_max"
    ]

    route = df.groupby(["origin_hub_id", "destination_hub_id"]).agg({
        "cumulative_commitments": ["mean", "std", "max", "count"],
        "cumulative_interest_signals": ["mean", "std", "max"]
    }).reset_index()

    route.columns = [
        "origin_hub_id", "destination_hub_id",
        "route_commit_mean", "route_commit_std", "route_commit_max",
        "route_freq", "route_interest_mean", "route_interest_std", "route_interest_max"
    ]

    return origin, dest, route


train = add_time_features(train)
test = add_time_features(test)

interaction_feats = build_interaction_features(interactions)
origin_stats, dest_stats, route_stats = hub_route_features(interactions)


def merge_features(df):
    df = df.merge(interaction_feats, on=["service_date", "origin_hub_id", "destination_hub_id"], how="left")
    df = df.merge(origin_stats, on="origin_hub_id", how="left")
    df = df.merge(dest_stats, on="destination_hub_id", how="left")
    df = df.merge(route_stats, on=["origin_hub_id", "destination_hub_id"], how="left")
    return df


train = merge_features(train)
test = merge_features(test)

cat_cols = ["origin_hub_id", "destination_hub_id"]
encoders = {}

for col in cat_cols:
    le = LabelEncoder()
    le.fit(pd.concat([train[col].astype(str), test[col].astype(str)]))
    train[col + "_enc"] = le.transform(train[col].astype(str))
    test[col + "_enc"] = le.transform(test[col].astype(str))
    encoders[col] = le


drop_cols = ["service_date", "final_service_units", "origin_hub_id", "destination_hub_id"]
features = [c for c in train.columns if c not in drop_cols]

X = train[features].fillna(0)
y = train["final_service_units"]
X_test = test[features].fillna(0)


kf = KFold(n_splits=5, shuffle=True, random_state=42)

oof_xgb = np.zeros(len(X))
oof_lgb = np.zeros(len(X))
oof_cat = np.zeros(len(X))

pred_xgb = np.zeros(len(X_test))
pred_lgb = np.zeros(len(X_test))
pred_cat = np.zeros(len(X_test))


for tr_idx, val_idx in kf.split(X):
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

    xgb_model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        tree_method="hist"
    )
    xgb_model.fit(X_tr, y_tr)
    oof_xgb[val_idx] = xgb_model.predict(X_val)
    pred_xgb += xgb_model.predict(X_test) / 5

    lgb_model = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    lgb_model.fit(X_tr, y_tr)
    oof_lgb[val_idx] = lgb_model.predict(X_val)
    pred_lgb += lgb_model.predict(X_test) / 5

    cat_model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.05,
        depth=7,
        random_state=42,
        verbose=False
    )
    cat_model.fit(X_tr, y_tr)
    oof_cat[val_idx] = cat_model.predict(X_val)
    pred_cat += cat_model.predict(X_test) / 5


best_mae = float("inf")
best_w = None

for w1 in np.arange(0.2, 0.5, 0.05):
    for w2 in np.arange(0.2, 0.5, 0.05):
        w3 = 1 - w1 - w2
        if w3 <= 0:
            continue
        mae = mean_absolute_error(y, w1 * oof_xgb + w2 * oof_lgb + w3 * oof_cat)
        if mae < best_mae:
            best_mae = mae
            best_w = (w1, w2, w3)


final_pred = (
    best_w[0] * pred_xgb +
    best_w[1] * pred_lgb +
    best_w[2] * pred_cat
)

final_pred = np.maximum(final_pred, 0)

submission = pd.DataFrame({
    "service_key": test["service_key"],
    "final_service_units": final_pred
})

submission.to_csv("submission.csv", index=False)