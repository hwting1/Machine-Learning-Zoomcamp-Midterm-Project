import pandas as pd
import category_encoders as ce
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
import joblib

SEED = 42

df = pd.read_csv("historical_data.csv", parse_dates=["created_at", "actual_delivery_time"])
df.drop(["estimated_order_place_duration", "estimated_store_to_consumer_driving_duration"], axis=1, inplace=True)
df = df.dropna(axis=0).reset_index(drop=True)
target = "order_fulfillment_time"
df[target] = (df['actual_delivery_time'] - df['created_at']).dt.total_seconds() / 60
df = df.loc[(df[target] <= 120) & (df[target] >= 10) , :].reset_index(drop=True)
y = df[target].copy()
X = df.drop([target, "created_at", "actual_delivery_time"], axis=1)

category = ["market_id", "store_id", "store_primary_category", "order_protocol"]
numeric = [col for col in X.columns if col not in category]

params = {
    "objective": "reg:squarederror",
    "random_state": SEED,
    "n_jobs": -1,
    "device": "cuda",
    "learning_rate": 0.1,
    "max_depth": 6,
    "n_estimators": 200,
    "subsample": 1,
    "min_child_weight": 1,
}

pipeline = Pipeline(
    [("encoder", ce.TargetEncoder(cols=category)),
     ("regressor", XGBRegressor(**params))]
)
pipeline.fit(X, y)
joblib.dump(pipeline, "pipeline.joblib")