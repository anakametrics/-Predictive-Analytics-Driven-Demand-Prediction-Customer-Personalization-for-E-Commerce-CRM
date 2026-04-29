import pandas as pd
import xgboost as xgb
import shap
import mysql.connector

# ---------------- DB ----------------
def get_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="next_basket"
    )

# ---------------- LOAD ----------------
def load_data():
    conn = get_connection()
    df = pd.read_sql("SELECT * FROM nb_product", conn)
    conn.close()
    return df

# ---------------- FEATURES ----------------
def feature_engineering(df):

    df["current_stock"] = df["quantity"]

    df["forecast_demand"] = df["current_stock"] * 0.6 + 20
    df["lead_time"] = 3
    df["avg_sales"] = df["forecast_demand"] / 2
    df["reorder_point"] = df["avg_sales"] * df["lead_time"]
    df["stock_ratio"] = df["current_stock"] / (df["forecast_demand"] + 1)

    return df

# ---------------- TARGET (YOUR RULE) ----------------
def create_target(df):

    def risk_logic(stock):
        if stock <= 5:
            return 2   # HIGH
        elif stock <= 50:
            return 1   # MEDIUM
        else:
            return 0   # LOW

    df["risk"] = df["current_stock"].apply(risk_logic)

    print("Risk Distribution:\n", df["risk"].value_counts())

    return df

# ---------------- TRAIN ----------------
def train_model(df):

    features = [
        "current_stock",
        "forecast_demand",
        "lead_time",
        "avg_sales",
        "reorder_point",
        "stock_ratio"
    ]

    X = df[features]
    y = df["risk"]

    model = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=3,
        n_estimators=120,
        max_depth=5
    )

    model.fit(X, y)

    return model, X

# ---------------- SHAP + PREDICT ----------------
def predict_and_explain(df, model, X):

    df["predicted_risk"] = model.predict(X)

    # Label mapping
    df["risk_label"] = df["predicted_risk"].map({
        2: "High",
        1: "Medium",
        0: "Low"
    })

    # SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    explanations = []

    for i in range(len(df)):
        exp = []
        class_idx = int(df.loc[i, "predicted_risk"])

        for j, col in enumerate(X.columns):

            val = shap_values[class_idx][i][j]

            if val > 0:
                exp.append(f"{col} ↑")
            else:
                exp.append(f"{col} ↓")

        explanations.append(", ".join(exp))

    df["explanation"] = explanations

    return df

# ---------------- REORDER ----------------
def reorder_calc(df):

    def calc(row):
        demand = row["forecast_demand"]
        stock = row["current_stock"]
        lead = row["lead_time"]

        return max(0, int(demand * lead + 10 - stock))

    df["reorder_qty"] = df.apply(calc, axis=1)

    return df

# ---------------- FULL PIPELINE ----------------
def run_pipeline():

    df = load_data()
    df = feature_engineering(df)
    df = create_target(df)

    model, X = train_model(df)

    df = predict_and_explain(df, model, X)
    df = reorder_calc(df)

    return df
