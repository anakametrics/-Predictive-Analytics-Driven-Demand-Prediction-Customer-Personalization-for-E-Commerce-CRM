import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import mysql.connector
from sklearn.model_selection import train_test_split

# -------------------------------
#  CONNECT TO MYSQL
# -------------------------------
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="next_basket"   # change this
)

query = "SELECT * FROM nb_product"
df = pd.read_sql(query, conn)

print("Data Loaded:", df.shape)

# -------------------------------
# FEATURE ENGINEERING
# -------------------------------

# Current stock
df["current_stock"] = df["quantity"]

# Demand (convert required_qty into usable signal)
df["forecast_demand"] = abs(df["required_qty"]) * 50 + 20

# Lead time (assume constant or random)
df["lead_time"] = 3

# Avg sales (approximation)
df["avg_sales"] = df["forecast_demand"] / 2

# Reorder point
df["reorder_point"] = df["avg_sales"] * df["lead_time"]

# -------------------------------
#  CREATE TARGET LABEL
# -------------------------------
# If stock < demand → HIGH RISK

df["risk"] = (df["current_stock"] < df["forecast_demand"]).astype(int)

# -------------------------------
# PREPARE DATA
# -------------------------------
features = [
    "current_stock",
    "forecast_demand",
    "lead_time",
    "avg_sales",
    "reorder_point"
]

X = df[features]
y = df["risk"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# -------------------------------
# 5. TRAIN XGBOOST MODEL
# -------------------------------
model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1
)

model.fit(X_train, y_train)

print("Model Training Completed")

# -------------------------------
# PREDICT RISK
# -------------------------------
df["predicted_risk"] = model.predict(X)

df["risk_label"] = df["predicted_risk"].apply(
    lambda x: "High" if x == 1 else "Low"
)

# -------------------------------
# 7. SHAP EXPLANATION
# -------------------------------
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

def explain_row(i):
    explanation = []
    
    for j, col in enumerate(features):
        val = shap_values[i][j]
        
        if val > 0:
            explanation.append(f"{col} increases risk")
        else:
            explanation.append(f"{col} reduces risk")
    
    return ", ".join(explanation)

# -------------------------------
# UPDATE MYSQL TABLE
# -------------------------------
cursor = conn.cursor()

for i in range(len(df)):
    
    product_id = int(df.loc[i, "id"])
    risk_label = df.loc[i, "risk_label"]
    explanation = explain_row(i)

    # Convert risk to status (1=High, 0=Low)
    status_val = 1 if risk_label == "High" else 0

    update_query = """
    UPDATE nb_product 
    SET status=%s, details=%s
    WHERE id=%s
    """

    cursor.execute(update_query, (status_val, explanation, product_id))

conn.commit()

print("Database Updated Successfully")

# -------------------------------
#  SHOW RESULTS
# -------------------------------
for i in range(len(df)):
    print("\n----------------------")
    print("Product:", df.loc[i, "product"])
    print("Stock:", df.loc[i, "current_stock"])
    print("Risk:", df.loc[i, "risk_label"])
    print("Explanation:", explain_row(i))

# Close connection
conn.close()
