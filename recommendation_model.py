import pandas as pd

def run_recommendation(user_id, path="data/sample_data.csv"):

    df = pd.read_csv(path)

    # Get user's purchase history
    user_data = df[df["user_id"] == user_id]

    # Recommend popular products not yet purchased
    popular = df.groupby("product")["purchase_count"].sum().sort_values(ascending=False)

    user_products = set(user_data["product"])

    recommendations = [
        p for p in popular.index if p not in user_products
    ][:5]

    return recommendations
