"""Student Performance Prediction
Predicts a student's final score from study habits using Pandas + Scikit-learn.
NOTE: the dataset is synthetic (generated below) so the project runs anywhere.
Replace students.csv with a real dataset (e.g. from Kaggle) to extend it."""
import numpy as np, pandas as pd, json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

rng = np.random.default_rng(42)
n = 500
df = pd.DataFrame({
    "study_hours": rng.uniform(0, 10, n).round(1),
    "attendance": rng.uniform(50, 100, n).round(0),
    "previous_score": rng.uniform(35, 98, n).round(0),
    "sleep_hours": rng.uniform(4, 9, n).round(1),
})
df["final_score"] = (0.45*df.previous_score + 3.2*df.study_hours + 0.25*df.attendance
                     + 1.0*df.sleep_hours + rng.normal(0, 4, n)).clip(0, 100).round(1)
df.to_csv("students.csv", index=False)                       # data cleaning/handling
df = df.dropna().drop_duplicates()

X, y = df.drop(columns="final_score"), df["final_score"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

lin = LinearRegression().fit(Xtr, ytr)
rf = RandomForestRegressor(n_estimators=200, random_state=42).fit(Xtr, ytr)
res = {}
for name, m in [("Linear Regression", lin), ("Random Forest", rf)]:
    p = m.predict(Xte); res[name] = {"R2": round(r2_score(yte, p), 3), "MAE": round(mean_absolute_error(yte, p), 2)}
    print(name, res[name])

imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values()
imp.plot.barh(color="#e07b00"); plt.title("Feature importance (Random Forest)"); plt.tight_layout()
plt.savefig("feature_importance.png", dpi=120)

json.dump({"intercept": lin.intercept_, "coef": dict(zip(X.columns, lin.coef_)), "metrics": res,
           "importance": imp.round(3).to_dict()}, open("model.json", "w"), indent=2)
