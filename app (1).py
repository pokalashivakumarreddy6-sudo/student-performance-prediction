import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

st.set_page_config(page_title="Student Performance Prediction", page_icon="🎓")
st.title("🎓 Student Performance Prediction")
st.caption("Python · Pandas · Scikit-learn · Streamlit. Trained on a synthetic dataset of 500 students.")

@st.cache_data
def load_data():
    return pd.read_csv("students.csv")

@st.cache_resource
def train(df):
    X, y = df.drop(columns="final_score"), df["final_score"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    lin = LinearRegression().fit(Xtr, ytr)
    rf = RandomForestRegressor(n_estimators=200, random_state=42).fit(Xtr, ytr)
    scores = {}
    for name, m in [("Linear Regression", lin), ("Random Forest", rf)]:
        p = m.predict(Xte)
        scores[name] = (r2_score(yte, p), mean_absolute_error(yte, p))
    imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values()
    return lin, rf, scores, imp

df = load_data()
lin, rf, scores, imp = train(df)

st.subheader("Enter student details")
c1, c2 = st.columns(2)
study = c1.slider("Study hours per day", 0.0, 10.0, 5.0, 0.5)
attend = c1.slider("Attendance %", 50, 100, 80)
prev = c2.slider("Previous score", 35, 98, 65)
sleep = c2.slider("Sleep hours", 4.0, 9.0, 7.0, 0.5)
model_name = st.radio("Model", ["Linear Regression", "Random Forest"], horizontal=True)

x = pd.DataFrame([[study, attend, prev, sleep]], columns=["study_hours", "attendance", "previous_score", "sleep_hours"])
pred = float((lin if model_name == "Linear Regression" else rf).predict(x)[0])
pred = max(0.0, min(100.0, pred))
st.metric("Predicted final score", f"{pred:.1f} / 100")
st.progress(int(pred))
st.write("Result: " + ("Excellent" if pred >= 75 else "Good" if pred >= 60 else "Needs improvement" if pred >= 45 else "At risk"))

st.subheader("Model performance (test data)")
st.table(pd.DataFrame(scores, index=["R²", "MAE"]).T.round(3))
st.subheader("Feature importance (Random Forest)")
st.bar_chart(imp)
with st.expander("View dataset"):
    st.dataframe(df.head(50))
