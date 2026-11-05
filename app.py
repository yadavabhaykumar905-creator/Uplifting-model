import joblib
import pandas as pd
import sklearn
import streamlit as st

st.set_page_config(page_title="Uplift Predictor", layout="wide")


@st.cache_resource(show_spinner="Loading saved model...")
def load_bundle():
    return joblib.load("uplift_model.pkl")


def score(raw, bundle):
    X = pd.get_dummies(raw[bundle["features"]]).reindex(columns=bundle["columns"], fill_value=0)
    p1 = bundle["m1"].predict_proba(X)[:, 1]
    p0 = bundle["m0"].predict_proba(X)[:, 1]
    return p1, p0, p1 - p0


def to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")


st.title("Uplift Predictor: Who Responds BECAUSE of the Email?")
st.write(
    "A normal model predicts who will visit the website. This model predicts who visits "
    "**because of** the email, so you email only the customers the email actually changes."
)

try:
    bundle = load_bundle()
except Exception as e:
    st.error(f"Could not load uplift_model.pkl: {e}")
    st.stop()

saved_v = bundle.get("sklearn_version")
if saved_v != sklearn.__version__:
    st.warning(
        f"This model was saved with scikit-learn {saved_v}, but your computer has {sklearn.__version__}. "
        f"To fix it, stop the app and run:  pip install scikit-learn=={saved_v}  "
        f"then start the app again."
    )

tab1, tab2 = st.tabs(["Predict uplift", "About"])

# ------------------------------------------------------------ Predict
with tab1:
    st.subheader("A. Score one customer")
    a, b = st.columns(2)
    recency = a.slider("Months since last purchase", 1, 12, 6)
    history = a.number_input("Money spent in the past year", 0.0, 5000.0, 200.0, 10.0)
    segment = a.selectbox("Past spending bracket", bundle["segments"])
    channel = a.selectbox("Channel", bundle["channels"])
    zip_code = b.selectbox("Area type", bundle["zips"])
    mens = b.checkbox("Bought men's items before", value=True)
    womens = b.checkbox("Bought women's items before", value=False)
    newbie = b.checkbox("New customer", value=False)

    if st.button("Predict uplift"):
        one = pd.DataFrame([{"recency": recency, "history_segment": segment, "history": history,
                             "mens": int(mens), "womens": int(womens), "zip_code": zip_code,
                             "newbie": int(newbie), "channel": channel}])
        p1, p0, u = score(one, bundle)
        x1, x2, x3 = st.columns(3)
        x1.metric("Visit chance WITH email", f"{p1[0]*100:.1f}%")
        x2.metric("Visit chance WITHOUT email", f"{p0[0]*100:.1f}%")
        x3.metric("Uplift", f"{u[0]*100:+.1f} pts")
        if u[0] > 0.05:
            st.success("Good target: the email is likely to make a real difference.")
        elif u[0] > 0:
            st.warning("Small effect: the email helps only a little for this customer.")
        else:
            st.error("Do not email: no expected gain (possibly a negative effect).")
        st.caption("Individual predictions are noisy. Use them as a ranking signal, not a guarantee.")

    st.divider()
    st.subheader("B. Score many customers from a CSV")
    template = pd.DataFrame([{"recency": 6, "history_segment": bundle["segments"][0], "history": 200.0,
                              "mens": 1, "womens": 0, "zip_code": bundle["zips"][0], "newbie": 0,
                              "channel": bundle["channels"][0]}])[bundle["features"]]
    st.download_button("Download CSV template", to_csv_bytes(template), "template.csv", "text/csv")
    up = st.file_uploader("Upload a CSV with the same columns as the template", type="csv")
    if up is not None:
        raw = pd.read_csv(up)
        missing = [c for c in bundle["features"] if c not in raw.columns]
        if missing:
            st.error("Your file is missing these columns: " + ", ".join(missing))
        else:
            _, _, u = score(raw, bundle)
            out = raw.copy()
            out["uplift_score"] = u
            out["rank"] = out["uplift_score"].rank(ascending=False, method="first").astype(int)
            out = out.sort_values("rank")
            st.write(f"Scored {len(out):,} customers. Highest uplift first:")
            st.dataframe(out.head(50), hide_index=True)
            st.download_button("Download all scores", to_csv_bytes(out), "uplift_scores.csv", "text/csv")

# ------------------------------------------------------------ About
with tab2:
    st.markdown(
        f"""
**Data:** Hillstrom MineThatData Email Experiment (64,000 customers, emails sent at random).
**Method:** T-Learner with Gradient Boosting. One model learns what happens with the email, one learns
what happens without it. Uplift = the difference between the two predictions.
**Average effect of the email in the data:** {bundle['ate']*100:.2f} percentage points.
**Saved as:** `uplift_model.pkl`.

**Limitations**
- The model understands only the columns it was trained on (see the template).
- Uplift estimates are noisy because visit rates are low.
- The target is website visits, not purchases.
"""
    )
