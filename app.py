import os
import joblib
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)

st.set_page_config(page_title="FraudShield AI", page_icon="🛡️", layout="wide")

MODEL_PATH = "models/fraud_model.joblib"
DATA_PATH = "data/creditcard.csv"
FEATURES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

if not os.path.exists(MODEL_PATH):
    st.error("Model file not found: models/fraud_model.joblib")
    st.stop()

if not os.path.exists(DATA_PATH):
    st.error("Dataset not found: data/creditcard.csv")
    st.stop()

model = joblib.load(MODEL_PATH)
data = pd.read_csv(DATA_PATH)

missing = [x for x in FEATURES if x not in data.columns]
if missing:
    st.error(f"Missing features: {missing}")
    st.stop()

def get_prediction(transaction_id):
    row = data.iloc[int(transaction_id)]
    input_data = pd.DataFrame(
        [[row[x] for x in FEATURES]],
        columns=FEATURES
    )
    probability = float(model.predict_proba(input_data)[0][1])
    score = probability * 100

    if score >= 70:
        risk = "HIGH RISK"
        decision = "🚨 FRAUD ALERT"
    elif score >= 30:
        risk = "MEDIUM RISK"
        decision = "⚠️ REVIEW"
    else:
        risk = "LOW RISK"
        decision = "✅ APPROVE"

    return row, score, risk, decision

st.sidebar.title("🛡️ FraudShield AI")
st.sidebar.write("AI-Powered Fraud Detection")

page = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "🔍 Analyze Transaction", "⚡ Live Monitoring",
     "📊 Model Performance", "ℹ️ About Project"]
)

st.sidebar.markdown("---")
st.sidebar.write("Python • Machine Learning • Streamlit • Plotly")

if page == "🏠 Home":
    st.title("🛡️ FraudShield AI")
    st.subheader("AI-Powered Real-Time Credit Card Fraud Detection")
    st.write("Detect suspicious transactions using a trained Random Forest model.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Transactions", f"{len(data):,}")
    c2.metric("Fraud Cases", f"{int(data['Class'].sum()):,}")
    c3.metric("Features", len(FEATURES))
    c4.metric("Model", "Random Forest")

    st.divider()
    st.subheader("🔄 Detection Flow")
    st.write("Transaction → Preprocessing → Random Forest → Risk Score → Decision → Alert")

    c1, c2, c3 = st.columns(3)
    c1.success("🟢 LOW RISK\n\n0% – 29.99%")
    c2.warning("🟡 MEDIUM RISK\n\n30% – 69.99%")
    c3.error("🔴 HIGH RISK\n\n70% – 100%")

elif page == "🔍 Analyze Transaction":
    st.title("🔍 Analyze Transaction")
    st.write("Select a transaction and analyze its fraud probability.")

    transaction_id = st.number_input(
        "Transaction Number", 0, len(data) - 1, 4920, 1
    )

    if st.button("🔎 ANALYZE TRANSACTION", use_container_width=True):
        row, score, risk, decision = get_prediction(transaction_id)

        c1, c2, c3 = st.columns(3)
        c1.metric("Transaction ID", int(transaction_id))
        c2.metric("Amount", f"${float(row['Amount']):,.2f}")
        c3.metric("Fraud Probability", f"{score:.2f}%")

        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            title={"text": "Fraud Risk Score"},
            number={"suffix": "%"},
            gauge={"axis": {"range": [0, 100]}}
        ))
        gauge.update_layout(height=350)
        st.plotly_chart(gauge, use_container_width=True)

        if risk == "HIGH RISK":
            st.error(f"🚨 HIGH RISK — FRAUD ALERT — {score:.2f}%")
        elif risk == "MEDIUM RISK":
            st.warning(f"⚠️ MEDIUM RISK — REVIEW — {score:.2f}%")
        else:
            st.success(f"✅ LOW RISK — APPROVE — {score:.2f}%")

        st.dataframe(pd.DataFrame({
            "Property": ["Transaction ID", "Time", "Amount", "Actual Dataset Label", "AI Decision"],
            "Value": [
                int(transaction_id), row["Time"], row["Amount"],
                "FRAUD" if row["Class"] == 1 else "LEGITIMATE", decision
            ]
        }), use_container_width=True, hide_index=True)

elif page == "⚡ Live Monitoring":
    st.title("⚡ Live Transaction Monitoring")
    st.write("Simulated real-time monitoring using transactions from the dataset.")
    st.success("🟢 MONITORING SYSTEM ONLINE")

    if "live_history" not in st.session_state:
        st.session_state.live_history = []

    if "live_index" not in st.session_state:
        st.session_state.live_index = 0

    demo_transactions = [610, 541, 4920, 6108, 6329]

    b1, b2 = st.columns(2)
    process = b1.button("⚡ PROCESS NEXT TRANSACTION", use_container_width=True)
    reset = b2.button("🔄 RESET MONITORING", use_container_width=True)

    if reset:
        st.session_state.live_history = []
        st.session_state.live_index = 0
        st.rerun()

    if process:
        tx = demo_transactions[st.session_state.live_index]
        row, score, risk, decision = get_prediction(tx)

        st.session_state.live_history.append({
            "Transaction": tx,
            "Amount": round(float(row["Amount"]), 2),
            "Risk Score": round(score, 2),
            "Risk": risk,
            "Decision": decision
        })

        st.session_state.live_index = (
            st.session_state.live_index + 1
        ) % len(demo_transactions)

    history = pd.DataFrame(
        st.session_state.live_history,
        columns=["Transaction", "Amount", "Risk Score", "Risk", "Decision"]
    )

    # -----------------------------------------------------
    # Dashboard counters
    # -----------------------------------------------------

    alert_count = int(
        (history["Risk"] == "HIGH RISK").sum()
    ) if not history.empty else 0

    processed_count = len(history)

    highest_risk = (
        float(history["Risk Score"].max())
        if not history.empty else 0.0
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("🚨 Fraud Alerts", alert_count)

    with c2:
        st.metric("📊 Transactions Processed", processed_count)

    with c3:
        st.metric("🎯 Highest Risk", f"{highest_risk:.2f}%")

    with c4:
        st.metric("🟢 System Status", "ONLINE")

    st.divider()

    # -----------------------------------------------------
    # Latest transaction / empty state
    # -----------------------------------------------------

    if history.empty:
        st.info(
            "No transactions processed yet. "
            "Click **⚡ PROCESS NEXT TRANSACTION** to start the live demo."
        )

        current_score = 0.0
        latest = None

    else:
        latest = history.iloc[-1]
        current_score = float(latest["Risk Score"])

        st.subheader("🔎 Latest Transaction")

        c1, c2, c3 = st.columns(3)
        c1.metric("Transaction ID", int(latest["Transaction"]))
        c2.metric("Amount", f"${latest['Amount']:,.2f}")
        c3.metric("AI Risk", f"{current_score:.2f}%")

    # -----------------------------------------------------
    # Risk Gauge
    # -----------------------------------------------------

    st.subheader("🎯 Live Fraud Risk Gauge")

    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=current_score,
        title={"text": "Current Fraud Risk"},
        number={"suffix": "%"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"thickness": 0.35},
            "steps": [
                {"range": [0, 30], "color": "lightgreen"},
                {"range": [30, 70], "color": "gold"},
                {"range": [70, 100], "color": "lightcoral"}
            ],
            "threshold": {
                "line": {"width": 5},
                "thickness": 0.8,
                "value": current_score
            }
        }
    ))

    gauge.update_layout(height=400)
    st.plotly_chart(gauge, use_container_width=True)

    # -----------------------------------------------------
    # Current alert
    # -----------------------------------------------------

    if latest is not None:
        if latest["Risk"] == "HIGH RISK":
            st.error(
                f"🚨 FRAUD ALERT! Transaction {int(latest['Transaction'])} "
                f"— {latest['Risk Score']:.2f}% risk"
            )
        elif latest["Risk"] == "MEDIUM RISK":
            st.warning(
                f"⚠️ REVIEW REQUIRED! Transaction {int(latest['Transaction'])} "
                f"— {latest['Risk Score']:.2f}% risk"
            )
        else:
            st.success(
                f"✅ TRANSACTION APPROVED — {latest['Risk Score']:.2f}% risk"
            )

    st.divider()

    # -----------------------------------------------------
    # Live Risk Score Chart
    # -----------------------------------------------------

    st.subheader("📈 Live Risk Score Trend")

    risk_chart = go.Figure()

    if history.empty:
        risk_chart.add_annotation(
            text="No transaction data yet",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False
        )
        risk_chart.update_xaxes(range=[0, 1])
        risk_chart.update_yaxes(range=[0, 100])
    else:
        risk_chart.add_trace(go.Scatter(
            x=list(range(1, len(history) + 1)),
            y=history["Risk Score"],
            mode="lines+markers",
            name="Risk Score"
        ))

        risk_chart.add_hline(
            y=70,
            line_dash="dash",
            annotation_text="High Risk"
        )

        risk_chart.add_hline(
            y=30,
            line_dash="dash",
            annotation_text="Medium Risk"
        )

    risk_chart.update_layout(
        xaxis_title="Transaction Sequence",
        yaxis_title="Fraud Probability (%)",
        yaxis={"range": [0, 100]},
        height=400
    )

    st.plotly_chart(risk_chart, use_container_width=True)

    # -----------------------------------------------------
    # Live Amount Chart
    # -----------------------------------------------------

    st.subheader("💰 Live Transaction Amount Chart")

    amount_chart = go.Figure()

    if history.empty:
        amount_chart.add_annotation(
            text="No transaction data yet",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False
        )
    else:
        amount_chart.add_trace(go.Bar(
            x=[f"TX {int(x)}" for x in history["Transaction"]],
            y=history["Amount"],
            text=history["Amount"].round(2),
            textposition="auto",
            name="Transaction Amount"
        ))

    amount_chart.update_layout(
        xaxis_title="Transaction",
        yaxis_title="Amount ($)",
        height=400
    )

    st.plotly_chart(amount_chart, use_container_width=True)

    # -----------------------------------------------------
    # Risk Distribution
    # -----------------------------------------------------

    st.subheader("🥧 Risk Distribution")

    if history.empty:
        st.info("Risk distribution will appear after transactions are processed.")
    else:
        counts = history["Risk"].value_counts().reset_index()
        counts.columns = ["Risk", "Count"]

        pie = px.pie(
            counts,
            names="Risk",
            values="Count",
            hole=0.4
        )

        pie.update_layout(height=400)
        st.plotly_chart(pie, use_container_width=True)

    # -----------------------------------------------------
    # Monitoring History
    # -----------------------------------------------------

    st.subheader("📋 Monitoring History")

    if history.empty:
        st.write("No transactions processed yet.")
    else:
        st.dataframe(
            history,
            use_container_width=True,
            hide_index=True
        )

    st.info(
        "Prototype note: transactions are simulated from dataset records. "
        "A production system could receive transactions through an API or streaming platform."
    )

elif page == "📊 Model Performance":
    st.title("📊 Model Performance")

    X = data[FEATURES]
    y = data["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)
    roc_auc = roc_auc_score(y_test, probabilities)
    pr_auc = average_precision_score(y_test, probabilities)

    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", f"{accuracy * 100:.2f}%")
    c2.metric("Precision", f"{precision * 100:.2f}%")
    c3.metric("Recall", f"{recall * 100:.2f}%")

    c1, c2, c3 = st.columns(3)
    c1.metric("F1 Score", f"{f1 * 100:.2f}%")
    c2.metric("ROC-AUC", f"{roc_auc:.4f}")
    c3.metric("PR-AUC", f"{pr_auc:.4f}")

    st.subheader("🔲 Confusion Matrix")
    cm = confusion_matrix(y_test, predictions)
    st.dataframe(
        pd.DataFrame(
            cm,
            index=["Actual Legitimate", "Actual Fraud"],
            columns=["Predicted Legitimate", "Predicted Fraud"]
        ),
        use_container_width=True
    )

elif page == "ℹ️ About Project":
    st.title("ℹ️ About FraudShield AI")
    st.subheader("🛡️ Project Overview")
    st.write(
        "FraudShield AI is a machine-learning prototype designed to identify suspicious credit-card transactions."
    )
    st.subheader("🤖 Machine Learning Model")
    st.write("Random Forest Classifier trained on a credit-card fraud dataset.")
    st.subheader("💻 Technology Stack")
    st.write("Python, Pandas, NumPy, Scikit-learn, Joblib, Plotly and Streamlit.")
    st.subheader("🚀 Future Scope")
    st.write("API integration • Kafka streaming • Explainable AI • Model monitoring • Human review • Cloud deployment")
    st.divider()
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Transactions", f"{len(data):,}")
    c2.metric("Fraud Transactions", f"{int(data['Class'].sum()):,}")
    c3.metric("Features", len(FEATURES))
