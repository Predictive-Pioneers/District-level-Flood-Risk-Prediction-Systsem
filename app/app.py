import os
import json
import pandas as pd
import joblib
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from streamlit_folium import st_folium
from email.mime.text import MIMEText
import smtplib
from twilio.rest import Client
import google.generativeai as genai
from dotenv import load_dotenv
import gdown

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize Gemini model
chatbot_model = genai.GenerativeModel("gemini-2.5-flash")
response = chatbot_model.generate_content("Hello!")
print(response.text)


try:
    import google.generativeai as genai
except Exception:
    genai = None

# configure Gemini
if genai is not None:
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    # initialize a separate chatbot client variable (do NOT overwrite your ML 'model')
    try:
        chatbot_model = genai.GenerativeModel("gemini-pro")   # typical initialization
    except Exception:
        # fallback: some versions use no GenerativeModel class; keep chatbot_model = None
        chatbot_model = None
else:
    chatbot_model = None

for m in genai.list_models():
    if "generateContent" in m.supported_generation_methods:
        print("Supported:", m.name)
    else:
        print("Other:", m.name, m.supported_generation_methods)


def send_to_gemini(prompt, model_name="gemini-pro", max_output_tokens=300):
    """
    Send prompt to Gemini and return plain text response.
    Tries multiple call styles depending on installed SDK version.
    """
    if genai is None:
        raise RuntimeError("google.generativeai (genai) not installed or import failed.")

    # If we have a GenerativeModel instance with generate_content
    if 'chatbot_model' in globals() and chatbot_model is not None and hasattr(chatbot_model, "generate_content"):
        resp = chatbot_model.generate_content(prompt)
        # many responses have .text
        return getattr(resp, "text", resp)

    # If genai provides a top-level generate function
    if hasattr(genai, "generate"):
        # new-style: genai.generate(model=model_name, input=prompt)
        resp = genai.generate(model=model_name, input=prompt, max_output_tokens=max_output_tokens)
        # try to extract text
        if hasattr(resp, "candidates") and resp.candidates:
            return resp.candidates[0].content
        if isinstance(resp, dict) and "candidates" in resp and resp["candidates"]:
            return resp["candidates"][0].get("content", "")
        return str(resp)

    # If no supported API found
    raise RuntimeError("No compatible Gemini call found in installed google-generativeai version.")

# Initialize session_state to store all predictions
if 'predictions' not in st.session_state:
    st.session_state['predictions'] = pd.DataFrame()

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(
    page_title="Flood Early Warning System 🌊",
    page_icon="🌊",
    layout="wide"
)
st.title("Flood Early Warning System 🌊")
st.header("District-Level Flood Risk Prediction")

# ---------------------------
# Load trained model
# ---------------------------
MODEL_PATH = "D:\\banglore internship\\major_project\\Early Flood Prediction System\\data\\flood_model.pkl"

# Google Drive file ID
FILE_ID = "1HBe-_a-wAcymSXg8KDsfJmZVSIt3mBJf"
URL = f"https://drive.google.com/uc?id={FILE_ID}"

# Download if not already present
if not os.path.exists(MODEL_PATH):
    os.makedirs("data", exist_ok=True)
    gdown.download(URL, MODEL_PATH, quiet=False)

# Load the model
model = joblib.load(MODEL_PATH)

# ---------------------------
# User file
# ---------------------------
USER_FILE = "users.json"

# Load or initialize users
if os.path.exists(USER_FILE):
    with open(USER_FILE, "r") as f:
        users = json.load(f)
else:
    users = {
        "admin": {"password": "admin123", "role": "admin"}
    }

# ---------------------------
# Initialize session_state for login
# ---------------------------
if 'role' not in st.session_state:
    st.session_state.role = None
if 'username' not in st.session_state:
    st.session_state.username = None

# ---------------------------
# Registration / Login Sidebar
# ---------------------------
# ---------------------------
# Registration / Login / Logout Sidebar
# ---------------------------
st.sidebar.header("User Access")

# If user already logged in → show logout
if st.session_state.role is not None:
    st.sidebar.success(f"Logged in as {st.session_state.username} ({st.session_state.role})")

    if st.sidebar.button("Logout"):
        st.session_state.role = None
        st.session_state.username = None
        st.success("You have been logged out.")
        st.rerun()

else:
    # If not logged in → show Register/Login options
    mode = st.sidebar.radio("Mode", ["Login","Register"])

    if mode=="Register":
        st.sidebar.subheader("Register (Public Only)")
        new_user = st.sidebar.text_input("New Username")
        new_pass = st.sidebar.text_input("New Password", type="password")
        if st.sidebar.button("Register"):
            if new_user in users:
                st.sidebar.error("Username already exists!")
            else:
                users[new_user] = {"password": new_pass, "role": "public"}
                with open(USER_FILE, "w") as f:
                    json.dump(users, f)
                st.sidebar.success("Registration successful! Please login.")

    elif mode=="Login":
        st.sidebar.subheader("Login")
        username_input = st.sidebar.text_input("Username", key="login_user")
        password_input = st.sidebar.text_input("Password", type="password", key="login_pass")
        login_btn = st.sidebar.button("Login")
        
        if login_btn:
            if username_input in users and password_input == users[username_input]["password"]:
                st.session_state.role = users[username_input]["role"]
                st.session_state.username = username_input
                st.sidebar.success(f"Welcome {st.session_state.username} ({st.session_state.role})")
                st.rerun()
            else:
                st.sidebar.error("Incorrect username or password")

# ---------------------------
# Show main app only if logged in
# ---------------------------
if st.session_state.role is not None:

    # ---------------------------
    # Helper functions
    # ---------------------------
    def predict_risk(df):
        df_features = df[model.feature_names_in_]  # Only trained features
        return model.predict(df_features)

    def map_risk_label(value):
        """Convert numeric prediction to categorical label"""
        if value >= 0.7:
            return "High"
        elif value >= 0.4:
            return "Medium"
        else:
            return "Low"

    def send_sms(to_number, message):
        account_sid = "YOUR_TWILIO_SID"
        auth_token = "YOUR_TWILIO_AUTH_TOKEN"
        client = Client(account_sid, auth_token)
        client.messages.create(body=message, from_="+YOUR_TWILIO_NUMBER", to=to_number)

    def send_email(to_email, subject, message):
        msg = MIMEText(message)
        msg['Subject'] = subject
        msg['From'] = "your_email@gmail.com"
        msg['To'] = to_email
        server = smtplib.SMTP_SSL("smtp.gmail.com", 465)
        server.login("your_email@gmail.com","your_password")
        server.sendmail("your_email@gmail.com", to_email, msg.as_string())
        server.quit()

    # Risk colors and icons
    risk_colors = {"Low":"green","Medium":"orange","High":"red"}
    risk_icons = {
        "High": '<i class="fa-solid fa-triangle-exclamation" style="color:red;"></i>',
        "Medium": '<i class="fa-solid fa-exclamation" style="color:orange;"></i>',
        "Low": '<i class="fa-solid fa-check-circle" style="color:green;"></i>'
    }

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Single District","Batch CSV","Visualization Report","Help Chatbot"])

    # ---------------------------
    # Tab 1: Single District
    # ---------------------------
    with tab1:
        st.subheader("Single District Prediction")
        with st.form("single_form"):
            features = {}
            for col in model.feature_names_in_:
                st.markdown(f"<span style='color:{risk_colors['Medium']}; font-weight:bold'>{col}</span>", unsafe_allow_html=True)
                features[col] = st.slider(col, 0.0, 1.0, 0.5)
            district_name = st.text_input("District Name")
            lat = st.number_input("Latitude", value=20.5937)
            lon = st.number_input("Longitude", value=78.9629)
            submit_btn = st.form_submit_button("Predict")

        if submit_btn:
            df_input = pd.DataFrame(features, index=[0])
            # Extra info for map/alerts
            df_input['District'] = district_name
            df_input['Latitude'] = lat
            df_input['Longitude'] = lon

            # ---------------------------
            # Predict and map to label
            # ---------------------------
            prediction_val = predict_risk(df_input)[0]  # numeric float
            prediction_label = map_risk_label(prediction_val)  # string label

            # Save prediction in session_state
            df_input_copy = df_input.copy()
            df_input_copy['PredictedRisk'] = prediction_label

            # Append to session_state
            st.session_state['predictions'] = pd.concat(
                [st.session_state['predictions'], df_input_copy],
                ignore_index=True
            )

            st.subheader("Predicted Flood Risk:")
            st.markdown(
                f"{risk_icons[prediction_label]} "
                f"<span style='color:{risk_colors[prediction_label]}; font-weight:bold'>{prediction_label}</span>",
                unsafe_allow_html=True
            )

            # Feature importance chart
            importances = model.feature_importances_
            feat_imp = pd.Series(importances, index=model.feature_names_in_).sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(10,5))
            sns.barplot(x=feat_imp.values, y=feat_imp.index, palette="viridis", ax=ax)
            st.pyplot(fig)

            # Admin alerts
            if st.session_state.role=="admin" and prediction_label=="High":
                send_sms("+919xxxxxxxxx", f"High flood risk in {district_name}")
                send_email("receiver@example.com","Flood Alert",f"High flood risk in {district_name}")
                st.success("Alerts sent!")

    # ---------------------------
    # Tab 2: Batch CSV Prediction
    # ---------------------------
    with tab2:
        st.subheader("Batch CSV Prediction")
        if st.session_state.role=="public":
            st.info("Public users cannot upload CSV")
        else:
            uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
            if uploaded_file:
                df_input = pd.read_csv(uploaded_file)
                if set(model.feature_names_in_) <= set(df_input.columns):
                    df_input['PredictedRisk'] = pd.Series(predict_risk(df_input)).apply(map_risk_label)
                    st.dataframe(df_input)

                    # Risk distribution chart
                    st.subheader("Flood Risk Distribution")
                    risk_counts = df_input['PredictedRisk'].value_counts()
                    fig2, ax2 = plt.subplots()
                    sns.barplot(x=risk_counts.index, y=risk_counts.values, palette=["green","orange","red"], ax=ax2)
                    st.pyplot(fig2)

                    # Alerts for high-risk districts
                    high_risk = df_input[df_input['PredictedRisk']=="High"]
                    for idx,row in high_risk.iterrows():
                        send_sms("+919xxxxxxxxx", f"High flood risk in {row['District']}")
                        send_email("receiver@example.com","Flood Alert",f"High flood risk in {row['District']}")
                    st.success("Alerts sent for high-risk districts!")
                else:
                    st.error(f"CSV must include columns: {list(model.feature_names_in_)}")

    # ---------------------------
    # Tab 3: Visualization Report
    # ---------------------------
    with tab3:
        st.subheader("Flood Prediction System Report & Visualizations")

        if st.session_state['predictions'].empty:
            st.info("No predictions yet. Please make a prediction in Tab 1 first.")
        else:
            df_report = st.session_state['predictions']

            # Show all predictions
            st.dataframe(df_report)

            # Risk distribution chart
            st.subheader("Flood Risk Distribution")
            risk_counts = df_report['PredictedRisk'].value_counts()
            fig, ax = plt.subplots()
            sns.barplot(x=risk_counts.index, y=risk_counts.values, palette=["green","orange","red"], ax=ax)
            st.pyplot(fig)

            # High-risk districts table
            st.subheader("High-Risk Districts")
            high_risk = df_report[df_report['PredictedRisk'] == "High"]
            if not high_risk.empty:
                st.table(high_risk[['District','PredictedRisk']])
            else:
                st.write("No high-risk districts predicted yet.")

            # Download report
            csv = df_report.to_csv(index=False).encode('utf-8')
            st.download_button("Download Report as CSV", data=csv, file_name="flood_predictions_report.csv", mime="text/csv")


    
    # ---------------------------
    # Tab 4: Help Chatbot
    # ---------------------------

    with tab4:
        st.subheader("Flood Prediction Chatbot 🤖")

        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []

        user_input = st.text_input("Ask a question about floods or using the system:")

        if st.button("Send"):
            if user_input.strip() != "":
                st.session_state.chat_history.append({"role": "user", "content": user_input})

                # Call Gemini API
                try:
                    chatbot_model = genai.GenerativeModel("gemini-2.5-flash")
                    response = chatbot_model.generate_content(user_input)
                    answer = response.text
                except Exception as e:
                    answer = f"Chatbot error: {e}"

                st.empty()  # Optional to clear previous outputs
                st.markdown(f"**You:** {user_input}")
                st.markdown(f"**Bot:** {answer}")
else:
    st.info("Please login or register from the sidebar to access the Flood Prediction System.")
