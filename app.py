import streamlit as st
import pandas as pd
import pickle
import nltk
import re
import os
import smtplib
from email.mime.text import MIMEText
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from textblob import TextBlob

nltk.download("stopwords")

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Smart Complaint System",
    layout="wide",
    page_icon="🎓"
)

# ================= STYLES =================
st.markdown("""
<style>

/* App background */
.stApp {
    background: linear-gradient(to right, #fdfbfb, #ebedee);
    font-family: 'Segoe UI', sans-serif;
}

/* Cards */
.card {
    padding: 25px;
    border-radius: 20px;
    margin-bottom: 25px;
    color: #2c2c2c;
    box-shadow: 0px 10px 25px rgba(0,0,0,0.08);
}

.student-card {
    background: linear-gradient(135deg, #84fab0, #8fd3f4);
}

.admin-card {
    background: linear-gradient(135deg, #a18cd1, #fbc2eb);
}

/* Headings */
.metric {
    font-size: 32px;
    font-weight: 700;
}

/* Labels */
.label {
    font-size: 16px;
    opacity: 0.9;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
    border-radius: 12px;
    height: 3em;
    font-size: 16px;
    border: none;
}

/* Inputs */
.stTextInput input, .stTextArea textarea {
    border-radius: 12px;
    border: 1px solid #ddd;
}

</style>
""", unsafe_allow_html=True)

# ================= FILE PATHS =================
USERS_FILE = "users.csv"
COMPLAINT_FILE = "complaints_log.csv"

# ================= FILE SAFETY =================
if not os.path.exists(USERS_FILE):
    pd.DataFrame(columns=["username", "password", "role"]).to_csv(USERS_FILE, index=False)

if not os.path.exists(COMPLAINT_FILE):
    pd.DataFrame(
        columns=["ID", "Username", "Complaint", "Category", "Urgency", "Status"]
    ).to_csv(COMPLAINT_FILE, index=False)

def load_complaints():
    try:
        return pd.read_csv(COMPLAINT_FILE)
    except:
        return pd.DataFrame(columns=["ID","Username","Complaint","Category","Urgency","Status"])

# ================= EMAIL =================
def send_email(complaint, category, urgency):
    sender = st.secrets["EMAIL"]
    password = st.secrets["PASSWORD"]

    msg = MIMEText(f"""
New High Urgency Complaint Received

Complaint: {complaint}
Category: {category}
Urgency: {urgency}
""")

    msg["Subject"] = "🚨 High Urgency Student Complaint"
    msg["From"] = sender
    msg["To"] = sender

    server = smtplib.SMTP_SSL("smtp.gmail.com", 465)
    server.login(sender, password)
    server.send_message(msg)
    server.quit()

# ================= NLP =================
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
stemmer = PorterStemmer()

def clean_text(text):
    text = re.sub("[^a-zA-Z]", " ", text).lower()
    words = [stemmer.stem(w) for w in text.split() if w not in stopwords.words("english")]
    return " ".join(words)

def get_urgency(text):
    text = text.lower()
    critical = ["no water", "no electricity", "broken", "emergency", "not working"]
    for c in critical:
        if c in text:
            return "High"
    polarity = TextBlob(text).sentiment.polarity
    if polarity < -0.4:
        return "High"
    elif polarity < 0:
        return "Medium"
    else:
        return "Low"

# ================= SESSION =================
if "login" not in st.session_state:
    st.session_state.login = False
    st.session_state.username = ""
    st.session_state.role = ""

users = pd.read_csv(USERS_FILE)

# ================= LOGIN PAGE =================
if not st.session_state.login:
    st.title("🎓 Smart Complaint System")

    role_choice = st.radio("Who are you?", ["Student", "Admin"])
    action = st.radio("Choose Action", ["Login", "Register"])

    if action == "Login":
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")

        if st.button("Login"):
            user = users[
                (users["username"] == u) &
                (users["password"] == p) &
                (users["role"].str.lower() == role_choice.lower())
            ]

            if not user.empty:
                st.session_state.login = True
                st.session_state.username = u
                st.session_state.role = role_choice.lower()
                st.rerun()
            else:
                st.error("Invalid credentials")

    else:
        u = st.text_input("Create Username")
        p = st.text_input("Create Password", type="password")
        r = st.selectbox("Role", ["student", "admin"])

        if st.button("Register"):
            if u in users["username"].values:
                st.error("Username already exists")
            else:
                users = pd.concat([users, pd.DataFrame([{
                    "username": u,
                    "password": p,
                    "role": r
                }])], ignore_index=True)
                users.to_csv(USERS_FILE, index=False)
                st.success("Registered successfully. Please login.")

# ================= DASHBOARD =================
else:
    st.sidebar.success(f"👤 {st.session_state.username}")
    if st.sidebar.button("Logout"):
        st.session_state.login = False
        st.rerun()

    df = load_complaints()

    # ---------- STUDENT ----------
    if st.session_state.role == "student":
        st.markdown("""
        <div class="card student-card">
            <div class="metric">🧑‍🎓 Student Dashboard</div>
            <div class="label">Submit & track your complaints</div>
        </div>
        """, unsafe_allow_html=True)

        complaint = st.text_area("Enter your complaint")

        if st.button("Submit Complaint"):
            if complaint.strip() == "":
                st.warning("Please enter a complaint")
            else:
                cleaned = clean_text(complaint)
                category = model.predict(vectorizer.transform([cleaned]))[0]
                urgency = get_urgency(complaint)

                new_id = len(df) + 1
                df.loc[len(df)] = [
                    new_id,
                    st.session_state.username,
                    complaint,
                    category,
                    urgency,
                    "Pending"
                ]

                df.to_csv(COMPLAINT_FILE, index=False)
                st.success("Complaint submitted successfully")

                if urgency == "High":
                    send_email(complaint, category, urgency)
                    st.info("📧 Admin notified")

        st.subheader("📄 My Complaints")
        st.dataframe(df[df["Username"] == st.session_state.username], use_container_width=True)

    # ---------- ADMIN ----------
    else:
        st.markdown("""
        <div class="card admin-card">
            <div class="metric">🧑‍💼 Admin Dashboard</div>
            <div class="label">Manage and resolve complaints</div>
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("Total", len(df))
        col2.metric("High Urgency", len(df[df["Urgency"] == "High"]))
        col3.metric("Pending", len(df[df["Status"] == "Pending"]))

        for i in df.index:
            st.write("----")
            st.write("👤 User:", df.loc[i,"Username"])
            st.write("📄 Complaint:", df.loc[i,"Complaint"])
            st.write("📌 Category:", df.loc[i,"Category"])
            st.write("⚠️ Urgency:", df.loc[i,"Urgency"])

            df.loc[i,"Status"] = st.selectbox(
                "Status",
                ["Pending", "In Progress", "Solved"],
                index=["Pending","In Progress","Solved"].index(df.loc[i,"Status"]),
                key=f"status_{i}"
            )

        df.to_csv(COMPLAINT_FILE, index=False)

        st.subheader("📊 Category Analytics")
        st.bar_chart(df["Category"].value_counts())

        st.subheader("🚨 High Urgency Complaints")
        st.dataframe(df[df["Urgency"] == "High"], use_container_width=True)
