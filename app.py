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

# ---------------- EMAIL -----------------
def send_email(complaint, category, urgency):
    sender = st.secrets["EMAIL"]
    password = st.secrets["PASSWORD"]
    receiver = sender

    msg = MIMEText(f"""
New High Urgency Complaint Received

Complaint: {complaint}
Category: {category}
Urgency: {urgency}
""")

    msg["Subject"] = "🚨 High Urgency Student Complaint"
    msg["From"] = sender
    msg["To"] = receiver

    server = smtplib.SMTP_SSL("smtp.gmail.com", 465)
    server.login(sender, password)
    server.send_message(msg)
    server.quit()

# ---------------- NLP -----------------
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
stemmer = PorterStemmer()

def clean_text(text):
    text = re.sub("[^a-zA-Z]", " ", text)
    text = text.lower()
    words = text.split()
    words = [stemmer.stem(w) for w in words if w not in stopwords.words("english")]
    return " ".join(words)

def get_urgency(text):
    text = text.lower()
    critical = ["no water","no electricity","broken","emergency","not working","too high"]
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

# ---------------- UI -----------------
st.set_page_config(page_title="Smart Complaint System", layout="wide")

if "login" not in st.session_state:
    st.session_state.login = False
    st.session_state.role = ""
    st.session_state.username = ""

# ---------------- USERS -----------------
users_file = "users.csv"
if not os.path.exists(users_file):
    pd.DataFrame(columns=["username","password","role"]).to_csv(users_file, index=False)

users = pd.read_csv(users_file)

# ---------------- COMPLAINT FILE -----------------
complaint_file = "complaints_log.csv"
if not os.path.exists(complaint_file):
    pd.DataFrame(columns=["ID","Username","Complaint","Category","Urgency","Status"]).to_csv(complaint_file, index=False)

# ---------------- LOGIN -----------------
if not st.session_state.login:
    st.title("🔐 Smart Complaint Login")

    option = st.radio("Choose", ["Login", "Register"])

    if option == "Login":
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")

        if st.button("Login"):
            user = users[(users["username"]==u) & (users["password"]==p)]
            if len(user) > 0:
                st.session_state.login = True
                st.session_state.role = user.iloc[0]["role"].strip().lower()
                st.session_state.username = u
                st.rerun()
            else:
                st.error("Invalid credentials")

    else:
        u = st.text_input("Create Username")
        p = st.text_input("Create Password", type="password")

        if st.button("Register"):
            if u in users["username"].values:
                st.error("Username exists")
            else:
                users = pd.concat([users, pd.DataFrame([{
                    "username":u,"password":p,"role":"student"
                }])], ignore_index=True)
                users.to_csv(users_file,index=False)
                st.success("Account created. Now login.")

# ---------------- DASHBOARD -----------------
else:
    st.sidebar.success(f"Logged in as {st.session_state.username}")
    if st.sidebar.button("Logout"):
        st.session_state.login=False
        st.session_state.role=""
        st.session_state.username=""
        st.rerun()

    df = pd.read_csv(complaint_file)

    # ---------- STUDENT ----------
    if st.session_state.role == "student":
        st.header("🧑‍🎓 Student Complaint Portal")

        complaint = st.text_area("Enter your complaint")

        if st.button("Submit Complaint"):
            cleaned = clean_text(complaint)
            category = model.predict(vectorizer.transform([cleaned]))[0]
            urgency = get_urgency(complaint)

            new_id = len(df) + 1

            df = pd.concat([df, pd.DataFrame([{
                "ID": new_id,
                "Username": st.session_state.username,
                "Complaint": complaint,
                "Category": category,
                "Urgency": urgency,
                "Status": "Pending"
            }])], ignore_index=True)

            df.to_csv(complaint_file, index=False)

            st.success("Complaint Submitted")
            st.write("📌 Category:", category)
            st.write("⚠️ Urgency:", urgency)

            if urgency == "High":
                send_email(complaint, category, urgency)
                st.info("📧 Admin notified")

        st.subheader("📄 My Complaints")
        st.dataframe(df[df["Username"] == st.session_state.username], use_container_width=True)

    # ---------- ADMIN ----------
    else:
        st.header("🧑‍💼 Admin Dashboard")

        for i in range(len(df)):
            st.write("----")
            st.write("User:", df.loc[i,"Username"])
            st.write("Complaint:", df.loc[i,"Complaint"])
            st.write("Category:", df.loc[i,"Category"])
            st.write("Urgency:", df.loc[i,"Urgency"])

            status = st.selectbox(
                "Status",
                ["Pending","In Progress","Solved"],
                index=["Pending","In Progress","Solved"].index(df.loc[i,"Status"]),
                key=i
            )

            df.loc[i,"Status"] = status

        df.to_csv(complaint_file, index=False)

        st.subheader("📊 Analytics")
        st.bar_chart(df["Category"].value_counts())

        st.subheader("🚨 High Urgency Complaints")
        st.dataframe(df[df["Urgency"]=="High"], use_container_width=True)
