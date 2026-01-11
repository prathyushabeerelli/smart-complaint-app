import smtplib
from email.mime.text import MIMEText
import streamlit as st
import pandas as pd
import pickle
import nltk
import re
import os
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from textblob import TextBlob
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

st.set_page_config(page_title="Smart Complaint System", layout="wide")
if "theme" not in st.session_state:
    st.session_state.theme = "light"

if st.sidebar.button("🌙 Toggle Dark Mode"):
    if st.session_state.theme == "light":
        st.session_state.theme = "dark"
    else:
        st.session_state.theme = "light"
if st.session_state.theme == "dark":
    st.markdown("""
<style>
/* Center the content */
.block-container {
    padding: 1rem 1rem 1rem 1rem;
}

/* Big buttons */
.stButton>button {
    width: 100%;
    border-radius: 10px;
    height: 3em;
    font-size: 18px;
}

/* Input fields */
.stTextInput>div>div>input, .stTextArea textarea {
    border-radius: 10px;
}

/* Cards */
.card {
    background-color: #f0f2f6;
    padding: 20px;
    border-radius: 15px;
    margin-bottom: 15px;
}
</style>
""", unsafe_allow_html=True)
nltk.download('stopwords')

# Load ML model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

stemmer = PorterStemmer()

def clean_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    words = text.split()
    words = [stemmer.stem(w) for w in words if w not in stopwords.words('english')]
    return " ".join(words)

def get_urgency(text):
    text = text.lower()
    critical_words = ["no water", "not working", "emergency", "broken", "no electricity", "too high"]
    for word in critical_words:
        if word in text:
            return "High"
    polarity = TextBlob(text).sentiment.polarity
    if polarity < -0.4:
        return "High"
    elif polarity < 0:
        return "Medium"
    else:
        return "Low"

# --------- LOGIN ----------
if "login" not in st.session_state:
    st.session_state.login = False
    st.session_state.role = ""

if not st.session_state.login:
    st.title("🔐 Login / Register")

    option = st.radio("Choose", ["Login", "Register"])

    if option == "Login":
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login", key="login_btn"):
            users = pd.read_csv("users.csv")

            user = users[(users["username"] == username) & (users["password"] == password)]

            if len(user) > 0:
                st.session_state.login = True
                st.session_state.role = user.iloc[0]["role"]
                st.session_state.username = username
                st.rerun()
            else:
                st.error("Invalid username or password")

    else:
        new_user = st.text_input("Create Username")
        new_pass = st.text_input("Create Password", type="password")

        if st.button("Register", key="register_btn"):
            users = pd.read_csv("users.csv")

            if new_user in users["username"].values:
                st.error("Username already exists")
            else:
                new_row = {
                    "username": new_user,
                    "password": new_pass,
                    "role": "student"
                }
                users = pd.concat([users, pd.DataFrame([new_row])], ignore_index=True)
                users.to_csv("users.csv", index=False)
                st.success("Account created. Now login.")

    if st.button("Login"):
        users = pd.read_csv("users.csv")

        user = users[(users["username"] == username) & (users["password"] == password)]

        if len(user) > 0:
            st.session_state.login = True
            st.session_state.role = user.iloc[0]["role"]
            st.rerun()
        else:
            st.error("Invalid username or password")

else:
    st.sidebar.markdown("### Navigation")
    if st.session_state.role == "admin":
        menu = st.sidebar.selectbox("Menu", ["Admin Dashboard"])
    else:
        menu = st.sidebar.selectbox("Menu", ["Student Portal"])

    if st.sidebar.button("Logout"):
        st.session_state.login = False
        st.session_state.role = ""
        st.rerun()

    # --------- STUDENT ----------
    if menu == "Student Portal":
        st.markdown("## 🧑‍🎓 Student Complaint Portal")

        complaint = st.text_area("Enter your complaint:")

        if st.button("Submit"):
            cleaned = clean_text(complaint)
            vector = vectorizer.transform([cleaned])
            category = model.predict(vector)[0]
            urgency = get_urgency(complaint)

            file = "complaints_log.csv"

            if os.path.exists(file):
                df = pd.read_csv(file)
                new_id = len(df) + 1
            else:
                df = pd.DataFrame(columns=["ID","Complaint","Category","Urgency","Status"])
                new_id = 1
                
                new_row = {
                    "ID": new_id,
                    "Username": st.session_state.username,
                    "Complaint": complaint,
                    "Category": category,
                    "Urgency": urgency,
                    "Status": "Pending"
                    }


            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            df.to_csv(file, index=False)

            st.success("Complaint Submitted")
            if urgency == "High":
                send_email(complaint, category, urgency)
                st.info("📧 Admin notified by email")

            st.write("📌 Category:", category)
            st.write("⚠️ Urgency:", urgency)

    # --------- ADMIN ----------
    if menu == "Admin Dashboard" and st.session_state.role == "admin":
        st.markdown("## 🧑‍💼 Admin Dashboard")

    if os.path.exists("complaints_log.csv"):
        df = pd.read_csv("complaints_log.csv")
    else:
        df = pd.DataFrame(columns=["ID","Complaint","Category","Urgency","Status"])

    st.subheader("📋 All Complaints")

    for i in range(len(df)):
        st.write("----")
        st.write("🆔 ID:", df.loc[i,"ID"])
        st.write("📄 Complaint:", df.loc[i,"Complaint"])
        st.write("📌 Category:", df.loc[i,"Category"])
        st.write("⚠️ Urgency:", df.loc[i,"Urgency"])

        status_list = ["Pending","In Progress","Solved"]
        current_status = df.loc[i,"Status"]

        new_status = st.selectbox(
            "Status",
            status_list,
            index=status_list.index(current_status),
            key=f"status_{i}"
        )

        if new_status != current_status:
            df.loc[i,"Status"] = new_status
            df.to_csv("complaints_log.csv", index=False)
            st.subheader("📄 My Complaints")

df = pd.read_csv("complaints_log.csv")
my_data = df[df["Username"] == st.session_state.username]
st.dataframe(my_data, use_container_width=True)
st.success("Status updated")

st.subheader("📊 Complaint Analytics")
st.bar_chart(df["Category"].value_counts())

st.subheader("🚨 High Urgency Complaints")
st.dataframe(df[df["Urgency"]=="High"], use_container_width=True)
