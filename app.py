
import os
import re
import hmac
import hashlib
import smtplib
import pickle
from datetime import datetime
from email.mime.text import MIMEText

import nltk
import pandas as pd
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from textblob import TextBlob


# =====================================================
# PAGE CONFIGURATION
# =====================================================
st.set_page_config(
    page_title="Smart Complaint Management",
    layout="wide",
    initial_sidebar_state="expanded"
)

USERS_FILE = "users.csv"
COMPLAINT_FILE = "complaints_log.csv"

USER_COLUMNS = ["username", "password", "role"]

COMPLAINT_COLUMNS = [
    "ID", "Username", "Complaint", "Category",
    "Urgency", "Status", "Created At"
]


# =====================================================
# PROFESSIONAL UI
# =====================================================
st.markdown("""
<style>
    .stApp {
        background: #f5f7fb;
        color: #172b4d;
    }

    [data-testid="stSidebar"] {
        background: #132744;
    }

    [data-testid="stSidebar"] * {
        color: #ffffff;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1450px;
    }

    h1, h2, h3 {
        color: #172b4d;
        font-weight: 650;
    }

    [data-testid="stMetric"] {
        background: #ffffff;
        padding: 20px;
        border: 1px solid #e1e7f0;
        border-radius: 12px;
    }

    .stButton > button {
        background: #2459a9;
        color: white;
        border: 1px solid #2459a9;
        border-radius: 8px;
        min-height: 42px;
        font-weight: 600;
    }

    .stButton > button:hover {
        background: #194681;
        color: white;
        border-color: #194681;
    }

    .stTextInput input,
    .stTextArea textarea,
    .stSelectbox div[data-baseweb="select"] {
        border-radius: 8px;
    }

    .page-subtitle {
        color: #64748b;
        font-size: 15px;
    }

    .section-card {
        background: white;
        border: 1px solid #e1e7f0;
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 16px;
    }
</style>
""", unsafe_allow_html=True)


# =====================================================
# FILE INITIALIZATION
# =====================================================
def initialize_files():
    if not os.path.exists(USERS_FILE):
        pd.DataFrame(columns=USER_COLUMNS).to_csv(
            USERS_FILE, index=False
        )

    if not os.path.exists(COMPLAINT_FILE):
        pd.DataFrame(columns=COMPLAINT_COLUMNS).to_csv(
            COMPLAINT_FILE, index=False
        )


def load_users():
    initialize_files()

    try:
        df = pd.read_csv(USERS_FILE, dtype=str).fillna("")
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        df = pd.DataFrame(columns=USER_COLUMNS)

    for column in USER_COLUMNS:
        if column not in df.columns:
            df[column] = ""

    return df[USER_COLUMNS]


def load_complaints():
    initialize_files()

    try:
        df = pd.read_csv(COMPLAINT_FILE)
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        df = pd.DataFrame(columns=COMPLAINT_COLUMNS)

    for column in COMPLAINT_COLUMNS:
        if column not in df.columns:
            df[column] = ""

    df = df[COMPLAINT_COLUMNS]
    df["ID"] = pd.to_numeric(df["ID"], errors="coerce")
    df["Urgency"] = df["Urgency"].fillna("Low")
    df["Status"] = df["Status"].fillna("Pending")

    return df


def save_complaints(df):
    df.to_csv(COMPLAINT_FILE, index=False)


initialize_files()


# =====================================================
# PASSWORD SECURITY
# =====================================================
def hash_password(password):
    salt = os.urandom(16).hex()

    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        bytes.fromhex(salt),
        300000
    ).hex()

    return f"pbkdf2_sha256${salt}${digest}"


def verify_password(password, stored):
    if stored.startswith("pbkdf2_sha256$"):
        try:
            _, salt, expected = stored.split("$", 2)

            actual = hashlib.pbkdf2_hmac(
                "sha256",
                password.encode("utf-8"),
                bytes.fromhex(salt),
                300000
            ).hex()

            return hmac.compare_digest(actual, expected)

        except (ValueError, TypeError):
            return False

    # Backward compatibility with existing plain-text accounts.
    # Successful login upgrades the stored password to a hash.
    return hmac.compare_digest(str(stored), password)


# =====================================================
# NLP RESOURCES AND MODEL
# =====================================================
@st.cache_resource
def load_nlp_resources():
    nltk.download("stopwords", quiet=True)
    words = set(stopwords.words("english"))
    return PorterStemmer(), words


@st.cache_resource
def load_model_files():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    return model, vectorizer


try:
    stemmer, stop_words = load_nlp_resources()
    model, vectorizer = load_model_files()

except Exception:
    st.error(
        "The NLP model could not be loaded. "
        "Check that model.pkl, vectorizer.pkl, and the required "
        "dependencies are available."
    )
    st.stop()


def clean_text(text):
    text = re.sub(r"[^a-zA-Z]", " ", str(text)).lower()

    words = [
        stemmer.stem(word)
        for word in text.split()
        if word not in stop_words
    ]

    return " ".join(words)


def get_urgency(text):
    text = str(text).lower()

    critical_phrases = [
        "no water",
        "no electricity",
        "emergency",
        "broken",
        "not working"
    ]

    if any(phrase in text for phrase in critical_phrases):
        return "High"

    polarity = TextBlob(text).sentiment.polarity

    if polarity < -0.4:
        return "High"
    elif polarity < 0:
        return "Medium"

    return "Low"


def predict_category(complaint):
    cleaned = clean_text(complaint)

    if not cleaned.strip():
        raise ValueError(
            "Please provide more descriptive complaint text."
        )

    features = vectorizer.transform([cleaned])
    return str(model.predict(features)[0])


# =====================================================
# EMAIL NOTIFICATIONS
# =====================================================
def send_email(complaint, category, urgency):
    try:
        sender = st.secrets["EMAIL"]
        password = st.secrets["PASSWORD"]

        msg = MIMEText(
            f"Complaint: {complaint}\n"
            f"Category: {category}\n"
            f"Urgency: {urgency}"
        )

        msg["Subject"] = "High Urgency Complaint"
        msg["From"] = sender
        msg["To"] = sender

        with smtplib.SMTP_SSL(
            "smtp.gmail.com", 465, timeout=15
        ) as server:
            server.login(sender, password)
            server.send_message(msg)

        return True

    except Exception:
        return False


# =====================================================
# SESSION STATE
# =====================================================
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.username = ""
    st.session_state.role = ""


# =====================================================
# LOGIN AND REGISTRATION
# =====================================================
if not st.session_state.authenticated:

    st.title("Smart Complaint Management System")
    st.markdown(
        '<p class="page-subtitle">'
        'A centralized platform for submitting, tracking, '
        'and managing institutional complaints.'
        '</p>',
        unsafe_allow_html=True
    )

    st.divider()

    left, right = st.columns([1.1, 1])

    with left:
        st.subheader("Complaint Management")
        st.write(
            "Submit concerns, monitor progress, and help "
            "administrators identify issues that require attention."
        )

        st.markdown("""
        <div class="section-card">
            <h3>Platform capabilities</h3>
            <p>Text-based complaint classification</p>
            <p>Urgency identification</p>
            <p>Complaint status tracking</p>
            <p>Administrative analytics</p>
        </div>
        """, unsafe_allow_html=True)

    with right:
        st.subheader("Account Access")

        login_tab, register_tab = st.tabs(["Sign in", "Student registration"])

        with login_tab:
            with st.form("login_form"):
                username = st.text_input("Username")
                password = st.text_input(
                    "Password", type="password"
                )
                role = st.selectbox(
                    "Account type", ["Student", "Administrator"]
                )

                login_clicked = st.form_submit_button(
                    "Sign in", use_container_width=True
                )

            if login_clicked:
                users = load_users()

                match = users[
                    (users["username"] == username.strip()) &
                    (users["role"].str.lower() == (
                        "admin" if role == "Administrator" else "student"
                    ))
                ]

                if not username.strip() or not password:
                    st.error("Enter your username and password.")

                elif match.empty:
                    st.error("Invalid username, password, or account type.")

                else:
                    row_index = match.index[0]
                    stored = users.loc[row_index, "password"]

                    if verify_password(password, stored):
                        # Upgrade legacy plain-text password storage.
                        if not stored.startswith("pbkdf2_sha256$"):
                            users.loc[row_index, "password"] = (
                                hash_password(password)
                            )
                            users.to_csv(USERS_FILE, index=False)

                        st.session_state.authenticated = True
                        st.session_state.username = username.strip()
                        st.session_state.role = (
                            "admin" if role == "Administrator"
                            else "student"
                        )
                        st.rerun()

                    else:
                        st.error(
                            "Invalid username, password, or account type."
                        )

        with register_tab:
            st.caption(
                "New accounts are registered as students. "
                "Administrator access must be provisioned separately."
            )

            with st.form("register_form"):
                new_username = st.text_input("Choose a username")
                new_password = st.text_input(
                    "Create a password", type="password"
                )
                confirm_password = st.text_input(
                    "Confirm password", type="password"
                )

                register_clicked = st.form_submit_button(
                    "Create student account",
                    use_container_width=True
                )

            if register_clicked:
                users = load_users()
                clean_username = new_username.strip()

                if len(clean_username) < 3:
                    st.error(
                        "Username must contain at least 3 characters."
                    )

                elif len(new_password) < 10:
                    st.error(
                        "Use a password with at least 10 characters."
                    )

                elif new_password != confirm_password:
                    st.error("The passwords do not match.")

                elif clean_username.lower() in (
                    users["username"].str.lower().values
                ):
                    st.error("That username already exists.")

                else:
                    new_user = pd.DataFrame([{
                        "username": clean_username,
                        "password": hash_password(new_password),
                        "role": "student"
                    }])

                    users = pd.concat(
                        [users, new_user], ignore_index=True
                    )

                    users.to_csv(USERS_FILE, index=False)
                    st.success(
                        "Account created. You can now sign in."
                    )


# =====================================================
# AUTHENTICATED APPLICATION
# =====================================================
else:

    df = load_complaints()

    with st.sidebar:
        st.title("Complaint Management")
        st.caption("Institutional Support Platform")
        st.divider()

        st.write("Signed in as")
        st.markdown(f"**{st.session_state.username}**")
        st.caption(st.session_state.role.title())

        if st.button("Sign out", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.username = ""
            st.session_state.role = ""
            st.rerun()

        st.divider()

        if st.session_state.role == "student":
            page = st.radio(
                "Navigation",
                ["Dashboard", "Submit Complaint", "My Complaints"]
            )
        else:
            page = st.radio(
                "Navigation",
                ["Dashboard", "Manage Complaints", "Analytics"]
            )

    st.title(page)
    st.markdown(
        '<p class="page-subtitle">'
        'Smart Complaint Management System'
        '</p>',
        unsafe_allow_html=True
    )

    # -------------------------------------------------
    # STUDENT DASHBOARD
    # -------------------------------------------------
    if st.session_state.role == "student":

        mine = df[
            df["Username"].astype(str) ==
            st.session_state.username
        ].copy()

        if page == "Dashboard":

            st.subheader(
                f"Welcome, {st.session_state.username}"
            )
            st.write(
                "Review your complaints and monitor their progress."
            )

            c1, c2, c3 = st.columns(3)

            c1.metric("Total Complaints", len(mine))
            c2.metric(
                "Pending",
                int((mine["Status"] == "Pending").sum())
            )
            c3.metric(
                "Resolved",
                int((mine["Status"] == "Solved").sum())
            )

            st.subheader("Recent Complaints")

            if mine.empty:
                st.info(
                    "You haven't submitted a complaint yet. "
                    "Use Submit Complaint to get started."
                )
            else:
                recent = mine.tail(5).iloc[::-1]
                st.dataframe(
                    recent[
                        ["ID", "Complaint", "Category",
                         "Urgency", "Status", "Created At"]
                    ],
                    use_container_width=True,
                    hide_index=True
                )

        elif page == "Submit Complaint":

            st.subheader("Submit a Complaint")

            with st.form("complaint_form"):
                complaint = st.text_area(
                    "Describe your concern",
                    placeholder=(
                        "Explain the issue, where it occurred, "
                        "and how it affects you."
                    ),
                    height=160,
                    max_chars=3000
                )

                submitted = st.form_submit_button(
                    "Submit Complaint",
                    use_container_width=True
                )

            if submitted:
                if len(complaint.strip()) < 10:
                    st.warning(
                        "Please provide at least 10 characters "
                        "so the complaint can be analyzed."
                    )

                else:
                    try:
                        category = predict_category(complaint)
                        urgency = get_urgency(complaint)

                        numeric_ids = pd.to_numeric(
                            df["ID"], errors="coerce"
                        ).dropna()

                        new_id = (
                            int(numeric_ids.max()) + 1
                            if not numeric_ids.empty else 1
                        )

                        new_row = {
                            "ID": new_id,
                            "Username": st.session_state.username,
                            "Complaint": complaint.strip(),
                            "Category": category,
                            "Urgency": urgency,
                            "Status": "Pending",
                            "Created At": datetime.now().strftime(
                                "%Y-%m-%d %H:%M:%S"
                            )
                        }

                        df = pd.concat(
                            [df, pd.DataFrame([new_row])],
                            ignore_index=True
                        )

                        save_complaints(df)

                        st.success(
                            f"Complaint #{new_id} submitted successfully."
                        )

                        st.write(f"**Predicted category:** {category}")
                        st.write(f"**Estimated urgency:** {urgency}")

                        if urgency == "High":
                            if send_email(
                                complaint, category, urgency
                            ):
                                st.info(
                                    "The administrator notification "
                                    "was sent."
                                )
                            else:
                                st.warning(
                                    "The complaint was saved, but the "
                                    "email notification could not be sent."
                                )

                    except Exception as exc:
                        st.error(
                            "The complaint could not be processed. "
                            "Check the model and vectorizer configuration."
                        )

        elif page == "My Complaints":

            st.subheader("Complaint History")

            search = st.text_input(
                "Search your complaints",
                placeholder="Search complaint text or category"
            )

            status_filter = st.selectbox(
                "Filter by status",
                ["All", "Pending", "In Progress", "Solved"]
            )

            if status_filter != "All":
                mine = mine[mine["Status"] == status_filter]

            if search.strip():
                mask = (
                    mine["Complaint"].astype(str).str.contains(
                        search, case=False, na=False
                    ) |
                    mine["Category"].astype(str).str.contains(
                        search, case=False, na=False
                    )
                )
                mine = mine[mask]

            st.dataframe(
                mine,
                use_container_width=True,
                hide_index=True
            )

    # -------------------------------------------------
    # ADMIN DASHBOARD
    # -------------------------------------------------
    else:

        if page == "Dashboard":

            st.subheader("Administrative Overview")

            total = len(df)
            pending = int((df["Status"] == "Pending").sum())
            progress = int(
                (df["Status"] == "In Progress").sum()
            )
            resolved = int((df["Status"] == "Solved").sum())
            high = int((df["Urgency"] == "High").sum())

            c1, c2, c3, c4 = st.columns(4)

            c1.metric("Total Complaints", total)
            c2.metric("Pending", pending)
            c3.metric("In Progress", progress)
            c4.metric("High Urgency", high)

            st.subheader("Latest Complaints")

            if df.empty:
                st.info("No complaints have been submitted.")
            else:
                st.dataframe(
                    df.tail(10).iloc[::-1],
                    use_container_width=True,
                    hide_index=True
                )

        elif page == "Manage Complaints":

            st.subheader("Complaint Register")

            status_filter = st.selectbox(
                "Status",
                ["All", "Pending", "In Progress", "Solved"]
            )

            urgency_filter = st.selectbox(
                "Urgency",
                ["All", "High", "Medium", "Low"]
            )

            search = st.text_input(
                "Search by username, complaint, or category"
            )

            filtered = df.copy()

            if status_filter != "All":
                filtered = filtered[
                    filtered["Status"] == status_filter
                ]

            if urgency_filter != "All":
                filtered = filtered[
                    filtered["Urgency"] == urgency_filter
                ]

            if search.strip():
                mask = (
                    filtered["Username"].astype(str).str.contains(
                        search, case=False, na=False
                    ) |
                    filtered["Complaint"].astype(str).str.contains(
                        search, case=False, na=False
                    ) |
                    filtered["Category"].astype(str).str.contains(
                        search, case=False, na=False
                    )
                )
                filtered = filtered[mask]

            st.dataframe(
                filtered,
                use_container_width=True,
                hide_index=True
            )

            st.divider()
            st.subheader("Update Complaint Status")

            if df.empty:
                st.info("There are no complaints to update.")
            else:
                valid_rows = df[df["ID"].notna()]

                if not valid_rows.empty:
                    options = {
                        f"#{int(row['ID'])} — "
                        f"{str(row['Complaint'])[:65]}": int(index)
                        for index, row in valid_rows.iterrows()
                    }

                    selected_label = st.selectbox(
                        "Select a complaint",
                        list(options.keys())
                    )

                    selected_index = options[selected_label]

                    with st.form("update_status_form"):
                        new_status = st.selectbox(
                            "New status",
                            ["Pending", "In Progress", "Solved"],
                            index=[
                                "Pending", "In Progress", "Solved"
                            ].index(
                                str(df.loc[selected_index, "Status"])
                                if str(df.loc[selected_index, "Status"])
                                in ["Pending", "In Progress", "Solved"]
                                else "Pending"
                            )
                        )

                        update_clicked = st.form_submit_button(
                            "Save Status"
                        )

                    if update_clicked:
                        df.loc[selected_index, "Status"] = new_status
                        save_complaints(df)
                        st.success("Complaint status updated.")
                        st.rerun()

        elif page == "Analytics":

            st.subheader("Complaint Analytics")

            c1, c2 = st.columns(2)

            with c1:
                st.markdown("#### Complaints by Category")
                if df.empty:
                    st.info("No category data available.")
                else:
                    st.bar_chart(
                        df["Category"].value_counts()
                    )

            with c2:
                st.markdown("#### Complaints by Urgency")
                if df.empty:
                    st.info("No urgency data available.")
                else:
                    st.bar_chart(
                        df["Urgency"].value_counts()
                    )

            st.markdown("#### Complaints by Status")
            if df.empty:
                st.info("No status data available.")
            else:
                st.bar_chart(df["Status"].value_counts())
