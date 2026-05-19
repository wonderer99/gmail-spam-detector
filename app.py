import streamlit as st
import pickle
import json
import tempfile
import os
import requests as http_requests
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

# Load model
with open('spam_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)

def get_config():
    c = dict(st.secrets["google_credentials"])
    return c["client_id"], c["client_secret"], st.secrets["redirect_uri"]

def get_auth_url(client_id, redirect_uri):
    import urllib.parse
    params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": " ".join(SCOPES),
        "access_type": "offline",
        "prompt": "consent"
    }
    return "https://accounts.google.com/o/oauth2/auth?" + urllib.parse.urlencode(params)

def exchange_code(code, client_id, client_secret, redirect_uri):
    resp = http_requests.post("https://oauth2.googleapis.com/token", data={
        "code": code,
        "client_id": client_id,
        "client_secret": client_secret,
        "redirect_uri": redirect_uri,
        "grant_type": "authorization_code"
    })
    return resp.json()

def get_service():
    if "creds" in st.session_state:
        creds = st.session_state["creds"]
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
        return build('gmail', 'v1', credentials=creds)

    if "token" in st.secrets:
        token_data = dict(st.secrets["token"])
        creds = Credentials(
            token=token_data.get("token"),
            refresh_token=token_data.get("refresh_token"),
            token_uri=token_data.get("token_uri"),
            client_id=token_data.get("client_id"),
            client_secret=token_data.get("client_secret"),
            scopes=SCOPES
        )
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
        st.session_state["creds"] = creds
        return build('gmail', 'v1', credentials=creds)

    return None


# --- UI ---
st.title("📧 Gmail Spam Detector")
st.write("Scan your Gmail inbox for spam using AI!")

client_id, client_secret, redirect_uri = get_config()

# Check for auth code in URL
query_params = st.query_params
auth_code = query_params.get("code", None)

service = get_service()

if service is None and auth_code:
    try:
        token = exchange_code(auth_code, client_id, client_secret, redirect_uri)
        if "error" in token:
            st.error(f"Authorization failed: {token}")
            st.stop()
        creds = Credentials(
            token=token["access_token"],
            refresh_token=token.get("refresh_token"),
            token_uri="https://oauth2.googleapis.com/token",
            client_id=client_id,
            client_secret=client_secret,
            scopes=SCOPES
        )
        st.session_state["creds"] = creds
        st.query_params.clear()
        st.success("✅ Authorized successfully!")
        st.info("Add these to Streamlit Secrets to skip this next time:")
        st.code(f"""[token]
token = "{creds.token}"
refresh_token = "{creds.refresh_token}"
token_uri = "https://oauth2.googleapis.com/token"
client_id = "{creds.client_id}"
client_secret = "{creds.client_secret}"
""")
        st.rerun()
    except Exception as e:
        st.error(f"Authorization failed: {e}")
        st.stop()

if service is None:
    auth_url = get_auth_url(client_id, redirect_uri)
    st.warning("#### Gmail Authorization Required")
    st.write("Click below to authorize access to your Gmail:")
    st.markdown(f"### [👉 Click here to authorize Gmail Access]({auth_url})")
    st.write("You will be redirected back automatically after authorizing.")
    st.stop()

# Authorized — show scanner
num_emails = st.slider("How many emails to scan?", 10, 100, 20)

if st.button("🔍 Scan My Emails"):
    with st.spinner("Scanning emails..."):
        results = service.users().messages().list(userId='me', maxResults=num_emails).execute()
        messages = results.get('messages', [])

    spam_count = 0
    ham_count = 0

    for msg in messages:
        txt = service.users().messages().get(userId='me', id=msg['id']).execute()
        headers = txt['payload']['headers']
        subject = next((h['value'] for h in headers if h['name'] == 'Subject'), 'No Subject')
        snippet = txt.get('snippet', '')

        result = model.predict(tfidf.transform([snippet]))

        if result[0] == 1:
            st.error(f"🚨 SPAM | {subject}")
            spam_count += 1
        else:
            st.success(f"✅ HAM | {subject}")
            ham_count += 1

    st.write("---")
    st.metric("Total Spam", spam_count)
    st.metric("Total Ham", ham_count)