import streamlit as st
import pickle
import json
import tempfile
import os
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build

SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

# Load model
with open('spam_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)

def get_redirect_uri():
    # Get the current app URL from Streamlit
    # On Streamlit Cloud this will be the public URL
    return st.secrets.get("redirect_uri", "http://localhost:8501")

def get_client_config():
    client_config = dict(st.secrets["google_credentials"])
    redirect_uri = get_redirect_uri()
    return {
        "web": {
            "client_id": client_config["client_id"],
            "client_secret": client_config["client_secret"],
            "redirect_uris": [redirect_uri],
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token"
        }
    }

def make_flow():
    config = get_client_config()
    redirect_uri = get_redirect_uri()
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config, f)
        temp_path = f.name
    flow = Flow.from_client_secrets_file(
        temp_path,
        scopes=SCOPES,
        redirect_uri=redirect_uri
    )
    os.unlink(temp_path)
    return flow

def get_service():
    # Already authorized in this session
    if "creds" in st.session_state:
        creds = st.session_state["creds"]
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
        return build('gmail', 'v1', credentials=creds)

    # Token stored in secrets (permanent fix)
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

# Check if we got a code back in the URL query params
query_params = st.query_params
auth_code = query_params.get("code", None)

service = get_service()

if service is None and auth_code:
    # Exchange code for token
    try:
        flow = make_flow()
        flow.fetch_token(code=auth_code)
        creds = flow.credentials
        st.session_state["creds"] = creds
        # Clear the code from URL
        st.query_params.clear()
        st.success("✅ Authorized successfully!")
        st.info("To avoid re-authorizing next time, add these to Streamlit Secrets:")
        st.code(f"""[token]
token = "{creds.token}"
refresh_token = "{creds.refresh_token}"
token_uri = "{creds.token_uri}"
client_id = "{creds.client_id}"
client_secret = "{creds.client_secret}"
""")
        service = build('gmail', 'v1', credentials=creds)
        st.rerun()
    except Exception as e:
        st.error(f"Authorization failed: {e}")
        st.stop()

if service is None:
    flow = make_flow()
    auth_url, _ = flow.authorization_url(prompt='consent', access_type='offline')
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