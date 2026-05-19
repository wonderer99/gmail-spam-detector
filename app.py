import streamlit as st
import pickle
import os
import json
import tempfile
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build

SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

def get_flow():
    client_config = dict(st.secrets["google_credentials"])
    client_config_dict = {
        "installed": {
            "client_id": client_config["client_id"],
            "client_secret": client_config["client_secret"],
            "redirect_uris": ["urn:ietf:wg:oauth:2.0:oob"],
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token"
        }
    }
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(client_config_dict, f)
        temp_path = f.name

    flow = Flow.from_client_secrets_file(
        temp_path,
        scopes=SCOPES,
        redirect_uri="urn:ietf:wg:oauth:2.0:oob"
    )
    os.unlink(temp_path)
    return flow

def authenticate():
    # Already have token in session
    if "creds" in st.session_state:
        creds = st.session_state["creds"]
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
            st.session_state["creds"] = creds
        return build('gmail', 'v1', credentials=creds)

    # Load token from Streamlit secrets if it exists
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

    # Need to authorize
    flow = get_flow()
    auth_url, _ = flow.authorization_url(prompt='consent')

    st.warning("### Gmail Authorization Required")
    st.write("**Step 1:** Click the link below to authorize:")
    st.markdown(f"[👉 Authorize Gmail Access]({auth_url})")
    st.write("**Step 2:** Paste the code you receive here and press Enter:")

    auth_code = st.text_input("Authorization code:", key="auth_code_input")

    if st.button("Submit Code"):
        if auth_code:
            try:
                flow.fetch_token(code=auth_code.strip())
                creds = flow.credentials
                st.session_state["creds"] = creds
                st.success("✅ Authorized successfully!")
                st.info("To avoid re-authorizing next time, add these to Streamlit Secrets:")
                st.code(f"""[token]
token = "{creds.token}"
refresh_token = "{creds.refresh_token}"
token_uri = "{creds.token_uri}"
client_id = "{creds.client_id}"
client_secret = "{creds.client_secret}"
""")
                st.rerun()
            except Exception as e:
                st.error(f"Authorization failed: {e}")
        else:
            st.warning("Please paste the authorization code first!")
    st.stop()


# Load model
with open('spam_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)

# UI
st.title("📧 Gmail Spam Detector")
st.write("Scan your Gmail inbox for spam using AI!")

num_emails = st.slider("How many emails to scan?", 10, 100, 20)

if st.button("🔍 Scan My Emails"):
    with st.spinner("Connecting to Gmail..."):
        service = authenticate()

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