import streamlit as st
import pickle
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
import os

SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

def authenticate():
    creds = None
    if os.path.exists('token.pkl'):
        with open('token.pkl', 'rb') as f:
            creds = pickle.load(f)
    
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        
        with open('token.pkl', 'wb') as f:
            pickle.dump(creds, f)
    
    return build('gmail', 'v1', credentials=creds)

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