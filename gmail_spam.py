import pickle
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
import os

# Gmail API scope
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

service = authenticate()
print("Gmail connected!")

# Load saved model and tfidf
with open('spam_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)

# Fetch 100 latest emails
results = service.users().messages().list(userId='me', maxResults=100).execute()
messages = results.get('messages', [])

print(f"Number of emails found: {len(messages)}")
print("Checking your emails...\n")

for msg in messages:
    txt = service.users().messages().get(userId='me', id=msg['id']).execute()
    
    headers = txt['payload']['headers']
    subject = next((h['value'] for h in headers if h['name'] == 'Subject'), 'No Subject')
    snippet = txt.get('snippet', '')
    
    result = model.predict(tfidf.transform([snippet]))
    label = "🚨 SPAM" if result[0] == 1 else "✅ HAM"
    
    print(f"{label} | {subject}")