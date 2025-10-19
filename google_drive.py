import streamlit as st
import os
import json
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.auth.transport.requests import Request
import webbrowser

class GoogleDriveService:
    def __init__(self):
        self.SCOPES = ['https://www.googleapis.com/auth/drive.file']
        self.redirect_uri = os.getenv('GOOGLE_REDIRECT_URI', 'http://localhost:8501')
        self.credentials_dir = '/app/data/credentials'
        os.makedirs(self.credentials_dir, exist_ok=True)
    
    def get_credentials_path(self, user_id):
        """Get path for user's credentials file"""
        return os.path.join(self.credentials_dir, f"{user_id}_google_drive_token.json")
    
    def get_credentials(self, user_id):
        """Get stored credentials for user"""
        creds_path = self.get_credentials_path(user_id)
        if os.path.exists(creds_path):
            try:
                creds = Credentials.from_authorized_user_file(creds_path, self.SCOPES)
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                    self.save_credentials(user_id, creds)
                return creds
            except Exception as e:
                st.error(f"Error loading credentials: {e}")
        return None
    
    def save_credentials(self, user_id, credentials):
        """Save credentials to file"""
        try:
            creds_path = self.get_credentials_path(user_id)
            with open(creds_path, 'w') as token:
                token.write(credentials.to_json())
            return True
        except Exception as e:
            st.error(f"Error saving credentials: {e}")
            return False
    
    def authenticate(self, user_id):
        """Start Google OAuth flow"""
        try:
            # Verify client secrets file exists
            if not os.path.exists('client_secrets.json'):
                st.error("client_secrets.json file not found")
                return False
            
            flow = Flow.from_client_secrets_file(
                'client_secrets.json',
                scopes=self.SCOPES,
                redirect_uri=self.redirect_uri
            )
            
            authorization_url, state = flow.authorization_url(
                access_type='offline',
                include_granted_scopes='true',
                prompt='consent'
            )
            
            # Store state in session
            st.session_state.google_oauth_state = state
            st.session_state.google_oauth_user_id = user_id
            
            # Open in new tab and show instructions
            st.markdown(f"""
            ### Google Drive Authentication
            Please complete the authentication in the new window.
            
            If the window doesn't open automatically, [click here]({authorization_url}).
            """)
            
            # Try to open browser
            try:
                webbrowser.open_new_tab(authorization_url)
            except:
                st.info("Please click the link above to open the authentication page.")
            
            return True
            
        except Exception as e:
            st.error(f"Authentication error: {str(e)}")
            return False
    
    def handle_oauth_callback(self):
        """Handle OAuth callback - call this from your main app"""
        query_params = st.query_params
        
        if 'code' in query_params and 'google_oauth_state' in st.session_state:
            try:
                st.info("Processing Google OAuth callback...")
                
                # Get the authorization code
                auth_code = query_params['code']
                state = st.session_state.google_oauth_state
                user_id = st.session_state.get('google_oauth_user_id')
                
                if not user_id and 'user' in st.session_state:
                    user_id = st.session_state.user['id']
                
                flow = Flow.from_client_secrets_file(
                    'client_secrets.json',
                    scopes=self.SCOPES,
                    state=state,
                    redirect_uri=self.redirect_uri
                )
                
                # Exchange code for tokens
                flow.fetch_token(code=auth_code)
                credentials = flow.credentials
                
                # Save credentials
                if user_id and self.save_credentials(user_id, credentials):
                    # Update user session
                    if 'user' in st.session_state:
                        st.session_state.user['drive_connected'] = True
                    
                    # Clear session state
                    if 'google_oauth_state' in st.session_state:
                        del st.session_state.google_oauth_state
                    if 'google_oauth_user_id' in st.session_state:
                        del st.session_state.google_oauth_user_id
                    
                    # Clear query params
                    st.query_params.clear()
                    
                    st.success("Google Drive connected successfully!")
                    st.rerun()
                else:
                    st.error("Failed to save credentials")
                    
            except Exception as e:
                st.error(f"OAuth callback error: {str(e)}")
    
    def upload_file(self, file_path, file_name):
        """Upload file to Google Drive"""
        if 'user' not in st.session_state:
            st.error("User not logged in")
            return None
        
        user_id = st.session_state.user['id']
        credentials = self.get_credentials(user_id)
        
        if not credentials:
            st.error("Google Drive not connected. Please connect in your profile settings.")
            return None
        
        try:
            service = build('drive', 'v3', credentials=credentials)
            
            file_metadata = {
                'name': file_name,
                'mimeType': 'application/pdf'
            }
            
            media = MediaFileUpload(
                file_path, 
                mimetype='application/pdf',
                resumable=True
            )
            
            file = service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id, webViewLink, name'
            ).execute()
            
            st.success(f"File '{file_name}' uploaded to Google Drive!")
            return file.get('webViewLink')
            
        except Exception as e:
            st.error(f"Upload failed: {str(e)}")
            return None
    
    def list_user_files(self, user_id, limit=10):
        """List user's files in Google Drive"""
        credentials = self.get_credentials(user_id)
        if not credentials:
            return []
        
        try:
            service = build('drive', 'v3', credentials=credentials)
            results = service.files().list(
                pageSize=limit,
                fields="files(id, name, webViewLink, createdTime)"
            ).execute()
            
            return results.get('files', [])
        except Exception as e:
            st.error(f"Error listing files: {e}")
            return []
    
    def disconnect_drive(self, user_id):
        """Disconnect Google Drive"""
        try:
            creds_path = self.get_credentials_path(user_id)
            if os.path.exists(creds_path):
                os.remove(creds_path)
            return True
        except Exception as e:
            st.error(f"Error disconnecting Drive: {e}")
            return False

def setup_google_drive_credentials():
    """Setup function to be called from auth0_auth.py"""
    # This ensures the credentials directory exists
    credentials_dir = '/app/data/credentials'
    os.makedirs(credentials_dir, exist_ok=True)
