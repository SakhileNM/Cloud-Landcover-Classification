import streamlit as st
import os
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

class GoogleDriveService:
    def __init__(self):
        self.SCOPES = ['https://www.googleapis.com/auth/drive.file']
        self.redirect_uri = os.getenv('AUTH0_REDIRECT_URI', 'http://localhost:8501')
    
    def get_credentials(self, user_id):
        """Get stored credentials for user"""
        if 'google_credentials' in st.session_state:
            return Credentials.from_authorized_user_info(st.session_state.google_credentials, self.SCOPES)
        return None
    
    def save_credentials(self, credentials):
        """Save credentials to session"""
        st.session_state.google_credentials = {
            'token': credentials.token,
            'refresh_token': credentials.refresh_token,
            'token_uri': credentials.token_uri,
            'client_id': credentials.client_id,
            'client_secret': credentials.client_secret,
            'scopes': credentials.scopes
        }
    
    def start_oauth_flow(self):
        """Start Google OAuth flow"""
        flow = Flow.from_client_secrets_file(
            'client_secrets.json',
            scopes=self.SCOPES,
            redirect_uri=self.redirect_uri
        )
        
        authorization_url, state = flow.authorization_url(
            access_type='offline',
            include_granted_scopes='true'
        )
        
        st.session_state.google_oauth_state = state
        st.markdown(f'<meta http-equiv="refresh" content="0; url={authorization_url}">', unsafe_allow_html=True)
        return authorization_url
    
    def handle_callback(self):
        """Handle OAuth callback"""
        if 'code' in st.query_params and 'google_oauth_state' in st.session_state:
            flow = Flow.from_client_secrets_file(
                'client_secrets.json',
                scopes=self.SCOPES,
                state=st.session_state.google_oauth_state,
                redirect_uri=self.redirect_uri
            )
            
            flow.fetch_token(authorization_response=st.query_params['code'])
            credentials = flow.credentials
            
            self.save_credentials(credentials)
            st.session_state.user['drive_connected'] = True
            
            # Clear the state
            del st.session_state.google_oauth_state
            st.query_params.clear()
            
            st.success("Google Drive connected successfully!")
            st.rerun()
    
    def upload_file(self, file_path, file_name):
        """Upload file to Google Drive"""
        credentials = self.get_credentials(st.session_state.user['id'])
        if not credentials:
            return None
        
        try:
            service = build('drive', 'v3', credentials=credentials)
            
            file_metadata = {'name': file_name}
            media = MediaFileUpload(file_path, mimetype='application/pdf')
            
            file = service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id, webViewLink'
            ).execute()
            
            return file.get('webViewLink')
        except Exception as e:
            st.error(f"Upload failed: {e}")
            return None
