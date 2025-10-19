import streamlit as st
import os
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

class GoogleDriveService:
    def __init__(self):
        self.SCOPES = ['https://www.googleapis.com/auth/drive.file']
        # Use the exact same redirect URI as registered in Google Cloud Console
        self.redirect_uri = os.getenv('GOOGLE_REDIRECT_URI', 'http://localhost:8501')
    
    def start_oauth_flow(self):
        """Start Google OAuth flow"""
        try:
            flow = Flow.from_client_secrets_file(
                'client_secrets.json',
                scopes=self.SCOPES,
                redirect_uri=self.redirect_uri
            )
            
            # For web credentials, use the exact redirect URI
            authorization_url, state = flow.authorization_url(
                access_type='offline',
                include_granted_scopes='true',
                prompt='consent'
            )
            
            st.session_state.google_oauth_state = state
            
            # Use Streamlit's native redirect instead of meta refresh
            st.markdown(f'[Click here to authenticate with Google Drive]({authorization_url})')
            st.stop()
            
        except Exception as e:
            st.error(f"OAuth configuration error: {e}")
            return None
    
    def handle_callback(self):
        """Handle OAuth callback"""
        if 'code' in st.query_params and 'google_oauth_state' in st.session_state:
            try:
                flow = Flow.from_client_secrets_file(
                    'client_secrets.json',
                    scopes=self.SCOPES,
                    state=st.session_state.google_oauth_state,
                    redirect_uri=self.redirect_uri
                )
                
                # Get the full URL for token exchange
                full_redirect_uri = f"{self.redirect_uri}?{st.query_params.to_dict()}"
                flow.fetch_token(authorization_response=full_redirect_uri)
                
                credentials = flow.credentials
                self.save_credentials(credentials)
                
                if 'user' in st.session_state:
                    st.session_state.user['drive_connected'] = True
                
                # Clear the state and query params
                del st.session_state.google_oauth_state
                st.query_params.clear()
                
                st.success("Google Drive connected successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"OAuth callback error: {e}")
    
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
