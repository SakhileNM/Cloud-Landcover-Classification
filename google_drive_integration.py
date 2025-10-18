import os
import pickle
import streamlit as st
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
import json
from datetime import datetime, timedelta

class GoogleDriveService:
    def __init__(self):
        self.SCOPES = [
            'https://www.googleapis.com/auth/drive.file',
            'https://www.googleapis.com/auth/userinfo.profile',
            'https://www.googleapis.com/auth/userinfo.email',
            'openid'
        ]
        self.creds = None
        self.service = None
        
    def get_credentials_path(self, user_id):
        """Get path for storing user credentials"""
        os.makedirs('/app/data/credentials', exist_ok=True)
        return f'/app/data/credentials/{user_id}_google_drive_token.json'
    
    def authenticate(self, user_id):
        """Authenticate with Google Drive API"""
        creds_path = self.get_credentials_path(user_id)
        
        # Load existing credentials
        if os.path.exists(creds_path):
            try:
                with open(creds_path, 'r') as token:
                    creds_data = json.load(token)
                self.creds = Credentials.from_authorized_user_info(creds_data, self.SCOPES)
            except Exception as e:
                st.error(f"Error loading credentials: {e}")
                return False
        
        # If there are no (valid) credentials available, start OAuth flow
        if not self.creds or not self.creds.valid:
            if self.creds and self.creds.expired and self.creds.refresh_token:
                try:
                    self.creds.refresh(Request())
                except Exception as e:
                    st.error(f"Error refreshing credentials: {e}")
                    return False
            else:
                return self.start_oauth_flow(user_id)
        
        # Build the service
        try:
            self.service = build('drive', 'v3', credentials=self.creds)
            return True
        except Exception as e:
            st.error(f"Error building Drive service: {e}")
            return False
    
    def start_oauth_flow(self, user_id):
        """Start OAuth 2.0 flow for Google Drive"""
        try:
            # Create the flow using the client secrets file
            flow = Flow.from_client_secrets_file(
                '/app/credentials/client_secrets.json',
                scopes=self.SCOPES,
                redirect_uri=os.getenv('GOOGLE_REDIRECT_URI', 'http://localhost:8501')
            )
            
            # Generate URL for request to Google's OAuth 2.0 server.
            authorization_url, state = flow.authorization_url(
                access_type='offline',
                include_granted_scopes='true',
                prompt='consent'
            )
            
            # Store the state and user ID in session
            st.session_state.google_oauth_state = state
            st.session_state.google_oauth_user_id = user_id
            
            # Redirect to authorization URL
            st.markdown(f'<meta http-equiv="refresh" content="0; url={authorization_url}">', unsafe_allow_html=True)
            st.success("Redirecting to Google authentication...")
            return False
            
        except Exception as e:
            st.error(f"Error starting OAuth flow: {e}")
            return False
    
    def handle_oauth_callback(self):
        """Handle OAuth 2.0 callback"""
        if 'code' in st.query_params and 'google_oauth_state' in st.session_state:
            try:
                state = st.session_state.google_oauth_state
                user_id = st.session_state.google_oauth_user_id
                
                flow = Flow.from_client_secrets_file(
                    '/app/credentials/client_secrets.json',
                    scopes=self.SCOPES,
                    state=state,
                    redirect_uri=os.getenv('GOOGLE_REDIRECT_URI', 'http://localhost:8501')
                )
                
                # Exchange authorization code for tokens
                flow.fetch_token(authorization_response=st.query_params['code'])
                
                # Get credentials
                creds = flow.credentials
                
                # Save credentials
                creds_path = self.get_credentials_path(user_id)
                with open(creds_path, 'w') as token:
                    token.write(creds.to_json())
                
                # Update user session
                if 'user' in st.session_state:
                    st.session_state.user['drive_connected'] = True
                
                # Clear OAuth state
                del st.session_state.google_oauth_state
                del st.session_state.google_oauth_user_id
                
                # Clear query parameters
                st.query_params.clear()
                
                st.success("Google Drive connected successfully!")
                st.rerun()
                
            except Exception as e:
                st.error(f"Error handling OAuth callback: {e}")
    
    def upload_file_to_drive(self, file_path, file_name, user_id):
        """Upload file to Google Drive"""
        if not self.authenticate(user_id):
            return None
        
        try:
            # Create folder for the app if it doesn't exist
            folder_id = self.get_or_create_folder('Geospatial Landcover Analysis', user_id)
            
            # File metadata
            file_metadata = {
                'name': file_name,
                'parents': [folder_id]
            }
            
            # Upload file
            media = MediaFileUpload(file_path, mimetype='application/pdf')
            file = self.service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id, webViewLink'
            ).execute()
            
            return file.get('webViewLink')
            
        except Exception as e:
            st.error(f"Error uploading to Google Drive: {e}")
            return None
    
    def get_or_create_folder(self, folder_name, user_id):
        """Get or create folder in Google Drive"""
        if not self.authenticate(user_id):
            return None
        
        try:
            # Check if folder exists
            query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
            results = self.service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
            folders = results.get('files', [])
            
            if folders:
                return folders[0]['id']
            else:
                # Create folder
                folder_metadata = {
                    'name': folder_name,
                    'mimeType': 'application/vnd.google-apps.folder'
                }
                folder = self.service.files().create(body=folder_metadata, fields='id').execute()
                return folder['id']
                
        except Exception as e:
            st.error(f"Error creating folder: {e}")
            return None
    
    def list_user_files(self, user_id, limit=10):
        """List user's files in Google Drive"""
        if not self.authenticate(user_id):
            return []
        
        try:
            folder_id = self.get_or_create_folder('Geospatial Landcover Analysis', user_id)
            query = f"'{folder_id}' in parents and trashed=false"
            results = self.service.files().list(
                q=query,
                spaces='drive',
                fields='files(id, name, createdTime, webViewLink)',
                orderBy='createdTime desc'
            ).execute()
            
            return results.get('files', [])[:limit]
            
        except Exception as e:
            st.error(f"Error listing files: {e}")
            return []
    
    def disconnect_drive(self, user_id):
        """Disconnect Google Drive and remove credentials"""
        try:
            creds_path = self.get_credentials_path(user_id)
            if os.path.exists(creds_path):
                os.remove(creds_path)
            
            if 'user' in st.session_state:
                st.session_state.user['drive_connected'] = False
            
            st.success("Google Drive disconnected successfully!")
            return True
            
        except Exception as e:
            st.error(f"Error disconnecting Google Drive: {e}")
            return False

def setup_google_drive_credentials():
    """Setup Google Drive credentials file"""
    credentials_data = {
        "web": {
            "client_id": os.getenv('GOOGLE_CLIENT_ID'),
            "project_id": os.getenv('GOOGLE_PROJECT_ID'),
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
            "client_secret": os.getenv('GOOGLE_CLIENT_SECRET'),
            "redirect_uris": [
                os.getenv('GOOGLE_REDIRECT_URI', 'http://localhost:8501')
            ]
        }
    }
    
    os.makedirs('/app/credentials', exist_ok=True)
    with open('/app/credentials/client_secrets.json', 'w') as f:
        json.dump(credentials_data, f)
