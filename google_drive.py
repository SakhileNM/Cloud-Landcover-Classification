# google_drive.py (patched)
import streamlit as st
import os
import json
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.auth.transport.requests import Request
import webbrowser
import traceback

class GoogleDriveService:
    def __init__(self):
        self.SCOPES = ['https://www.googleapis.com/auth/drive.file']
        # file path can be overridden with env var if needed
        self.client_secrets_file = os.getenv('GOOGLE_CLIENT_SECRETS', 'client_secrets.json')
        self.credentials_dir = '/app/data/credentials'
        os.makedirs(self.credentials_dir, exist_ok=True)

        # load client_secrets.json to get registered redirect URIs
        self.registered_redirects = []
        try:
            with open(self.client_secrets_file, 'r') as f:
                data = json.load(f)
                webcfg = data.get('web') or data.get('installed') or {}
                self.registered_redirects = webcfg.get('redirect_uris', [])
        except FileNotFoundError:
            st.warning(f"client_secrets.json not found at {self.client_secrets_file}. Make sure it is present inside the container.")
        except Exception as e:
            st.warning(f"Could not read client_secrets.json: {e}")

        # choose redirect_uri: environment variable takes priority, otherwise the first registered redirect
        env_redirect = os.getenv('GOOGLE_REDIRECT_URI')
        if env_redirect:
            self.redirect_uri = env_redirect
        elif self.registered_redirects:
            self.redirect_uri = self.registered_redirects[0]
        else:
            # fallback to localhost (useful for local testing)
            self.redirect_uri = 'http://localhost:8501'

        # warn if using a redirect URI not listed in client_secrets.json
        if self.registered_redirects and self.redirect_uri not in self.registered_redirects:
            st.warning(
                "The redirect URI you're using is not present in client_secrets.json registered redirect URIs.\n"
                "This will cause Google to return a 400 error. Make sure the redirect URI EXACTLY matches one in\n"
                "the OAuth client settings (including trailing slash). Registered URIs: "
                f"{self.registered_redirects} ; Using: {self.redirect_uri}"
            )

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
                st.error(traceback.format_exc())
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
            if not os.path.exists(self.client_secrets_file):
                st.error(f"client_secrets.json file not found at {self.client_secrets_file}")
                return False

            # Important: Flow will reject the redirect URI if it isn't registered exactly.
            flow = Flow.from_client_secrets_file(
                self.client_secrets_file,
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
            
            # Show link and try to open automatically
            st.markdown(f"""
            ### Google Drive Authentication
            Please complete the authentication in the new window.

            If the window doesn't open automatically, [click here]({authorization_url}).
            **Redirect URI used for this request:** `{self.redirect_uri}`
            """)
            
            try:
                webbrowser.open_new_tab(authorization_url)
            except Exception:
                st.info("Please click the link above to open the authentication page.")
            
            return True
            
        except Exception as e:
            st.error(f"Authentication error: {str(e)}")
            st.error(traceback.format_exc())
            return False
    
    def handle_oauth_callback(self):
        """Handle OAuth callback - call this from your main app"""
        query_params = st.query_params

        # streamlit returns lists for query params (e.g. {'code': ['...']})
        code = query_params.get('code', [None])[0]
        state = query_params.get('state', [None])[0]
        
        # we expect to only handle this callback if our state is present in session
        if code and 'google_oauth_state' in st.session_state:
            try:
                st.info("Processing Google OAuth callback...")
                
                user_id = st.session_state.get('google_oauth_user_id')
                if not user_id and 'user' in st.session_state:
                    user_id = st.session_state.user['id']
                
                flow = Flow.from_client_secrets_file(
                    self.client_secrets_file,
                    scopes=self.SCOPES,
                    state=st.session_state.get('google_oauth_state'),
                    redirect_uri=self.redirect_uri
                )
                
                # Exchange code for tokens
                flow.fetch_token(code=code)
                credentials = flow.credentials
                
                # Save credentials
                if user_id and self.save_credentials(user_id, credentials):
                    # Update user session
                    if 'user' in st.session_state:
                        st.session_state.user['drive_connected'] = True
                    
                    # Clear session state
                    st.session_state.pop('google_oauth_state', None)
                    st.session_state.pop('google_oauth_user_id', None)
                    
                    # Clear query params
                    st.query_params.clear()
                    
                    st.success("Google Drive connected successfully!")
                    st.rerun()
                else:
                    st.error("Failed to save credentials")
                    
            except Exception as e:
                st.error(f"OAuth callback error: {str(e)}")
                st.error(traceback.format_exc())
