# auth0_auth.py - FIXED VERSION
import streamlit as st
import requests
import jwt
import json
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

load_dotenv()

class Auth0Service:
    def __init__(self):
        self.domain = os.getenv("AUTH0_DOMAIN")
        self.client_id = os.getenv("AUTH0_CLIENT_ID")
        self.client_secret = os.getenv("AUTH0_CLIENT_SECRET")
        self.redirect_uri = os.getenv("AUTH0_REDIRECT_URI", "http://earthgo.work.gd:8501")
    
    def get_auth_url(self, include_drive_scope=False):
        """Generate Auth0 authorization URL"""
        scope = "openid profile email"
        if include_drive_scope:
            scope += " https://www.googleapis.com/auth/drive.file"
        
        return (
            f"https://{self.domain}/authorize?"
            f"response_type=code&"
            f"client_id={self.client_id}&"
            f"redirect_uri={self.redirect_uri}&"
            f"scope={scope}&"
            f"audience=https://{self.domain}/api/v2/"
        )
    
    def get_token(self, code):
        """Exchange authorization code for tokens"""
        url = f"https://{self.domain}/oauth/token"
        headers = {"content-type": "application/x-www-form-urlencoded"}
        data = {
            "grant_type": "authorization_code",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "code": code,
            "redirect_uri": self.redirect_uri
        }
        
        response = requests.post(url, headers=headers, data=data)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Token exchange failed: {response.text}")
            return None
    
    def get_user_info(self, access_token):
        """Get user profile information"""
        url = f"https://{self.domain}/userinfo"
        headers = {"authorization": f"Bearer {access_token}"}
        
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"User info failed: {response.text}")
            return None

def show_auth0_login():
    if 'auth0_service' not in st.session_state:
        st.session_state.auth0_service = Auth0Service()
    
    st.markdown("""
    <style>
    .login-container {
        max-width: 400px;
        margin: 0 auto;
        padding: 2rem;
        border: 1px solid #ddd;
        border-radius: 10px;
        background: white;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    .google-button {
        background-color: #4285F4;
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 4px;
        cursor: pointer;
        width: 100%;
        font-size: 16px;
        font-weight: 500;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 12px;
        transition: background-color 0.3s;
    }
    .google-button:hover {
        background-color: #357ae8;
    }
    .google-icon {
        width: 18px;
        height: 18px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="login-container">', unsafe_allow_html=True)
        
        st.subheader("Sign In")
        st.write("Access the Geospatial Landcover Classification Platform")
        
        # Google Login via Auth0
        auth_url = st.session_state.auth0_service.get_auth_url(include_drive_scope=False)
        
        google_button_html = f'''
        <a href="{auth_url}" target="_self" style="text-decoration: none;">
            <button class="google-button">
                <svg class="google-icon" viewBox="0 0 24 24">
                    <path fill="#fff" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/>
                    <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/>
                    <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/>
                    <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/>
                </svg>
                Sign in with Google
            </button>
        </a>
        '''
        
        st.markdown(google_button_html, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

def handle_auth0_callback():
    """Handle Auth0 callback after login"""
    # Use new st.query_params instead of experimental_get_query_params
    query_params = st.query_params
    
    if 'code' in query_params:
        code = query_params['code']
        auth_service = Auth0Service()
        
        # Exchange code for tokens
        tokens = auth_service.get_token(code)
        if tokens:
            # Get user info
            user_info = auth_service.get_user_info(tokens['access_token'])
            
            if user_info:
                # Store user in session
                st.session_state.user = {
                    'id': user_info['sub'],
                    'email': user_info['email'],
                    'name': user_info.get('name', user_info['email']),
                    'picture': user_info.get('picture', ''),
                    'auth0_data': user_info,
                    'access_token': tokens.get('access_token'),
                    'refresh_token': tokens.get('refresh_token'),
                    'drive_connected': False
                }
                st.session_state.authenticated = True
                
                st.success(f"Welcome {user_info.get('name', user_info['email'])}!")
                # Clear query parameters using new API
                st.query_params.clear()
                st.rerun()

def show_auth0_profile():
    user = st.session_state.user
    
    st.title("User Profile")
    
    # Navigation buttons at the top
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("Home", key="profile_home_btn"):
            st.session_state.show_profile = False
            st.rerun()
    with col2:
        if st.button("Profile Settings", key="profile_settings_btn", disabled=True):
            pass  # Already on profile page
    with col3:
        if st.button("Logout", key="profile_logout_btn"):
            auth0_domain = os.getenv("AUTH0_DOMAIN")
            client_id = os.getenv("AUTH0_CLIENT_ID")
            return_to = "http://earthgo.work.gd:8501"
            
            logout_url = f"https://{auth0_domain}/v2/logout?client_id={client_id}&returnTo={return_to}"
            st.markdown(f'[Click here to logout completely]({logout_url})')
            
            for key in ['authenticated', 'user', 'access_token']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
    
    st.markdown("---")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Profile Information")
        st.write(f"**Name:** {user['name']}")
        st.write(f"**Email:** {user['email']}")
        st.write(f"**User ID:** {user['id']}")
        
        # Cloud storage status
        st.subheader("Cloud Storage")
        if user.get('drive_connected'):
            st.success("Google Drive: Connected")
        else:
            st.info("Google Drive: Not Connected")
        
        if user.get('picture'):
            st.image(user['picture'], width=100)
    
    with col2:
        st.subheader("Application Settings")
        
        with st.form("user_settings_form", key="user_settings_form"):
            theme = st.selectbox(
                "Interface Theme",
                ["light", "dark", "auto"],
                index=0,
                key="theme_select"
            )
            
            save_location = st.radio(
                "Save results to:",
                ["local", "google_drive"],
                format_func=lambda x: {
                    "local": "Local Storage",
                    "google_drive": "Google Drive"
                }[x],
                index=0,
                key="save_location_radio"
            )
            
            if st.form_submit_button("Save Settings", key="save_settings_btn"):
                st.success("Settings saved successfully!")
    
    st.markdown("---")
    
    # Google Drive integration section
    st.subheader("Google Drive Integration")
    st.info("""
    **Google Drive Integration - Coming Soon**
    
    Future features will include:
    - Save PDF reports directly to your Google Drive
    - Access your analysis history from any device  
    - Automatic backups of your results
    - Share results with team members
    """)
    
    if st.button("Notify Me When Available", key="notify_drive_btn"):
        st.success("We will notify you when Google Drive integration is ready!")
