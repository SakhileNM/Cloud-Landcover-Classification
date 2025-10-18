# auth0_auth.py - READY FOR GOOGLE DRIVE
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
    
    def refresh_token(self, refresh_token):
        """Refresh access token"""
        url = f"https://{self.domain}/oauth/token"
        headers = {"content-type": "application/x-www-form-urlencoded"}
        data = {
            "grant_type": "refresh_token",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "refresh_token": refresh_token
        }
        
        response = requests.post(url, headers=headers, data=data)
        if response.status_code == 200:
            return response.json()
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
    }
    .auth0-badge {
        background: linear-gradient(45deg, #EB5424, #2A2E43);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
    }
    .feature-badge {
        background: #4CAF50;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 10px;
        font-size: 0.8rem;
        margin-left: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="login-container">', unsafe_allow_html=True)
        
        st.markdown('<div class="auth0-badge">🔐 Secure Auth0 Login</div>', unsafe_allow_html=True)
        
        # Regular login (without Drive scope)
        auth_url = st.session_state.auth0_service.get_auth_url(include_drive_scope=False)
        st.markdown(f'<a href="{auth_url}" target="_self"><button style="background-color: #4285F4; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; width: 100%; margin-bottom: 10px;">Sign in with Google</button></a>', unsafe_allow_html=True)
        
        # Option for future Drive integration
        st.markdown("---")
        st.info("""
        **Coming Soon:**
        - Save results directly to Google Drive
        - Automatic cloud backups  
        - Organize your analysis files
        """)
        
        st.markdown('</div>', unsafe_allow_html=True)

def handle_auth0_callback():
    """Handle Auth0 callback after login"""
    query_params = st.experimental_get_query_params()
    
    if 'code' in query_params:
        code = query_params['code'][0]
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
                    'drive_connected': False  # Will be True when we add Drive scope
                }
                st.session_state.authenticated = True
                
                st.success(f"Welcome {user_info.get('name', user_info['email'])}!")
                # Clear URL parameters
                st.experimental_set_query_params()
                st.rerun()

def show_auth0_profile():
    user = st.session_state.user
    
    st.title("👤 Profile (Auth0)")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Profile Info")
        st.write(f"**Name:** {user['name']}")
        st.write(f"**Email:** {user['email']}")
        st.write(f"**ID:** {user['id']}")
        
        # Drive connection status
        st.subheader("Cloud Storage")
        if user.get('drive_connected'):
            st.success("Google Drive Connected")
        else:
            st.info("Google Drive - Coming Soon")
        
        if user.get('picture'):
            st.image(user['picture'], width=100)
    
    with col2:
        st.subheader("Application Settings")
        
        with st.form("user_settings"):
            theme = st.selectbox(
                "Theme",
                ["light", "dark", "auto"],
                index=0
            )
            
            save_location = st.radio(
                "Save results to:",
                ["local", "google_drive"],
                format_func=lambda x: {
                    "local": "Local Storage",
                    "google_drive": "Google Drive (Coming Soon)"
                }[x]
            )
            
            if st.form_submit_button("Save Settings"):
                st.success("Settings saved! (Google Drive integration coming soon)")
    
    st.markdown("---")
    
    # Google Drive integration placeholder
    st.subheader("Connect Google Drive")
    st.info("""
    **Google Drive Integration - Coming Soon**
    
    Future features:
    - Save PDF reports directly to your Google Drive
    - Access your analysis history from any device  
    - Automatic backups of your results
    - Share results with team members
    """)
    
    if st.button("Notify Me When Available"):
        st.success("We'll notify you when Google Drive integration is ready!")
    
    st.markdown("---")
    if st.button("Logout"):
        auth0_domain = os.getenv("AUTH0_DOMAIN")
        client_id = os.getenv("AUTH0_CLIENT_ID")
        return_to = "http://earthgo.work.gd:8501"
        
        logout_url = f"https://{auth0_domain}/v2/logout?client_id={client_id}&returnTo={return_to}"
        st.markdown(f'[Click here to logout completely]({logout_url})')
        
        for key in ['authenticated', 'user', 'access_token']:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()
