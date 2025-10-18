import streamlit as st
import requests
import jwt
import json
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import sqlite3
from contextlib import contextmanager
import hashlib
from google_drive_integration import GoogleDriveService, setup_google_drive_credentials

load_dotenv()

@contextmanager
def get_db_connection():
    """Context manager for database connections"""
    os.makedirs('/app/data', exist_ok=True)
    
    conn = sqlite3.connect('/app/data/user_preferences.db', check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

class Auth0Service:
    def __init__(self):
        self.domain = os.getenv("AUTH0_DOMAIN")
        self.client_id = os.getenv("AUTH0_CLIENT_ID")
        self.client_secret = os.getenv("AUTH0_CLIENT_SECRET")
        self.redirect_uri = os.getenv("AUTH0_REDIRECT_URI", "http://localhost:8501")
        self.google_drive_service = GoogleDriveService()
        self.init_database()
        
        # Setup Google Drive credentials
        setup_google_drive_credentials()

    def init_database(self):
        """Initialize SQLite database for user preferences and history"""
        with get_db_connection() as conn:
            # User preferences table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS user_preferences (
                    user_id TEXT PRIMARY KEY,
                    theme TEXT DEFAULT 'light',
                    default_model TEXT DEFAULT 'Random Forest',
                    auto_save BOOLEAN DEFAULT 1,
                    email_notifications BOOLEAN DEFAULT 0,
                    save_location TEXT DEFAULT 'local',
                    drive_connected BOOLEAN DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # User analysis history table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS user_analysis_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    analysis_type TEXT NOT NULL,
                    location_lat REAL,
                    location_lon REAL,
                    years TEXT,
                    model_used TEXT,
                    results_data TEXT,
                    drive_file_id TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES user_preferences (user_id)
                )
            ''')
            
            conn.commit()

    def get_auth_url(self, include_drive_scope=False):
        """Generate Auth0 authorization URL"""
        scope = "openid profile email"
        return (f"https://{self.domain}/authorize?"
                f"response_type=code&"
                f"client_id={self.client_id}&"
                f"redirect_uri={self.redirect_uri}&"
                f"scope={scope}&"
                f"audience=https://{self.domain}/api/v2/")

    def get_token(self, code):
        """Exchange authorization code for tokens"""
        try:
            response = requests.post(
                f"https://{self.domain}/oauth/token",
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                data={
                    "grant_type": "authorization_code",
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                    "code": code,
                    "redirect_uri": self.redirect_uri
                }
            )
            return response.json() if response.status_code == 200 else None
        except Exception as e:
            st.error(f"Token exchange failed: {e}")
            return None

    def get_user_info(self, access_token):
        """Get user information from Auth0"""
        try:
            response = requests.get(
                f"https://{self.domain}/userinfo",
                headers={"Authorization": f"Bearer {access_token}"}
            )
            return response.json() if response.status_code == 200 else None
        except Exception as e:
            st.error(f"User info fetch failed: {e}")
            return None

    def get_user_preferences(self, user_id):
        """Get user preferences from database"""
        try:
            with get_db_connection() as conn:
                cursor = conn.execute(
                    '''SELECT theme, default_model, auto_save, email_notifications, save_location, drive_connected 
                       FROM user_preferences WHERE user_id = ?''',
                    (user_id,)
                )
                result = cursor.fetchone()
                if result:
                    return {
                        'theme': result[0],
                        'default_model': result[1],
                        'auto_save': bool(result[2]),
                        'email_notifications': bool(result[3]),
                        'save_location': result[4],
                        'drive_connected': bool(result[5])
                    }
        except Exception as e:
            st.error(f"Error loading preferences: {e}")
        return None

    def save_user_preferences(self, user_id, preferences):
        """Save user preferences to database"""
        try:
            with get_db_connection() as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO user_preferences 
                    (user_id, theme, default_model, auto_save, email_notifications, save_location, drive_connected, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ''', (
                    user_id,
                    preferences.get('theme', 'light'),
                    preferences.get('default_model', 'Random Forest'),
                    int(preferences.get('auto_save', True)),
                    int(preferences.get('email_notifications', False)),
                    preferences.get('save_location', 'local'),
                    int(preferences.get('drive_connected', False))
                ))
                conn.commit()
                return True
        except Exception as e:
            st.error(f"Error saving preferences: {e}")
            return False

    def save_analysis_history(self, user_id, analysis_data):
        """Save user analysis history"""
        try:
            with get_db_connection() as conn:
                conn.execute('''
                    INSERT INTO user_analysis_history 
                    (user_id, analysis_type, location_lat, location_lon, years, model_used, results_data, drive_file_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    user_id,
                    analysis_data.get('analysis_type', 'landcover'),
                    analysis_data.get('lat'),
                    analysis_data.get('lon'),
                    json.dumps(analysis_data.get('years', [])),
                    analysis_data.get('model_type'),
                    json.dumps(analysis_data.get('results', {})),
                    analysis_data.get('drive_file_id', '')
                ))
                conn.commit()
                return True
        except Exception as e:
            st.error(f"Error saving analysis history: {e}")
            return False

    def get_analysis_history(self, user_id, limit=10):
        """Get user analysis history"""
        try:
            with get_db_connection() as conn:
                cursor = conn.execute('''
                    SELECT * FROM user_analysis_history 
                    WHERE user_id = ? 
                    ORDER BY created_at DESC 
                    LIMIT ?
                ''', (user_id, limit))
                
                results = []
                for row in cursor.fetchall():
                    results.append(dict(row))
                return results
        except Exception as e:
            st.error(f"Error loading analysis history: {e}")
            return []

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
    </style>
    """, unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="login-container">', unsafe_allow_html=True)
        st.subheader("Sign In")
        st.write("Access the Geospatial Landcover Classification Platform")
        
        auth_url = st.session_state.auth0_service.get_auth_url(include_drive_scope=False)
        
        if st.button("Sign in with Auth0", key="auth0_login_btn"):
            st.markdown(f'<meta http-equiv="refresh" content="0; url={auth_url}">', unsafe_allow_html=True)
            st.success("Redirecting to Auth0...")
        
        st.markdown('</div>', unsafe_allow_html=True)

def handle_auth0_callback():
    """Handle Auth0 callback after login"""
    # First handle Google Drive OAuth callback if present
    if 'google_drive_service' not in st.session_state:
        st.session_state.google_drive_service = GoogleDriveService()
    
    st.session_state.google_drive_service.handle_oauth_callback()
    
    # Then handle Auth0 callback
    query_params = st.query_params
    
    if 'code' in query_params and 'google_oauth_state' not in st.session_state:
        code = query_params['code']
        
        if 'auth0_service' not in st.session_state:
            st.session_state.auth0_service = Auth0Service()
        
        auth_service = st.session_state.auth0_service
        
        # Exchange code for tokens
        tokens = auth_service.get_token(code)
        if tokens:
            # Get user info
            user_info = auth_service.get_user_info(tokens['access_token'])
            
            if user_info:
                # Load or create user preferences
                user_id = user_info['sub']
                preferences = auth_service.get_user_preferences(user_id)
                
                if not preferences:
                    # Create default preferences for new user
                    preferences = {
                        'theme': 'light',
                        'default_model': 'Random Forest',
                        'auto_save': True,
                        'email_notifications': False,
                        'save_location': 'local',
                        'drive_connected': False
                    }
                    auth_service.save_user_preferences(user_id, preferences)
                
                # Check if Google Drive is connected
                drive_connected = os.path.exists(f'/app/data/credentials/{user_id}_google_drive_token.json')
                preferences['drive_connected'] = drive_connected
                
                # Store user in session
                st.session_state.user = {
                    'id': user_id,
                    'email': user_info['email'],
                    'name': user_info.get('name', user_info['email']),
                    'picture': user_info.get('picture', ''),
                    'auth0_data': user_info,
                    'access_token': tokens.get('access_token'),
                    'refresh_token': tokens.get('refresh_token'),
                    'member_since': datetime.now().strftime('%Y-%m-%d'),
                    # Load preferences into session
                    **preferences
                }
                
                # Initialize analysis count
                analysis_history = auth_service.get_analysis_history(user_id)
                st.session_state.user['analysis_count'] = len(analysis_history)
                st.session_state.user['analysis_history'] = analysis_history
                
                st.session_state.authenticated = True
                
                st.success(f"Welcome {user_info.get('name', user_info['email'])}!")
                st.query_params.clear()
                st.rerun()

def show_auth0_profile():
    user = st.session_state.user
    auth_service = st.session_state.auth0_service
    drive_service = st.session_state.google_drive_service
    
    st.title("User Profile & Settings")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Profile Information")
        st.write(f"**Name:** {user['name']}")
        st.write(f"**Email:** {user['email']}")
        st.write(f"**Member Since:** {user.get('member_since', 'Recent')}")
        st.write(f"**Analyses Completed:** {user.get('analysis_count', 0)}")
        
        if user.get('picture'):
            st.image(user['picture'], width=120)
    
    with col2:
        st.subheader("Personal Settings")
        
        with st.form(key="user_preferences_form"):
            # Interface preferences
            st.write("**Interface Preferences**")
            theme = st.selectbox(
                "Theme",
                ["light", "dark", "auto"],
                index=["light", "dark", "auto"].index(user.get('theme', 'light'))
            )
            
            # Analysis preferences
            st.write("**Analysis Preferences**")
            default_model = st.selectbox(
                "Default Model",
                ["Random Forest", "Gradient Boosting"],
                index=0 if user.get('default_model', 'Random Forest') == "Random Forest" else 1
            )
            
            auto_save = st.checkbox(
                "Auto-save results",
                value=user.get('auto_save', True),
                help="Automatically save analysis results to your history"
            )
            
            email_notifications = st.checkbox(
                "Email notifications",
                value=user.get('email_notifications', False),
                help="Receive email notifications when analyses are completed"
            )
            
            save_location = st.radio(
                "Default save location",
                ["local", "google_drive"],
                index=0 if user.get('save_location', 'local') == "local" else 1,
                format_func=lambda x: "Local Storage" if x == "local" else "Google Drive"
            )
            
            submitted = st.form_submit_button("Save Preferences")
            if submitted:
                new_preferences = {
                    'theme': theme,
                    'default_model': default_model,
                    'auto_save': auto_save,
                    'email_notifications': email_notifications,
                    'save_location': save_location,
                    'drive_connected': user.get('drive_connected', False)
                }
                
                # Save to database
                success = auth_service.save_user_preferences(user['id'], new_preferences)
                
                if success:
                    # Update session state
                    for key, value in new_preferences.items():
                        user[key] = value
                    st.session_state.user = user
                    
                    st.success("Preferences saved successfully!")
                else:
                    st.error("Failed to save preferences")
        
        # Analysis History
        st.subheader("Recent Analysis History")
        history = user.get('analysis_history', [])
        if history:
            for analysis in history[:5]:
                with st.expander(f"Analysis {analysis['id']} - {analysis['created_at']}"):
                    st.write(f"**Model:** {analysis.get('model_used', 'N/A')}")
                    st.write(f"**Years:** {analysis.get('years', 'N/A')}")
                    st.write(f"**Location:** {analysis.get('location_lat', 'N/A')}, {analysis.get('location_lon', 'N/A')}")
                    if analysis.get('drive_file_id'):
                        st.write("Saved to Google Drive")
        else:
            st.info("No analysis history yet.")
    
    st.markdown("---")
    
    # Google Drive Integration Section
    st.subheader("Google Drive Integration")
    
    if user.get('drive_connected'):
        st.success("Google Drive is connected to your account")
        
        # Show recent files
        st.write("**Recent Files in Google Drive:**")
        files = drive_service.list_user_files(user['id'])
        if files:
            for file in files:
                st.write(f"- [{file['name']}]({file['webViewLink']}) ({file['createdTime'][:10]})")
        else:
            st.info("No files found in Google Drive")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Refresh File List", key="refresh_drive_btn"):
                st.rerun()
        
        with col2:
            if st.button("Disconnect Google Drive", key="disconnect_drive_btn"):
                if drive_service.disconnect_drive(user['id']):
                    user['drive_connected'] = False
                    st.session_state.user = user
                    auth_service.save_user_preferences(user['id'], user)
                    st.rerun()
    else:
        st.info("Connect Google Drive to automatically save your analysis reports and access them from anywhere.")
        
        if st.button("Connect Google Drive", key="connect_drive_btn"):
            if drive_service.authenticate(user['id']):
                st.success("Google Drive authentication initiated!")
            else:
                st.info("Please complete the Google authentication in the new window.")
    
    st.markdown("---")
    
    # Account actions
    st.subheader("Account Actions")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Export My Data", key="export_data_btn"):
            st.info("Data export feature coming soon")
    
    with col2:
        if st.button("Clear History", key="clear_history_btn"):
            if st.checkbox("I understand this will delete all my analysis history"):
                st.warning("History clearing feature coming soon")
