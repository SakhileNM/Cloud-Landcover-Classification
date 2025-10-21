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
# Use device-code flow helpers instead of web redirect flow
from device_oauth_drive import (
    streamlit_connect_button,
    build_drive_service_for_user,
    credentials_from_saved_token
)


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
        self.init_database()

    def init_database(self):
        """Initialize SQLite database for user preferences and history"""
        with get_db_connection() as conn:
            # User preferences table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS user_preferences (
                    user_id TEXT PRIMARY KEY,
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
                    '''SELECT default_model, auto_save, email_notifications, save_location, drive_connected 
                       FROM user_preferences WHERE user_id = ?''',
                    (user_id,)
                )
                result = cursor.fetchone()
                if result:
                    return {
                        # result columns correspond to the SELECT order:
                        # default_model (0), auto_save (1), email_notifications (2), save_location (3), drive_connected (4)
                        'default_model': result[0],
                        'auto_save': bool(result[1]),
                        'email_notifications': bool(result[2]),
                        'save_location': result[3],
                        'drive_connected': bool(result[4])
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
                    (user_id, default_model, auto_save, email_notifications, save_location, drive_connected, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ''', (
                    user_id,
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
        max-width: 800px;
        margin: 0 auto;
        padding: 2rem;
    }
    .google-login-btn {
        display: block;
        margin: 2rem auto;
        background: white;
        color: #4285F4;
        border: 2px solid #4285F4;
        border-radius: 25px;
        padding: 12px 30px;
        font-size: 16px;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.3s ease;
        text-align: center;
        width: 220px;
        text-decoration: none;
    }
    .google-login-btn:hover {
        background: #4285F4;
        color: white;
    }
    .project-description {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin: 2rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .project-title {
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="login-container">', unsafe_allow_html=True)
        
        # Project description
        st.markdown("""
<div class="project-description">
    <div class="project-title">Geospatial Landcover Classification Platform</div>
    <p>This research project was developed by Sakhile Mkhize during his BEngTech Honours in Computer Engineering at CPUT. The research demonstrates how complex geospatial analysis tasks can be transformed into accessible, efficient processes through modern machine learning and cloud technologies.</p>
    
    <p>The platform features dual machine learning models - Random Forest and Gradient Boosting - trained on Western Cape regional data. Leveraging Digital Earth Africa's STAC dataset and deployed via Docker on Oracle Cloud infrastructure, the application enables stakeholders to accurately classify landcover using both Landsat and Sentinel satellite imagery from 1995 to 2023.</p>
    
    <p>Users can select specific locations, choose between individual models or automated model selection, analyze multiple years, and receive comprehensive results via email notifications or downloadable PDF reports stored locally or in Google Drive.</p>
</div>
        """, unsafe_allow_html=True)
        
        # Centered Google login button
        auth_url = st.session_state.auth0_service.get_auth_url(include_drive_scope=False)

        st.markdown("**DEBUG auth_url (Auth0 → Google request):**")
        st.code(auth_url)

        # Add debug info
        st.write("**Current session state:**", {
            'authenticated': st.session_state.get('authenticated', False),
            'user': bool(st.session_state.get('user')),
            'query_params': dict(st.query_params)
        })

        google_button_html = f'''
        <a href="{auth_url}" class="google-login-btn" style="text-decoration: none;">
            <div style="display: flex; align-items: center; justify-content: center; gap: 10px;">
                <svg width="18" height="18" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
                    <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/>
                    <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/>
                    <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/>
                    <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/>
                </svg>
                Login with Google
            </div>
        </a>
        '''
        st.markdown(google_button_html, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

def handle_auth0_callback():
    """Handle Auth0 callback after login"""
    
    # Handle Auth0 callback
    query_params = st.query_params
    
    # Debug: show what we received
    st.write("DEBUG: Query params received:", dict(query_params))
    
    if 'code' in query_params:
        code = query_params['code']
        
        # If code is a list, take the first element
        if isinstance(code, list):
            code = code[0]
            
        st.write(f"DEBUG: Processing Auth0 callback with code: {code[:20]}...")
        
        if 'auth0_service' not in st.session_state:
            st.session_state.auth0_service = Auth0Service()
        
        auth_service = st.session_state.auth0_service
        
        # Exchange code for tokens
        tokens = auth_service.get_token(code)
        if tokens:
            st.write("DEBUG: Token exchange successful")
            # Get user info
            user_info = auth_service.get_user_info(tokens['access_token'])
            
            if user_info:
                st.write(f"DEBUG: User info received for: {user_info.get('email')}")
                # Load or create user preferences
                user_id = user_info['sub']
                preferences = auth_service.get_user_preferences(user_id)
                
                if not preferences:
                    # Create default preferences for new user
                    preferences = {
                        'default_model': 'Random Forest',
                        'auto_save': True,
                        'email_notifications': False,
                        'save_location': 'local',
                        'drive_connected': False
                    }
                    auth_service.save_user_preferences(user_id, preferences)
                
                # Check if Google Drive device-flow token exists (device_oauth_drive default dir)
                creds_dir = os.getenv("GOOGLE_CREDS_DIR", "/app/data/google_tokens")
                drive_connected = os.path.exists(os.path.join(creds_dir, f"{user_id}_google_token.json"))
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
            else:
                st.error("DEBUG: Failed to get user info from Auth0")
        else:
            st.error("DEBUG: Token exchange failed")
    else:
        st.write("DEBUG: No code in query params or already processed")

def show_auth0_profile():
    user = st.session_state.user
    auth_service = st.session_state.auth0_service
    
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
        
        # Show recent files using device-flow saved credentials (build Drive service)
        st.write("**Recent Files in Google Drive (most recent 10):**")
        try:
            drive_service = build_drive_service_for_user(user['id'])
            if drive_service:
                resp = drive_service.files().list(pageSize=10, fields="files(id,name,webViewLink,createdTime)").execute()
                files = resp.get('files', [])
                if files:
                    for file in files:
                        created = file.get('createdTime', '')[:10]
                        st.write(f"- [{file['name']}]({file.get('webViewLink')}) ({created})")
                else:
                    st.info("No files found in Google Drive")
            else:
                st.warning("No Google Drive credentials found for this account.")
        except Exception as e:
            st.error(f"Could not fetch Drive files: {e}")
    
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Refresh File List", key="refresh_drive_btn"):
                st.experimental_rerun()
        
        with col2:
            if st.button("Disconnect Google Drive", key="disconnect_drive_btn"):
                # delete the device-flow token file and update prefs
                creds_dir = os.getenv("GOOGLE_CREDS_DIR", "/app/data/google_tokens")
                token_path = os.path.join(creds_dir, f"{user['id']}_google_token.json")
                try:
                    if os.path.exists(token_path):
                        os.remove(token_path)
                    user['drive_connected'] = False
                    st.session_state.user = user
                    auth_service.save_user_preferences(user['id'], user)
                    st.success("Google Drive disconnected for your account.")
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"Disconnect failed: {e}")

    else:
        st.info("Connect Google Drive to automatically save your analysis reports and access them from anywhere.")
    
        # Use device-flow connect UI (no redirect required)
        streamlit_connect_button(user['id'])

    
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
