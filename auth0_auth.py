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

load_dotenv()

@contextmanager
def get_db_connection():
    """Context manager for database connections"""
    # Ensure data directory exists
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
                    theme TEXT DEFAULT 'light',
                    default_model TEXT DEFAULT 'Random Forest',
                    auto_save BOOLEAN DEFAULT 1,
                    email_notifications BOOLEAN DEFAULT 0,
                    save_location TEXT DEFAULT 'local',
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
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES user_preferences (user_id)
                )
            ''')
            
            # User session table for persistence
            conn.execute('''
                CREATE TABLE IF NOT EXISTS user_sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    session_data TEXT,
                    last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES user_preferences (user_id)
                )
            ''')
            
            conn.commit()

    def get_auth_url(self, include_drive_scope=False):
        """Generate Auth0 authorization URL"""
        scope = "openid profile email"
        if include_drive_scope:
            scope += " https://www.googleapis.com/auth/drive.file"
        
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
                    '''SELECT theme, default_model, auto_save, email_notifications, save_location 
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
                        'save_location': result[4]
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
                    (user_id, theme, default_model, auto_save, email_notifications, save_location, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ''', (
                    user_id,
                    preferences.get('theme', 'light'),
                    preferences.get('default_model', 'Random Forest'),
                    int(preferences.get('auto_save', True)),
                    int(preferences.get('email_notifications', False)),
                    preferences.get('save_location', 'local')
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
                    (user_id, analysis_type, location_lat, location_lon, years, model_used, results_data)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    user_id,
                    analysis_data.get('analysis_type', 'landcover'),
                    analysis_data.get('lat'),
                    analysis_data.get('lon'),
                    json.dumps(analysis_data.get('years', [])),
                    analysis_data.get('model_type'),
                    json.dumps(analysis_data.get('results', {}))
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

    def save_user_session(self, user_id, session_data):
        """Save user session data"""
        try:
            session_id = hashlib.md5(f"{user_id}{datetime.now()}".encode()).hexdigest()
            expires_at = datetime.now() + timedelta(days=7)
            
            with get_db_connection() as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO user_sessions 
                    (session_id, user_id, session_data, expires_at)
                    VALUES (?, ?, ?, ?)
                ''', (
                    session_id,
                    user_id,
                    json.dumps(session_data),
                    expires_at
                ))
                conn.commit()
            return session_id
        except Exception as e:
            st.error(f"Error saving session: {e}")
            return None

    def load_user_session(self, session_id):
        """Load user session data"""
        try:
            with get_db_connection() as conn:
                cursor = conn.execute(
                    'SELECT session_data FROM user_sessions WHERE session_id = ? AND expires_at > CURRENT_TIMESTAMP',
                    (session_id,)
                )
                result = cursor.fetchone()
                if result:
                    return json.loads(result[0])
        except Exception as e:
            st.error(f"Error loading session: {e}")
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
    query_params = st.query_params
    
    if 'code' in query_params:
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
                        'save_location': 'local'
                    }
                    auth_service.save_user_preferences(user_id, preferences)
                
                # Store user in session
                st.session_state.user = {
                    'id': user_id,
                    'email': user_info['email'],
                    'name': user_info.get('name', user_info['email']),
                    'picture': user_info.get('picture', ''),
                    'auth0_data': user_info,
                    'access_token': tokens.get('access_token'),
                    'refresh_token': tokens.get('refresh_token'),
                    'drive_connected': False,
                    'member_since': datetime.now().strftime('%Y-%m-%d'),
                    # Load preferences into session
                    **preferences
                }
                
                # Initialize analysis count
                analysis_history = auth_service.get_analysis_history(user_id)
                st.session_state.user['analysis_count'] = len(analysis_history)
                st.session_state.user['analysis_history'] = analysis_history
                
                st.session_state.authenticated = True
                
                # Save session to database for persistence
                session_data = {
                    'user': st.session_state.user,
                    'authenticated': True,
                    'last_login': datetime.now().isoformat()
                }
                auth_service.save_user_session(user_id, session_data)
                
                st.success(f"Welcome {user_info.get('name', user_info['email'])}!")
                st.query_params.clear()
                st.rerun()

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
                    'save_location': save_location
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
            for analysis in history[:5]:  # Show last 5 analyses
                with st.expander(f"Analysis {analysis['id']} - {analysis['created_at']}"):
                    st.write(f"**Model:** {analysis.get('model_used', 'N/A')}")
                    st.write(f"**Years:** {analysis.get('years', 'N/A')}")
                    st.write(f"**Location:** {analysis.get('location_lat', 'N/A')}, {analysis.get('location_lon', 'N/A')}")
        else:
            st.info("No analysis history yet.")
    
    st.markdown("---")
    
    # Account actions
    st.subheader("Account Actions")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Export My Data", key="export_data_btn"):
            # Export user data functionality
            st.info("Data export feature coming soon")
    
    with col2:
        if st.button("Clear History", key="clear_history_btn"):
            if st.checkbox("I understand this will delete all my analysis history"):
                st.warning("History clearing feature coming soon")
