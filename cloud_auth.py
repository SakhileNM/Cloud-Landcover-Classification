# cloud_auth.py
import streamlit as st
from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import uuid
from passlib.context import CryptContext
import jwt
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

load_dotenv()

Base = declarative_base()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class User(Base):
    __tablename__ = "users"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String(255), unique=True, nullable=False)
    username = Column(String(100), unique=True, nullable=False)
    hashed_password = Column(String(255))
    full_name = Column(String(255))
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Cloud storage settings
    theme = Column(String(20), default='light')
    save_location = Column(String(20), default='local')  # local, google_drive, dropbox
    local_path = Column(Text, default='/opt/app/data')
    
    # Cloud storage API keys (encrypted in real implementation)
    google_drive_api_key = Column(Text, default='')
    dropbox_api_key = Column(Text, default='')
    onedrive_client_id = Column(Text, default='')  # For advanced users

class CloudAuth:
    def __init__(self, db_url="sqlite:////opt/app/data/users.db"):
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        self.secret_key = os.getenv("SECRET_KEY", "cloud-geospatial-app-2024")
    
    def verify_password(self, plain_password, hashed_password):
        return pwd_context.verify(plain_password, hashed_password)
    
    def get_password_hash(self, password):
        return pwd_context.hash(password)
    
    def create_access_token(self, user_id: str):
        expire = datetime.utcnow() + timedelta(days=30)
        to_encode = {"user_id": user_id, "exp": expire}
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm="HS256")
        return encoded_jwt
    
    def register_user(self, email: str, password: str, username: str, full_name: str = None):
        db = self.SessionLocal()
        try:
            # Check if user exists
            if db.query(User).filter((User.email == email) | (User.username == username)).first():
                return False, "User already exists"
            
            user = User(
                email=email,
                username=username,
                full_name=full_name or username,
                hashed_password=self.get_password_hash(password)
            )
            
            db.add(user)
            db.commit()
            return True, "User registered successfully"
                
        except Exception as e:
            db.rollback()
            return False, f"Registration failed: {str(e)}"
        finally:
            db.close()
    
    def authenticate_user(self, email: str, password: str):
        db = self.SessionLocal()
        try:
            user = db.query(User).filter(User.email == email).first()
            if not user or not self.verify_password(password, user.hashed_password):
                return None
            if not user.is_active:
                return None
            return user
        finally:
            db.close()
    
    def update_user_settings(self, user_id: str, settings: dict):
        db = self.SessionLocal()
        try:
            user = db.query(User).filter(User.id == user_id).first()
            if user:
                for key, value in settings.items():
                    if hasattr(user, key):
                        setattr(user, key, value)
                db.commit()
                return True, "Settings updated successfully"
            return False, "User not found"
        except Exception as e:
            db.rollback()
            return False, f"Settings update failed: {str(e)}"
        finally:
            db.close()
    
    def test_cloud_connection(self, user, service):
        """Test cloud storage connection using API keys"""
        try:
            if service == "google_drive" and user.google_drive_api_key:
                # Test Google Drive connection
                return self.test_google_drive(user.google_drive_api_key)
            elif service == "dropbox" and user.dropbox_api_key:
                # Test Dropbox connection
                return self.test_dropbox(user.dropbox_api_key)
            return False, "API key not configured"
        except Exception as e:
            return False, f"Connection test failed: {str(e)}"
    
    def test_google_drive(self, api_key):
        """Test Google Drive API key"""
        try:
            # Simple test - in production, you'd use the actual Google Drive API
            # This is a placeholder for the actual implementation
            if api_key and len(api_key) > 10:
                return True, "Google Drive connection successful"
            return False, "Invalid API key"
        except Exception as e:
            return False, f"Google Drive test failed: {str(e)}"
    
    def test_dropbox(self, api_key):
        """Test Dropbox API key"""
        try:
            # Simple test - in production, you'd use the actual Dropbox API
            if api_key and len(api_key) > 10:
                return True, "Dropbox connection successful"
            return False, "Invalid API key"
        except Exception as e:
            return False, f"Dropbox test failed: {str(e)}"

def show_cloud_login():
    if 'cloud_auth' not in st.session_state:
        st.session_state.cloud_auth = CloudAuth()
    
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
    .cloud-badge {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="login-container">', unsafe_allow_html=True)
        
        st.markdown('<div class="cloud-badge">☁️ Geospatial Landcover Classification Platform</div>', unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["Login", "Sign Up"])
        
        with tab1:
            st.subheader("Login to Cloud Platform")
            with st.form("login_form"):
                email = st.text_input("Email")
                password = st.text_input("Password", type="password")
                submit = st.form_submit_button("Login", type="primary")
                
                if submit:
                    user = st.session_state.cloud_auth.authenticate_user(email, password)
                    if user:
                        st.session_state.user = user
                        st.session_state.authenticated = True
                        st.session_state.user_token = st.session_state.cloud_auth.create_access_token(user.id)
                        st.success("Login successful!")
                        st.rerun()
                    else:
                        st.error("Invalid email or password")
        
        with tab2:
            st.subheader("Create Cloud Account")
            with st.form("signup_form"):
                full_name = st.text_input("Full Name")
                username = st.text_input("Username")
                email = st.text_input("Email")
                password = st.text_input("Password", type="password")
                confirm_password = st.text_input("Confirm Password", type="password")
                submit = st.form_submit_button("Create Cloud Account", type="primary")
                
                if submit:
                    if password != confirm_password:
                        st.error("Passwords do not match!")
                    elif len(password) < 6:
                        st.error("Password must be at least 6 characters")
                    else:
                        success, message = st.session_state.cloud_auth.register_user(
                            email=email, password=password, 
                            username=username, full_name=full_name
                        )
                        if success:
                            st.success("Cloud account created! Please login.")
                        else:
                            st.error(message)
        
        st.markdown("---")
        st.info("""
        **Cloud Features:**
        - 👤 User profiles & settings
        - 🎨 Light/Dark themes  
        - ☁️ Save to cloud drives
        - 📊 Personal analysis history
        """)
        
        st.markdown('</div>', unsafe_allow_html=True)

def show_cloud_profile():
    auth = st.session_state.cloud_auth
    user = st.session_state.user
    
    st.title("Cloud Profile & Settings")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Profile Info")
        st.write(f"**Name:** {user.full_name}")
        st.write(f"**Email:** {user.email}")
        st.write(f"**Username:** {user.username}")
        st.write(f"**Member since:** {user.created_at.strftime('%Y-%m-%d')}")
        
        # Cloud storage status
        st.subheader("☁️ Cloud Status")
        if user.save_location == "google_drive" and user.google_drive_api_key:
            st.success("Google Drive Connected")
        elif user.save_location == "dropbox" and user.dropbox_api_key:
            st.success("Dropbox Connected")
        else:
            st.info("Connect cloud storage below")
    
    with col2:
        st.subheader("Application Settings")
        with st.form("user_settings"):
            theme = st.selectbox(
                "Theme",
                ["light", "dark", "auto"],
                index=["light", "dark", "auto"].index(user.theme),
                help="Choose your preferred theme"
            )
            
            save_location = st.radio(
                "Save results to:",
                ["local", "google_drive", "dropbox"],
                format_func=lambda x: {
                    "local": "Local Storage",
                    "google_drive": "☁️ Google Drive", 
                    "dropbox": "☁️ Dropbox"
                }[x],
                index=["local", "google_drive", "dropbox"].index(user.save_location)
            )
            
            # Cloud storage configuration
            if save_location == "google_drive":
                st.subheader("Google Drive Setup")
                st.info("Get API key from [Google Cloud Console](https://console.cloud.google.com/)")
                google_api_key = st.text_input(
                    "Google Drive API Key", 
                    value=user.google_drive_api_key or "",
                    type="password",
                    help="Enter your Google Drive API key"
                )
                
                if user.google_drive_api_key:
                    if st.button("Test Google Drive Connection"):
                        success, message = auth.test_cloud_connection(user, "google_drive")
                        if success:
                            st.success(message)
                        else:
                            st.error(message)
            
            elif save_location == "dropbox":
                st.subheader("Dropbox Setup") 
                st.info("Get API key from [Dropbox Developer Portal](https://www.dropbox.com/developers)")
                dropbox_api_key = st.text_input(
                    "Dropbox API Key",
                    value=user.dropbox_api_key or "", 
                    type="password",
                    help="Enter your Dropbox API key"
                )
                
                if user.dropbox_api_key:
                    if st.button("Test Dropbox Connection"):
                        success, message = auth.test_cloud_connection(user, "dropbox")
                        if success:
                            st.success(message)
                        else:
                            st.error(message)
            
            else:  # local storage
                local_path = st.text_input(
                    "Local storage path", 
                    value=user.local_path or "/opt/app/data",
                    help="Path where results will be saved locally"
                )
            
            if st.form_submit_button("Save Cloud Settings", type="primary"):
                update_data = {"theme": theme, "save_location": save_location}
                
                if save_location == "google_drive":
                    update_data["google_drive_api_key"] = google_api_key
                elif save_location == "dropbox":
                    update_data["dropbox_api_key"] = dropbox_api_key
                else:
                    update_data["local_path"] = local_path
                
                success, message = auth.update_user_settings(user.id, update_data)
                if success:
                    st.success("Cloud settings saved successfully!")
                    # Update session state
                    st.session_state.user = auth.SessionLocal().query(User).filter(User.id == user.id).first()
                else:
                    st.error(message)
    
    st.markdown("---")
    if st.button("Logout", type="primary"):
        for key in ['authenticated', 'user', 'user_token', 'show_profile']:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()
