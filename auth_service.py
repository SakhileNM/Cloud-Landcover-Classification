# auth_service.py
import streamlit as st
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta
import uuid
from models import User, UserSession, init_db
from auth_config import AuthConfig
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import requests
from authlib.integrations.requests_client import OAuth2Session
import msal
import json

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class AuthService:
    def __init__(self):
        self.SessionLocal = init_db()
        self.config = AuthConfig()
    
    def verify_password(self, plain_password, hashed_password):
        return pwd_context.verify(plain_password, hashed_password)
    
    def get_password_hash(self, password):
        return pwd_context.hash(password)
    
    def create_access_token(self, data: dict, expires_delta: timedelta = None):
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=15)
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, self.config.SECRET_KEY, algorithm=self.config.ALGORITHM)
        return encoded_jwt
    
    def verify_token(self, token: str):
        try:
            payload = jwt.decode(token, self.config.SECRET_KEY, algorithms=[self.config.ALGORITHM])
            return payload
        except JWTError:
            return None
    
    def send_verification_email(self, email: str, token: str):
        """Send verification email to user"""
        verification_url = f"{st.secrets.get('APP_URL', 'http://localhost:8501')}/verify?token={token}"
        
        message = MimeMultipart("alternative")
        message["Subject"] = "Verify Your Email - Geospatial Classification App"
        message["From"] = self.config.FROM_EMAIL
        message["To"] = email
        
        html = f"""
        <html>
          <body>
            <h2>Welcome to the Cloud-Based Geospatial Classification App!</h2>
            <p>Please verify your email address by clicking the link below:</p>
            <p><a href="{verification_url}">Verify Email Address</a></p>
            <p>If you didn't create an account, please ignore this email.</p>
            <br>
            <p>Best regards,<br>Geospatial Classification Team</p>
          </body>
        </html>
        """
        
        message.attach(MimeText(html, "html"))
        
        try:
            with smtplib.SMTP(self.config.SMTP_SERVER, self.config.SMTP_PORT) as server:
                server.starttls()
                server.login(self.config.SMTP_USERNAME, self.config.SMTP_PASSWORD)
                server.send_message(message)
            return True
        except Exception as e:
            print(f"Email sending failed: {e}")
            return False
    
    def register_user(self, email: str, password: str, username: str, full_name: str = None):
        db = self.SessionLocal()
        try:
            # Check if user already exists
            if db.query(User).filter((User.email == email) | (User.username == username)).first():
                return False, "User already exists"
            
            # Create new user
            verification_token = str(uuid.uuid4())
            user = User(
                email=email,
                username=username,
                full_name=full_name or username,
                hashed_password=self.get_password_hash(password),
                verification_token=verification_token,
                is_active=True,  # Auto-activate for now, can require email verification
                is_verified=False
            )
            
            db.add(user)
            db.commit()
            
            # Send verification email
            if self.send_verification_email(email, verification_token):
                return True, "User registered successfully. Please check your email for verification."
            else:
                return True, "User registered successfully, but verification email failed to send."
                
        except Exception as e:
            db.rollback()
            return False, f"Registration failed: {str(e)}"
        finally:
            db.close()
    
    def verify_user_email(self, token: str):
        db = self.SessionLocal()
        try:
            user = db.query(User).filter(User.verification_token == token).first()
            if user:
                user.is_verified = True
                user.verification_token = None
                db.commit()
                return True, "Email verified successfully!"
            return False, "Invalid verification token"
        except Exception as e:
            db.rollback()
            return False, f"Verification failed: {str(e)}"
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
    
    def get_google_auth_url(self):
        oauth = OAuth2Session(
            self.config.GOOGLE_CLIENT_ID,
            redirect_uri=self.config.GOOGLE_REDIRECT_URI,
            scope=["openid", "email", "profile"]
        )
        authorization_url, state = oauth.create_authorization_url(
            "https://accounts.google.com/o/oauth2/v2/auth",
            access_type="offline",
            prompt="select_account"
        )
        return authorization_url, state
    
    def handle_google_callback(self, code: str, state: str):
        try:
            oauth = OAuth2Session(
                self.config.GOOGLE_CLIENT_ID,
                redirect_uri=self.config.GOOGLE_REDIRECT_URI,
                state=state
            )
            token = oauth.fetch_token(
                "https://oauth2.googleapis.com/token",
                client_secret=self.config.GOOGLE_CLIENT_SECRET,
                code=code
            )
            
            # Get user info
            user_info = oauth.get("https://www.googleapis.com/oauth2/v1/userinfo").json()
            
            return self.create_or_get_oauth_user(
                email=user_info["email"],
                oauth_provider="google",
                oauth_id=user_info["id"],
                full_name=user_info.get("name", ""),
                username=user_info["email"].split("@")[0]
            )
        except Exception as e:
            return None, f"Google OAuth failed: {str(e)}"
    
    def get_microsoft_auth_url(self):
        msal_app = msal.ConfidentialClientApplication(
            self.config.MICROSOFT_CLIENT_ID,
            client_credential=self.config.MICROSOFT_CLIENT_SECRET,
            authority=f"https://login.microsoftonline.com/{self.config.MICROSOFT_TENANT}"
        )
        
        auth_url = msal_app.get_authorization_request_url(
            scopes=["https://graph.microsoft.com/User.Read"],
            redirect_uri=self.config.MICROSOFT_REDIRECT_URI
        )
        return auth_url
    
    def handle_microsoft_callback(self, code: str):
        try:
            msal_app = msal.ConfidentialClientApplication(
                self.config.MICROSOFT_CLIENT_ID,
                client_credential=self.config.MICROSOFT_CLIENT_SECRET,
                authority=f"https://login.microsoftonline.com/{self.config.MICROSOFT_TENANT}"
            )
            
            result = msal_app.acquire_token_by_authorization_code(
                code,
                scopes=["https://graph.microsoft.com/User.Read"],
                redirect_uri=self.config.MICROSOFT_REDIRECT_URI
            )
            
            if "access_token" in result:
                # Get user info from Microsoft Graph
                graph_data = requests.get(
                    "https://graph.microsoft.com/v1.0/me",
                    headers={"Authorization": f"Bearer {result['access_token']}"}
                ).json()
                
                return self.create_or_get_oauth_user(
                    email=graph_data["mail"] or graph_data["userPrincipalName"],
                    oauth_provider="microsoft",
                    oauth_id=graph_data["id"],
                    full_name=graph_data.get("displayName", ""),
                    username=graph_data["userPrincipalName"].split("@")[0]
                )
            else:
                return None, "Microsoft authentication failed"
        except Exception as e:
            return None, f"Microsoft OAuth failed: {str(e)}"
    
    def create_or_get_oauth_user(self, email: str, oauth_provider: str, oauth_id: str, full_name: str, username: str):
        db = self.SessionLocal()
        try:
            # Check if user exists by email or oauth_id
            user = db.query(User).filter(
                (User.email == email) | 
                ((User.oauth_provider == oauth_provider) & (User.oauth_id == oauth_id))
            ).first()
            
            if user:
                # Update last login
                user.last_login = datetime.utcnow()
                db.commit()
                return user, "Login successful"
            else:
                # Create new user
                user = User(
                    email=email,
                    username=username,
                    full_name=full_name,
                    oauth_provider=oauth_provider,
                    oauth_id=oauth_id,
                    is_active=True,
                    is_verified=True  # OAuth users are automatically verified
                )
                db.add(user)
                db.commit()
                return user, "User created successfully"
        except Exception as e:
            db.rollback()
            return None, f"OAuth user creation failed: {str(e)}"
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
    
    def get_user_by_id(self, user_id: str):
        db = self.SessionLocal()
        try:
            return db.query(User).filter(User.id == user_id).first()
        finally:
            db.close()
