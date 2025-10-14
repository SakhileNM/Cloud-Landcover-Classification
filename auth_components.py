# auth_components.py
import streamlit as st
from auth_service import AuthService
import requests

def init_auth_service():
    if 'auth_service' not in st.session_state:
        st.session_state.auth_service = AuthService()
    return st.session_state.auth_service

def show_login_form():
    auth_service = init_auth_service()
    
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
    .oauth-buttons {
        display: flex;
        flex-direction: column;
        gap: 10px;
        margin: 20px 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="login-container">', unsafe_allow_html=True)
        
        st.title("Login")
        
        # Email/Password Login
        with st.form("login_form"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login")
            
            if submit:
                user = auth_service.authenticate_user(email, password)
                if user:
                    st.session_state.user = user
                    st.session_state.authenticated = True
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error("Invalid email or password")
        
        st.markdown("---")
        st.subheader("Or login with:")
        
        # OAuth Buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Google", use_container_width=True):
                auth_url, state = auth_service.get_google_auth_url()
                st.session_state.oauth_state = state
                st.markdown(f'<meta http-equiv="refresh" content="0; url={auth_url}">', unsafe_allow_html=True)
        
        with col2:
            if st.button("Microsoft", use_container_width=True):
                auth_url = auth_service.get_microsoft_auth_url()
                st.markdown(f'<meta http-equiv="refresh" content="0; url={auth_url}">', unsafe_allow_html=True)
        
        with col3:
            if st.button("Apple", use_container_width=True):
                st.info("Apple OAuth coming soon!")
        
        st.markdown("---")
        st.markdown("Don't have an account? [Sign up here](#signup)")
        
        st.markdown('</div>', unsafe_allow_html=True)

def show_signup_form():
    auth_service = init_auth_service()
    
    with st.container():
        st.markdown('<div class="login-container">', unsafe_allow_html=True)
        
        st.title("Sign Up")
        
        with st.form("signup_form"):
            col1, col2 = st.columns(2)
            with col1:
                full_name = st.text_input("Full Name")
            with col2:
                username = st.text_input("Username")
            
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            
            submit = st.form_submit_button("Create Account")
            
            if submit:
                if password != confirm_password:
                    st.error("Passwords do not match!")
                elif len(password) < 8:
                    st.error("Password must be at least 8 characters long")
                else:
                    success, message = auth_service.register_user(
                        email=email,
                        password=password,
                        username=username,
                        full_name=full_name
                    )
                    if success:
                        st.success(message)
                    else:
                        st.error(message)
        
        st.markdown("---")
        st.markdown("Already have an account? [Login here](#login)")
        
        st.markdown('</div>', unsafe_allow_html=True)

def show_user_profile():
    auth_service = init_auth_service()
    user = st.session_state.user
    
    st.title("User Profile")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Profile Info")
        st.write(f"**Name:** {user.full_name}")
        st.write(f"**Email:** {user.email}")
        st.write(f"**Username:** {user.username}")
        if user.oauth_provider:
            st.write(f"**Connected via:** {user.oauth_provider.title()}")
        st.write(f"**Member since:** {user.created_at.strftime('%Y-%m-%d')}")
    
    with col2:
        st.subheader("Settings")
        
        with st.form("user_settings"):
            # Theme selection
            theme = st.selectbox(
                "Theme",
                ["light", "dark", "auto"],
                index=["light", "dark", "auto"].index(user.theme)
            )
            
            # Save location
            save_location = st.radio(
                "Save plots to:",
                ["local", "google_drive", "onedrive"],
                format_func=lambda x: {
                    "local": "Local Storage",
                    "google_drive": "Google Drive",
                    "onedrive": "Microsoft OneDrive"
                }[x],
                index=["local", "google_drive", "onedrive"].index(user.save_location)
            )
            
            # Local path (only show if local storage selected)
            if save_location == "local":
                local_path = st.text_input("Local path", value=user.local_path or "")
            else:
                local_path = user.local_path
            
            # OAuth connection status
            st.subheader("Connected Accounts")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                google_connected = user.oauth_provider == "google"
                st.write(f"Google: {'Connected' if google_connected else '❌ Not connected'}")
            
            with col2:
                ms_connected = user.oauth_provider == "microsoft"
                st.write(f"Microsoft: {'Connected' if ms_connected else '❌ Not connected'}")
            
            with col3:
                apple_connected = user.oauth_provider == "apple"
                st.write(f"Apple: {'Connected' if apple_connected else '❌ Not connected'}")
            
            if st.form_submit_button("Save Settings"):
                success, message = auth_service.update_user_settings(user.id, {
                    "theme": theme,
                    "save_location": save_location,
                    "local_path": local_path
                })
                if success:
                    st.success(message)
                    # Update session state
                    st.session_state.user = auth_service.get_user_by_id(user.id)
                else:
                    st.error(message)
    
    st.markdown("---")
    if st.button("Logout"):
        st.session_state.authenticated = False
        st.session_state.user = None
        st.rerun()

def handle_oauth_callback():
    auth_service = init_auth_service()
    
    query_params = st.experimental_get_query_params()
    
    if 'code' in query_params and 'state' in query_params:
        # Google OAuth callback
        code = query_params['code'][0]
        state = query_params['state'][0]
        
        if 'oauth_state' in st.session_state and st.session_state.oauth_state == state:
            user, message = auth_service.handle_google_callback(code, state)
            if user:
                st.session_state.user = user
                st.session_state.authenticated = True
                st.success("Google login successful!")
                # Clear query params
                st.experimental_set_query_params()
                st.rerun()
            else:
                st.error(message)
    
    elif 'code' in query_params:
        # Microsoft OAuth callback
        code = query_params['code'][0]
        user, message = auth_service.handle_microsoft_callback(code)
        if user:
            st.session_state.user = user
            st.session_state.authenticated = True
            st.success("Microsoft login successful!")
            st.experimental_set_query_params()
            st.rerun()
        else:
            st.error(message)
    
    elif 'token' in query_params:
        # Email verification callback
        token = query_params['token'][0]
        success, message = auth_service.verify_user_email(token)
        if success:
            st.success(message)
        else:
            st.error(message)
        st.experimental_set_query_params()
