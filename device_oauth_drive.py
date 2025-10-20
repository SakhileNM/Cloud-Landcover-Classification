# device_oauth_drive.py
import os
import time
import requests
import json
import streamlit as st
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

# Config from env
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")  # optional for device flow but include if your client has one
SCOPES = "openid email profile https://www.googleapis.com/auth/drive.file"
DEVICE_CODE_URL = "https://oauth2.googleapis.com/device/code"
TOKEN_URL = "https://oauth2.googleapis.com/token"

# storage functions: adapt to your DB or file layout; below uses filesystem
CREDENTIALS_DIR = os.getenv("GOOGLE_CREDS_DIR", "/app/data/google_tokens")
os.makedirs(CREDENTIALS_DIR, exist_ok=True)

def _save_token_for_user(user_id, token_json):
    path = f"{CREDENTIALS_DIR}/{user_id}_google_token.json"
    with open(path, "w") as f:
        json.dump(token_json, f)
    return path

def _load_token_for_user(user_id):
    path = f"{CREDENTIALS_DIR}/{user_id}_google_token.json"
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None

def start_device_flow_for_user(user_id):
    """Kick off device authorization. Returns dict containing user_code, verification_url, device_code, interval, expires_in"""
    if not GOOGLE_CLIENT_ID:
        raise RuntimeError("Set GOOGLE_CLIENT_ID env var")
    data = {
        "client_id": GOOGLE_CLIENT_ID,
        "scope": SCOPES
    }
    r = requests.post(DEVICE_CODE_URL, data=data, timeout=10)
    r.raise_for_status()
    device_info = r.json()
    # store device_info in session so you can poll
    st.session_state[f"google_device_{user_id}"] = device_info
    return device_info

def poll_device_token(user_id, timeout=600):
    """Polls Google's token endpoint until token returned or timeout (seconds). Returns token JSON on success."""
    key = f"google_device_{user_id}"
    device_info = st.session_state.get(key)
    if not device_info:
        raise RuntimeError("No device flow started for this user")
    device_code = device_info["device_code"]
    interval = int(device_info.get("interval", 5))
    expires_in = int(device_info.get("expires_in", 600))
    deadline = time.time() + min(timeout, expires_in)

    while time.time() < deadline:
        data = {
            "client_id": GOOGLE_CLIENT_ID,
            "device_code": device_code,
            "grant_type": "urn:ietf:params:oauth:grant-type:device_code"
        }
        if GOOGLE_CLIENT_SECRET:
            data["client_secret"] = GOOGLE_CLIENT_SECRET

        r = requests.post(TOKEN_URL, data=data, timeout=10)
        # If user hasn't yet authorized, we get error 'authorization_pending' or 'slow_down'
        if r.status_code == 200:
            token_json = r.json()
            # token_json contains: access_token, refresh_token (devices always return refresh), expires_in, scope, token_type
            _save_token_for_user(user_id, token_json)
            # clear session device state
            st.session_state.pop(key, None)
            return token_json
        else:
            try:
                err = r.json()
            except Exception:
                r.raise_for_status()
            errcode = err.get("error")
            if errcode in ("authorization_pending",):
                time.sleep(interval)
                continue
            elif errcode == "slow_down":
                # back off a bit more
                interval += 5
                time.sleep(interval)
                continue
            else:
                # any other error -> bail
                raise RuntimeError(f"Device flow error: {err}")
    raise TimeoutError("Device flow polling timed out")

def credentials_from_saved_token(user_id):
    tok = _load_token_for_user(user_id)
    if not tok:
        return None
    creds = Credentials(
        token=tok["access_token"],
        refresh_token=tok.get("refresh_token"),
        token_uri=TOKEN_URL,
        client_id=GOOGLE_CLIENT_ID,
        client_secret=GOOGLE_CLIENT_SECRET,
        scopes=tok.get("scope").split()
    )
    # google oauth2 credentials object handles refresh when used with googleapiclient if you call build(...)
    return creds

def build_drive_service_for_user(user_id):
    creds = credentials_from_saved_token(user_id)
    if not creds:
        return None
    service = build("drive", "v3", credentials=creds)
    return service

# Example Streamlit wiring
def streamlit_connect_button(user_id):
    st.write("Connect Google Drive (device flow) — no HTTPS required on server")
    if st.button("Start Google Device Authorization"):
        info = start_device_flow_for_user(user_id)
        st.session_state[f"visible_device_info_{user_id}"] = info
        st.experimental_rerun()

    device_info = st.session_state.get(f"visible_device_info_{user_id}")
    if device_info:
        st.markdown("**Open this link on your phone or desktop and enter the code:**")
        st.code(device_info["user_code"])
        st.markdown(f"[Open verification page]({device_info['verification_url']}) (opens in a new tab)")
        if st.button("Poll for token now"):
            try:
                tok = poll_device_token(user_id)
                st.success("Connected to Google Drive")
                st.json(tok)
            except Exception as e:
                st.error(f"Error polling token: {e}")
