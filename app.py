import streamlit as st
from streamlit_folium import st_folium
import folium
from inference import init_dask_cluster, predict_for_years, create_prediction_pdf
from auth0_auth import show_auth0_login, show_auth0_profile, handle_auth0_callback
import resource
import os
from datetime import datetime
import smtplib
from email.mime.text import MIMEText  # Fixed import
from email.mime.multipart import MIMEMultipart  # Fixed import

# Set memory limits
try:
    resource.setrlimit(resource.RLIMIT_AS, (20 * 1024**3, 20 * 1024**3))
except:
    pass

# Page configuration
st.set_page_config(
    page_title="Geospatial Landcover Classification Platform",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

class EmailService:
    def __init__(self):
        self.smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("SMTP_PORT", 587))
        self.smtp_username = os.getenv("SMTP_USERNAME")
        self.smtp_password = os.getenv("SMTP_PASSWORD")
        self.from_email = os.getenv("FROM_EMAIL", "noreply@geospatialplatform.com")
    
    def send_analysis_completion_email(self, user_email, user_name, analysis_details):
        """Send email notification when analysis is completed"""
        if not all([self.smtp_username, self.smtp_password]):
            return False, "Email configuration not set up"
        
        try:
            message = MIMEMultipart("alternative")  # Fixed class name
            message["Subject"] = "Analysis Completed - Geospatial Platform"
            message["From"] = self.from_email
            message["To"] = user_email
            
            html = f"""
            <html>
              <body>
                <h2>Analysis Completed Successfully!</h2>
                <p>Hello {user_name},</p>
                <p>Your geospatial analysis has been completed successfully.</p>
                
                <h3>Analysis Details:</h3>
                <ul>
                  <li><strong>Location:</strong> {analysis_details.get('location', 'N/A')}</li>
                  <li><strong>Years Analyzed:</strong> {analysis_details.get('years', 'N/A')}</li>
                  <li><strong>Model Used:</strong> {analysis_details.get('model', 'N/A')}</li>
                  <li><strong>Completion Time:</strong> {analysis_details.get('completion_time', 'N/A')}</li>
                </ul>
                
                <p>You can view the results by logging into the platform.</p>
                
                <p>Best regards,<br>
                Geospatial Classification Team</p>
              </body>
            </html>
            """
            
            message.attach(MIMEText(html, "html"))  # Fixed class name
            
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_username, self.smtp_password)
                server.send_message(message)
            
            return True, "Email sent successfully"
        except Exception as e:
            return False, f"Failed to send email: {str(e)}"

# Custom CSS with theme support
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .user-welcome {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .status-badge {
        background: #4CAF50;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.8rem;
        margin-left: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

def save_analysis_to_history(lat, lon, years, model_type, results):
    """Save analysis to user's history"""
    if 'auth0_service' in st.session_state and st.session_state.user:
        auth_service = st.session_state.auth0_service
        user_id = st.session_state.user['id']
        
        analysis_data = {
            'analysis_type': 'landcover',
            'lat': lat,
            'lon': lon,
            'years': years,
            'model_type': model_type,
            'results': results
        }
        
        success = auth_service.save_analysis_history(user_id, analysis_data)
        if success:
            # Update analysis count in session
            st.session_state.user['analysis_count'] = st.session_state.user.get('analysis_count', 0) + 1
            
            # Reload analysis history
            history = auth_service.get_analysis_history(user_id)
            st.session_state.user['analysis_history'] = history

def main_application():
    """Main application for authenticated users"""
    
    # Apply user theme if set
    user_theme = st.session_state.user.get('theme', 'light')
    if user_theme == 'dark':
        st.markdown("""
        <style>
            .main {
                background-color: #0E1117;
                color: white;
            }
        </style>
        """, unsafe_allow_html=True)
    
    # Initialize services
    if 'dask_client' not in st.session_state:
        try:
            st.session_state.dask_client = init_dask_cluster()
        except Exception as e:
            st.warning(f"Could not start Dask cluster: {e}")
    
    if 'email_service' not in st.session_state:
        st.session_state.email_service = EmailService()

    # User welcome bar
    if st.session_state.user:
        col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
        with col1:
            user = st.session_state.user
            
            welcome_text = f"Welcome, {user['name']}! | Professional Geospatial Platform"
            if user.get('drive_connected'):
                welcome_text += " <span class='status-badge'>Google Drive Connected</span>"
            if user.get('email_notifications'):
                welcome_text += " <span class='status-badge'>Email Notifications ON</span>"
            
            st.markdown(f'<div class="user-welcome">{welcome_text}</div>', unsafe_allow_html=True)
        
        with col2:
            if st.button("Home", key="home_btn_main"):
                st.session_state.show_profile = False
                st.rerun()
        with col3:
            if st.button("Profile", key="profile_btn_main"):
                st.session_state.show_profile = True
                st.rerun()
        with col4:
            if st.button("Logout", key="logout_btn_main"):
                for key in ['authenticated', 'user', 'access_token', 'show_profile']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()

    # Show profile if requested
    if st.session_state.get('show_profile', False):
        show_auth0_profile()
        return

    # Sidebar with user preferences
    with st.sidebar:
        st.title("Navigation")
        
        # Show user info
        if st.session_state.user:
            user = st.session_state.user
            st.write(f"**User:** {user['name']}")
            st.write(f"**Analyses:** {user.get('analysis_count', 0)} completed")
            
            # Show user preferences status
            if user.get('email_notifications'):
                st.success("📧 Email Notifications: Enabled")
            else:
                st.info("📧 Email Notifications: Disabled")
            
            if user.get('drive_connected'):
                st.success("☁️ Google Drive: Connected")
            else:
                st.info("☁️ Google Drive: Available")
            
            # Quick settings toggle
            with st.expander("Quick Settings"):
                current_theme = user.get('theme', 'light')
                new_theme = st.selectbox(
                    "Theme",
                    ["light", "dark", "auto"],
                    index=["light", "dark", "auto"].index(current_theme)
                )
                
                if new_theme != current_theme:
                    user['theme'] = new_theme
                    st.session_state.user = user
                    st.session_state.auth0_service.save_user_preferences(user['id'], user)
                    st.rerun()
        
        st.markdown("---")
        st.subheader("Analysis Settings")
        
        # Use user's saved preferences
        user_default_model = st.session_state.user.get('default_model', 'Random Forest')
        user_auto_save = st.session_state.user.get('auto_save', True)
        
        st.write(f"**Default Model:** {user_default_model}")
        st.write(f"**Auto-save:** {'Enabled' if user_auto_save else 'Disabled'}")
        
        st.markdown("---")
        st.subheader("Platform Features")
        st.markdown("""
        - User profiles and authentication
        - Persistent preferences
        - Analysis history tracking
        - Cloud storage integration
        - Custom theme settings
        - Progress tracking
        """)

    # Main content
    st.markdown('<div class="main-header">Geospatial Landcover Classification Platform</div>', unsafe_allow_html=True)
    
    st.markdown(
        """
        **Instructions:**  
        1. Click on the map to select a geographic location  
        2. Choose one or more years (1995-2022)  
        3. Select a machine learning model  
        4. Click Run Predictions to view classification maps and area summaries
        
        *All your preferences and analysis history are automatically saved*
        """
    )

    # Main application content
    st.subheader("Single Location Analysis")
    
    st.write("Select a location on the map")

    # Create a Folium map
    folium_map = folium.Map(location=[-28.0, 24.0], zoom_start=5, tiles="OpenStreetMap")
    m = st_folium(folium_map, width=700, height=450)

    if m and m.get("last_clicked"):
        lat = m["last_clicked"]["lat"]
        lon = m["last_clicked"]["lng"]
        st.write(f"**Selected coordinates:** {lat:.4f}, {lon:.4f}")
    else:
        lat = lon = None
        st.info("Click on the map to choose a location.")

    st.write("Choose year(s) to process")
    all_years = list(range(1995, 2023))
    selected_years = st.multiselect("Select one or more years", all_years, default=[2020, 2022], key="years_multiselect")

    st.write("Select Machine Learning Model")
    # Use user's default model preference
    user_default_model = st.session_state.user.get('default_model', 'Random Forest')
    default_index = 0 if user_default_model == "Random Forest" else 1
    
    model_type = st.selectbox(
        "Choose classification model:",
        ["Random Forest", "Gradient Boosting"],
        index=default_index,
        help="Random Forest: Generally more robust and requires less tuning. Gradient Boosting: Can achieve higher accuracy but may be more sensitive to parameters.",
        key="model_selectbox"
    )

    # Add PDF generation option - use user's auto_save preference
    user_auto_save = st.session_state.user.get('auto_save', True)
    generate_pdf = st.checkbox("Generate PDF Report", value=user_auto_save, 
                            help="Create a comprehensive PDF report with all results and analysis",
                            key="pdf_checkbox")

    if st.button("Run Predictions", type="primary", key="run_predictions_btn"):
        if lat is None or not selected_years:
            st.error("Please select a location on the map and at least one year.")
        else:
            status_box = st.empty()
            status_messages = []

            def update_status(msg):
                status_messages.append(msg)
                status_box.markdown("**Status:**<br>" + "<br>".join(status_messages), unsafe_allow_html=True)

            with st.spinner("Running inference... this may take a few minutes"):
                try:
                    update_status(f"Starting predictions using {model_type} model...")
                    
                    # Run predictions
                    predictions, figures, areas_per_class, transition_matrices = predict_for_years(
                        lat, lon, selected_years, model_type, status_callback=update_status
                    )
                    
                    # Save analysis to history
                    save_analysis_to_history(
                        lat, lon, selected_years, model_type, 
                        {
                            'predictions': len(predictions),
                            'figures': len(figures),
                            'areas_per_class': areas_per_class,
                            'transition_matrices': len(transition_matrices)
                        }
                    )
                    
                    # Send email notification if enabled
                    if st.session_state.user.get('email_notifications'):
                        update_status("Sending email notification...")
                        analysis_details = {
                            'location': f"Lat: {lat:.4f}, Lon: {lon:.4f}",
                            'years': f"{min(selected_years)} to {max(selected_years)}",
                            'model': model_type,
                            'completion_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        }
                        
                        success, message = st.session_state.email_service.send_analysis_completion_email(
                            st.session_state.user['email'], st.session_state.user['name'], analysis_details
                        )
                        
                        if success:
                            update_status("Email notification sent successfully")
                        else:
                            update_status(f"Email notification failed: {message}")
                    
                    # Store results
                    st.session_state.predictions = predictions
                    st.session_state.figures = figures
                    st.session_state.areas_per_class = areas_per_class
                    st.session_state.transition_matrices = transition_matrices
                    st.session_state.lat = lat
                    st.session_state.lon = lon
                    st.session_state.selected_years = selected_years
                    st.session_state.model_type = model_type
                    
                except Exception as e:
                    st.error(f"An error occurred during prediction: {str(e)}")
                    st.stop()

            st.success(f"Predictions completed successfully using {model_type}!")
            
            # Display model information
            st.info(f"""
            **Model Information:**
            - **Selected Model:** {model_type}
            - **Years Processed:** {len(selected_years)} years ({min(selected_years)} to {max(selected_years)})
            - **Sensors Used:** {'Landsat' if min(selected_years) < 2017 else 'Sentinel-2'} for older years, {'Sentinel-2' if max(selected_years) >= 2017 else 'Landsat'} for recent years
            - **Total Analyses:** {st.session_state.user.get('analysis_count', 0)} completed
            """)

            st.subheader("Per-Year Results")
            for i, fig in enumerate(st.session_state.figures):
                if i < len(st.session_state.selected_years):
                    year = st.session_state.selected_years[i]
                    st.write(f"### {year} Results")
                    st.pyplot(fig)
                else:
                    st.pyplot(fig)

            st.subheader("Class Distribution Table")
            import pandas as pd
            df_areas = pd.DataFrame(st.session_state.areas_per_class).T.fillna(0)
            
            styled_df = df_areas.style.format("{:.2f}%").background_gradient(cmap='Blues')
            st.dataframe(styled_df)
            
            st.subheader("Distribution Summary")
            col1, col2, col3 = st.columns(3)
            with col1:
                avg_total = df_areas.sum(axis=1).mean()
                st.metric("Mean Coverage Total", f"{avg_total:.1f}%")
            
            with col2:
                avg_class_share = df_areas.mean().mean()
                st.metric("Average Class Share", f"{avg_class_share:.2f}%")
            
            with col3:
                dominant_class = df_areas.iloc[-1].idxmax() if len(df_areas) > 0 else "N/A"
                st.metric("Current Dominant Class", dominant_class)

            # PDF Generation Section with Google Drive integration
            if generate_pdf:
                st.subheader("PDF Report Generation")
                with st.spinner("Generating PDF report..."):
                    try:
                        pdf_path = create_prediction_pdf(
                            st.session_state.predictions, 
                            st.session_state.figures, 
                            st.session_state.areas_per_class, 
                            st.session_state.transition_matrices,
                            st.session_state.lat, 
                            st.session_state.lon, 
                            st.session_state.selected_years
                        )
                        
                        if pdf_path and os.path.exists(pdf_path):
                            file_name = f"landcover_analysis_{st.session_state.lat}_{st.session_state.lon}_{min(st.session_state.selected_years)}_{max(st.session_state.selected_years)}.pdf"
                            
                            # Check if Google Drive is connected
                            if st.session_state.user.get('drive_connected'):
                                # Save to Google Drive
                                from google_drive_integration import GoogleDriveService
                                drive_service = GoogleDriveService()
                                
                                drive_url = drive_service.upload_file_to_drive(
                                    pdf_path, 
                                    file_name, 
                                    st.session_state.user['id']
                                )
                                
                                if drive_url:
                                    st.success(f"PDF report saved to Google Drive!")
                                    st.markdown(f"[📎 View in Google Drive]({drive_url})")
                                    
                                    # Also provide local download
                                    with open(pdf_path, "rb") as pdf_file:
                                        st.download_button(
                                            label="📥 Download Local Copy",
                                            data=pdf_file,
                                            file_name=file_name,
                                            mime="application/pdf",
                                            key="download_pdf_local_btn"
                                        )
                                else:
                                    st.warning("Google Drive upload failed. Downloading locally.")
                                    with open(pdf_path, "rb") as pdf_file:
                                        st.download_button(
                                            label="Download PDF Report",
                                            data=pdf_file,
                                            file_name=file_name,
                                            mime="application/pdf",
                                            key="download_pdf_fallback_btn"
                                        )
                            else:
                                # Just local download
                                with open(pdf_path, "rb") as pdf_file:
                                    st.download_button(
                                        label="Download PDF Report",
                                        data=pdf_file,
                                        file_name=file_name,
                                        mime="application/pdf",
                                        key="download_pdf_btn"
                                    )
                                st.success("PDF report generated successfully!")
                            
                            # Show what's included in the PDF
                            with st.expander("What's included in the PDF report?"):
                                st.markdown(f"""
                                - Cover page with prediction details
                                - Model information: {model_type}
                                - Automated analysis of land cover changes
                                - All classification maps for {len(selected_years)} years
                                - True color satellite imagery 
                                - Probability maps and confidence levels
                                - Area analysis and change detection
                                - Transition matrices between years
                                - Summary tables and statistics
                                """)
                        else:
                            st.error("PDF generation failed - file not created")
                            
                    except Exception as e:
                        st.error(f"PDF generation failed: {str(e)}")
                        st.info("You can still view all results above in the interactive display.")

    st.markdown("---")
    st.markdown("**Google Drive Integration** | **Secure Authentication** | **Geospatial AI**")

def main():
    # Initialize session state
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'user' not in st.session_state:
        st.session_state.user = None
    if 'show_profile' not in st.session_state:
        st.session_state.show_profile = False

    # Handle Auth0 callback
    handle_auth0_callback()

    # Check authentication
    if not st.session_state.authenticated:
        st.markdown('<div class="main-header">Geospatial Landcover Classification Platform</div>', unsafe_allow_html=True)
        show_auth0_login()
        st.markdown("---")
        st.markdown("**Google Drive Integration** | **Secure Authentication** | **Geospatial AI**")
    else:
        main_application()

if __name__ == "__main__":
    main()
