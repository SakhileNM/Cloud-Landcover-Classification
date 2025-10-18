# app.py - FIXED VERSION
import streamlit as st
from streamlit_folium import st_folium
import folium
from inference import init_dask_cluster, predict_for_years, create_prediction_pdf
from auth0_auth import show_auth0_login, show_auth0_profile, handle_auth0_callback
import resource
import os
from datetime import datetime

# Set memory limits for Oracle Cloud (24GB available)
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

# Custom CSS
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

def main_application():
    """Main application for authenticated users"""
    
    # Initialize Dask cluster
    if 'dask_client' not in st.session_state:
        try:
            st.session_state.dask_client = init_dask_cluster()
        except Exception as e:
            st.warning(f"Could not start Dask cluster: {e}")

    # User welcome bar with three buttons
    if st.session_state.user:
        col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
        with col1:
            user = st.session_state.user
            
            welcome_text = f"Welcome, {user['name']}! | Professional Geospatial Platform"
            if user.get('drive_connected'):
                welcome_text += " <span class='status-badge'>Google Drive Connected</span>"
            
            st.markdown(f'<div class="user-welcome">{welcome_text}</div>', unsafe_allow_html=True)
        
        with col2:
            if st.button("Home", key="home_btn_main"):
                st.session_state.show_profile = False
                st.rerun()
        with col3:
            if st.button("Profile Settings", key="profile_btn_main"):
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

    # Sidebar
    with st.sidebar:
        st.title("Navigation")
        
        # Show user info
        if st.session_state.user:
            user = st.session_state.user
            st.write(f"**User:** {user['name']}")
            st.write(f"**Analyses:** {user.get('analysis_count', 0)} completed")
            
            if user.get('drive_connected'):
                st.success("Google Drive: Connected")
            else:
                st.info("Google Drive: Available")
        
        st.markdown("---")
        st.subheader("Machine Learning Models")
        st.markdown("""
        **Random Forest**
        - Ensemble of decision trees
        - Robust to overfitting
        
        **Gradient Boosting**  
        - Sequential tree building
        - Often higher accuracy
        
        **Sensor Selection**
        - **Landsat**: 1995-2016 (30m resolution)
        - **Sentinel-2**: 2017+ (10-20m resolution)
        """)
        
        st.markdown("---")
        st.subheader("Platform Features")
        st.markdown("""
        - User profiles and authentication
        - Cloud storage integration
        - Custom theme settings
        - Analysis history tracking
        - Oracle Cloud deployment
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
        
        *Deployed on Oracle Cloud with 4 OCPUs and 24GB RAM*
        """
    )

    # Main application content
    st.subheader("Single Location Analysis")
    
    st.write("Select a location on the map")

    # Create a Folium map centered on South Africa
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
    model_type = st.selectbox(
        "Choose classification model:",
        ["Random Forest", "Gradient Boosting"],
        index=0,
        help="Random Forest: Generally more robust and requires less tuning. Gradient Boosting: Can achieve higher accuracy but may be more sensitive to parameters.",
        key="model_selectbox"
    )

    # Add PDF generation option
    generate_pdf = st.checkbox("Generate PDF Report", value=True, 
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
                    
                    # Run predictions with model type selection
                    predictions, figures, areas_per_class, transition_matrices = predict_for_years(
                        lat, lon, selected_years, model_type, status_callback=update_status
                    )
                    
                    # Update user analysis count
                    user = st.session_state.user
                    user['analysis_count'] = user.get('analysis_count', 0) + 1
                    st.session_state.user = user
                    
                    # Store results for potential PDF generation
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
            """)

            st.subheader("Per-Year Results")
            for i, fig in enumerate(figures):
                if i < len(selected_years):
                    year = selected_years[i]
                    st.write(f"### {year} Results")
                    st.pyplot(fig)
                else:
                    st.pyplot(fig)

            st.subheader("Class Distribution Table")
            import pandas as pd
            df_areas = pd.DataFrame(areas_per_class).T.fillna(0)
            
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

            # PDF Generation Section
            if generate_pdf:
                st.subheader("PDF Report Generation")
                with st.spinner("Generating PDF report..."):
                    try:
                        pdf_path = create_prediction_pdf(
                            predictions, 
                            figures, 
                            areas_per_class, 
                            transition_matrices,
                            lat, 
                            lon, 
                            selected_years
                        )
                        
                        if pdf_path and os.path.exists(pdf_path):
                            # Check if user has Google Drive connected
                            user = st.session_state.user
                            file_name = f"landcover_analysis_{lat}_{lon}_{min(selected_years)}_{max(selected_years)}.pdf"
                            
                            if user.get('drive_connected'):
                                # Future: Save to Google Drive
                                st.info("Google Drive integration coming soon - downloading locally for now")
                            
                            with open(pdf_path, "rb") as pdf_file:
                                st.download_button(
                                    label="Download PDF Report",
                                    data=pdf_file,
                                    file_name=file_name,
                                    mime="application/pdf",
                                    help="Download comprehensive report with all analysis and plots",
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
    st.markdown("**Oracle Cloud Deployment** | **Secure Authentication** | **Geospatial AI**")

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
        st.markdown("**Oracle Cloud** | **Secure Authentication** | **Geospatial AI**")
    else:
        # User is authenticated, show main application
        main_application()

if __name__ == "__main__":
    main()
