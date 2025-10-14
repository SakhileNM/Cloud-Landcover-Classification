# app.py
import streamlit as st
from streamlit_folium import st_folium
import folium
from inference import init_dask_cluster, predict_for_years, create_prediction_pdf
from auth_components import show_login_form, show_signup_form, show_user_profile, handle_oauth_callback
import resource
import os

# Set memory limits for Oracle Cloud (24GB available)
try:
    resource.setrlimit(resource.RLIMIT_AS, (20 * 1024**3, 20 * 1024**3))  # 20GB limit
except:
    pass

# Page configuration must be first
st.set_page_config(
    page_title="Cloud Geospatial Classification - Oracle Cloud",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2e86ab;
        margin: 1rem 0;
    }
    .login-container {
        max-width: 400px;
        margin: 0 auto;
        padding: 2rem;
        border: 1px solid #ddd;
        border-radius: 10px;
        background: white;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
</style>
""", unsafe_allow_html=True)

def main_application():
    """Main application that only authenticated users can access"""
    
    # Initialize Dask cluster once per Streamlit session
    if 'dask_client' not in st.session_state:
        try:
            st.session_state.dask_client = init_dask_cluster()
        except Exception as e:
            st.warning(f"Could not start Dask cluster automatically: {e}")

    # Sidebar with user info and navigation
    with st.sidebar:
        st.title("Navigation")
        
        # User profile section
        if st.session_state.user:
            st.write(f"Welcome, **{st.session_state.user.full_name}**!")
            if st.button("Profile Settings"):
                st.session_state.show_profile = True
            
            # Show user's save preference
            save_location = st.session_state.user.save_location
            st.info(f"Save to: {save_location.replace('_', ' ').title()}")
            
            st.markdown("---")
        
        # Main navigation
        st.subheader("Analysis")
        analysis_type = st.radio(
            "Select Analysis Type:",
            ["Single Location", "Time Series", "Batch Processing"]
        )
        
        st.markdown("---")
        st.subheader("About the Models")
        st.markdown("""
        **Random Forest**
        - Ensemble of decision trees
        - Robust to overfitting
        - Good for complex datasets
        
        **Gradient Boosting**
        - Sequential tree building
        - Often higher accuracy
        - More sensitive to parameters
        
        **Sensor Selection**
        - **Landsat**: 1995-2016 (30m resolution)
        - **Sentinel-2**: 2017+ (10-20m resolution)
        """)
        
        st.markdown("---")
        st.subheader("System Info")
        st.markdown("""
        - **Instance**: 4 OCPU, 24GB RAM
        - **Location**: Oracle Cloud
        - **Data**: Digital Earth Africa
        - **Classes**: 6 land cover types
        """)

    # Show profile if requested
    if st.session_state.get('show_profile', False):
        show_user_profile()
        return

    # Main content area
    st.markdown('<div class="main-header">Cloud-Based Geospatial Classification</div>', unsafe_allow_html=True)
    
    st.markdown(
        """
        **Instructions:**  
        1. Click on the map to select a geographic location.  
        2. Choose one or more years (1995–2022).  
        3. Select a machine learning model.  
        4. Click **Run Predictions** to view classification maps and area summaries.
        
        *Deployed on Oracle Cloud with 4 OCPUs and 24GB RAM*
        """
    )

    # Main application content based on analysis type
    if analysis_type == "Single Location":
        st.markdown('<div class="sub-header">Single Location Analysis</div>', unsafe_allow_html=True)
        
        st.subheader("1. Select a location on the map")

        # Create a Folium map centered on South Africa
        folium_map = folium.Map(location=[-28.0, 24.0], zoom_start=5, tiles="OpenStreetMap")
        m = st_folium(folium_map, width=700, height=450)

        if m and m.get("last_clicked"):
            lat = m["last_clicked"]["lat"]
            lon = m["last_clicked"]["lng"]
            st.markdown(f"**Selected coordinates:** {lat:.4f}, {lon:.4f}")
        else:
            lat = lon = None
            st.info("Click on the map to choose a location.")

        st.subheader("2. Choose year(s) to process")
        all_years = list(range(1995, 2023))
        selected_years = st.multiselect("Select one or more years", all_years, default=[2020, 2022])

        st.subheader("3. Select Machine Learning Model")
        model_type = st.selectbox(
            "Choose classification model:",
            ["Random Forest", "Gradient Boosting"],
            index=0,
            help="Random Forest: Generally more robust and requires less tuning. Gradient Boosting: Can achieve higher accuracy but may be more sensitive to parameters."
        )

        # Add PDF generation option
        generate_pdf = st.checkbox("Generate PDF Report", value=True, 
                                help="Create a comprehensive PDF report with all results and analysis")

        if st.button("Run Predictions", type="primary"):
            if lat is None or not selected_years:
                st.error("Please select a location on the map *and* at least one year.")
            else:
                status_box = st.empty()
                status_messages = []

                def update_status(msg):
                    status_messages.append(msg)
                    status_box.markdown("**Status:**<br>" + "<br>".join(status_messages), unsafe_allow_html=True)

                with st.spinner("Running inference… this may take a few minutes"):
                    try:
                        update_status(f"Starting predictions using {model_type} model...")
                        
                        # Run predictions with model type selection
                        predictions, figures, areas_per_class, transition_matrices = predict_for_years(
                            lat, lon, selected_years, model_type, status_callback=update_status
                        )
                        
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
                        st.error(f"An error occurred during prediction:\n{str(e)}")
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
                    # Only show classification plots (not the area summary plots)
                    if i < len(selected_years):
                        year = selected_years[i]
                        st.markdown(f"### {year} Results")
                        st.pyplot(fig)
                    else:
                        # For area summary and transition matrix plots
                        st.pyplot(fig)

                st.subheader("Class Distribution Table")
                import pandas as pd
                df_areas = pd.DataFrame(areas_per_class).T.fillna(0)
                
                # Format the dataframe for better display (two decimal places for percentages)
                styled_df = df_areas.style.format("{:.2f}%").background_gradient(cmap='Blues')
                st.dataframe(styled_df)
                
                # Add summary statistics
                st.subheader("Distribution Summary")
                col1, col2, col3 = st.columns(3)
                with col1:
                    # Total area now always sums to 100% for each year
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
                    st.subheader("📄 PDF Report Generation")
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
                                with open(pdf_path, "rb") as pdf_file:
                                    st.download_button(
                                        label="Download PDF Report",
                                        data=pdf_file,
                                        file_name=os.path.basename(pdf_path),
                                        mime="application/pdf",
                                        help="Download comprehensive report with all analysis and plots"
                                    )
                                st.success("PDF report generated successfully!")
                                
                                # Show what's included in the PDF
                                with st.expander("What's included in the PDF report?"):
                                    st.markdown(f"""
                                    - **Cover page** with prediction details
                                    - **Model information**: {model_type}
                                    - **Automated analysis** of land cover changes
                                    - **All classification maps** for {len(selected_years)} years
                                    - **True color satellite imagery** 
                                    - **Probability maps** and confidence levels
                                    - **Area analysis** and change detection
                                    - **Transition matrices** between years
                                    - **Summary tables** and statistics
                                    """)
                            else:
                                st.error("PDF generation failed - file not created")
                                
                        except Exception as e:
                            st.error(f"PDF generation failed: {str(e)}")
                            st.info("You can still view all results above in the interactive display.")
    
    elif analysis_type == "Time Series":
        st.markdown('<div class="sub-header">Time Series Analysis</div>', unsafe_allow_html=True)
        st.info("Time Series Analysis feature is under development")
        
    elif analysis_type == "Batch Processing":
        st.markdown('<div class="sub-header">Batch Processing</div>', unsafe_allow_html=True)
        st.info("Batch Processing feature is under development")

    st.markdown("---")
    st.markdown("**Oracle Cloud Deployment** | **Secure Authentication** | **Digital Earth Africa**")

def main():
    # Initialize session state
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'user' not in st.session_state:
        st.session_state.user = None
    if 'show_profile' not in st.session_state:
        st.session_state.show_profile = False

    # Handle OAuth callbacks
    handle_oauth_callback()

    # Check authentication
    if not st.session_state.authenticated:
        # Show authentication interface
        st.markdown('<div class="main-header">Cloud Geospatial Classification</div>', unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["Login", "Sign Up"])
        
        with tab1:
            show_login_form()
        
        with tab2:
            show_signup_form()
            
        st.markdown("---")
        st.markdown("**Oracle Cloud Deployment** | **Secure Authentication** | **Digital Earth Africa**")
    else:
        # User is authenticated, show main application
        main_application()

if __name__ == "__main__":
    main()
