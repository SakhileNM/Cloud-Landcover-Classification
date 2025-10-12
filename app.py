# app.py
import streamlit as st
from streamlit_folium import st_folium
import folium
from inference import init_dask_cluster, predict_for_years, create_prediction_pdf
import resource
import os

# Set memory limits for Oracle Cloud (24GB available)
try:
    resource.setrlimit(resource.RLIMIT_AS, (20 * 1024**3, 20 * 1024**3))  # 20GB limit
except:
    pass

# page config must be set before any other st.* call
st.set_page_config(
    page_title="Land-Cover Classification - Oracle Cloud",
    layout="wide",
    initial_sidebar_state="auto"
)

# Initialize Dask cluster once per Streamlit session
if 'dask_client' not in st.session_state:
    try:
        st.session_state.dask_client = init_dask_cluster()
    except Exception as e:
        st.warning(f"Could not start Dask cluster automatically: {e}")

# --- UI ---
st.title("🌍 Land-Cover Classification & Analysis")
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
                # Import the necessary functions
                from inference import predict_for_years, create_prediction_pdf
                has_pdf_support = True
                
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

        st.subheader("Numeric Area Table")
        import pandas as pd
        df_areas = pd.DataFrame(areas_per_class).T.fillna(0)
        
        # Format the dataframe for better display
        styled_df = df_areas.style.format("{:,.0f}").background_gradient(cmap='Blues')
        st.dataframe(styled_df)
        
        # Add summary statistics
        st.subheader("Area Summary")
        col1, col2, col3 = st.columns(3)
        with col1:
            total_area = df_areas.sum().sum()
            st.metric("Total Area Analyzed", f"{total_area:,.0f} m²")
        with col2:
            avg_area = df_areas.mean().mean()
            st.metric("Average Class Area", f"{avg_area:,.0f} m²")
        with col3:
            dominant_class = df_areas.iloc[-1].idxmax() if len(df_areas) > 0 else "N/A"
            st.metric("Current Dominant Class", dominant_class)
        
        # PDF Generation Section
        if generate_pdf and has_pdf_support:
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

# Add information about the models
with st.sidebar:
    st.header("About the Models")
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
    
    st.header("System Info")
    st.markdown("""
    - **Instance**: 4 OCPU, 24GB RAM
    - **Location**: Oracle Cloud
    - **Data**: Digital Earth Africa
    - **Classes**: 6 land cover types
    """)

st.markdown("---")
st.markdown("🚀 **Oracle Cloud Deployment** | 💻 **4 OCPU, 24GB RAM** | 🌍 **Digital Earth Africa**")
