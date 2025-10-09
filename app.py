# app.py
import streamlit as st
from streamlit_folium import st_folium
import folium
from inference import init_dask_cluster, predict_for_years
import resource
# Cloud compatibility fixes
import os
if os.getenv('RAILWAY_ENVIRONMENT'):
    # Disable resource limits in cloud
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (1 * 1024**3, 1 * 1024**3))  # More conservative limit for cloud
    except:
        pass
    
    # Disable Dask cluster in cloud (memory constraints)
    if 'dask_client' not in st.session_state:
        st.session_state.dask_client = None

try:
    # Limit memory usage to 16GB
    resource.setrlimit(resource.RLIMIT_AS, (16 * 1024**3, 16 * 1024**3))
except:
    pass

# page config must be set before any other st.* call
st.set_page_config(
    page_title="Land-Cover Predictor",
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
st.title("Land-Cover Classification & Analysis")
st.markdown(
    """
    **Instructions:**  
    1. Click on the map to select a geographic location.  
    2. Choose one or more years (1995–2022).  
    3. Click **Run Predictions** to view classification maps and area summaries.
    """
)

st.subheader("1. Select a location on the map")

# Create a Folium map
folium_map = folium.Map(location=[-34.0, 20.0], zoom_start=6, tiles="OpenStreetMap")
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
selected_years = st.multiselect("Select one or more years", all_years)

# Add PDF generation option
generate_pdf = st.checkbox("Generate PDF Report", value=True, 
                          help="Create a comprehensive PDF report with all results and analysis")

if st.button("Run Predictions"):
    if lat is None or not selected_years:
        st.error("Please select a location on the map *and* at least one year.")
    else:
        status_box = st.empty()
        status_messages = []

        def update_status(msg):
            status_messages.append(msg)
            status_box.markdown("**Status:**<br>" + "<br>".join(status_messages), unsafe_allow_html=True)

        with st.spinner("Running inference… this may take a minute"):
            try:
                # Import the PDF function safely
                try:
                    from inference import create_prediction_pdf, generate_analysis_text
                    has_pdf_support = True
                except ImportError as e:
                    st.warning(f"PDF generation not available: {e}")
                    has_pdf_support = False
                
                # Run predictions (existing functionality)
                predictions, figures, areas_per_class, transition_matrices = predict_for_years(
                    lat, lon, selected_years, status_callback=update_status
                )
                
                # Store results for potential PDF generation
                st.session_state.predictions = predictions
                st.session_state.figures = figures
                st.session_state.areas_per_class = areas_per_class
                st.session_state.transition_matrices = transition_matrices
                st.session_state.lat = lat
                st.session_state.lon = lon
                st.session_state.selected_years = selected_years
                
            except Exception as e:
                st.error(f"An error occurred during prediction:\n{e}")
                st.stop()

        st.subheader("Per-Year Results")
        for fig in figures:
            st.pyplot(fig)

        st.subheader("Numeric Area Table")
        import pandas as pd
        df_areas = pd.DataFrame(areas_per_class).T.fillna(0)
        st.dataframe(df_areas.style.format("{:,.0f}"))
        
        # PDF Generation Section
        if generate_pdf and has_pdf_support:
            st.subheader("PDF Report")
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
                                label="📄 Download PDF Report",
                                data=pdf_file,
                                file_name=os.path.basename(pdf_path),
                                mime="application/pdf",
                                help="Download comprehensive report with all analysis and plots"
                            )
                        st.success("PDF report generated successfully!")
                        
                        # Show what's included in the PDF
                        with st.expander("What's included in the PDF report?"):
                            st.markdown("""
                            - **Cover page** with prediction details
                            - **Automated analysis** of land cover changes
                            - **All classification maps** for each year
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

st.markdown("---")
st.markdown("ℹ️ Built with [Streamlit](https://streamlit.io) & [Open Data Cube](https://www.opendatacube.org)")

# Health check endpoint for Railway
if __name__ == "__main__":
    # This allows Railway to health check the container
    import requests
    try:
        response = requests.get("http://localhost:8501", timeout=5)
        print(f"Health check: {response.status_code}")
    except:
        print("Health check: Starting up...")