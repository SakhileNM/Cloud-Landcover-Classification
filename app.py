# app.py - WITH CLOUD PROFILES
import streamlit as st
from streamlit_folium import st_folium
import folium
from inference import init_dask_cluster, predict_for_years, create_prediction_pdf
from cloud_auth import show_cloud_login, show_cloud_profile
from cloud_storage import cloud_storage
import resource
import os

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
    .cloud-badge {
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

    # User welcome bar
    if st.session_state.user:
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            user = st.session_state.user
            storage_info = cloud_storage.get_user_storage_info(user)
            
            welcome_text = f"""
            Welcome, **{user.full_name}**! | 
            {user.theme.title()} Theme | 
            {storage_info['service'].replace('_', ' ').title()} Storage
            """
            if storage_info['connected']:
                welcome_text += " <span class='cloud-badge'>Connected</span>"
            
            st.markdown(f'<div class="user-welcome">{welcome_text}</div>', unsafe_allow_html=True)
        
        with col2:
            if st.button("Profile"):
                st.session_state.show_profile = True
                st.rerun()
        with col3:
            if st.button("Logout"):
                for key in ['authenticated', 'user', 'user_token', 'show_profile']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()

    # Show profile if requested
    if st.session_state.get('show_profile', False):
        show_cloud_profile()
        return

    # Sidebar
    with st.sidebar:
        st.title("Navigation")
        
        # Show cloud storage info
        if st.session_state.user:
            storage_info = cloud_storage.get_user_storage_info(st.session_state.user)
            st.info(f"**Storage:** {storage_info['service'].replace('_', ' ').title()}")
            if storage_info['connected']:
                st.success(f"{storage_info['total_space']}")
            
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
        
        **Gradient Boosting**  
        - Sequential tree building
        - Often higher accuracy
        
        **Sensor Selection**
        - **Landsat**: 1995-2016 (30m)
        - **Sentinel-2**: 2017+ (10-20m)
        """)
        
        st.markdown("---")
        st.subheader("Cloud Platform")
        st.markdown("""
        - 👤 User profiles
        - ☁️ Cloud storage
        - 🎨 Custom themes
        - 📊 Analysis history
        - 🚀 Oracle Cloud
        """)

    # Main content
    st.markdown('<div class="main-header">☁️ Cloud Geospatial Platform</div>', unsafe_allow_html=True)
    
    st.markdown(
        """
        **Instructions:**  
        1. Click on the map to select a location  
        2. Choose years to analyze (1995–2022)  
        3. Select ML model  
        4. Run predictions and save to your cloud storage
        
        *Results automatically saved to your configured cloud storage*
        """
    )

    # Analysis interface (your existing code)
    if analysis_type == "Single Location":
        # ... your existing map and prediction code ...
        
        # MODIFIED: After generating PDF, save to cloud
        if generate_pdf and st.session_state.user:
            with st.spinner("Generating PDF report..."):
                try:
                    pdf_path = create_prediction_pdf(
                        predictions, figures, areas_per_class, 
                        transition_matrices, lat, lon, selected_years
                    )
                    
                    if pdf_path and os.path.exists(pdf_path):
                        # Save to user's cloud storage
                        user = st.session_state.user
                        file_name = f"landcover_analysis_{lat}_{lon}_{min(selected_years)}_{max(selected_years)}.pdf"
                        
                        success, message = cloud_storage.save_to_cloud(user, pdf_path, file_name)
                        
                        if success:
                            st.success(f"{message}")
                            
                            with open(pdf_path, "rb") as pdf_file:
                                st.download_button(
                                    label="📥 Download Local Copy",
                                    data=pdf_file,
                                    file_name=file_name,
                                    mime="application/pdf"
                                )
                        else:
                            st.warning(f"Cloud save failed: {message}")
                            # Fallback to local download
                            with open(pdf_path, "rb") as pdf_file:
                                st.download_button(
                                    label="Download PDF Report",
                                    data=pdf_file,
                                    file_name=file_name,
                                    mime="application/pdf"
                                )
                    else:
                        st.error("PDF generation failed")
                        
                except Exception as e:
                    st.error(f"PDF generation failed: {str(e)}")

def main():
    # Initialize session state
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'user' not in st.session_state:
        st.session_state.user = None
    if 'show_profile' not in st.session_state:
        st.session_state.show_profile = False

    # Check authentication
    if not st.session_state.authenticated:
        st.markdown('<div class="main-header">☁️ Cloud Geospatial Platform</div>', unsafe_allow_html=True)
        show_cloud_login()
        st.markdown("---")
        st.markdown("🚀 **Oracle Cloud** | ☁️ **Cloud Storage** | 🌍 **Geospatial AI**")
    else:
        main_application()

if __name__ == "__main__":
    main()
