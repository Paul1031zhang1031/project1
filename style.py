import streamlit as st
import base64  

def create_header(main_title, logo_path, banner_path):
    """
    Creates and displays a styled header with a logo, title, and banner.

    Args:
        main_title (str): The main title for the header.
        logo_path (str): The file path to the logo image.
        banner_path (str): The file path to the banner image.
    """
    # --- CSS for styling the header elements ---
    st.markdown(
        """
        <style>
        /* Main container for the top bar (logo + title) */
        .header-top-bar {
            display: flex;
            align-items: center; /* Vertically center items */
            padding-bottom: 15px; /* Space below the top bar */
        }
        .header-logo img {
            width: 100px; /* Logo size */
            margin-right: 15px; /* Space between logo and title */
        }
        .header-title h2 {
            margin: 0;
            font-weight: 500;
            font-size: 26px;
            color: #333;
        }

        /* Banner Image Styling */
        .banner-image img {
            width: 100%;
            max-height: 180px; /* Control the banner height */
            object-fit: cover; /* Prevents distortion */
            border-radius: 8px; /* Optional: adds rounded corners */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # --- Header Layout ---
    # Top bar with Logo and Title
    st.markdown(
        f"""
        <div class="header-top-bar">
            <div class="header-logo">
                <img src="data:image/png;base64,{get_base64_of_bin_file(logo_path)}">
            </div>
            <div class="header-title">
                <h2>{main_title}</h2>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Banner Image
    st.markdown(
        f"""
        <div class="banner-image">
            <img src="data:image/png;base64,{get_base64_of_bin_file(banner_path)}">
        </div>
        """,
        unsafe_allow_html=True
    )

# Helper function to load local files and encode them
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

#set button sytle
def apply_global_styles():
    """
    Applies CSS for common elements across the entire app.
    """
    st.markdown(
        """
        <style>
        /* Violet Button Styling for all secondary buttons */
        button[kind="secondary"] {
            background-color: #8A2BE2 !important;
            color: white !important;
            border: none !important;
            border-radius: 5px !important;
        }
        button[kind="secondary"]:hover {
            background-color: #7a1fd1 !important;
            color: white !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

