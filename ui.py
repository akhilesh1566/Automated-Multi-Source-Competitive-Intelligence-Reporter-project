# ui.py
import streamlit as st

def create_input_form():
    """Creates the input form using Streamlit widgets."""
    with st.sidebar:
        st.header("Competitor Analysis Setup")
        competitor_name = st.text_input("Enter Competitor Name:", placeholder="e.g., Google")
        competitor_url = st.text_input("Enter Competitor Website URL (Optional):", placeholder="https://www.example.com")
        days_back = st.slider("Select Timeframe (days back):", min_value=1, max_value=30, value=7)
        generate_button = st.button("Generate Report", type="primary")
    # --- RETURN URL ---
    return competitor_name, competitor_url, days_back, generate_button

def display_report(report_text: str):
    """Displays the generated report."""
    st.subheader("Competitive Intelligence Summary")
    st.markdown(report_text) # Use markdown for better formatting potential

def display_info(message: str):
    """Displays informational messages."""
    st.info(message)

def display_error(message: str):
    """Displays error messages."""
    st.error(message)

def display_warning(message: str):
    """Displays warning messages."""
    st.warning(message)