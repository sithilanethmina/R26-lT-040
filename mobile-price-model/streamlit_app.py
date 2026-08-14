"""
Streamlit entrypoint forwarder.

Main app logic is maintained in `app/streamlit_app.py`.
Run: `streamlit run streamlit_app.py` or `streamlit run app/streamlit_app.py`
"""

from app.streamlit_app import main

if __name__ == "__main__":
    main()
