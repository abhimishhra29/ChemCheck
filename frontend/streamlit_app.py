import os

import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="ChemCheck", page_icon="C")

st.title("ChemCheck")
st.write("Upload front and/or back label images to start the identification flow.")

col1, col2 = st.columns(2)
with col1:
    front_image = st.file_uploader(
        "Front image",
        type=["png", "jpg", "jpeg"],
        key="front_image",
    )
with col2:
    back_image = st.file_uploader(
        "Back image (optional)",
        type=["png", "jpg", "jpeg"],
        key="back_image",
    )

if st.button("Submit"):
    if not front_image and not back_image:
        st.error("Upload at least one image.")
        st.stop()

    files = {}
    if front_image:
        files["front_image"] = (front_image.name, front_image.getvalue(), front_image.type)
    if back_image:
        files["back_image"] = (back_image.name, back_image.getvalue(), back_image.type)

    try:
        with st.spinner("Uploading..."):
            response = requests.post(
                f"{API_BASE_URL}/api/v1/identify-chemical",
                files=files,
                timeout=60,
            )
        response.raise_for_status()
    except requests.RequestException as exc:
        st.error(f"Request failed: {exc}")
    else:
        st.success("Response received")
        st.json(response.json())
