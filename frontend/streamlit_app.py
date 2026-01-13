"""
A Streamlit web frontend for the ChemCheck application.

This UI allows users to upload chemical label images and view the results
of the SDS (Safety Data Sheet) identification workflow.
"""
import streamlit as st

from client import APIError, identify_chemical


def display_results(result: dict):
    """Renders the API response in a user-friendly format."""
    st.subheader("Workflow Result")

    status = result.get("sds", {}).get("status", "unknown")
    if status == "found":
        st.success(f"**Status:** {status.upper()}")
    else:
        st.warning(f"**Status:** {status.upper()}")

    st.markdown(f"""**Message:** {result.get("sds", {}).get("message", "No message.")}""")

    st.divider()

    with st.expander("OCR Details"):
        ocr_data = result.get("ocr", {})
        st.markdown(f"**Product Name:** {ocr_data.get('product_name') or '_Not found_'}")
        st.markdown(f"**Product Code:** {ocr_data.get('product_code') or '_Not found_'}")
        st.markdown(f"**CAS Number:** {ocr_data.get('cas_number') or '_Not found_'}")
        st.markdown(f"**Manufacturer:** {ocr_data.get('manufacturer_name') or '_Not found_'}")
        st.text_area("Full Extracted Text", ocr_data.get("full_text", ""), height=200)

    with st.expander("SDS Search Details"):
        sds_data = result.get("sds", {})
        st.markdown(f"**Manufacturer URL:** {sds_data.get('manufacturer_url') or '_Not found_'}")
        st.markdown(f"**SDS URL:** {sds_data.get('sds_url') or '_Not found_'}")
        st.markdown(f"**Search Query:** `{sds_data.get('sds_query') or '_Not applicable_'}`")
        
        st.write("**Search Attempts:**")
        st.json(sds_data.get("sds_attempts", []))

    with st.expander("Validation Details"):
        st.json(result.get("sds", {}).get("validation", {}))


# --- Main UI ---

st.set_page_config(page_title="ChemCheck", page_icon="🧪", layout="wide")

st.title("🧪 ChemCheck")
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

if st.button("Submit", type="primary"):
    if not front_image and not back_image:
        st.error("Upload at least one image.")
        st.stop()

    with st.spinner("Processing... This may take a minute."):
        try:
            result_data = identify_chemical(front_image, back_image)
            st.success("Processing complete!")
            display_results(result_data)
        except APIError as exc:
            st.error(f"An error occurred: {exc}")
        except ValueError as exc:
            st.error(f"Invalid input: {exc}")
