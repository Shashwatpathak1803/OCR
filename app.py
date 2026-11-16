"""
Multilingual (Hindi + English) Visiting Card OCR to Excel System.
Scans visiting cards, performs OpenCV preprocessing, recognizes Devanagari and Latin text
via PaddleOCR, classifies contacts into separate Hindi and English fields using deterministic rules,
provides an interactive verification interface, and exports to Excel.
100% OFFLINE - ZERO AI / LLM / CLOUD DEPENDENCIES.
"""

import json
import time
import streamlit as st
from PIL import Image
import pandas as pd
from dotenv import load_dotenv

# Import internal modules
from ocr_engine import get_engine_metrics
from processing_queue import ProcessingQueue, QueueWorker
from excel_generator import (
    create_new_excel,
    read_existing_excel_sheets,
    preview_existing_excel_info,
    append_to_existing_excel,
    EXCEL_COLUMNS,
)

load_dotenv()


@st.cache_resource
def get_queue_components():
    queue = ProcessingQueue()
    return queue, QueueWorker(queue)


@st.cache_data(show_spinner=False)
def create_excel_cached(records_json: str):
    started = time.perf_counter()
    workbook = create_new_excel(json.loads(records_json))
    return workbook.getvalue(), time.perf_counter() - started


# Page configuration
st.set_page_config(
    page_title="Multilingual Visiting Card OCR (Hindi & English)",
    page_icon="🪪",
    layout="wide",
    initial_sidebar_state="expanded",
)

queue_store, queue_worker = get_queue_components()

# Custom Styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.0rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .badge-completed {
        background-color: #DCFCE7;
        color: #166534;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .badge-verification {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .badge-failed {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .badge-lang {
        background-color: #EFF6FF;
        color: #1D4ED8;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-right: 4px;
    }
    .section-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #1E293B;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
        border-bottom: 2px solid #E2E8F0;
        padding-bottom: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize Session State
if "extracted_records" not in st.session_state:
    st.session_state.extracted_records = []
if "last_camera_key" not in st.session_state:
    st.session_state.last_camera_key = 0


# ==========================================
# Sidebar Configuration
# ==========================================
with st.sidebar:
    st.markdown("### ⚙️ System Settings")

    st.markdown("#### 🔤 Offline Multilingual OCR")
    st.markdown("• **Engine**: PaddleOCR (Devanagari & Latin)")
    st.markdown("• **Supported**: English & Hindi (हिंदी)")
    st.markdown("• **Mode**: Local CPU Accelerated (Offline)")
    st.markdown("• **Segmentation**: Unicode Script Detection")
    st.success("🔒 100% Offline - Zero AI / Translation APIs")

    st.markdown("---")
    st.markdown("#### 🖼️ Image Preprocessing")
    enable_perspective = st.checkbox("Perspective Auto-Warp", value=True, help="Detect card boundary and straighten card.")
    enable_contrast = st.checkbox("Contrast Enhancement (CLAHE)", value=True, help="Equalize contrast for faint/low-contrast text.")
    enable_deskew = st.checkbox("Auto-Deskew Text Lines", value=True, help="Correct slight angle rotations.")

    st.markdown("---")
    if st.button(
        "🗑️ Reset Application",
        use_container_width=True,
        disabled=queue_worker.is_running(),
    ):
        if queue_worker.is_running():
            queue_worker.pause()
        queue_store.clear()
        st.session_state.extracted_records = []
        st.rerun()


# ==========================================
# Main App Header
# ==========================================
st.markdown('<div class="main-title">🪪 Visiting Card OCR to Excel (Hindi & English)</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Offline multilingual card scanner supporting Hindi (Devanagari) & English (Latin) text on the same card with segregated field extraction.</div>',
    unsafe_allow_html=True,
)


# ==========================================
# Section 1: Upload Visiting Cards
# ==========================================
st.markdown('<div class="section-header">1. Upload Visiting Cards</div>', unsafe_allow_html=True)

upload_tab, camera_tab = st.tabs(["📁 Bulk Image Upload", "📷 Camera Scanning"])

with upload_tab:
    uploaded_files = st.file_uploader(
        "Upload visiting card images in English, Hindi, or Bilingual (JPG, JPEG, PNG, WEBP):",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
        key="bulk_uploader",
    )
    if uploaded_files:
        new_count = 0
        duplicate_count = 0
        for f in uploaded_files:
            try:
                _, added = queue_store.add_image(f.name, f.getvalue())
                new_count += int(added)
                duplicate_count += int(not added)
            except Exception as exc:
                st.error(f"Failed to add image '{f.name}' to the local queue: {exc}")
        if new_count > 0:
            st.success(f"Added {new_count} new card(s) to the persistent queue.")
        if duplicate_count:
            st.info(
                f"Skipped {duplicate_count} duplicate image(s); completed images are not reprocessed."
            )

with camera_tab:
    st.write("Align the visiting card within the camera frame and capture a photograph.")
    camera_photo = st.camera_input("Capture visiting card", key=f"camera_input_{st.session_state.last_camera_key}")
    if camera_photo:
        photo_bytes = camera_photo.getvalue()
        cam_filename = f"camera_card_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
        _, added = queue_store.add_image(cam_filename, photo_bytes)
        if added:
            st.success(f"Captured photograph added as '{cam_filename}'.")
        else:
            st.info("This photograph is already in the persistent queue.")


# Staged Cards Gallery
queue_cards = [
    card for card in queue_store.list_cards() if card["image_path"]
]
st.session_state.extracted_records = queue_store.get_completed_records()
staged_count = len(queue_cards)
if staged_count:
    counts = queue_store.summary()["counts"]
    st.markdown(
        f"**Persistent Queue ({staged_count}):** "
        f"{counts.get('Pending', 0)} pending · "
        f"{counts.get('Processing', 0)} processing · "
        f"{counts.get('Failed', 0)} failed"
    )
    col_clear, _ = st.columns([1, 4])
    with col_clear:
        if st.button(
            "❌ Clear Pending Queue",
            use_container_width=True,
            disabled=queue_worker.is_running(),
        ):
            queue_store.clear_queue()
            st.session_state.last_camera_key += 1
            st.rerun()

    cols_per_row = 4
    preview_cards = queue_cards[:8]
    for i in range(0, len(preview_cards), cols_per_row):
        row_cards = preview_cards[i:i + cols_per_row]
        cols = st.columns(cols_per_row)
        for col, card in zip(cols, row_cards):
            with col:
                with Image.open(card["image_path"]) as image:
                    st.image(image.copy(), use_container_width=True)
                st.caption(f"📄 {card['filename']} · {card['status']}")
                if st.button(
                    "🗑️ Remove",
                    key=f"remove_{card['id']}",
                    use_container_width=True,
                    disabled=queue_worker.is_running()
                    or card["status"] == "Processing",
                ):
                    queue_store.remove_card(card["id"])
                    st.rerun()
    if staged_count > len(preview_cards):
        st.caption(
            f"Showing 8 of {staged_count} queued images. All queue entries remain available in the database."
        )
else:
    st.info("No visiting cards staged yet. Upload images above or use the camera.")


# ==========================================
# Section 2: Process Cards
# ==========================================
st.markdown('<div class="section-header">2. Process Cards</div>', unsafe_allow_html=True)

processing_options = {
    "auto_perspective": enable_perspective,
    "auto_deskew": enable_deskew,
    "enhance_contrast": enable_contrast,
}
queue_summary = queue_store.summary()
pending_count = queue_summary["counts"].get("Pending", 0)
failed_count = queue_summary["counts"].get("Failed", 0)

if queue_worker.is_running():
    if st.button("⏸️ Pause after current card", use_container_width=True):
        queue_worker.pause()
        st.rerun()
else:
    control_cols = st.columns(2)
    if control_cols[0].button(
        "🚀 Start / Resume Processing",
        type="primary",
        disabled=pending_count == 0,
        use_container_width=True,
    ):
        queue_worker.start(processing_options)
        st.rerun()
    if control_cols[1].button(
        f"🔁 Retry Failed Cards ({failed_count})",
        disabled=failed_count == 0,
        use_container_width=True,
    ):
        queue_worker.retry_failed(processing_options)
        st.rerun()


@st.fragment(run_every="2s")
def render_queue_metrics():
    summary = queue_store.summary()
    counts = summary["counts"]
    total = summary["total"]
    processed = summary["completed"] + counts.get("Failed", 0)
    progress = processed / total if total else 0.0
    last_seen = st.session_state.get("last_queue_completed")
    last_failed = st.session_state.get("last_queue_failed")
    if last_seen is None or last_failed is None:
        st.session_state.last_queue_completed = summary["completed"]
        st.session_state.last_queue_failed = counts.get("Failed", 0)
    elif (
        summary["completed"] != last_seen
        or counts.get("Failed", 0) != last_failed
    ):
        st.session_state.last_queue_completed = summary["completed"]
        st.session_state.last_queue_failed = counts.get("Failed", 0)
        st.rerun(scope="app")
    metric_cols = st.columns(5)
    metric_cols[0].metric("Queued", total)
    metric_cols[1].metric("Completed", summary["completed"])
    metric_cols[2].metric("Pending", counts.get("Pending", 0))
    metric_cols[3].metric("Failed", counts.get("Failed", 0))
    metric_cols[4].metric("Average / Card", f"{summary['average_card_s']:.2f} s")
    st.progress(progress, text=f"Queue progress: {processed}/{total}")
    st.caption(
        f"Total processing time: {summary['total_processing_s']:.2f} s · "
        f"Estimated remaining: "
        f"{summary['average_card_s'] * counts.get('Pending', 0):.1f} s · "
        f"Status: {'Paused' if summary['paused'] else 'Running'}"
    )
    stages = summary["stage_averages"]
    engine_metrics = get_engine_metrics()
    st.caption(
        "Average stages (seconds/card): "
        f"image load {stages.get('image_loading_s', 0):.3f} · "
        f"preprocess {stages.get('preprocessing_s', 0):.3f} · "
        f"detection {stages.get('detection_s', 0):.3f} · "
        f"recognition {stages.get('recognition_s', 0):.3f} · "
        f"total OCR {stages.get('ocr_s', 0):.3f} · "
        f"extraction {stages.get('data_extraction_s', 0):.3f} · "
        f"database {stages.get('database_saving_s', 0):.3f}"
    )
    if engine_metrics:
        st.caption(
            f"Engine initialization: {engine_metrics.get('initialization_s', 0):.2f} s · "
            f"device: {engine_metrics.get('device', 'unknown')} · "
            f"CPU cores: {engine_metrics.get('cpu_cores', 'unknown')} · "
            f"RAM available at startup: {engine_metrics.get('ram_available_gb', 0):.1f} GB"
        )


render_queue_metrics()

with st.expander("🔍 View Queue Log & Original OCR Text"):
    log_cards = queue_store.list_cards()
    recent_cards = log_cards[-12:]
    if not recent_cards:
        st.caption("No cards have been queued yet.")
    for card in reversed(recent_cards):
        st.markdown(f"**{card['filename']}** — {card['status']}")
        if card["error"]:
            st.error(card["error"])
        if card["raw_text"]:
            st.text_area(
                "Original OCR Text (Raw)",
                value=card["raw_text"],
                height=100,
                key=f"raw_{card['id']}",
            )
        timings = json.loads(card["timings_json"] or "{}")
        if timings:
            st.caption(
                f"Card time: {timings.get('total_card_s', 0):.2f} s · "
                f"OCR: {timings.get('ocr_s', 0):.2f} s · "
                f"attempts: {card['attempts']}"
            )


# ==========================================
# Section 3: Review Extracted Details & Verification
# ==========================================
st.markdown('<div class="section-header">3. Review Extracted Details & Verification</div>', unsafe_allow_html=True)

if st.session_state.extracted_records:
    total_recs = len(st.session_state.extracted_records)
    verified_count = sum(1 for r in st.session_state.extracted_records if not r.get("verification_needed", False))
    awaiting_count = total_recs - verified_count
    with_hi = sum(1 for r in st.session_state.extracted_records if r.get("full_name_hi") or r.get("company_name_hi"))
    with_en = sum(1 for r in st.session_state.extracted_records if r.get("full_name_en") or r.get("company_name_en"))

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Cards", total_recs)
    m2.metric("English Content", f"{with_en}/{total_recs}")
    m3.metric("Hindi Content", f"{with_hi}/{total_recs}")
    m4.metric("Awaiting Verification", awaiting_count)

    # Search & Filter
    search_col, filter_col = st.columns([3, 1])
    with search_col:
        search_query = st.text_input("🔍 Search records (English / Hindi name, company, city, email, phone):", "")
    with filter_col:
        status_filter = st.selectbox("Filter Status:", ["All Records", "Awaiting Verification", "Verified"])

    # Prepare DataFrame matching EXCEL_COLUMNS
    columns_order = [col[1] for col in EXCEL_COLUMNS]
    column_headers_map = {col[1]: col[0] for col in EXCEL_COLUMNS}

    filtered_records = []
    for r in st.session_state.extracted_records:
        is_awaiting = r.get("verification_needed", False)
        if status_filter == "Awaiting Verification" and not is_awaiting:
            continue
        if status_filter == "Verified" and is_awaiting:
            continue

        if search_query:
            query = search_query.lower()
            haystack = f"{r.get('full_name_en','')} {r.get('full_name_hi','')} {r.get('company_name_en','')} {r.get('company_name_hi','')} {r.get('email','')} {r.get('mobile_number','')} {r.get('city_en','')} {r.get('city_hi','')}".lower()
            if query not in haystack:
                continue

        filtered_records.append(r)

    df = pd.DataFrame(filtered_records)
    for k in columns_order:
        if k not in df.columns:
            df[k] = ""
    df = df[columns_order]
    df_display = df.rename(columns=column_headers_map)

    st.write("Double-click any cell below to edit English or Hindi values independently:")
    edited_df = st.data_editor(
        df_display,
        use_container_width=True,
        num_rows="dynamic",
        key="cards_data_editor",
    )

    # Sync back edits
    reverse_map = {v: k for k, v in column_headers_map.items()}
    edited_rows = edited_df.rename(columns=reverse_map).to_dict(orient="records")
    for edited_row in edited_rows:
        sno = edited_row.get("serial_number")
        for orig in st.session_state.extracted_records:
            if orig.get("serial_number") == sno:
                orig.update(edited_row)
                queue_id = orig.get("_queue_id")
                if queue_id:
                    queue_store.save_record(queue_id, orig)
                break

    # Side-by-Side Card Verification
    st.markdown("#### 🔎 Side-by-Side Multilingual Card Verification")
    filenames = [r.get("image_filename", f"Card #{idx+1}") for idx, r in enumerate(st.session_state.extracted_records)]
    selected_idx = st.selectbox(
        "Select card to inspect against photograph and original OCR text:",
        range(len(filenames)),
        format_func=lambda i: f"#{i+1}: {filenames[i]} ({'⚠️ Awaiting Verification' if st.session_state.extracted_records[i].get('verification_needed') else '✅ Verified'})",
    )

    if 0 <= selected_idx < len(st.session_state.extracted_records):
        inspected_rec = st.session_state.extracted_records[selected_idx]
        inspected_fname = inspected_rec.get("image_filename")
        queue_id = inspected_rec.get("_queue_id")
        image_path = queue_store.get_image_path(queue_id) if queue_id else None
        card_data = queue_store.get_card(queue_id) if queue_id else None

        col_img, col_info = st.columns([1, 1])

        with col_img:
            st.markdown(f"**Original Photograph:** `{inspected_fname}`")
            if image_path and image_path.exists():
                with Image.open(image_path) as card_img:
                    st.image(card_img.copy(), use_container_width=True)
            else:
                st.info("Original image is no longer available in the local queue.")
            if card_data and card_data["raw_text"]:
                with st.expander("Original OCR text"):
                    st.text(card_data["raw_text"])

        with col_info:
            is_awaiting = inspected_rec.get("verification_needed", False)
            unverified = inspected_rec.get("unverified_fields", [])

            if is_awaiting:
                st.warning(f"⚠️ **Awaiting Manual Verification** — Missing/Uncertain: {', '.join(unverified) if unverified else 'Low confidence OCR'}")
                if st.button("✅ Mark Record as Verified", key=f"verify_btn_{selected_idx}"):
                    inspected_rec["verification_needed"] = False
                    inspected_rec["unverified_fields"] = []
                    if queue_id:
                        queue_store.save_record(queue_id, inspected_rec)
                    st.success("Record marked as verified!")
                    st.rerun()
            else:
                st.success("✅ **Record Verified**")

            # Comparative 2-column view of English vs Hindi fields
            c_en, c_hi = st.columns(2)
            with c_en:
                st.markdown("**🇬🇧 English Fields**")
                st.write(f"**Name:** {inspected_rec.get('full_name_en') or '—'}")
                st.write(f"**Designation:** {inspected_rec.get('designation_en') or '—'}")
                st.write(f"**Company:** {inspected_rec.get('company_name_en') or '—'}")
                st.write(f"**Address:** {inspected_rec.get('address_en') or '—'}")
                st.write(f"**City/State:** {inspected_rec.get('city_en') or ''} {inspected_rec.get('state_en') or ''}".strip() or "—")

            with c_hi:
                st.markdown("**🇮🇳 Hindi Fields (हिंदी)**")
                st.write(f"**नाम (Name):** {inspected_rec.get('full_name_hi') or '—'}")
                st.write(f"**पद (Designation):** {inspected_rec.get('designation_hi') or '—'}")
                st.write(f"**कंपनी (Company):** {inspected_rec.get('company_name_hi') or '—'}")
                st.write(f"**पता (Address):** {inspected_rec.get('address_hi') or '—'}")
                st.write(f"**शहर/राज्य (City/State):** {inspected_rec.get('city_hi') or ''} {inspected_rec.get('state_hi') or ''}".strip() or "—")

            st.markdown("---")
            st.markdown("**📞 Contact Details & Links**")
            st.write(f"**Mobile:** `{inspected_rec.get('mobile_number') or '—'}` | **Alternate:** `{inspected_rec.get('alternate_number') or '—'}`")
            st.write(f"**Email:** {inspected_rec.get('email') or '—'} | **Web:** {inspected_rec.get('website') or '—'}")
            st.write(f"**PIN Code:** `{inspected_rec.get('pin_code') or '—'}` | **Country:** {inspected_rec.get('country') or '—'}")
            st.write(f"**LinkedIn:** {inspected_rec.get('linkedin') or '—'}")

else:
    st.info("No extracted records yet. Process cards in Section 2 to populate the table.")


# ==========================================
# Section 4: Excel Export
# ==========================================
st.markdown('<div class="section-header">4. Excel Export</div>', unsafe_allow_html=True)

if not st.session_state.extracted_records:
    st.info("Extract card information above to unlock Excel export.")
else:
    export_option = st.radio(
        "Choose Excel Export Mode:",
        ["Option 1: Create New Excel Workbook", "Option 2: Append to Existing Excel Workbook"],
        horizontal=True,
    )

    if export_option == "Option 1: Create New Excel Workbook":
        col_name, col_btn = st.columns([2, 1])
        with col_name:
            out_filename = st.text_input("Output Excel Filename", value="visiting_cards_multilingual.xlsx")
            if not out_filename.endswith(".xlsx"):
                out_filename += ".xlsx"

        records_json = json.dumps(
            st.session_state.extracted_records,
            ensure_ascii=False,
            sort_keys=True,
        )
        excel_bytes, excel_generation_s = create_excel_cached(records_json)
        st.caption(f"Excel generation: {excel_generation_s:.3f} s")

        with col_btn:
            st.write("")
            st.write("")
            st.download_button(
                label=f"📥 Download '{out_filename}'",
                data=excel_bytes,
                file_name=out_filename,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary",
                use_container_width=True,
            )

    else:
        st.write("Upload an existing `.xlsx` file to append newly extracted multilingual records without altering existing data:")
        existing_file = st.file_uploader(
            "Upload Existing Excel File (.xlsx)",
            type=["xlsx"],
            key="existing_excel_uploader",
        )

        if existing_file:
            try:
                file_bytes = existing_file.getvalue()
                sheet_names = read_existing_excel_sheets(file_bytes)
                selected_sheet = st.selectbox("Select Target Worksheet:", sheet_names)

                preview_info = preview_existing_excel_info(file_bytes, selected_sheet)
                st.caption(
                    f"Selected Sheet: **{selected_sheet}** | Existing Records: **{preview_info['record_count']}** | "
                    f"Existing Columns: {len(preview_info['existing_headers'])}"
                )

                if st.button("🔄 Append Records to Excel Workbook", type="primary"):
                    excel_started = time.perf_counter()
                    updated_buf, count = append_to_existing_excel(
                        file_bytes,
                        selected_sheet,
                        st.session_state.extracted_records,
                    )
                    st.success(f"Appended {count} records to worksheet '{selected_sheet}'!")
                    st.caption(
                        f"Excel append generation: {time.perf_counter() - excel_started:.3f} s"
                    )

                    download_name = existing_file.name.replace(".xlsx", "_updated.xlsx")
                    st.download_button(
                        label=f"📥 Download Updated Excel ({download_name})",
                        data=updated_buf,
                        file_name=download_name,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        type="primary",
                    )
            except Exception as e:
                st.error(f"Error reading existing Excel workbook: {e}")
