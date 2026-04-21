"""
frontend/streamlit_app.py
─────────────────────────
AI Resume Recommendation System — Ingestion Dashboard

Provides a clean, dashboard-style Streamlit UI with two tabs:
  • Tab 1 — Upload Job Description
  • Tab 2 — Upload & Process Resumes

Backend base URL is resolved from (in priority order):
  1. Environment variable  API_BASE_URL
  2. st.secrets["API_BASE_URL"]
  3. Hard-coded fallback    http://localhost:8000

Run:
    streamlit run frontend/streamlit_app.py
"""

from __future__ import annotations

import os
import time
from io import BytesIO

import httpx
import streamlit as st

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Resume Recommendation",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── resolve backend URL ───────────────────────────────────────────────────────
def _get_api_base() -> str:
    if url := os.getenv("API_BASE_URL"):
        return url.rstrip("/")
    try:
        return str(st.secrets["API_BASE_URL"]).rstrip("/")
    except (KeyError, FileNotFoundError):
        return "http://localhost:8000"


API_BASE = _get_api_base()

# ── custom CSS for a polished look ────────────────────────────────────────────
st.markdown(
    """
    <style>
        /* Main title gradient */
        .main-title {
            font-size: 2.4rem;
            font-weight: 800;
            background: linear-gradient(90deg, #4f8ef7, #a259f7);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0;
        }
        .subtitle {
            color: #8a93a2;
            font-size: 1rem;
            margin-top: 0.2rem;
            margin-bottom: 1.5rem;
        }
        /* Metric cards */
        div[data-testid="metric-container"] {
            background: #1e2130;
            border: 1px solid #2e3250;
            border-radius: 10px;
            padding: 1rem 1.5rem;
        }
        /* Log panel */
        .log-panel {
            background: #0e1117;
            border: 1px solid #2e3250;
            border-radius: 8px;
            padding: 0.8rem 1rem;
            font-family: monospace;
            font-size: 0.85rem;
            color: #c8d0e0;
            max-height: 220px;
            overflow-y: auto;
        }
        /* Step badge */
        .step-badge {
            display: inline-block;
            background: #1f3a5f;
            color: #4f8ef7;
            border-radius: 999px;
            padding: 2px 12px;
            font-size: 0.78rem;
            font-weight: 600;
            margin-right: 6px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── header ────────────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">🤖 AI Resume Recommendation System</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Ingestion Pipeline — upload a JD and resumes, '
    'generate embeddings, and index them into FAISS.</p>',
    unsafe_allow_html=True,
)

# Backend URL indicator in sidebar
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.code(f"Backend: {API_BASE}", language="text")
    st.caption("Set `API_BASE_URL` env var or `secrets.toml` to change.")
    if st.button("🔍 Health Check"):
        try:
            r = httpx.get(f"{API_BASE}/health", timeout=5)
            if r.status_code == 200:
                st.success("Backend is online ✅")
            else:
                st.warning(f"HTTP {r.status_code}")
        except Exception as e:
            st.error(f"Cannot reach backend: {e}")

st.divider()

# ── tabs ──────────────────────────────────────────────────────────────────────
tab_jd, tab_resumes = st.tabs(["📄 Upload Job Description", "📂 Upload Resumes"])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — JOB DESCRIPTION
# ═══════════════════════════════════════════════════════════════════════════════
with tab_jd:
    st.subheader("Upload Job Description")
    st.markdown(
        "Upload the **Job Description** file for this hiring round.  "
        "Accepted formats: **PDF**, **TXT**."
    )

    col_upload, col_info = st.columns([2, 1])

    with col_upload:
        jd_file = st.file_uploader(
            "Choose a JD file",
            type=["pdf", "txt"],
            key="jd_uploader",
            help="PDF or plain-text job description",
        )

        upload_jd_btn = st.button(
            "⬆️  Upload JD",
            disabled=jd_file is None,
            use_container_width=True,
        )

    with col_info:
        st.markdown("#### What happens here")
        st.markdown(
            "- File is saved to `data/jd/`\n"
            "- No parsing is done at this stage\n"
            "- JD parsing is handled by your teammates' modules"
        )

    if upload_jd_btn and jd_file is not None:
        with st.status("Uploading Job Description…", expanded=True) as status_box:
            st.write("📤 Sending file to backend…")
            try:
                response = httpx.post(
                    f"{API_BASE}/upload-jd",
                    files={"file": (jd_file.name, BytesIO(jd_file.getvalue()), jd_file.type)},
                    timeout=30,
                )
                if response.status_code == 200:
                    data = response.json()
                    status_box.update(label="JD uploaded successfully ✅", state="complete")
                    st.success(
                        f"**{data['filename']}** saved "
                        f"({data['size_bytes']:,} bytes)"
                    )
                else:
                    status_box.update(label="Upload failed ❌", state="error")
                    st.error(f"Backend returned HTTP {response.status_code}: {response.text}")
            except httpx.ConnectError:
                status_box.update(label="Connection error ❌", state="error")
                st.error(
                    f"Could not connect to the backend at **{API_BASE}**.  "
                    "Is `uvicorn` running?"
                )
            except Exception as exc:
                status_box.update(label="Unexpected error ❌", state="error")
                st.error(f"Error: {exc}")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — RESUME UPLOAD & INGESTION
# ═══════════════════════════════════════════════════════════════════════════════
with tab_resumes:
    st.subheader("Upload & Process Resumes")
    st.markdown(
        "Upload one or more **PDF resumes**.  "
        "The pipeline will extract text, generate 384-dim embeddings, "
        "and index them into FAISS."
    )

    col_files, col_opts = st.columns([2, 1])

    with col_files:
        resume_files = st.file_uploader(
            "Choose resume PDFs",
            type=["pdf", "txt"],
            accept_multiple_files=True,
            key="resume_uploader",
            help="Upload one or more PDF / TXT resumes",
        )

    with col_opts:
        st.markdown("#### Ingestion Mode")
        ingest_mode = st.radio(
            "Mode",
            options=["append", "replace"],
            index=0,
            help=(
                "**append** — add to existing FAISS index\n\n"
                "**replace** — wipe the index and rebuild from uploaded files"
            ),
            label_visibility="collapsed",
        )
        st.caption(
            "🔄 **append**: keeps existing vectors and adds new ones.\n\n"
            "⚠️ **replace**: deletes the current index entirely."
        )

    if resume_files:
        st.info(f"**{len(resume_files)} file(s) selected:** " + ", ".join(f.name for f in resume_files))

    process_btn = st.button(
        "⚙️  Process Resumes",
        disabled=not resume_files,
        use_container_width=True,
        type="primary",
    )

    if process_btn and resume_files:
        st.markdown("---")
        st.markdown("### 🔄 Processing Pipeline")

        # ── pipeline steps ────────────────────────────────────────────────────
        STEPS = [
            ("📤", "Uploading files"),
            ("📝", "Extracting resume text"),
            ("🧠", "Generating embeddings"),
            ("🗄️", "Indexing vectors in FAISS"),
        ]
        progress_bar = st.progress(0, text="Starting…")
        log_placeholder = st.empty()
        log_lines: list[str] = []

        def append_log(msg: str) -> None:
            log_lines.append(msg)
            log_html = "<br>".join(log_lines[-20:])   # keep last 20 lines visible
            log_placeholder.markdown(
                f'<div class="log-panel">{log_html}</div>',
                unsafe_allow_html=True,
            )

        result_placeholder = st.empty()

        # ── step 1: show upload intent ────────────────────────────────────────
        progress_bar.progress(10, text=f"<span class='step-badge'>1/4</span> {STEPS[0][1]}")
        for f in resume_files:
            append_log(f"📄 Queued: <b>{f.name}</b>")
        time.sleep(0.3)

        # ── step 2: send to backend ───────────────────────────────────────────
        progress_bar.progress(25, text=f"2/4 — {STEPS[1][1]}")
        append_log("🚀 Sending files to backend…")

        try:
            files_payload = [
                ("files", (f.name, BytesIO(f.getvalue()), "application/octet-stream"))
                for f in resume_files
            ]

            with httpx.Client(timeout=300) as client:
                response = client.post(
                    f"{API_BASE}/upload-resumes",
                    files=files_payload,
                    params={"mode": ingest_mode},
                )

            # ── step 3: embeddings (happened server-side; simulate progress) ──
            progress_bar.progress(60, text=f"3/4 — {STEPS[2][1]}")
            append_log("🧠 Generating embeddings… (running on server)")
            time.sleep(0.4)

            # ── step 4: indexing ──────────────────────────────────────────────
            progress_bar.progress(85, text=f"4/4 — {STEPS[3][1]}")
            append_log("🗄️  Indexing vectors into FAISS…")
            time.sleep(0.3)

            if response.status_code == 200:
                data = response.json()
                progress_bar.progress(100, text="✅ Ingestion complete!")

                for fname in data.get("processed_files", []):
                    append_log(f"✔ Indexed: <b>{fname}</b>")
                if warnings := data.get("warnings"):
                    for w in warnings:
                        append_log(f"⚠️ {w}")

                append_log(
                    f"<b style='color:#4ade80'>Done — "
                    f"{data['resumes_processed']} resume(s) processed, "
                    f"{data['vectors_in_index']} vector(s) in index.</b>"
                )

                # ── results panel ─────────────────────────────────────────────
                st.markdown("---")
                st.markdown("### ✅ Ingestion Complete")

                m1, m2, m3 = st.columns(3)
                m1.metric(
                    label="📄 Total Resumes Indexed",
                    value=data["resumes_processed"],
                )
                m2.metric(
                    label="🗄️ Total Vectors Stored",
                    value=data["vectors_in_index"],
                )
                m3.metric(
                    label="⚙️ Ingestion Mode",
                    value=data["mode"].capitalize(),
                )

                with st.expander("📋 Processed Files", expanded=False):
                    for i, fname in enumerate(data.get("processed_files", []), start=1):
                        st.write(f"{i}. `{fname}`")

                if data.get("warnings"):
                    with st.expander("⚠️ Warnings", expanded=True):
                        for w in data["warnings"]:
                            st.warning(w)

            else:
                progress_bar.progress(100, text="❌ Error during ingestion")
                append_log(f"❌ Backend returned HTTP {response.status_code}")
                st.error(
                    f"Backend returned **HTTP {response.status_code}**:\n\n"
                    f"```\n{response.text}\n```"
                )

        except httpx.ConnectError:
            progress_bar.progress(100, text="❌ Connection failed")
            append_log(f"❌ Cannot connect to {API_BASE}")
            st.error(
                f"Could not connect to the backend at **{API_BASE}**.  "
                "Make sure `uvicorn` is running."
            )
        except Exception as exc:
            progress_bar.progress(100, text="❌ Unexpected error")
            append_log(f"❌ Error: {exc}")
            st.error(f"Unexpected error: {exc}")

