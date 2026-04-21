"""
frontend/streamlit_app.py
─────────────────────────
AI Resume Recommendation System — Complete Dashboard

Enhanced UI with Ajay's hybrid matching integration:
  • Tab 1 — Upload Job Descriptions
  • Tab 2 — Upload Resumes
  • Tab 3 — Search & Match (NEW!)

Run:
    streamlit run frontend/streamlit_app.py
"""

from __future__ import annotations

import os
import time
from io import BytesIO

import httpx
import streamlit as st
import streamlit.components.v1 as components

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

# ── custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
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
        div[data-testid="metric-container"] {
            background: #1e2130;
            border: 1px solid #2e3250;
            border-radius: 10px;
            padding: 1rem 1.5rem;
        }
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
        .candidate-card {
            background: linear-gradient(135deg, #1a1d2e 0%, #242837 100%);
            border: 1px solid #2e3250;
            border-left: 4px solid #4f8ef7;
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            transition: all 0.3s ease;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
        }
        .candidate-card:hover {
            border-left-color: #a259f7;
            box-shadow: 0 4px 16px rgba(79, 142, 247, 0.2);
            transform: translateY(-2px);
        }
        .score-badge {
            display: inline-block;
            background: linear-gradient(135deg, #1f3a5f 0%, #2a4a6f 100%);
            color: #4f8ef7;
            border: 1px solid #3a5a7f;
            border-radius: 6px;
            padding: 4px 10px;
            font-size: 0.8rem;
            font-weight: 600;
            margin-right: 6px;
            margin-bottom: 4px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── header ────────────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">🤖 AI Resume Recommendation System</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Hybrid Matching with FAISS + Skill Overlap + Experience Alignment</p>',
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
tab_jd, tab_resumes, tab_search = st.tabs([
    "📄 Upload Job Descriptions",
    "📂 Upload Resumes",
    "🔍 Search & Match"
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — JOB DESCRIPTION
# ═══════════════════════════════════════════════════════════════════════════════
with tab_jd:
    st.subheader("Upload Job Descriptions")
    st.markdown(
        "Upload **Job Description** files (PDF or TXT). "
        "The system will parse them and generate embeddings for matching."
    )

    col_upload, col_info = st.columns([2, 1])

    with col_upload:
        jd_files = st.file_uploader(
            "Choose JD file(s)",
            type=["pdf", "txt"],
            accept_multiple_files=True,
            key="jd_uploader",
            help="PDF or plain-text job descriptions",
        )

        upload_jd_btn = st.button(
            "⬆️  Upload & Process JDs",
            disabled=not jd_files,
            use_container_width=True,
        )

    with col_info:
        st.markdown("#### What happens here")
        st.markdown(
            "- Files saved to `data/jd/`\n"
            "- Text extraction & parsing\n"
            "- Embedding generation\n"
            "- Artifact creation for matching"
        )

    if upload_jd_btn and jd_files:
        with st.status("Uploading Job Descriptions…", expanded=True) as status_box:
            st.write(f"📤 Sending {len(jd_files)} file(s) to backend…")
            try:
                files_payload = [
                    ("files", (f.name, BytesIO(f.getvalue()), f.type))
                    for f in jd_files
                ]

                with httpx.Client(timeout=300) as client:
                    response = client.post(
                        f"{API_BASE}/upload_jd",
                        files=files_payload,
                    )

                if response.status_code == 200:
                    data = response.json()
                    status_box.update(label="JDs uploaded successfully ✅", state="complete")
                    st.success(f"**{data.get('message', 'Success')}**")

                    if summary := data.get("pipeline_summary"):
                        with st.expander("📊 Pipeline Summary"):
                            st.json(summary)
                else:
                    status_box.update(label="Upload failed ❌", state="error")
                    st.error(f"Backend returned HTTP {response.status_code}: {response.text}")
            except httpx.ConnectError:
                status_box.update(label="Connection error ❌", state="error")
                st.error(f"Could not connect to backend at **{API_BASE}**")
            except Exception as exc:
                status_box.update(label="Unexpected error ❌", state="error")
                st.error(f"Error: {exc}")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — RESUME UPLOAD
# ═══════════════════════════════════════════════════════════════════════════════
with tab_resumes:
    st.subheader("Upload & Process Resumes")
    st.markdown(
        "Upload **resume PDFs**. The pipeline will extract sections, "
        "generate embeddings, and index them for matching."
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
        st.markdown("#### Pipeline Info")
        st.markdown(
            "📝 **Section extraction**\n\n"
            "🧠 **Embedding generation**\n\n"
            "🗄️ **FAISS indexing**"
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
        with st.status("Processing Resumes…", expanded=True) as status_box:
            st.write(f"📤 Uploading {len(resume_files)} resume(s)…")

            try:
                files_payload = [
                    ("files", (f.name, BytesIO(f.getvalue()), "application/octet-stream"))
                    for f in resume_files
                ]

                with httpx.Client(timeout=300) as client:
                    response = client.post(
                        f"{API_BASE}/upload_resumes",
                        files=files_payload,
                    )

                if response.status_code == 200:
                    data = response.json()
                    status_box.update(label="Resumes processed successfully ✅", state="complete")

                    st.success(f"**{data.get('message', 'Success')}**")

                    if summary := data.get("pipeline_summary"):
                        st.markdown("### ✅ Processing Complete")

                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("📄 Resumes", summary.get("num_resumes", 0))
                        col2.metric("📋 Jobs", summary.get("num_jobs", 0))
                        col3.metric("📦 Chunks", summary.get("num_chunks", 0))
                        col4.metric("🔤 Skills", summary.get("num_skills", 0))

                        with st.expander("📊 Artifacts Generated"):
                            for name, path in summary.get("artifacts", {}).items():
                                st.code(f"{name}: {path}")
                else:
                    status_box.update(label="Processing failed ❌", state="error")
                    st.error(f"Backend returned HTTP {response.status_code}: {response.text}")

            except httpx.ConnectError:
                status_box.update(label="Connection failed ❌", state="error")
                st.error(f"Could not connect to backend at **{API_BASE}**")
            except Exception as exc:
                status_box.update(label="Unexpected error ❌", state="error")
                st.error(f"Error: {exc}")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — SEARCH & MATCH
# ═══════════════════════════════════════════════════════════════════════════════
with tab_search:
    st.subheader("🔍 Search for Matching Resumes")
    st.markdown("Select a job and find the best matching candidates using hybrid scoring.")

    # ── Load available jobs ───────────────────────────────────────────────────
    try:
        jobs_response = httpx.get(f"{API_BASE}/search/jobs", timeout=10)
        if jobs_response.status_code == 200:
            jobs_data = jobs_response.json()
            jobs = jobs_data.get("jobs", [])

            if not jobs:
                st.warning("No jobs found. Please upload job descriptions first.")
            else:
                col_search, col_params = st.columns([2, 1])

                with col_search:
                    # Job selection
                    job_options = {f"{j['job_id']} - {j['job_title']}": j['job_id'] for j in jobs}
                    selected_job = st.selectbox(
                        "Select Job",
                        options=list(job_options.keys()),
                        help="Choose a job to find matching candidates"
                    )

                    job_id = job_options[selected_job]

                    # Number of results
                    top_k = st.slider("Number of results", min_value=1, max_value=50, value=10)

                    # LLM reranking option
                    use_llm = st.checkbox(
                        "Enable semantic reranking (requires Ollama)",
                        value=False,
                        help="Use Ollama's multi-qa-MiniLM model for advanced semantic reranking"
                    )

                with col_params:
                    st.markdown("#### Scoring Weights")
                    with st.expander("⚙️ Adjust weights", expanded=False):
                        semantic_weight = st.slider("Semantic", 0.0, 1.0, 0.5, 0.1)
                        skill_weight = st.slider("Skills", 0.0, 1.0, 0.3, 0.1)
                        experience_weight = st.slider("Experience", 0.0, 1.0, 0.1, 0.1)
                        domain_weight = st.slider("Domain", 0.0, 1.0, 0.1, 0.1)

                    st.caption(
                        f"**Current weights:**\n\n"
                        f"🧠 Semantic: {semantic_weight:.1f}\n\n"
                        f"🔤 Skills: {skill_weight:.1f}\n\n"
                        f"📅 Experience: {experience_weight:.1f}\n\n"
                        f"🏢 Domain: {domain_weight:.1f}"
                    )

                # Search button
                search_btn = st.button("🚀 Search Candidates", use_container_width=True, type="primary")

                if search_btn:
                    with st.spinner("Searching for best candidates…"):
                        try:
                            search_payload = {
                                "job_id": job_id,
                                "top_k": top_k,
                                "use_llm_reranking": use_llm,
                                "semantic_weight": semantic_weight,
                                "skill_weight": skill_weight,
                                "experience_weight": experience_weight,
                                "domain_weight": domain_weight,
                            }

                            search_response = httpx.post(
                                f"{API_BASE}/search",
                                json=search_payload,
                                timeout=120,
                            )

                            if search_response.status_code == 200:
                                search_data = search_response.json()
                                results = search_data.get("results", [])

                                st.success(f"Found {len(results)} matching candidates!")

                                st.markdown("---")
                                st.markdown("### 🏆 Top Candidates")

                                for i, candidate in enumerate(results, 1):
                                    candidate_name = candidate.get('candidate_name', candidate.get('resume_id', 'Unknown'))
                                    match_summary = candidate.get('match_summary', '')

                                    # Rank badge color based on position
                                    if i == 1:
                                        rank_color = "#FFD700"  # Gold
                                    elif i == 2:
                                        rank_color = "#C0C0C0"  # Silver
                                    elif i == 3:
                                        rank_color = "#CD7F32"  # Bronze
                                    else:
                                        rank_color = "#4f8ef7"  # Default blue

                                    # Build HTML card
                                    card_html = f"""
                                    <style>
                                        .candidate-card {{
                                            background: linear-gradient(135deg, #1a1d2e 0%, #242837 100%);
                                            border: 1px solid #2e3250;
                                            border-left: 4px solid #4f8ef7;
                                            border-radius: 12px;
                                            padding: 1.5rem;
                                            margin-bottom: 1rem;
                                            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
                                            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                                        }}
                                        .score-badge {{
                                            display: inline-block;
                                            background: linear-gradient(135deg, #1f3a5f 0%, #2a4a6f 100%);
                                            color: #4f8ef7;
                                            border: 1px solid #3a5a7f;
                                            border-radius: 6px;
                                            padding: 4px 10px;
                                            font-size: 0.8rem;
                                            font-weight: 600;
                                            margin-right: 6px;
                                            margin-bottom: 4px;
                                        }}
                                    </style>
                                    <div class="candidate-card">
                                        <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                                            <div style="background: {rank_color}; color: #000; border-radius: 50%; width: 32px; height: 32px; display: flex; align-items: center; justify-content: center; font-weight: bold; margin-right: 12px;">
                                                {i}
                                            </div>
                                            <h3 style="margin: 0; color: #fff;">{candidate_name}</h3>
                                        </div>
                                        
                                        <div style="color: #a8b2c0; font-size: 0.95rem; margin: 0.5rem 0 1rem 0; line-height: 1.5;">
                                            {match_summary}
                                        </div>
                                        
                                        <div style="margin-bottom: 0.5rem;">
                                            <span class="score-badge">Overall: {candidate['hybrid_score']:.3f}</span>
                                            <span class="score-badge">Semantic: {candidate['semantic_score']:.3f}</span>
                                            <span class="score-badge">Skills: {candidate['skill_score']:.3f}</span>
                                            <span class="score-badge">Experience: {candidate['experience_score']:.3f}</span>
                                            <span class="score-badge">Domain: {candidate['domain_score']:.3f}</span>
                                        </div>
                                        
                                        <p style="color: #6b7280; font-size: 0.85rem; margin: 0.5rem 0 0 0;">
                                            <strong>Matched Sections:</strong> {', '.join(candidate.get('matched_sections', []))}
                                        </p>
                                        
                                        <p style="color: #6b7280; font-size: 0.75rem; margin: 0.3rem 0 0 0;">
                                            <em>Resume ID: {candidate['resume_id']}</em>
                                        </p>
                                    </div>
                                    """

                                    components.html(card_html, height=280)

                                # Download results
                                import json
                                results_json = json.dumps(results, indent=2)
                                st.download_button(
                                    "📥 Download Results (JSON)",
                                    data=results_json,
                                    file_name=f"candidates_{job_id}.json",
                                    mime="application/json"
                                )

                            else:
                                st.error(f"Search failed: HTTP {search_response.status_code}\n\n{search_response.text}")

                        except Exception as e:
                            st.error(f"Search error: {e}")

        else:
            st.error("Failed to load jobs. Please make sure you've uploaded job descriptions.")

    except httpx.ConnectError:
        st.error(f"Cannot connect to backend at **{API_BASE}**. Make sure the server is running.")
    except Exception as e:
        st.warning(f"Could not load jobs: {e}")

