import streamlit as st
import os
import re
import time

from langchain_community.retrievers import WikipediaRetriever
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
# Using centered layout to keep the UI readable on large screens.
# Wide layout caused content to spread too thin on high-res monitors.
st.set_page_config(
    page_title="Market Research Assistant",
    page_icon="📊",
    layout="centered"
)


# =============================================================================
# GLOBAL CSS
# =============================================================================
# Custom styling to give the app a professional, card-based look.
# I chose DM Sans for readability and DM Mono for timing displays.
# The step badges, connectors, and report area are all styled here
# so the workflow feels like a guided process rather than a plain form.
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

:root {
    --border: #e2e4e9;
    --text-secondary: #5f6672;
    --text-muted: #9199a5;
    --accent: #2563eb;
    --accent-light: #eff4ff;
    --accent-border: #bfdbfe;
    --success: #16a34a;
    --success-light: #f0fdf4;
    --success-border: #bbf7d0;
}

/* ── Hero ── */
.hero-section {
    text-align: center;
    padding-bottom: 24px;
}
.hero-section h1 {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 1.75rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em;
    margin-bottom: 6px !important;
    padding-top: 0 !important;
}
.hero-section p {
    color: var(--text-secondary);
    font-size: 0.95rem;
    max-width: 520px;
    margin: 0 auto;
}

/* ── Reset hint ── */
.reset-hint {
    font-size: 0.88rem;
    color: var(--text-secondary);
    line-height: 2.4;
}

/* ── Step header ── */
.step-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 4px;
}
.step-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 30px;
    height: 30px;
    border-radius: 8px;
    font-size: 0.8rem;
    font-weight: 700;
    background: var(--accent-light);
    color: var(--accent);
    border: 1px solid var(--accent-border);
    flex-shrink: 0;
}
.step-badge.done {
    background: var(--success-light);
    color: var(--success);
    border-color: var(--success-border);
}
.step-title {
    font-size: 1.05rem;
    font-weight: 600;
    letter-spacing: -0.01em;
    color: #1a1d23;
}
.step-subtitle {
    font-size: 0.82rem;
    color: var(--text-muted);
    margin-left: auto;
    font-weight: 400;
    font-family: 'DM Mono', monospace;
}
.step-desc {
    font-size: 0.88rem;
    color: var(--text-secondary);
    margin-bottom: 8px;
    line-height: 1.55;
}

/* ── Connector ── */
.connector {
    width: 2px;
    height: 18px;
    background: var(--border);
    margin: 4px auto;
    border-radius: 2px;
}

/* ── Report area ── */
.report-area {
    background: #fafbfc;
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 24px 28px;
    margin-top: 8px;
    margin-bottom: 8px;
}
.report-area h1 {
    font-size: 1.15rem !important;
    font-weight: 600 !important;
    color: #1a1d23;
    margin-bottom: 12px !important;
}
.report-area h2 {
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    color: var(--accent) !important;
    margin-top: 16px !important;
    margin-bottom: 6px !important;
}
.report-area p {
    font-size: 0.88rem;
    color: var(--text-secondary);
    margin-bottom: 10px;
    line-height: 1.6;
}

/* ── Footer ── */
.app-footer {
    text-align: center;
    margin-top: 40px;
    padding-bottom: 20px;
    font-size: 0.75rem;
    color: var(--text-muted);
}
</style>
""", unsafe_allow_html=True)


# =============================================================================
# CONSTANTS (SAFETY & PERFORMANCE)
# =============================================================================
# These constants act as guardrails for the entire pipeline.
# Each timeout is tuned based on testing — validation is fast (single LLM call),
# retrieval is slower (multiple scoring calls), and generation depends on report length.
# MAX_CONTEXT_CHARS prevents sending too much text to the LLM, which would
# increase cost, slow down generation, and risk hitting token limits.
MAX_VALIDATION_TIME = 15        # seconds — single LLM call, should be quick
MAX_RETRIEVAL_TIME = 60         # seconds — includes Wikipedia API + scoring each page
MAX_GENERATION_TIME = 60        # seconds — generating a 500-word structured report
MAX_CONTEXT_CHARS = 12000       # total Wikipedia text sent to LLM (avoids token overflow)
MAX_REPORT_WORDS = 500          # enforced both in the prompt and post-generation
REQUIRED_WIKI_PAGES = 5         # target number of sources for a well-grounded report


# =============================================================================
# API KEY CONFIGURATION
# =============================================================================
# Secrets-based approach for deployment (Streamlit Cloud handles this securely).
# The fallback text input is only for local development — it's password-masked
# so the key is never visible on screen. The key is never stored in the code.
try:
    api_key = st.secrets["OPENAI_API_KEY"]
    using_secrets = True
except (KeyError, FileNotFoundError):
    using_secrets = False
    api_key = None


# =============================================================================
# SIDEBAR
# =============================================================================
st.sidebar.header("⚙️ Settings")

if not using_secrets:
    api_key = st.sidebar.text_input(
        "OpenAI API Key",
        type="password",
        help="Used only for local testing."
    )
else:
    st.sidebar.success("🔐 Using secure API key")

# Model and page count are shown but disabled — this keeps the UI transparent
# about what settings are being used, while preventing the user from changing
# them and potentially breaking the pipeline or running up costs.
st.sidebar.selectbox(
    "Model",
    ["gpt-5-mini"],
    index=0,
    disabled=True,
    help="Model is set to gpt-5-mini for optimal performance"
)

st.sidebar.number_input(
    "Number of Wikipedia pages",
    value=5,
    min_value=3,
    max_value=10,
    disabled=True,
    help="Target number of Wikipedia sources"
)


# =============================================================================
# MAIN HEADER
# =============================================================================
st.markdown(
    """
    <div class="hero-section">
        <h1>📊 Market Research Assistant</h1>
        <p>AI-powered industry market research assistant that provides a report on an industry chosen by the user.</p>
    </div>
    """,
    unsafe_allow_html=True
)


# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================
# Streamlit re-runs the entire script on every interaction, so session state
# is essential to persist data between steps. Each key tracks a specific part
# of the workflow. The retrieval_message fields exist because st.rerun() wipes
# any st.success/st.warning messages — storing them in state lets us re-display
# them after the rerun.
DEFAULT_STATE = {
    "industry_valid": False,
    "sources_ready": False,
    "report_generated": False,
    "industry_feedback": "",
    "docs": [],
    "urls": [],
    "report": "",
    "current_industry": "",
    "validation_time": 0,
    "retrieval_time": 0,
    "generation_time": 0,
    "retrieval_message": "",         # persists feedback across st.rerun()
    "retrieval_message_type": ""     # "success" or "warning"
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =============================================================================
# START OVER
# =============================================================================
# Resets all session state so the user can research a different industry
# without refreshing the browser. Also clears the text input widget.
col_reset_btn, col_reset_hint = st.columns([1, 4])
with col_reset_btn:
    if st.button("🔄 Start Over"):
        for key, value in DEFAULT_STATE.items():
            st.session_state[key] = value
        if "industry_input" in st.session_state:
            del st.session_state["industry_input"]
        st.rerun()
with col_reset_hint:
    st.markdown(
        '<span class="reset-hint">Reset the workflow to research a different industry</span>',
        unsafe_allow_html=True
    )


# =============================================================================
# CHECK API KEY
# =============================================================================
# Hard stop if no key is configured — there's no point rendering the rest
# of the UI if the LLM calls will fail anyway.
if not api_key:
    st.warning("⚠️ Please configure an OpenAI API key to proceed.")
    st.stop()


# =============================================================================
# HELPER: Render step header HTML
# =============================================================================
# Generates the numbered badge (1/2/3) that turns into a green checkmark
# when the step completes. Also shows elapsed time for completed steps.
def step_header_html(number, title, done=False, elapsed=None):
    badge = f'<span class="step-badge done">✓</span>' if done else f'<span class="step-badge">{number}</span>'
    time_html = f'<span class="step-subtitle">⏱ {elapsed:.2f}s</span>' if done and elapsed and elapsed > 0 else ''
    return f'<div class="step-header">{badge}<span class="step-title">{title}</span>{time_html}</div>'


# =============================================================================
# HELPER: Convert markdown report to HTML
# =============================================================================
# The LLM outputs markdown (# and ## headers), but we need to render the report
# inside a styled <div>. Streamlit's st.markdown() can't nest content inside
# custom HTML divs across multiple calls, so we convert the markdown to HTML
# and render everything in a single st.markdown() call.
# We also strip backticks as a safety net — the prompt tells the LLM not to
# use them, but LLMs don't always follow instructions perfectly.
def report_to_html(report_text):
    """Convert the LLM markdown report to HTML for rendering inside a styled div."""
    html = report_text.replace('`', '')
    # Must convert ## before # to avoid ## being matched by the # pattern
    html = re.sub(r'^## (.+)$', r'<h2>\1</h2>', html, flags=re.MULTILINE)
    html = re.sub(r'^# (.+)$', r'<h1>\1</h1>', html, flags=re.MULTILINE)
    # Wrap remaining text lines in <p> tags for consistent styling
    lines = html.split('\n')
    converted = []
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith('<h') and not stripped.startswith('</'):
            converted.append(f'<p>{stripped}</p>')
        else:
            converted.append(line)
    return '\n'.join(converted)


# =============================================================================
# STEP 1 — INDUSTRY INPUT & VALIDATION
# =============================================================================
# This is the first guardrail in the pipeline. The LLM validates whether the
# input is a real, researchable industry BEFORE any Wikipedia calls are made.
# This prevents wasted API calls and ensures the retrieval step gets clean input.
with st.container(border=True):
    st.markdown(
        step_header_html(1, "Validate Industry",
                         done=st.session_state.industry_valid,
                         elapsed=st.session_state.validation_time)
        + '<p class="step-desc">Enter an industry name. We\'ll check it\'s specific enough for structured market research.</p>',
        unsafe_allow_html=True
    )

    # Caching validation results so the same industry isn't re-validated on every
    # Streamlit rerun. The api_key_value parameter is included in the cache hash
    # so that different API keys produce fresh results.
    @st.cache_data(show_spinner=False)
    def validate_industry(industry_name: str, api_key_value: str) -> str:
        # temperature=0.0 because validation needs to be deterministic —
        # the same input should always get the same VALID/INVALID/INAPPROPRIATE result.
        llm = ChatOpenAI(
            model="gpt-5-mini",
            temperature=0.0,
            api_key=api_key_value
        )

        # The prompt handles three categories of bad input:
        # 1. INAPPROPRIATE — offensive/harmful content (blocked immediately)
        # 2. INVALID — abstract concepts, typos, non-words, overly broad terms
        # 3. VALID — real industries that can support market structure analysis
        # SUGGESTED_ALTERNATIVES helps the user recover from invalid input
        # without having to guess what a "valid industry" looks like.
        system_prompt = (
            "You are validating user input for a corporate market research application.\n\n"
            "First, check if the input contains inappropriate, offensive, or sensitive content:\n"
            "- If it contains profanity, hate speech, violence, sexual content, or illegal activities → respond with: VALIDITY: INAPPROPRIATE\n\n"
            "- If it's clearly not a business/industry term → continue validation\n\n"
            "Determine whether the input represents a VALID, CORRECTLY WRITTEN, SPECIFIC, RESEARCHABLE INDUSTRY.\n\n"
            "VALID industries:\n"
            "- Clearly defined economic sectors or value chains\n"
            "- Imply a set of firms producing similar goods or services\n"
            "- Can support analysis of competition, trends, regulation, and market structure\n\n"
            "INVALID industries:\n"
            "- Abstract concepts (e.g. luxury, innovation, music)\n"
            "- Adjectives without a value chain\n"
            "- Misspellings or typographical errors\n"
            "- Non-words or malformed terms\n"
            "- Overly broad terms spanning many unrelated sectors\n"
            "- Inappropriate, offensive, or sensitive content\n\n"
            "Respond STRICTLY in this format:\n"
            "VALIDITY: VALID or INVALID or INAPPROPRIATE\n\n"
            "EXPLANATION: Brief explanation in 1-2 sentences.\n"
            "SUGGESTED_ALTERNATIVES: List up to 3 industry names, separated by commas."
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "Industry input: {industry}")
        ])

        response = llm.invoke(
            prompt.format_messages(industry=industry_name)
        )

        return response.content

    # Using st.form so that pressing Enter in the text field triggers validation
    # directly, without needing to click the button separately.
    with st.form(key="validation_form"):
        industry = st.text_input(
            "Industry name",
            placeholder="e.g. Automotive, Retail",
            key="industry_input",
            label_visibility="collapsed"
        )
        submitted = st.form_submit_button("🔍 Validate Industry")

    if not industry and not st.session_state.industry_valid:
        st.info("ℹ️ Please enter an industry name above to begin validation.")

    if submitted and industry:

        # If the user changed the industry, reset downstream steps so stale
        # sources/report from the previous industry don't carry over.
        if industry != st.session_state.current_industry:
            st.session_state.sources_ready = False
            st.session_state.report_generated = False
            st.session_state.docs = []
            st.session_state.urls = []
            st.session_state.report = ""
            st.session_state.current_industry = industry

        start = time.time()
        with st.spinner("Validating industry input..."):
            try:
                feedback = validate_industry(industry, api_key)
            except Exception as e:
                st.error(f"❌ Validation error: {str(e)}")
                st.stop()

        elapsed = time.time() - start
        st.session_state.validation_time = elapsed

        # Timeout guardrail — if validation takes too long, the input is likely
        # too vague and is causing the LLM to struggle.
        if elapsed > MAX_VALIDATION_TIME:
            st.error(
                "⚠️ Validation took too long. This usually means the industry is too vague.\n\n"
                "Please try a more specific industry name."
            )
            st.stop()

        st.session_state.industry_feedback = feedback

        # Route to the appropriate response based on the LLM's verdict.
        # INAPPROPRIATE is checked first since it's the most critical guardrail.
        if "VALIDITY: INAPPROPRIATE" in feedback:
            st.error("❌ The input contains inappropriate or offensive content. Please enter a valid industry name.\n\n")
            st.session_state.industry_valid = False
        elif "VALIDITY: VALID" in feedback:
            st.success("✅ Industry validated successfully.\n\nYou can now proceed to retrieve Wikipedia sources in Step 2.")
            st.session_state.industry_valid = True
            st.caption(f"⏱️ Validation completed in {elapsed:.2f} seconds")
        else:
            st.error("❌ Industry is not suitable for structured market research.\n\n")
            st.session_state.industry_valid = False

        # Validation details are in an expander so the user can see the full
        # LLM reasoning (including suggested alternatives) without cluttering the UI.
        with st.expander("📋 Validation Details", expanded=True):
            st.markdown(feedback)

    # On rerun (after the page refreshes), re-display the validation status
    # from session state so the user doesn't lose context.
    elif st.session_state.industry_valid and st.session_state.industry_feedback:
        st.success(f"✅ Industry '{st.session_state.current_industry}' is validated.")
        with st.expander("📋 View Validation Details"):
            st.markdown(st.session_state.industry_feedback)


# Visual connector between steps
st.markdown('<div class="connector"></div>', unsafe_allow_html=True)


# =============================================================================
# STEP 2 — WIKIPEDIA RETRIEVAL & RELEVANCE SCORING
# =============================================================================
# This step retrieves Wikipedia pages and scores each one for industry-level
# relevance. The scoring is a critical guardrail — without it, Wikipedia returns
# a mix of company pages, country economy pages, and tangential topics that
# would produce a low-quality report.
with st.container(border=True):
    st.markdown(
        step_header_html(2, "Retrieve Wikipedia Sources",
                         done=st.session_state.sources_ready,
                         elapsed=st.session_state.retrieval_time)
        + '<p class="step-desc">Retrieve and rank relevant Wikipedia pages to build a grounded evidence base.</p>',
        unsafe_allow_html=True
    )

    # Button is disabled until Step 1 completes — enforces the sequential workflow.
    if not st.session_state.industry_valid:
        st.button("📚 Retrieve Wikipedia Pages", disabled=True)

    else:

        def score_relevance(docs, industry_name, api_key):
            """Score each Wikipedia page for industry-level relevance using the LLM.
            
            This is the source quality guardrail. Without it, the report generator
            would receive irrelevant pages and either hallucinate or produce a
            generic report that doesn't actually cover the target industry.
            """
            # temperature=0.0 for consistent, deterministic scoring.
            # Each page should always get the same relevance score.
            llm = ChatOpenAI(
                model="gpt-5-mini",
                temperature=0.0,
                api_key=api_key
            )

            scored_docs = []

            for doc in docs:
                title = doc.metadata.get("title", "")
                # Only send first 1000 chars of each page to keep scoring fast
                # and avoid hitting token limits when processing many pages.
                summary = doc.page_content[:1000]

                # The scoring prompt explicitly penalises common failure modes:
                # - Country economy pages (e.g. "Economy of Puerto Rico") → score 1
                # - General technology pages (e.g. "Digital twin") → score 1
                # - Individual company pages (e.g. "McLaren Automotive") → score 2
                # This was tuned through testing — earlier versions scored these
                # too high, which led to poor source selection.
                prompt = ChatPromptTemplate.from_messages([
                    ("system",
                     "You are evaluating how relevant a Wikipedia page is for "
                     "professional INDUSTRY-LEVEL market research on a SPECIFIC industry.\n\n"
                     "The page must be DIRECTLY about or closely related to the named industry.\n\n"
                     "Scoring guide:\n"
                     "5 = Direct industry page (e.g. 'Retail industry', 'Automotive industry')\n"
                     "4 = Core industry topic (e.g. 'Health insurance' for Healthcare, 'Electric vehicle' for Automotive)\n"
                     "3 = Closely related industry concept (e.g. 'Medical device' for Healthcare)\n"
                     "2 = Single company in the industry, or loosely related sector\n"
                     "1 = Country economy page, general technology page, or only mentions the industry in passing\n"
                     "0 = Unrelated\n\n"
                     "IMPORTANT: Pages about a country's economy, general technologies, or broad topics that merely "
                     "mention the industry should score 1 or below.\n\n"
                     "Return ONLY the number."),
                    ("human",
                     f"Industry: {industry_name}\n\n"
                     f"Title: {title}\n\n"
                     f"Summary:\n{summary}")
                ])

                try:
                    score = int(llm.invoke(prompt.format_messages()).content.strip())
                    scored_docs.append((score, doc))
                # ValueError/TypeError if LLM returns non-numeric text — skip that page
                except (ValueError, TypeError) as e:
                    continue
                # Catch API errors (rate limits, network issues) with a visible warning
                except Exception as e:
                    st.warning(f"⚠️ Could not score '{title}': {str(e)}")
                    continue

            scored_docs.sort(reverse=True, key=lambda x: x[0])
            return scored_docs


        # Cache version key: changing this invalidates all cached retrieval results.
        # This is necessary because the cache is keyed on function arguments only,
        # so changes to the scoring prompt (inside the function body) wouldn't
        # otherwise trigger fresh results. Bump this whenever the prompt changes.
        _RETRIEVAL_CACHE_VERSION = "v7_revert_industry"

        @st.cache_data(show_spinner=False)
        def retrieve_wikipedia(industry_name: str, api_key_value: str, _cache_version: str = _RETRIEVAL_CACHE_VERSION):
            """Retrieve Wikipedia pages and score them for industry relevance.
            
            Results are cached so repeated searches for the same industry
            don't re-call Wikipedia and the scoring LLM unnecessarily.
            """
            # top_k_results=8 and load_max_docs=10 are tuned for balance:
            # enough candidates to find 5 good ones, but not so many that
            # scoring takes too long (each page = one LLM call).
            retriever = WikipediaRetriever(
                top_k_results=8,
                doc_content_chars_max=4000,
                load_max_docs=10
            )

            # Appending "industry" to the search query biases Wikipedia's results
            # toward pages like "Automotive industry" instead of company pages.
            # Tested alternatives like "sector overview" but "industry" performed best.
            initial_docs = retriever.invoke(f"{industry_name} industry")

            if len(initial_docs) == 0:
                return []

            scored_docs = score_relevance(initial_docs, industry_name, api_key_value)

            return scored_docs


        if st.button("📚 Retrieve Wikipedia Pages", disabled=st.session_state.sources_ready):

            start = time.time()

            with st.spinner("🔍 Retrieving and evaluating Wikipedia sources..."):
                try:
                    scored_docs = retrieve_wikipedia(
                        st.session_state.current_industry,
                        api_key
                    )
                except Exception as e:
                    st.error(f"❌ Retrieval error: {str(e)}")
                    st.stop()

            elapsed = time.time() - start
            st.session_state.retrieval_time = elapsed

            if elapsed > MAX_RETRIEVAL_TIME:
                st.warning(
                    "⚠️ Retrieval took longer than expected. "
                    "The industry may be broad or loosely structured on Wikipedia."
                )

            if len(scored_docs) == 0:
                st.error(
                    "❌ No Wikipedia sources were retrieved. "
                    "Please refine the industry name."
                )
                st.session_state.sources_ready = False

            else:
                # Adaptive threshold selection: try to get the best sources available.
                # Prefer score >= 4 (direct industry pages), fall back to >= 3,
                # then >= 2. This ensures we always proceed with something usable
                # while communicating source quality honestly to the user.
                high_quality = [doc for score, doc in scored_docs if score >= 4]
                medium_quality = [doc for score, doc in scored_docs if score >= 3]
                low_quality = [doc for score, doc in scored_docs if score >= 2]

                if len(high_quality) >= REQUIRED_WIKI_PAGES:
                    selected = high_quality[:REQUIRED_WIKI_PAGES]
                    st.session_state.retrieval_message = f"✅ {REQUIRED_WIKI_PAGES} high-quality sources selected."
                    st.session_state.retrieval_message_type = "success"

                elif len(medium_quality) >= REQUIRED_WIKI_PAGES:
                    selected = medium_quality[:REQUIRED_WIKI_PAGES]
                    st.session_state.retrieval_message = (
                        "⚠️ Fewer top-tier sources found. "
                        "Using strongly relevant contextual sources."
                    )
                    st.session_state.retrieval_message_type = "warning"

                elif len(low_quality) >= 3:
                    selected = low_quality[:REQUIRED_WIKI_PAGES]
                    st.session_state.retrieval_message = (
                        f"⚠️ Only {len(low_quality)} moderately relevant sources found. "
                        "Report may be less comprehensive."
                    )
                    st.session_state.retrieval_message_type = "warning"

                else:
                    selected = [doc for _, doc in scored_docs[:3]]
                    st.session_state.retrieval_message = (
                        "⚠️ Limited relevance detected. Proceeding with best available sources."
                    )
                    st.session_state.retrieval_message_type = "warning"

                st.session_state.docs = selected
                st.session_state.urls = [
                    doc.metadata.get("source", "")
                    for doc in selected
                ]
                st.session_state.sources_ready = True
                # Force a clean re-render so sources display properly inside
                # the container. Without this, Streamlit's rendering can show
                # duplicate content from the current run mixed with the rerun.
                st.rerun()

        # These messages are stored in session state (not displayed directly)
        # because st.rerun() above wipes any st.success/st.warning from the
        # current run. Re-displaying from state ensures the user always sees them.
        if st.session_state.retrieval_message:
            if st.session_state.retrieval_message_type == "success":
                st.success(st.session_state.retrieval_message)
            else:
                st.warning(st.session_state.retrieval_message)

        # Show all source URLs so the analyst can verify them — transparency
        # is important for a business tool where decisions depend on the output.
        if st.session_state.sources_ready and st.session_state.urls:
            st.markdown("**Wikipedia Sources Used:**")
            for i, url in enumerate(st.session_state.urls, 1):
                st.markdown(f"{i}. [{url}]({url})")


# Visual connector between steps
st.markdown('<div class="connector"></div>', unsafe_allow_html=True)


# =============================================================================
# STEP 3 — REPORT GENERATION
# =============================================================================
# The report generator is the final step. It takes the filtered Wikipedia sources
# and produces a structured executive report. Multiple guardrails operate here:
# hallucination prevention (source-only instruction), output length control
# (prompt + post-generation check), and content moderation in the prompt.
with st.container(border=True):
    st.markdown(
        step_header_html(3, "Generate Report",
                         done=st.session_state.report_generated,
                         elapsed=st.session_state.generation_time)
        + '<p class="step-desc">Generate a ≤ 500-word executive report based solely on the retrieved sources.</p>',
        unsafe_allow_html=True
    )

    if not st.session_state.sources_ready:
        st.button("📝 Generate Report", disabled=True)
    else:

        if st.button("📝 Generate Report", disabled=st.session_state.report_generated):

            start = time.time()

            with st.spinner("✍️ Generating industry report..."):

                # Build the context from selected sources, respecting the character
                # limit. This prevents sending too much text to the LLM, which would
                # increase latency and risk hitting the model's context window.
                context_parts = []
                total_chars = 0

                for doc in st.session_state.docs:
                    if total_chars + len(doc.page_content) > MAX_CONTEXT_CHARS:
                        break
                    context_parts.append(doc.page_content)
                    total_chars += len(doc.page_content)

                context = "\n\n".join(context_parts)

                # temperature=0.2 for report generation — slightly creative for
                # natural-sounding prose, but low enough to stay factual and
                # not invent information. This is higher than validation (0.0)
                # because we want readable text, not just classification.
                llm = ChatOpenAI(
                    model="gpt-5-mini",
                    temperature=0.2,
                    api_key=api_key
                )

                from datetime import date
                today = date.today().strftime("%B %d, %Y")
                
                # The report prompt has several layers of guardrails:
                # 1. "Use ONLY the provided Wikipedia sources" — hallucination prevention
                # 2. "Do NOT introduce external knowledge" — reinforces #1
                # 3. Word limit in the prompt + post-generation verification
                # 4. "Do NOT use backticks" — prevents formatting issues in the UI
                # 5. Content moderation section — ensures professional output
                # 6. Structured sections with distinct content rules to avoid repetition
                report_prompt = ChatPromptTemplate.from_messages([
                    ("system",
                     f"You are a senior market research analyst at a large multinational corporation.\n"
                     f"Your task is to write a FORMAL INDUSTRY REPORT suitable for executives and strategy teams.\n\n"
                     f"CRITICAL RULES (must follow all):\n"
                     f"- Write EXACTLY {MAX_REPORT_WORDS} words or fewer - do NOT exceed this limit\n"
                     f"- Use ONLY the provided Wikipedia sources\n"
                     f"- Do NOT introduce external knowledge\n"
                     f"- Write in professional, factual, neutral business language\n"
                     f"- Use complete sentences and paragraphs\n"
                     f"- Do NOT use bullet points\n"
                     f"- Do NOT use backticks, inline code, or code formatting of any kind\n"
                     f"- End all sentences properly - never leave incomplete thoughts\n"
                     f"- Each section must contain UNIQUE content - do NOT overlap or repeat information between sections\n\n"
                     f"REPORT STRUCTURE (use these exact sections with DISTINCT content):\n"
                     f"1. Title: [Industry Name] Industry Report\n"
                     f"2. Prepared on: {today}\n\n"
                     f"3. ## Industry Scope and Definition\n"
                     f"   - Define what the industry is and what it encompasses\n"
                     f"   - Explain the industry's boundaries and core activities\n"
                     f"   - DO NOT discuss market structure or players here\n\n"
                     f"4. ## Market Overview\n"
                     f"   - Describe the overall market size, geography, and structure\n"
                     f"   - Discuss market segments, channels, or formats\n"
                     f"   - Mention major companies only in general terms (e.g., 'major retailers', 'leading manufacturers')\n"
                     f"   - DO NOT list specific company names or go into competitive dynamics\n\n"
                     f"5. ## Key Trends and Dynamics\n"
                     f"   - Identify current trends affecting the industry (technology, consumer behavior, regulations)\n"
                     f"   - Focus on industry-wide changes and developments\n"
                     f"   - Discuss barriers to entry, competitive advantages, or risks\n"
                     f"   - DO NOT discuss competitive strategies or strategic considerations\n\n"
                     f"6. ## Conclusion\n"
                     f"   - Summarize key findings and insights from the report focused on the industry's strategic outlook\n"
                     f"FORMAT REQUIREMENTS:\n"
                     f"- Start with # for title (e.g., # Retail Industry Report)\n"
                     f"- Use ## for section headers (e.g., ## Industry Scope and Definition)\n"
                     f"- Write regular text for content (no special formatting)\n"
                     f"- Separate sections with blank lines\n"
                     f"- Keep paragraphs concise (2-4 sentences each)\n\n"
                     f"CONTENT MODERATION:\n"
                     f"- Ensure all content is appropriate, factual, and professional\n"
                     f"- Avoid any inappropriate, offensive, or sensitive content\n"
                     f"- Focus on business and market analysis only"),
                    ("human",
                     "Industry: {industry}\n\n"
                     "Wikipedia sources:\n{context}\n\n"
                     f"- Remember: Maximum {MAX_REPORT_WORDS} words. Do NOT exceed this limit."
                     f"- Remember: Write complete sentences and end properly. Today's date is {today}."
                     f' Before finalizing the response, check:'
                     f' - Title starts with "# "'
                     f' - All section headers start with "## "'
                     f' - There is a blank line after every header'
                     f' - There is a blank line between sections\n')
                ])

                try:
                    response = llm.invoke(
                        report_prompt.format_messages(
                            industry=st.session_state.current_industry,
                            context=context
                        )
                    )

                    st.session_state.report = response.content.strip()
                    st.session_state.report_generated = True

                except Exception as e:
                    st.error(f"❌ Report generation error: {str(e)}")
                    st.stop()

            elapsed = time.time() - start
            st.session_state.generation_time = elapsed

            if elapsed > MAX_GENERATION_TIME:
                st.error(
                    "⚠️ Report generation took too long.\n\n"
                    "Try refining the industry or restarting the workflow."
                )
                st.stop()

            st.rerun()

    # Display the report (persists across reruns)
    if st.session_state.report:

        # Convert markdown to HTML and render inside the styled report area.
        report_html = report_to_html(st.session_state.report)
        st.markdown(
            f'<div class="report-area">{report_html}</div>',
            unsafe_allow_html=True
        )

        # Post-generation word count check — this is a second layer of defence
        # beyond the prompt instruction. If the LLM exceeds the limit despite
        # being told not to, the user sees a clear warning.
        clean_report = st.session_state.report.replace('`', '')
        wc = len(clean_report.split())
        
        if wc <= MAX_REPORT_WORDS:
            st.success(f"✅ **Word count:** {wc} / {MAX_REPORT_WORDS}")
        else:
            st.warning(f"⚠️ **Word count:** {wc} / {MAX_REPORT_WORDS} (exceeds limit)")

        # Download as markdown so the analyst can use the report elsewhere —
        # paste into a deck, email it, or convert to PDF.
        st.download_button(
            label="📥 Download Report",
            data=st.session_state.report,
            file_name=f"{st.session_state.current_industry.replace(' ', '_')}_report.md",
            mime="text/markdown",
            help="Download the report as a Markdown file"
        )


# =============================================================================
# FOOTER
# =============================================================================
st.markdown(
    """
    <div class="app-footer">
    Market Research Assistant · Built with Streamlit & LangChain · MSIN0231 Assignment
    </div>
    """,
    unsafe_allow_html=True
)