import re
from datetime import datetime

import streamlit as st

from agents import build_search_agent, writer_chain, critic_chain
from pipeline import safe_invoke, extract_urls, looks_blocked
from tools import scrape_url

# ---------------------------------------------------------------- page setup
st.set_page_config(page_title="Research Agent", page_icon="🔎", layout="wide")

st.markdown(
    """
    <style>
    .hero {
        padding: 1.6rem 2rem;
        border-radius: 16px;
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        color: white;
        margin-bottom: 1.2rem;
    }
    .hero h1 { margin: 0; font-size: 2rem; color: white; }
    .hero p  { margin: .3rem 0 0 0; opacity: .9; }
    .stTabs [data-baseweb="tab"] { font-size: 1rem; }
    </style>
    <div class="hero">
        <h1>🔎 Multi-Agent Research System</h1>
        <p>Search agent → Web scraper → Writer → Critic</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    max_urls = st.slider("URLs to try scraping", 1, 5, 5)
    max_chars = st.slider("Max scraped characters", 1000, 8000, 5000, step=500)
    st.caption("Lower values use fewer tokens (helpful on Groq's free tier).")
    st.divider()
    st.markdown(
        "**How it works**\n\n"
        "1. 🔍 Search agent finds sources\n"
        "2. 🌐 Scraper reads a page\n"
        "3. ✍️ Writer drafts the report\n"
        "4. 🧐 Critic scores it"
    )

# ---------------------------------------------------------------- input
EXAMPLES = [
    "Impact of AI on jobs",
    "Solid-state battery technology",
    "How do mRNA vaccines work",
]

if "topic" not in st.session_state:
    st.session_state["topic"] = ""


def set_topic(t: str):
    st.session_state["topic"] = t


topic = st.text_input(
    "Research topic",
    key="topic",
    placeholder="e.g. Impact of AI on jobs",
)

cols = st.columns(len(EXAMPLES))
for col, ex in zip(cols, EXAMPLES):
    col.button(ex, on_click=set_topic, args=(ex,), use_container_width=True)

run = st.button("🚀 Start Research", type="primary")


# ---------------------------------------------------------------- pipeline
def run_pipeline(topic: str, max_urls: int, max_chars: int) -> dict:
    state = {}
    year = datetime.now().year

    with st.status("Agents are working...", expanded=True) as status:
        try:
            # Step 1: search
            st.write("🔍 **Step 1/4** — Search agent is searching the web...")
            search_agent = build_search_agent()
            result = safe_invoke(search_agent, {
                "messages": [("user",
                    f"Find the most recent (2025-{year}), reliable and detailed "
                    f"information about: {topic}. The current year is {year}.")]
            })
            state["search_results"] = result["messages"][-1].content

            # Step 2: scrape
            st.write("🌐 **Step 2/4** — Scraping top resources...")
            state["scraped_content"] = ""
            state["scraped_url"] = None
            for url in extract_urls(state["search_results"])[:max_urls]:
                try:
                    st.write(f"&nbsp;&nbsp;&nbsp;↳ trying {url}")
                    content = str(scrape_url.invoke({"url": url}))
                    if content.strip() and not looks_blocked(content):
                        state["scraped_content"] = content[:max_chars]
                        state["scraped_url"] = url
                        break
                    st.write("&nbsp;&nbsp;&nbsp;↳ blocked or too short, next...")
                except Exception as e:
                    st.write(f"&nbsp;&nbsp;&nbsp;↳ failed ({e}), next...")
            if not state["scraped_content"]:
                state["scraped_content"] = (
                    "No page could be scraped. Use the search results only."
                )

            combined = (
                f"SEARCH RESULTS:\n{state['search_results']}\n\n"
                f"DETAILED SCRAPED CONTENT:\n{state['scraped_content']}"
            )

            # Step 3: writer
            st.write("✍️ **Step 3/4** — Writer is drafting the report...")
            state["report"] = writer_chain.invoke({
                "topic": topic,
                "research": combined,
            })

            # Step 4: critic
            st.write("🧐 **Step 4/4** — Critic is reviewing the report...")
            state["feedback"] = critic_chain.invoke({
                "topic": topic,
                "research": combined,
                "report": state["report"],
            })

            status.update(label="✅ Research complete!", state="complete", expanded=False)
            return state

        except Exception as e:
            status.update(label="❌ Something went wrong", state="error")
            st.error(f"{type(e).__name__}: {e}")
            if "rate" in str(e).lower() or "429" in str(e):
                st.info("You hit a Groq free-tier rate limit. Wait a minute and try again.")
            return {}


if run:
    if not topic.strip():
        st.warning("Please enter a topic first.")
    else:
        result = run_pipeline(topic.strip(), max_urls, max_chars)
        if result:
            result["topic"] = topic.strip()
            st.session_state["result"] = result

# ---------------------------------------------------------------- results
result = st.session_state.get("result")

if result:
    st.divider()
    st.subheader(f"Results: {result['topic']}")

    m = re.search(r"Score:\s*\**\s*(\d+)\s*/\s*10", result["feedback"])
    score = int(m.group(1)) if m else None

    c1, c2, c3 = st.columns(3)
    c1.metric("Critic score", f"{score}/10" if score is not None else "n/a")
    c2.metric("Sources found", len(extract_urls(result["search_results"])))
    c3.metric("Page scraped", "Yes" if result.get("scraped_url") else "No")

    tab_report, tab_critic, tab_sources, tab_raw = st.tabs(
        ["📄 Report", "🧐 Critic feedback", "🔗 Sources", "🧾 Raw data"]
    )

    with tab_report:
        st.markdown(result["report"])
        st.download_button(
            "⬇️ Download report (.md)",
            data=result["report"],
            file_name="research_report.md",
            mime="text/markdown",
        )

    with tab_critic:
        st.markdown(result["feedback"])

    with tab_sources:
        urls = extract_urls(result["search_results"])
        if result.get("scraped_url"):
            st.success(f"Scraped: {result['scraped_url']}")
        for i, u in enumerate(urls, 1):
            st.markdown(f"{i}. {u}")

    with tab_raw:
        with st.expander("Search agent output"):
            st.markdown(result["search_results"])
        with st.expander("Scraped content"):
            st.text(result["scraped_content"])