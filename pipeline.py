import re
import time

from groq import BadRequestError

from agents import build_search_agent, writer_chain, critic_chain
from tools import scrape_url


BLOCK_MARKERS = ("access denied", "403 forbidden", "enable javascript",
                 "verify you are human", "captcha")


def looks_blocked(text: str) -> bool:
    """True if the scraped text looks like an error / blocked page."""
    t = text.lower()[:500]
    return len(text) < 200 or any(m in t for m in BLOCK_MARKERS)


def safe_invoke(agent, payload, retries=4):
    """Retry when the model produces a malformed tool call."""
    for attempt in range(retries):
        try:
            return agent.invoke(payload)
        except BadRequestError as e:
            if "tool_use_failed" in str(e) and attempt < retries - 1:
                print(f"  Tool call failed, retrying ({attempt + 1}/{retries - 1})...")
                time.sleep(2)
                continue
            raise


def extract_urls(text: str) -> list:
    """Pull unique URLs out of text, keeping their order."""
    urls = re.findall(r"https?://[^\s\)\]\>\"']+", text)
    cleaned = []
    for u in urls:
        u = u.rstrip(".,;:")
        if u not in cleaned:
            cleaned.append(u)
    return cleaned


def run_research_pipeline(topic: str) -> dict:

    state = {}

    # step 1 - search agent
    print("\n" + "=" * 50)
    print("step 1 - search agent is working ...")
    print("=" * 50)

    search_agent = build_search_agent()
    search_result = safe_invoke(search_agent, {
        "messages": [("user", f"Find recent, reliable and detailed information about: {topic}")]
    })
    state["search_results"] = search_result["messages"][-1].content

    print("\n search result ", state["search_results"])

    # step 2 - scrape (plain Python, no LLM)
    print("\n" + "=" * 50)
    print("step 2 - scraping top resources ...")
    print("=" * 50)

    state["scraped_content"] = ""
    for url in extract_urls(state["search_results"])[:5]:   # try up to 5 URLs
        try:
            print(f"  Scraping: {url}")
            content = str(scrape_url.invoke({"url": url}))
            if content.strip() and not looks_blocked(content):
                state["scraped_content"] = content[:5000]   # keep small for free-tier limits
                state["scraped_url"] = url
                break
            print("  Page blocked or too short, trying next URL...")
        except Exception as e:
            print(f"  Failed ({e}), trying next URL...")

    if not state["scraped_content"]:
        state["scraped_content"] = "No page could be scraped. Use the search results only."

    print("\nscraped content: \n", state["scraped_content"])

    # step 3 - writer chain
    print("\n" + "=" * 50)
    print("step 3 - Writer is drafting the report ...")
    print("=" * 50)

    research_combined_result = (
        f"SEARCH RESULTS : \n {state['search_results']} \n\n"
        f"DETAILED SCRAPED CONTENT : \n {state['scraped_content']}"
    )

    state["report"] = writer_chain.invoke({
        "topic": topic,
        "research": research_combined_result,
    })

    print("\n Final Report\n", state["report"])

    # step 4 - critic
    print("\n" + "=" * 50)
    print("step 4 - critic is reviewing the report ")
    print("=" * 50)

    state["feedback"] = critic_chain.invoke({
        "topic": topic,
        "research": research_combined_result,
        "report": state["report"],
    })

    print("\n critic report \n", state["feedback"])

    return state


if __name__ == "__main__":
    topic = input("\n Enter a research topic : ")
    run_research_pipeline(topic)