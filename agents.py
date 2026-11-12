from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tools import web_search, scrape_url
from dotenv import load_dotenv

load_dotenv()

tool_llm = ChatGroq(model="qwen/qwen3.8-27b", temperature=0, max_tokens=800) #search + reader agents
text_llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)   # writer + critic


#pehla(1st)agent 
def build_search_agent():
  return create_agent(
    model=tool_llm,
    tools=[web_search],
    system_prompt ="You are a research assistant. Use web_search to find sources. "
                      "Reply concisely: at most 250 words of key facts, each with its URL."
  )
  
  
#doosra(2nd)agent 
def build_reader_agent():
  return create_agent(
    model = tool_llm,
    tools = [scrape_url]
  )
  

writer_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an expert research writer. Write clear, structured and insightful reports. "
     "Use ONLY the information in the research provided. Do not add facts from memory, "
     "and do not invent sources or statistics. If the research does not cover something, "
     "say so briefly."),
    ("human", """Write a detailed research report on the topic below.

Topic: {topic}

Research Gathered:
{research}

Structure the report as:
- Introduction
- Key Findings (minimum 3 well-explained points, with inline citations like [1], [2])
- Conclusion
- Sources (numbered list of URLs that appear in the research; do not add any others)

Be detailed, factual and professional."""),
])

writer_chain = writer_prompt | text_llm | StrOutputParser() # ye ek chain hai writer prompt llm ko send hoga jo answer dega aur pir ek structure me oska output milega StrOP se


#Critics chain

critic_prompt = ChatPromptTemplate.from_messages([
    ("system",
 "You are a sharp and constructive research critic. Be honest and specific. "
 "Judge the report ONLY against the research provided. Do not use outside knowledge "
 "or claim that a source does not exist. Check whether every claim is supported by "
 "the research, and flag unsupported claims, missing citations, or invented sources. "
 "Note: some pages may have failed to scrape; that is not the writer's fault."),
    ("human", """Review the research report below and evaluate it strictly.

Topic: {topic}

Research Gathered:
{research}

Report:
{report}

Respond in this exact format:

Score: X/10

Strengths:
- ...
- ...

Areas to Improve:
- ... (be specific and actionable, so the writer can fix it)
- ...

One line verdict:
..."""),
])

critic_chain = critic_prompt | text_llm | StrOutputParser()