import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv()
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent

st.set_page_config(layout = "wide" , page_icon = "📃" , page_title="AI Research Assistant")
st.title("AI Research Assistant")
st.write("An autonomous ReAct agent that searches the web to answer complex questions.")

try:
    if not os.getenv("GROQ_API_KEY"):
        st.error("Missing GROQ Api Key In env file")
    llm = ChatGroq(model = "openai/gpt-oss-120b" , temperature = "0.2")
except Exception  as e:
    st.error(f"Error Initializing LLM :  {e}")

if not os.getenv("TAVILY_API_KEY"):
        st.warning("Missing Tavily Api Key In env file")

search_tool = TavilySearchResults(max_results = 3)
tools = [search_tool]

agent_executor = create_react_agent(llm , tools)

if "messages" not in st.session_state:
    st.session_state.messages = [{
          "role":"assistant",
          "content":"Hello! Ask me any research question, and I'll scour the web to provide an accurate answer",                         
}]

for message in st.session_state.messages:
    with st.chat_message(message['role']):
          st.write(message['content'])

if user_query := st.chat_input("Enter Your Research Topic : "):
    st.session_state.messages.append({"role":"user","content":user_query})

    with st.chat_message('user'):
               st.write(user_query)

    with st.chat_message('assistant'):
         status_placeholder = st.empty()
         response_placeholder = st.empty()

         with status_placeholder.status("🧠 Assistant is thinking and searching...", expanded=True) as status:
            try:       
                response = agent_executor.invoke({"messages":[("user",user_query)]})
                final_Answer = response["messages"][-1].content
                status.update(label="Research Completed !",state="complete",expanded=False)

                response_placeholder.write(final_Answer)
                st.session_state.messages.append({"role":"assistant","content":final_Answer})

            except Exception as e:
                 status.update(label="❌ An Error Occured", state="error")
                 st.error(f"Failed To Execute Agent Loop : {str(e)}")