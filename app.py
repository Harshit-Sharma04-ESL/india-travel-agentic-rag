import json, os
import streamlit as st
from rag import Agent, HERE

st.set_page_config(page_title="India Travel Agent", page_icon="🧭")
st.title("🧭 India Travel Agentic RAG")


@st.cache_resource
def get_agent():
    return Agent()


agent = get_agent()
if "msgs" not in st.session_state:
    st.session_state.msgs = []

synth = os.path.join(HERE, "data", "synthetic_questions.jsonl")
if os.path.exists(synth):
    qs = [json.loads(l)["question"] for l in open(synth)]
    with st.sidebar:
        st.header("Sample questions")
        pick = st.selectbox("Synthetic questions", [""] + qs)
else:
    pick = ""

for m in st.session_state.msgs:
    st.chat_message(m["role"]).write(m["content"])

q = st.chat_input("Ask about travelling in India...") or (pick if st.sidebar.button("Ask selected") else None)
if q:
    st.chat_message("user").write(q)
    with st.spinner("Thinking..."):
        ans, trace = agent.run(q, history=st.session_state.msgs[-6:])
    with st.chat_message("assistant"):
        st.write(ans)
        with st.expander(f"Agent trace ({len(trace)} tool calls)"):
            st.json(trace)
    st.session_state.msgs += [{"role": "user", "content": q}, {"role": "assistant", "content": ans}]
