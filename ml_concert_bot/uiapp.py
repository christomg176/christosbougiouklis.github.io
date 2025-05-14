import streamlit as st
from rag_engine import ingest_document, answer_query
from search_mode import search_artist_tour
st.title(" ML Concert Tour Assistant")

st.sidebar.header("Choose Action")
mode = st.sidebar.radio("Action:", ["Ingest Document", "Ask a Question"])

if mode == "Ingest Document":
    text = st.text_area("Paste concert info here:")
    if st.button("Ingest"):
        result = ingest_document(text)
        st.success(f"Document Ingested:\n{result}")

elif mode == "Ask a Question":
    query = st.text_input("Your question:")
    use_online = st.checkbox(" Use online search if no documents found")

    if st.button("Ask"):
        answer = answer_query(query)

        # If no documents ingested, optionally search online
        if "No documents have been ingested" in answer and use_online:
            artist = st.text_input("Artist name for search:")
            if artist:
                search_result = search_artist_tour(artist)
                st.info(f"Online Search Result:\n{search_result}")
            else:
                st.warning("Please enter an artist name.")
        else:
            st.info(f"Answer:\n{answer}")
