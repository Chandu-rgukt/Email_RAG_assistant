# streamlit_app.py
import streamlit as st
from rag import get_rag_response

st.set_page_config(page_title="Email RAG Assistant", page_icon="📧", layout="centered")

st.title("📧 Email Response Generator (RAG)")
st.write(
    """
    Enter an email or query below and get a professional suggested response.
    """
)

# User input
user_query = st.text_area("✉️ Enter the email or query:", height=150)

if st.button("Generate Response"):
    if not user_query.strip():
        st.warning("Please enter an email or query!")
    else:
        with st.spinner("Generating response..."):
            try:
                result = get_rag_response(user_query)
                st.success("✅ Response generated!")
                
                # Show AI-generated email
                st.subheader("💌 Suggested Email Response:")
                st.write(result['answer'])

                # Show source documents used
                if result['source_documents']:
                    st.subheader("📚 Source Documents:")
                    for i, doc in enumerate(result['source_documents']):
                        st.markdown(f"**Document {i+1}:**")
                        st.write(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)

            except Exception as e:
                st.error(f"❌ Error generating response: {str(e)}")
