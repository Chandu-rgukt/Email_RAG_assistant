import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_classic.schema import Document
# from langchain_core.documents import Document
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv


os.environ.get("GOOGLE_API_KEY")

load_dotenv()
# -----------------------
# Initialize Models
# -----------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3
)

# -----------------------
# Documents (Outreach Emails)
# -----------------------
documents = [
    Document(
        page_content="Email Subject: Interview Invitation - Software Engineer Position\nHi [Candidate Name],\n\nYour resume has been shortlisted for the Software Engineer position...",
        metadata={"type": "interview_invitation", "category": "job_application", "urgency": "high"}
    ),
    Document(
        page_content="Email Subject: Technical Interview Schedule\nDear [Candidate],\n\nCongratulations! Your application has been selected for the next round...",
        metadata={"type": "interview_scheduling", "category": "job_application", "urgency": "high"}
    ),
    Document(
        page_content="Email Subject: Final Round Interview\nHello [Candidate Name],\n\nYou have progressed to the final round of interviews...",
        metadata={"type": "final_round_interview", "category": "job_application", "urgency": "high"}
    ),
    Document(
        page_content="Email Subject: Coffee Chat Request\nHi [Name],\n\nI came across your profile and would love to schedule a 15-20 min virtual coffee chat...",
        metadata={"type": "networking_request", "category": "professional_networking", "urgency": "medium"}
    ),
    Document(
        page_content="Email Subject: Partnership Opportunity\nDear [Contact Name],\n\nWe believe there's a strong opportunity for collaboration...",
        metadata={"type": "business_proposal", "category": "partnership", "urgency": "medium"}
    ),
    Document(
        page_content="Email Subject: Product Demo Request\nHi [Team Name],\n\nWould it be possible to schedule a demo to learn more about the product features?",
        metadata={"type": "product_inquiry", "category": "sales", "urgency": "medium"}
    ),
    Document(
        page_content="Response Template: Accepting Interview\nThank you for the interview invitation! I'm very interested...",
        metadata={"type": "response_template", "category": "acceptance", "template": "interview_acceptance"}
    ),
    Document(
        page_content="Response Template: Polite Decline\nThank you for considering me for the opportunity. After careful consideration...",
        metadata={"type": "response_template", "category": "decline", "template": "polite_decline"}
    ),
    Document(
        page_content="Response Template: Scheduling Call\nThanks for your message! I'm available for a call on the following days...",
        metadata={"type": "response_template", "category": "scheduling", "template": "availability"}
    ),
    # Add more documents/templates as needed
]

# -----------------------
# Vector Store & Retriever
# -----------------------
vectorstore = Chroma.from_documents(documents, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# -----------------------
# Retrieval Chain Setup
# -----------------------
prompt = ChatPromptTemplate.from_template("""
You are helping a professional respond to emails. Generate a complete email response as the recipient.

Context from similar email responses:
{context}

Email received:
{input}

Write a professional email response as the person receiving this email. Do not mention you are an AI.
1️⃣ Agenda:
I am applying for a job position. 
If the lead or recruiter is interested, share the meeting booking link: https://cal.com/example.
Always maintain a professional tone.
  
2️⃣ Tone:
Be polite, concise, and express gratitude.
Avoid long sentences or informal words.

3️⃣ Templates:
- Thank you for shortlisting my profile! I'm available for a technical interview. You can book a slot here: https://cal.com/example.
- I appreciate your interest in my profile. Please use this link to schedule our discussion: https://cal.com/example.
- Thanks for reaching out! I’d be happy to connect. You can pick a time here: https://cal.com/example.

4️⃣ Notes:
If the email mentions an “interview”, respond with availability.
If the email mentions “meeting” or “call”, share the scheduling link.
Email Response:""")


document_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# -----------------------
# Helper Function
# -----------------------
def get_rag_response(query: str) -> str:
    """
    Input: user query (email text)
    Output: dict with 'answer' and 'source_documents'
    """
    result = retrieval_chain.invoke({"input": query})
    return {
        "answer": result.get("answer", ""),
        "source_documents": result.get("source_documents", [])
    }