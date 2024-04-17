import uuid
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores.faiss import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv


load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))



texts = []
id = 100
if 'texts' not in st.session_state:
    st.session_state['texts'] = texts

if 'id' not in st.session_state:
    st.session_state['id'] = id

def set_application_text(text):
    if len(st.session_state.texts) == 0:
        text += "This is an application with SBI Bank."
    else :
        text += "This is another application with SBI Bank."
    return text


def set_name_in_application_text(text, name):
    text += f"The application is created by {name}."
    return text

def set_ssn_in_application_text(text, name, ssn):
    text += f"The {name} in this application has an personal id of {ssn}."
    return text

def set_type_and_identifier_in_application_text(text, type, identifier):
    text += f"The id of this application is {identifier} and it is for a {type}.\n\n"
    return text

def set_email_and_phone_in_application_text(text,name, email, phone):
    text += f"The {name} in this application has the email address {email} and phone number {phone}."
    return text

def get_application_text(name,ssn,type,identifier,email,phone):
    text=""
    text = set_application_text(text)
    text = set_name_in_application_text(text,name)
    text = set_ssn_in_application_text(text, name, ssn)
    text = set_type_and_identifier_in_application_text(text, type, identifier)
    text = set_email_and_phone_in_application_text(text,name,email,phone)
    return text



def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain



def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)

    st.write("Reply: ", response["output_text"])


def get_all_application_texts():
            final_text = ""
            print(st.session_state.texts)
            for application_text in st.session_state.texts:
                final_text += application_text
            text_chunks = get_text_chunks(final_text)
            get_vector_store(text_chunks)



def main():
    
    st.set_page_config("Chat Originate")
    st.header("Chat with Originate using Gemini💁")


    name = st.text_input("Full Name")
    ssn = st.text_input("Personal ID")
    account_type = st.selectbox(label="Type of account",options=["Current Account", "Savings Account"])
    email = st.text_input("Email Id")
    phone = st.text_area("Phone Number")



    if st.button("Create account"):
        with st.spinner("Processing"):
            st.session_state.id += 1
            application_text = get_application_text(name,ssn,account_type,f"ORIG-{str(st.session_state.id)}",email,phone)
            print(st.session_state.texts)
            st.session_state.texts.append(application_text)
            st.write(f"Your application id is ORIG-{st.session_state.id} ")
            st.write(application_text)
            st.success("Done")
            get_all_application_texts()
            
    user_question = st.text_input("Ask a Question")
    if user_question:
        user_input(user_question)



if __name__ == "__main__":
    main()