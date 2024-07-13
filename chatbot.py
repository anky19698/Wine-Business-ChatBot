from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from nltk.tokenize import sent_tokenize
from pdfminer.high_level import extract_text
from langchain_core.documents import Document
import spacy
import streamlit as st 
import os
from langchain_huggingface import HuggingFaceEndpoint
import google.generativeai as genai


def get_chunks():
    nlp = spacy.load("en_core_web_sm")
    pdf_path = "Corpus.pdf"
    # loader = PyPDFLoader(pdf_path)
    pdf_text = extract_text(pdf_path)
    doc = nlp(pdf_text)
    sentences =  [sent.text.strip() for sent in doc.sents]
    chunk_size = 5
    combined_chunks = []

    for i in range(0, len(sentences), 5):
        if len(sentences) - i < 5:
            chunk = sentences[i:]
        else:
            chunk = sentences[i:i+5]
        
        # Join the sentences in the chunk, then create a Document
        chunk_text = " ".join(chunk)
        combined_chunks.append(Document(page_content=chunk_text))
    
    return combined_chunks


def load_existing_chroma_collection():
    # Initialize the HuggingFace embeddings
    embeddings = HuggingFaceEmbeddings()

    # Load the Chroma database from disk
    chroma_db = Chroma(
        persist_directory="data",
        embedding_function=embeddings,
        collection_name="lc_chroma_demo"
    )

    # Get the collection from the Chroma database
    collection = chroma_db.get()

    return chroma_db, collection

def create_and_persist_chroma_collection(chunks):
    # Initialize the HuggingFace embeddings
    embeddings = HuggingFaceEmbeddings()

    # Create a new Chroma database from the documents
    chroma_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="data",
        collection_name="lc_chroma_demo"
    )

    # Save the Chroma database to disk
    chroma_db.persist()

    return chroma_db

def load_chroma_collection(chunks):
    chroma_db, collection = load_existing_chroma_collection()

    # If the collection is empty, create a new one
    if len(collection['ids']) == 0:
        chroma_db = create_and_persist_chroma_collection(chunks)

    return chroma_db


def get_llm_model():
    # Set up Hugging Face Token
    hf_token = st.secrets['hf_token']
    os.environ['HUGGINGFACEHUB_API_TOKEN'] = hf_token

    # Set up the LLAMA model
    model_id = "mistralai/Mistral-7B-Instruct-v0.3"
    # model_id = "meta-llama/Llama-2-7b-chat-hf"
    hf_endpoint = HuggingFaceEndpoint(
        endpoint_url=model_id,
        task="text-generation",
        temperature= 1,
        max_new_tokens= 1000,
        token=hf_token
    )

    return hf_endpoint


def get_gemini_model():
# Set up gemini pro key
    gemini_pro_key = st.secrets['gemini_api_key']
    genai.configure(api_key=gemini_pro_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
    return model



def get_samples():
    samples = [
    {
        "question": "What white wines do you have?",
        "answer": "We offer the following wines: The Jessup Cellars 2022 Chardonnay and the 2023  Sauvignon Blanc"
    },
    {
        "question": "What red wines is Jessup Cellars offering in 2024?",
        "answer": "Jessup Cellars offers a number of red wines across a range of varietals, from Pinot Noir and Merlot blends from the Truchard Vineyard, to blended Cabernet Sauvignon from both the Napa and Alexander Valleys, our Mendocino Rougette combining Grenache and Carignane varietals which we refer to as our 'Summer Red', to the ultimate expression of the 'Art of the Blend\" with our Juel and Table for Four Red Wines. We also offer 100% Zinfandel from 134 year old vines in the Mendocino region and our own 100% Petite Sirah grown in the Wooden Valley in Southeastern Napa County. We also offer some seasonal favorites, led by the popular whimsical Manny's Blend which should be released later in 2024 with a very special label."
    },
    {
        "question": "Please tell me more about your consulting winemaker Rob Lloyd?",
        "answer": "Rob Lloyd \nConsulting Winemaker\n\nBIOGRAPHY\nHometown: All of California\n\nFavorite Jessup Wine Pairing: Carneros Chardonnay with freshly caught Mahi-Mahi\n\nAbout: Rob\u2019s foray into wine started directly after graduating college when he came to Napa to work in a tasting room for the summer \u2013 before getting a \u2018real job\u2019. He became fascinated with wine and the science of winemaking and began to learn everything he could about the process.\n\nWhile interviewing for that \u201creal job\u201d, the interviewer asked him what he had been doing with his time since graduation. After speaking passionately and at length about wine, the interviewer said, \u201cYou seem to love that so much. Why do you want this job?\u201d Rob realized he didn't want it, actually. He thanked the man, and thus began a career in the wine industry.\n\nRob has since earned his MS in Viticulture & Enology from the University of California Davis and worked for many prestigious wineries including Cakebread, Stag\u2019s Leap Wine Cellars, and La Crema. Rob began crafting Jessup Cellars in the 2009 season and took the position of Director of Winemaking at Jessup Cellars in 2010. He now heads up our winemaking for the Good Life Wine Collective, which also includes Handwritten Wines."
    },
    {
        "question": "Tell me an interesting fact about Rob Llyod",
        "answer": "While interviewing for that \u201creal job\u201d, the interviewer asked him what he had been doing with his time since graduation. After speaking passionately and at length about wine, the interviewer said, \u201cYou seem to love that so much. Why do you want this job?\u201d Rob realized he didn't want it, actually. He thanked the man, and thus began a career in the wine industry."
    },
    {
        "question": "Are walk ins allowed?",
        "answer": " Walk-ins are welcome anytime for our Light Flight experience. Our most popular tastinig experience is the jessup Classic Tasting, which includes a flight of 5 wines perfectly paired with cheeses, accompanied by palate cleansing Marcona almonds and a chocolate surprise. The Classic Tasting is $60 perperson, but is waived with a purchase of two or more bottles of wine per person."
    }
    ]

    return samples


def get_similar_document(chroma_db, user_question):
    
    results = chroma_db.similarity_search_with_relevance_scores(user_question)

    document, score = results[0]
    if score < 0:
        return "Empty"
    else:
        return document.page_content


def get_prompt(corpus, samples, user_question, chat_history):
    try:
        recent_history = chat_history[-5:]
        history_text = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in recent_history])
    except:
        history_text = ""
    prompt = f"""
    You are a helpful assistant for a wine business. Your task is to answer customer questions based solely on the information provided in the following corpus. If the question cannot be answered using this corpus, politely inform the customer to contact the business directly.

    Corpus:
    {corpus}

    Here are some sample questions and answers to guide your responses:
    {samples}

    Chat History:
    {history_text}

    Remember:
    1. Only use information from the provided corpus to answer questions.
    2. If the question cannot be answered using the corpus, politely direct the customer to contact the business.
    3. Provide concise yet informative answers.
    4. Consider Chat History to keep up with Context
    5: Answer in 3-5 Lines

    User Question: {user_question}

    Assistant: """

    return prompt


def get_answer(model, prompt):
    answer = model.invoke(prompt)
    return answer.strip()


def get_gemini_response(model, prompt):
    try:
        response = model.generate_content(prompt)
        return response.text
    except:
        return "Empty"



def main():
    # Get Chunks
    chunks = get_chunks()
    print("Chunks:", chunks)
    # Load Chroma Collection
    # chroma_db = load_chroma_collection(chunks)
    chroma_db, collection = load_existing_chroma_collection()

    # Get LLM Model
    # model = get_llm_model()
    model = get_gemini_model()

    # Get Samples
    samples = get_samples()

    # Streamlit App
    st.title(":wine_glass: Wine Business Chatbot")

    user_question = "What hotels are located near Jessup Cellars?"

    

    # Streamlit Chat App
    # Initialize chat history in session state if it doesn't exist
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

    # Display chat history
    for i, message in enumerate(st.session_state.chat_history):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input at the bottom
    user_question = st.chat_input("Ask a Question about the Wine Business:")

    if user_question:
        # Add user question to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_question})

        # Display user question
        with st.chat_message("user"):
            st.markdown(user_question)

        # Generate and display AI response
        with st.chat_message("assistant"):
            with st.spinner("Generating response..."):

                # try:
                    # Get Corpus
                    corpus = get_similar_document(chroma_db, user_question)

                    # Get Prompt
                    prompt = get_prompt(corpus, samples, user_question, st.session_state.chat_history)

                    # Get Answer
                    # answer = get_answer(model, prompt)
                    answer = get_gemini_response(model, prompt)

                    st.write(answer)

                    # Add assistant response to chat history
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})

                # except:
                #     st.warning("Please Contact Business Directly")


if __name__ == "__main__":
    main()    
