import streamlit as st
import random
import string
import requests
import time
import re
import os
from sentence_transformers import SentenceTransformer, util
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from dotenv import load_dotenv

load_dotenv()

@st.cache_data
def setup():
    nltk.download('wordnet')
    nltk.download('punkt')
    nltk.download('punkt_tab')
    return
    

def tokenize(sentence):
    return nltk.word_tokenize(sentence)


def stem(word):
    return st.session_state.stemmer.stem(word.lower())


def authenticate(url):
    try:
        print("AUTHENTICATE")
        resp = requests.post(url, json={ "email": f"{os.environ.get('EMAIL')}", "password": f"{os.environ.get('PASSWORD')}"})
        resp.raise_for_status()
        token = resp.json()["token"]
           
        return token
    except Exception as e:
        print("ERROR: ", e)

@st.cache_data
def get_data(url, token):
    try:
        print("RETRIEVE DATA")
        resp = requests.get(url, headers={"authorization" : f"Bearer {token}"})
        resp.raise_for_status()
        data = resp.json()
    
        return data["FAQ"]
    except Exception as e:
        print("ERROR: ", e)
        
def remove_punc(text):
    stop_punc_words = set(list(string.punctuation))
    filtered_text = [token for token in text.split() if token not in stop_punc_words]
    
    return " ".join(filtered_text)       

def check_response(tag, patterns, question, response):
    '''
    Determines the validity of the chatbot's response
    '''
    
    question = tokenize(question)
    patterns = ' '.join(patterns)
    ignore_words = ['?', '!', '.', ',', 'are', 'you', 'can', 'and', 'let','where', 'why', 'what', 'how' , 'when', 'who', 'the' , 'need', 'for', 'have', 'but']
    stemmed_words  = [stem(w) for w in question if w.lower() not in ignore_words and len(w) > 2 ] # avoid punctuation or words like I , a , or 

    if len(stemmed_words) == 0:
        stemmed_words = [stem(w) for w in question]

    found = [ w for w in stemmed_words if re.search(w.lower(), response.lower()) or re.search(w.lower(),tag.lower() ) != None or re.search(w.lower(), patterns.lower())] #check if the question has words related in the response
    print("FOUND", found)
    return len(found) > 0
    
def get_response(query, data):
    default_answer = ("", "Hmm... I do not understand that question. Please try again or ask a different question")
    sentence_tag_map = {}
    all_sentences = []
    
    for faq in data :
        tag = faq["tag"]
        patterns = faq["patterns"]

        for sent in patterns:
            clean_sent = remove_punc(sent.lower())
            sentence_tag_map[clean_sent] = tag
            all_sentences.append(clean_sent)
            
    query_embedding = st.session_state.transformer_model.encode(query)
    pattern_embeddings = st.session_state.transformer_model.encode(all_sentences)

    # similarity = self.sentence_transformer_model.similarity(query_embedding, pattern_embeddings)
    # print("SIMILARITY", similarity)
    scores = util.dot_score(query_embedding, pattern_embeddings)[0].cpu().tolist()
    
    pattern_score_pairs = list(zip(all_sentences, scores))
    
    #Sort by decreasing score
    pattern_score_pairs = sorted(pattern_score_pairs, key=lambda x: x[1], reverse=True)

    # #Output passages & scores
    # print("DOT SCORE")
    # for pattern, score in pattern_score_pairs[:20]:
    #     print(score, pattern)
        
    target_pattern, target_score = pattern_score_pairs[0]
    target_tag = sentence_tag_map[target_pattern]
    result = (target_tag, target_pattern, target_score)
    print("FINAL ANSWER", result)
    
    for faq in data :
        tag = faq["tag"]
        patterns = faq["patterns"]
        responses = faq["responses"]
        if target_tag == tag:
            resp = random.choice(responses)     
            if check_response(tag, patterns, query, resp):
                return  (target_tag, f"{resp}")
        
    return default_answer        

setup()

print("HELLO WORLD")
url = "https://ask-fyeo-chatbot-68o6.onrender.com"
if "token" not in st.session_state:
    st.session_state.token = authenticate(f"{url}/login")


data = get_data(f"{url}/faq", st.session_state.token)


if "transformer_model" not in st.session_state: 
    st.session_state.transformer_model = SentenceTransformer("multi-qa-mpnet-base-cos-v1")

if "stemmer" not in st.session_state:
    st.session_state.stemmer = PorterStemmer()   


# Streamed response emulator
def response_generator(response):
    for word in response.split():
        yield word + " "
        time.sleep(0.05)


st.title("Ask FYEO")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({"role": "assistant", "content": "Hello, it's nice to meet you! I am the FYEO chatbot and I'm here to answer any of your questions about your first year of engineering." })   

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
     

# Accept user input
if prompt := st.chat_input("Ask me your question!"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        tag, response = get_response(prompt, data)
        st.markdown(response, unsafe_allow_html=True)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response, "tag" : tag })


# print(st.session_state.messages)