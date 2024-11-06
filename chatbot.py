import streamlit as st
import random
import string
import requests
import time
import re
import os
from sentence_transformers import SentenceTransformer, util
import nltk
from nltk.stem.porter import PorterStemmer
from dotenv import load_dotenv

    
load_dotenv()

@st.cache_data
def setup():
    nltk.download('wordnet')
    nltk.download('punkt')
    nltk.download('punkt_tab')
    return


def authenticate(url):
    try:
        print("AUTHENTICATE")
        resp = requests.post(url, json={ "email": st.secrets["email"], "password": st.secrets["password"]})
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
        
@st.cache_data
def get_pattern_embeddings(_transformer_model, patterns):      
    print("GET PATTERNS")  
    return transformer_model.encode(patterns)

@st.cache_resource            
def load_transformer_model():
    print("LOAD MODEL")
    return SentenceTransformer("multi-qa-mpnet-base-cos-v1")

@st.cache_resource            
def load_stemmer_model():
    return PorterStemmer() 

def tokenize(sentence):
    return nltk.word_tokenize(sentence)
       
def remove_punc(text):
    stop_punc_words = set(list(string.punctuation))
    filtered_text = [token for token in text.split() if token not in stop_punc_words]
    
    return " ".join(filtered_text)       

def check_response(tag, patterns, question, response, stemmer):
    '''
    Determines the validity of the chatbot's response
    '''
    
    question = tokenize(question.lower())
    patterns = ' '.join(patterns).lower()
    response = response.lower()
    tag =tag.lower()
    ignore_words = ['?', '!', '.', ',', 'are', 'you', 'can', 'and', 'let','where', 'why', 'what', 'how' , 'when', 'who', 'the' , 'need', 'for', 'have', 'but']
    stemmed_words  = [stemmer.stem(w) for w in question if w not in ignore_words and len(w) > 2 ] # avoid punctuation or words like I , a , or 

    if len(stemmed_words) == 0:
        stemmed_words = [stemmer.stem(w) for w in question]

    found = [ w for w in stemmed_words if re.search(w, response) or re.search(w,tag ) != None or re.search(w, patterns)] #check if the question has words related in the response
    print("FOUND", found)
    return len(found) > 0
    
def get_response(query, transformer_model, stemmer_model, data, pattern_embeddings, all_patterns):
    default_answer = ("", "Hmm... I do not understand that question. Please try again or ask a different question")
 
    query_embedding = transformer_model.encode(query)    

    # similarity = self.sentence_transformer_model.similarity(query_embedding, pattern_embeddings)
    # print("SIMILARITY", similarity)
    scores = util.dot_score(query_embedding, pattern_embeddings)[0].cpu().tolist()
    
    pattern_score_pairs = list(zip(all_patterns, scores))
    
    #Sort by decreasing score
    pattern_score_pairs = sorted(pattern_score_pairs, key=lambda x: x[1], reverse=True)

    # #Output passages & scores
    # print("DOT SCORE")
    # for pattern, score in pattern_score_pairs[:20]:
    #     print(score, pattern)
        
    target_pattern, target_score = pattern_score_pairs[0]
    target_tag = pattern_tag_map[target_pattern]
    result = (target_tag, target_pattern, target_score)
    print("FINAL ANSWER", result)
    
    for faq in data :
        tag = faq["tag"]
        patterns = faq["patterns"]
        responses = faq["responses"]
        if target_tag == tag:
            resp = random.choice(responses)     
            if check_response(tag, patterns, query, resp, stemmer_model):
                return  (target_tag, f"{resp}")
        
    return default_answer    


# Streamed response emulator
def response_generator(response):
    for word in response.split():
        yield word + " "
        time.sleep(0.05)   

def write_stream(stream):
    result = ""
    container = st.empty()
    for chunk in stream:
        result += chunk
        container.write(result, unsafe_allow_html=True) 
        
def startConversation(url):
    
    resp = requests.post(url, json={
        "student_id": st.session_state.form_student_number, 
        "firstname" : st.session_state.form_first_name,
        "lastname" : st.session_state.form_last_name,
        "program" : st.session_state.form_program,
        "email" : st.session_state.form_email
    })
    
    resp.raise_for_status() 
    data = resp.json()
    conversation = data["conversation"]
    
    
    return False

def form_callback():
    print("FORM: ", st.session_state.form_student_number, st.session_state.form_first_name, st.session_state.form_last_name, st.session_state.form_program, st.session_state.form_email)
    if not st.session_state.form_student_number or not st.session_state.form_first_name or not st.session_state.form_last_name or not st.session_state.form_program or not st.session_state.form_email:
        st.session_state.form_error = f":red[Error: Missing Information]" 
    
    elif not st.session_state.form_student_number.isnumeric():
        st.session_state.form_error =  f":red[Error: Invalid Student Number]"
    
    elif st.session_state.form_email.find("@") == -1 or st.session_state.form_email.lower().split("@")[1]  != "ryerson.ca" and st.session_state.form_email.lower().split("@")[1] != "torontomu.ca":
        st.session_state.form_error = f":red[Error: Invalid Email]"
        
    else:    
        try:
        #     startConversation()
        #     raise Exception('Invalid details')
            
            st.session_state.student_number = st.session_state.form_student_number
            st.session_state.first_name = st.session_state.form_first_name
            st.session_state.last_name = st.session_state.form_last_name
            st.session_state.program = st.session_state.form_program
            st.session_state.email = st.session_state.form_email
            st.session_state.conversation_mode = True
            st.session_state.disabled = True
        except Exception as e:
            st.write(f":red[Error: {str(e)}]")
            
def feedback_callback():
    feedback_response = None
    if st.session_state.form_feedback is not None:
        if st.session_state.form_feedback == 1:
            feedback_response = f"You selected: Yes. Ask me another question!" 
        else:
            feedback_response = f"You selected: No. Try rewording your question and make sure you are asking one question at a time. Ask me again!" 
    else:
        feedback_response = f"You provided no feedback. Ask me another question!" 
            
    st.session_state.messages.append({"role": "assistant", "content": feedback_response }) 
            
            
setup()

print("HELLO WORLD")
url = "https://ask-fyeo-chatbot-68o6.onrender.com"
if "token" not in st.session_state:
    st.session_state.token = authenticate(f"{url}/login")


data = get_data(f"{url}/faq", st.session_state.token)

pattern_tag_map = {}
all_patterns = []

for faq in data:
    tag = faq["tag"]
    patterns = faq["patterns"]

    for sent in patterns:
        clean_sent = remove_punc(sent.lower())
        pattern_tag_map[clean_sent] = tag
        all_patterns.append(clean_sent)    

transformer_model = load_transformer_model()
stemmer_model = load_stemmer_model()

pattern_embeddings = get_pattern_embeddings(transformer_model, all_patterns)

st.title("Ask FYEO")

if "conversation_mode" not in st.session_state:
    st.session_state.conversation_mode = False
    
if "feedback_mode" not in st.session_state:
    st.session_state.feedback_mode = False    
    
if "form_error" not in st.session_state:
    st.session_state.form_error = ""    

if not st.session_state.conversation_mode:
    with st.form("student_details", clear_on_submit = True):
        student_number = st.text_input("Student Number", key="form_student_number")
        first_name = st.text_input("First Name", key="form_first_name")
        last_name = st.text_input("Last Name", key="form_last_name")
        program = st.selectbox(
            "Select Program",
            ("Aerospace", "Biomedical", "Chemical", "Civil", "Computer", "Electrical" , "Industrial", "Mechanical" ),
            key="form_program"
        )
  
        email = st.text_input("Email", key="form_email")

        if st.session_state.form_error:
            st.write(st.session_state.form_error)    
        # Every form must have a submit button.
        submitted = st.form_submit_button("Submit" , on_click=form_callback)         
          
elif st.session_state.conversation_mode:
    print(st.session_state.student_number, st.session_state.first_name, st.session_state.last_name, st.session_state.program, st.session_state.email)
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({"role": "assistant", "content": f"Hello {st.session_state.first_name}, it's nice to meet you! I am the FYEO chatbot and I'm here to answer any of your questions about your first year of engineering." })   

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"], unsafe_allow_html=True)
            
    # Accept user input
    if prompt := st.chat_input("Ask me your question!"):
        
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        
        tag, response = get_response(prompt, transformer_model, stemmer_model, data, pattern_embeddings, all_patterns)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            write_stream(response_generator(response))
            
        st.session_state.feedback_mode = True
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response, "tag" : tag })
        
        
    if st.session_state.feedback_mode:
        get_feedback = "Was I able to answer your question?"    
        feedback_response = None
        with st.chat_message("assistant"):
            write_stream(response_generator(get_feedback))     
        st.session_state.messages.append({"role": "assistant", "content": get_feedback })
          
        with st.form("student_feedback"):
            feedback = st.feedback("thumbs", key="form_feedback")  
             
            # Every form must have a submit button.
            submitted = st.form_submit_button("Submit", on_click=feedback_callback)
           
        
        st.session_state.feedback_mode = False
            
