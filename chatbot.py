
#Meet Robo: your friend

#import necessary libraries
import io
import random
import string # to process standard python strings
import warnings
import numpy
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('popular', quiet=True) # for downloading packages

# descomentar lo siguiente solo la primera vez
#nltk.download('punkt') # uso solo por primera vez
#nltk.download('wordnet') # uso solo por primera vez


#Lectura en el corpus
with open('LegalHub.txt','r', encoding='utf8', errors ='ignore') as fin:
    raw = fin.read().lower()

#TOkenisación
sent_tokens = nltk.sent_tokenize(raw)# se convierte en una lista de oraciones
word_tokens = nltk.word_tokenize(raw)# se convierte en una lista de palabras

# Preprocesamiento
lemmer = WordNetLemmatizer()
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))


# Coincidencia de palabras clave
GREETING_INPUTS = ("hola", "que tal?", "saludos", "como va?", "buenos dias!","hey",)
GREETING_RESPONSES = ["Hola!", "hey humano", "*asiente*", "Hola señor/a", "hello", "Me alegra que estés hablando conmigo!"]

def greeting(sentence):
    """If user's input is a greeting, return a greeting response"""
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)


# Generando respuesta
def response(user_response):
    robo_response=''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx=vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf==0):
        robo_response=robo_response+"Disculpame! No logro entenderte"
        return robo_response
    else:
        robo_response = robo_response+sent_tokens[idx]
        return robo_response


flag=True
print("Te responderé tus dudas acerca de Legal Hub. Si quieres salir, escribe Bye!")
while(flag==True):
    user_response = input()
    user_response=user_response.lower()
    if(user_response!='bye'):
        if(user_response=='gracias' or user_response=='te agradezco' ):
            flag=False
            print("Staff: De nada..")
        else:
            if(greeting(user_response)!=None):
                print("Staff: "+greeting(user_response))
            else:
                print("Staff: ",end="")
                print(response(user_response))
                sent_tokens.remove(user_response)
    else:
        flag=False
        print("Staff: Adios! cuidate..")    
        
        

