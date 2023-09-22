from flask import Flask, render_template, request
import pickle

import re
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize as st
from nltk.stem import WordNetLemmatizer
import re
lemmatizer = WordNetLemmatizer()

import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

nltk.download("stopwords")

model = pickle.load(open("log.pkl", 'rb'))
le = pickle.load(open("le.pkl", 'rb'))
def clean_texts(text):
    """Removes digits, words containing digits, punctuations, special characters, extra spaces and links from text

    Args:
        text (str): raw texts
    Returns:
        text (str) : cleaned texts
    """
    #removes links
    text = re.sub(r'http*\S+', '', text)
    #removes digits 
    text = re.sub('\d+','', text)
    #removes punctuations and special characters
    text = re.sub('[^a-zA-Z]+', ' ', text)
    return text



@app.route("/", methods = ['GET', 'POST'])
def home():
    message = ""
    try:
        if request.method == 'POST':
            dic = request.form.to_dict()
            news = dic['news']
            if len(news) == 0:
                raise Exception
            news = clean_texts(news).split()
            text_stopwords_lemmatized = [lemmatizer.lemmatize(word) for word in news if not word in stopwords.words('english')]
            text_stopwords_lemmatized = ' '.join(text_stopwords_lemmatized)
            pred = model.predict([text_stopwords_lemmatized])
            k1=le.inverse_transform(pred)
            message = k1[0]
    except:
        message = "Please enter some text"
    return render_template("index.html", message = message)

if __name__ == '__main__':
    app.run(debug = True)