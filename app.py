# IMPORTING THE REQUIRED LIBRARIES 

from flask import Flask, request, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
from string import punctuation
from nltk.corpus import stopwords


# Downloading the stopwords
nltk.download('stopwords')

set(stopwords.words('english'))

# Setting up flask application
app = Flask(__name__)

@app.route('/')
def my_form():
    return render_template('main.html')

@app.route('/', methods=['POST'])
def my_form_post():
    stop_words = stopwords.words('english')

    #convert to lowercase
    text1 = request.form['text1'].lower()
    text_final = ''.join(c for c in text1 if not c.isdigit())

    #remove stopwords    
    processed_doc1 = ' '.join([word for word in text1.split() if word not in stop_words])

    # Using the model to get the values i.e positive , negative and neutral scores
    sa = SentimentIntensityAnalyzer()
    dd = sa.polarity_scores(text=processed_doc1)
    compound = round((1 + dd['compound'])/2, 2)

    return render_template('main.html', final=compound*100 ,text1=text_final,text2=dd['pos'],text5=dd['neg'],text4=compound,text3=dd['neu'])

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000, threaded=True)
