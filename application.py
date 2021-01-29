from flask import Flask
from flask_restful import reqparse, abort, Api, Resource
import pickle
import numpy as np
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import re

application = Flask(__name__)

clf_path = 'sentiment_analyser_keras.hdf5'
model = load_model(clf_path)


spam_clf_path = 'spam_detector.hdf5'
spam_model = load_model(spam_clf_path)

vec_path = 'tokenizer.pickle'
with open(vec_path, 'rb') as f:
    tokenizer = pickle.load(f)

spam_vec_path = 'spam-tokenizer.pickle'
with open(spam_vec_path, 'rb') as f:
    spam_tokenizer = pickle.load(f)
    

# argument parsing
parser = reqparse.RequestParser()
parser.add_argument('query')


NON_ALPHANUM = re.compile(r'[\W]')
NON_ASCII = re.compile(r'[^a-z0-1\s]')

def normalize_texts(texts):
    normalized_texts = []
    for text in texts:
        lower = text.lower()
        no_punctuation = NON_ALPHANUM.sub(r' ', lower)
        no_non_ascii = NON_ASCII.sub(r'', no_punctuation)
        normalized_texts.append(no_non_ascii)
    return normalized_texts

@application.route('/', methods=['GET'])
def sa_trpp():
    # use parser and find the user's query
    args = parser.parse_args()
    user_query = args['query']
    
    MAX_LENGTH = 255        
    SPAM_MAX_LENGTH = 32

    # vectorize the user's query and make a prediction
    #user_query = 'Stuning even for the non-gamer: This sound track was beautiful! It paints the senery in your mind so well I would recomend it even to people who hate vid. game music! I have played the game Chrono Cross but out of all of the games I have ever played it has the best music! It backs away from crude keyboarding and takes a fresher step with grate guitars and soulful orchestras. It would impress anyone who cares to listen!'
    #user_query='Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat...'
    #user_query = 'XXXMobileMovieClub: To use your credit, click the WAP link in the next txt message or click here>> http://wap. xxxmobilemovieclub.com?n=QJKGIGHJJGCBL'
    user_query = normalize_texts([str(user_query)])
            
    uq_vectorized = tokenizer.texts_to_sequences(user_query)
    uq_vectorized_spam = spam_tokenizer.texts_to_sequences(user_query)
    
    uq_vectorized = pad_sequences(uq_vectorized, maxlen=MAX_LENGTH)
    uq_vectorized_spam = pad_sequences(uq_vectorized_spam, maxlen = SPAM_MAX_LENGTH)

    prediction = model.predict(uq_vectorized)
    prediction_spam = spam_model.predict(uq_vectorized_spam)
    
    pred_proba = prediction[0]
    pred_proba_spam = prediction_spam[0]       
    
    
    # Output either 'Negative' or 'Positive' along with the score
    if pred_proba < 0.4:
        pred_text = 'Negative'
    elif pred_proba > 0.6:
        pred_text = 'Positive'
    else:
        pred_text = 'Neutral'     
        
    if pred_proba_spam < 0.4:
        pred_text_spam = 'Valid'
    elif pred_proba_spam > 0.6:
        pred_text_spam = 'Spam'
    else:
        pred_text_spam = 'Not Sure' 
        
    # round the predict proba value and set to new variable
    confidence = round(pred_proba[0], 3)
    confidence_spam = round(pred_proba_spam[0], 3)

    # create JSON object
    output = {'sentiment_prediction': pred_text, 'sentiment_confidence': str(confidence),
              'spam_prediction': pred_text_spam, 'spam_confidence': str(confidence_spam)}       
    return output


if __name__ == '__main__':
    application.run(debug=False)