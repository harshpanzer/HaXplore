from flask import Flask, request, jsonify
import numpy as np
from keras.models import load_model  
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
import json
import random

app = Flask(__name__)

def lo_model(model_add):
    try:
        model = load_model(model_add)
        return model
    except Exception as e:
        raise Exception(f"Failed to load model: {str(e)}")

pre_model = lo_model('.\Coginitive Disorder Pridiction\exp_01.h5')
chatbot = load_model('.\Mental health chatbot\model.h5')

intents = json.loads(open('.\Mental health chatbot\intents.json').read())
words = pickle.load(open('.\Mental health chatbot\lexts.pkl','rb'))
classes = pickle.load(open('.\Mental health chatbot\labels.pkl','rb'))
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words
def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))
def predict_class(sentence, model):
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list
def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result
def chatbot_response(msg):
    ints = predict_class(msg, chatbot)
    res = getResponse(ints, intents)
    return res



@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        if 'input_data' not in data:
            return jsonify({'error': 'Input data is missing'}), 400
        input_data = np.array(data['input_data'][0])
        prediction = pre_model.predict(input_data.reshape(1,14,1)) 
        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/chat', methods=['POST'])
def get_bot_response():
    try:
        chat = request.json
        if 'user_data' not in chat:
           return jsonify({'error': 'User data is missing'}), 400
        userText = chat['user_data']
        return chatbot_response(userText)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
