
from datasets import Dataset, DatasetDict
import pandas as pd
from transformers import AutoTokenizer
from transformers import TrainingArguments, Trainer
from transformers import AutoModelForSequenceClassification
from flask import Flask, render_template, request, jsonify
import numpy as np

import google.generativeai as genai

import json


class bertweet_sentiment_analysis:

    def __init__(self):
        self.trainer = self.init_model()

    def init_model(self):
        model = AutoModelForSequenceClassification.from_pretrained("Models/bertweet")
        trainer = Trainer(
            model=model
        )
        return trainer
    
    def tokenize_function(self, input_text):
        tokenizer = AutoTokenizer.from_pretrained(
            cls_token= "<s>",
            eos_token= "</s>",
            mask_token= "<mask>",
            model_max_length= 128,
            pretrained_model_name_or_path= "vinai/bertweet-base",
            normalization= False,
            pad_token= "<pad>",
            sep_token= "</s>",
            tokenizer_class= "BertweetTokenizer",
            unk_token= "<unk>"
        )
        return tokenizer(input_text["text"], padding='max_length', truncation=True,max_length=120)


    #1 is positive, 0 is negative
    def check_sentiment(self, prompt):
        data = Dataset.from_dict({'text': [prompt]})
        data_tok = data.map(self.tokenize_function, batched=True)
        result = self.trainer.predict(test_dataset=data_tok)
        result_logits = result[0]
        prediction = np.argmax(result_logits, axis=-1)[0]
        return prediction==0
    
class dehatebert_hate_speech:

    def __init__(self):
        self.trainer = self.init_model()

    def init_model(self):

        model = AutoModelForSequenceClassification.from_pretrained("Models/dehatebert")
        trainer = Trainer(
            model=model,
        )
        return trainer
    
    def tokenize_function(self, input_text):

        tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased", do_lower_case= False, max_len= 512, unk_token= "[UNK]", sep_token= "[SEP]", pad_token= "[PAD]", cls_token= "[CLS]", mask_token= "[MASK]")
        return tokenizer(input_text["text"], padding='max_length', truncation=True,max_length=512)


    def check_sentiment(self, prompt):
        data = Dataset.from_dict({'text': [prompt]})
        data_tok = data.map(self.tokenize_function, batched=True)
        result = self.trainer.predict(test_dataset=data_tok)
        result_logits = result[0]
        prediction = np.argmax(result_logits, axis=-1)[0]

        return prediction==1
    
class Gemini():
    def __init__(self):
        GOOGLE_API_KEY = 'AIzaSyA5aOEHO2OyoYXinHgsCNnZDflYYMMohlE'
        genai.configure(api_key=GOOGLE_API_KEY)
        self.model = genai.GenerativeModel('gemini-pro')
        self.filter_pipeline_models = {
            "dehatebert": dehatebert_hate_speech(),
            "bertweet": bertweet_sentiment_analysis()
        }
        self.select_filter_model("bertweet")

    def select_filter_model(self, filter_model):
        self.filter_pipeline = self.filter_pipeline_models[filter_model]

    def request_response(self, prompt):
        hate_speech = self.filter_pipeline.check_sentiment(prompt)
        if hate_speech:
            return "Hate Speech Detected"
        else:
            response = self.model.generate_content(prompt)
            try:
                return response.text
            except: 
                return "Response Blocked by Gemini"


app = Flask(__name__, template_folder='template')

gpt = Gemini()

recent_request = ""

try:
    reported_issues = json.load(open("reported_issues.json"))
except:
    reported_issues = {}

@app.route('/')
def index():
    return render_template('index_style_v4.html')

@app.route('/select_model', methods=['POST'])
def select_model():
    global gpt
    selected_model = request.form['models']
    gpt.select_filter_model(selected_model)
    return jsonify({'message': selected_model})

requesting = False

@app.route('/query_models', methods=['POST'])
def query_models():
    global requesting
    global recent_request
    if not requesting:
        requesting = True
        text = request.form['text']
        recent_request = text
        result = gpt.request_response(text)
        requesting = False
        return jsonify({'message': result, 'blocked' : result == "Response Blocked by Gemini" or result == "Hate Speech Detected"})
    else:
        return jsonify({'message': "Please Don't Spam Request, Still Loading", 'blocked' : True})

@app.route('/report_issue', methods=['POST'])
def report_issue():
    global reported_issues
    issue = request.form['issue']
    reported_issues[recent_request] = issue
    json.dump(reported_issues, open("reported_issues.json", "w"))
    return jsonify({'message': "Issue reported"})

if __name__ == '__main__':
    app.run(debug=True)
