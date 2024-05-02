# HAII-Project: Socially-Aware Prompt Filtering

## Description

In order to prevent hateful content from being generated, large language models typically anaylze the input prompt before running the model to predict if it is hateful. However, many filtering models over-censor inputs that relate to "controversial" topics such as race, gender, or religion. Topics like these are essential to expressing one's identity, and therefore censoring innocent prompts relating to these topics is harmful towards people's ability to express themselves. 

Therefore, our goal is to create a prompt filtering algorithm that censors hateful content, while allowing more socially nuanced conversation of these controversial topics. The key idea behind our algorithm is that it is fine-tuned on socially-nuanced data. Namely, we used the [DIALOCONAN](https://github.com/marcoguerini/CONAN?tab=readme-ov-file#dialoconan) dataset, which simulates conversations between a hateful and neutral person talking about controversial topics. We fine-tuned many versions of two models, and picked the best versions by analyzing their performance on DIALOCONAN, a [twitter sentiment analysis dataset](https://www.kaggle.com/datasets/arkhoshghalb/twitter-sentiment-analysis-hatred-speech) and manual testing.

We are expressing our algorithm in the form of a demo accessible via a web interface. After following the setup mentioned below, one can interact with our algorithm via a web page. Look at the linked video for further demonstrations and instructions. The purpose of format would be that a person who represents company that hosts a large language model would contact us after trying the demo, and then implement the algorithm into their actual product instead of using the web interface.

## Setup:
1. Clone the repo
```bash
git clone https://github.com/aLyonsGH/HAII-Project.git
```
2. Enter the repo
```bash
cd HAII-Project
```
3. Create an anaconda environment and activate it
```bash
conda env create -f environment.yml
conda activate HAII_Project
```
4. Download the models with the following line
```bash
gdown --folder https://drive.google.com/drive/folders/12NrNo4fvMrXIBqpJnQRUQIlr4rBTByRC
```
## Running:
Run the Sentiment_Analysis_Flask.py file via
```bash
python Sentiment_Analysis_Flask.py
```
In your console, a link will soon appear that looks similar to the following
```text
http://127.0.0.1:5000
```
Copy and paste the generated link into your favorite browser, and the webpage will appear. To close the program, press Ctrl+c in the console.

## Repo Files
- Fine-Tuning: Scripts used for fine-tuning and evaluating the different models
- Models: Where the models are stored after the setup mentioned above
- template: Contains the style file for the web interface
- Sentiment_Analysis_Flask.py: Runs the main program
- environment.yml: File for setting up the anaconda environment

## Resources

- For learning about how to fine tune with Huggingface, we looked at this tutorial: https://huggingface.co/blog/sentiment-analysis-python
- For learning about web functionality (ex. How to make buttons, drop-down selections, etc) we adapted examples generated by ChatGPT
- For finding a dataset, we used this database: https://hatespeechdata.com

