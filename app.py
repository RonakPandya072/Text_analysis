from flask import Flask, render_template, request, jsonify,session
import numpy as np
import pandas as pd
from numpy import save
from numpy import load
import matplotlib.pyplot as plt
from collections import Counter
from bs4 import BeautifulSoup
import urllib.request
from urllib.request import Request, urlopen
from htmldate import find_date
import textstat
textstat.set_lang("en")
import re
import nltk
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.collocations import *
import random
from textblob import TextBlob
from textblob import Word
from wordcloud import WordCloud, STOPWORDS
wcstopwords = set(STOPWORDS)
import requests
#from gensim.summarization.summarizer import summarize
#from gensim.summarization import keywords
from wordcloud import WordCloud, STOPWORDS
wcstopwords = set(STOPWORDS)


class imgcounterclass():
    """Class container for processing stuff."""

    _counter = 0

    def addcounter(self):
        self._counter += 1

image_counter=imgcounterclass()


class cleanedtext_store:
    def __init__(self):
        self.text = None

    def addtext(self, newtext):
        self.text = newtext

    def cleartext(self):
        self.text = None


cleanedtextstore = cleanedtext_store()

def text_summarizer(url):
  text=[]
  page = requests.get(url)
  soup = BeautifulSoup(page.content,'html.parser')
  para = soup.find_all('p',id=None)
  for par in para:
    text.append(par.get_text())
  text=" ".join(text)
  return summarize(text,split=True, ratio=0.5)

def text_summarizer_keyword(url):
      text = []
      page = requests.get(url)
      soup = BeautifulSoup(page.content, 'html.parser')
      para = soup.find_all('p', id=None)
      for par in para:
          text.append(par.get_text())
      text = " ".join(text)
      return keywords(text, words=5, lemmatize=True)


  #wordcloud = WordCloud(background_color="white").generate(text)
  #plt.imshow(wordcloud,interpolation='bilinear')
  #plt.axis("off")
  #plt.show()


def wordcloud(tokensalpha):
    comment_words = " ".join(tokensalpha) + " "

    wordcloud = WordCloud(width=800, height=500,
                          background_color='white',
                          stopwords=wcstopwords,
                          min_font_size=8).generate(comment_words)

    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)

    return plt


def get_html(url):
    user_agent_list = [
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_5) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/13.1.1 Safari/605.1.15',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:77.0) Gecko/20100101 Firefox/77.0',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:77.0) Gecko/20100101 Firefox/77.0',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36',
        ]

    for i in range(1, 4):
        # Pick a random user agent
        user_agent = random.choice(user_agent_list)
        # Set the headers
        headers = {'User-Agent': user_agent}
        req = Request(url, headers=headers)

    return urlopen(req).read()


def cleaned_text(htmlinput):
    cleaned_text = BeautifulSoup(htmlinput, "html.parser").get_text(" ").replace("\r", " ").replace("\t", " ").replace(
        "\n", " ").replace(u'\xa0', u' ')
    return cleaned_text


def tokens_alpha(htmlinput):
    raw = BeautifulSoup(htmlinput, 'html.parser').get_text(strip=True)
    words = nltk.word_tokenize(raw)
    tokens_alpha = [word for word in words if word.isalpha()]  # or use option of := if word.isalnum()
    return tokens_alpha


def extract_title(htmlinput):
    return BeautifulSoup(htmlinput, 'html.parser').title.string


def get_date(urlinput):
    return find_date(urlinput)


def extract_tags(htmlinput):
    tags = []
    tags = [x.text for x in BeautifulSoup(htmlinput, 'html.parser').find_all("a", href=re.compile(".*tag.*"))]
    if len(tags) == 0:
        return 'None'
    else:
        return tags


def image_count(htmlinput):
    try:
        article_soup = BeautifulSoup(htmlinput, 'html.parser').find('article')
        figures = article_soup.find_all('figure')
        if len(figures) > 0:
            return len(figures)
        else:
            return len(BeautifulSoup(htmlinput, 'html.parser').find_all('img'))
    except:
        return len(BeautifulSoup(htmlinput, 'html.parser').find_all('img'))


def embedded_vids_count(htmlinput):
    return str(htmlinput).count("embed&display_name=YouTube")


def other_embedded_items_count(htmlinput, embedded_vids_count):
    return int(str(htmlinput).count("iframeSrc")) - int(embedded_vids_count)


def word_count(cleanedtext):
    return textstat.lexicon_count(cleanedtext, removepunct=True)


def tokenize_excl_stopwords(
        cleanedtext):  # a bit of repetition but this version unlike the other tokenize text removes stop words
    lst = re.findall(r'\b\w+', cleanedtext)
    lst = [x.lower() for x in lst]
    counter = Counter(lst)
    occs = [(word, count) for word, count in counter.items() if count > 1]
    occs.sort(key=lambda x: x[1])
    WordListWithCommonWordsRemoved = [(occs[i][0], occs[i][1]) for i in range(len(occs)) if
                                      not occs[i][0] in stopwords.words()]
    return WordListWithCommonWordsRemoved


def vocab_count_excl_commonwords(tokenize_excl_stopwords):
    return len(tokenize_excl_stopwords)


def most_common_words(tokenize_excl_stopwords):
    return tokenize_excl_stopwords()[-5:]


def sentence_count(cleanedtext):
    blob = TextBlob(cleanedtext)
    split_text = blob.sentences
    No_Of_Sentences = len(split_text)
    # Initially used this but split_text tends to overcount
    return textstat.sentence_count(cleanedtext)


def content_summary(cleanedtext):
    try:
        blob = TextBlob(cleanedtext)
        split_text = blob.sentences

        nouns = list()
        for word, tag in blob.tags:
            if tag == 'NN':
                nouns.append(word.lemmatize())
        text_summary = ["This text is about..."]

        for item in random.sample(nouns, 5):
            word = Word(item)
            if len(word) > 3:
                text_summary.append(word)  # Can also use word.pluralize()
    except:
        text_summary = []

    return text_summary


def sentiment(cleanedtext):
    blob = TextBlob(cleanedtext)
    split_text = blob.sentences
    df = pd.DataFrame((''.join(split_text[i]) for i in range(len(split_text))), columns=['Sentences'])
    df[["TextBlob_Polarity", "TextBlob_Subjectivity"]] = pd.DataFrame(
        (split_text[i].sentiment for i in range(len(split_text))))
    df = df[df['Sentences'].map(len) > 15]  # Remove all short sentences
    # Avoid counting any sentences with Polarity 0 or Subjectivity 0
    TextBlob_Overall_Polarity = df[df["TextBlob_Polarity"] != 0]['TextBlob_Polarity'].median()
    TextBlob_Overall_Subjectivity = df[df["TextBlob_Subjectivity"] != 0]['TextBlob_Subjectivity'].median()
    return TextBlob_Overall_Polarity, TextBlob_Overall_Subjectivity


def polarity(sentiment):
    return sentiment()[0] * 100


def subjectivity(self):
    return sentiment()[1] * 100


def common_trigrams(tokensalpha):
    trigram_measures = nltk.collocations.TrigramAssocMeasures()
    finder = TrigramCollocationFinder.from_words(tokensalpha)
    finder.apply_freq_filter(2)
    finder.nbest(trigram_measures.pmi, 5)  # doctest: +NORMALIZE_WHITESPACE
    FiveMostCommonlyOccuringTrigrams = finder.nbest(trigram_measures.pmi, 5)
    return FiveMostCommonlyOccuringTrigrams


def FS_ReadingEaseScore(cleanedtext):
    FS_GradeScore = textstat.flesch_reading_ease(cleanedtext)
    return FS_GradeScore


def FS_ReadingEaseLevel(FS_GradeScore):
    if FS_GradeScore >= 90:
        FS_Grade = "Very easy to read"
    elif FS_GradeScore < 90 and FS_GradeScore >= 80:
        FS_Grade = "Easy"
    elif FS_GradeScore < 80 and FS_GradeScore >= 70:
        FS_Grade = "Fairly easy"
    elif FS_GradeScore < 70 and FS_GradeScore >= 60:
        FS_Grade = "Standard"
    elif FS_GradeScore < 60 and FS_GradeScore >= 50:
        FS_Grade = "Fairly Difficult"
    elif FS_GradeScore < 50 and FS_GradeScore >= 30:
        FS_Grade = "Difficult"
    else:
        FS_Grade = "Very Confusing"
    return FS_Grade

def sentence_by_sentence_analysis(cleanedtext):
    blob = TextBlob(cleanedtext)
    split_text = blob.sentences
    df = pd.DataFrame((''.join(split_text[i]) for i in range(len(split_text))), columns=['Sentences'])
    df["Sentence Word Count"] = pd.DataFrame(len(df["Sentences"][i].split()) for i in range(len(df)))
    df["FS_GradeScore"] = pd.DataFrame((textstat.flesch_reading_ease(df["Sentences"][i]) for i in range(len(df))))
    df[["TextBlob_Polarity", "TextBlob_Subjectivity"]] = round(
        pd.DataFrame((split_text[i].sentiment for i in range(len(split_text)))) * 100, 1)
    return df


def plot_sentence_by_sentence(cleanedtext, type="polarity"):
    blob = TextBlob(cleanedtext)
    split_text = blob.sentences
    df = pd.DataFrame((''.join(split_text[i]) for i in range(len(split_text))), columns=['Sentences'])
    df["Sentence Word Count"] = pd.DataFrame(len(df["Sentences"][i].split()) for i in range(len(df)))
    df["FS_GradeScore"] = pd.DataFrame((textstat.flesch_reading_ease(df["Sentences"][i]) for i in range(len(df))))
    df[["TextBlob_Polarity", "TextBlob_Subjectivity"]] = round(
        pd.DataFrame((split_text[i].sentiment for i in range(len(split_text)))) * 100, 1)

    # gca stands for 'get current axis'
    figure = plt.figure()
    ax = plt.gca()

    if type == "polarity":
        plotvar = 'TextBlob_Polarity'
    elif type == "subjectivity":
        plotvar = 'TextBlob_Subjectivity'
    elif type == "readability":
        plotvar = "FS_GradeScore"
    elif type == "wordcount":
        plotvar = "Sentence Word Count"
    else:
        plotvar = "polarity"

    print("Plot Of ", plotvar, " By Sentence")
    df.plot(kind='line', y=plotvar, ax=ax)
    # set the x-spine
    ax.spines['left'].set_position('zero')

    # turn off the right spine/ticks
    ax.spines['right'].set_color('none')
    # ax.yaxis.tick_left()

    # set the y-spine
    ax.spines['bottom'].set_position('zero')

    # turn off the top spine/ticks
    ax.spines['top'].set_color('none')
    ax.xaxis.tick_bottom()

    print("The X-Axis represents sentences within the document")
    return plt  # plt.show()

def plot_subjectivity_by_sentence(cleanedtext, type="polarity"):
    blob = TextBlob(cleaned_text)
    split_text = blob.sentences
    df = pd.DataFrame((''.join(split_text[i]) for i in range(len(split_text))), columns=['Sentences'])
    df[["TextBlob_Polarity", "TextBlob_Subjectivity"]] = round(pd.DataFrame((split_text[i].sentiment for i in range(len(split_text)))) * 100, 1)

    # gca stands for 'get current axis'
    ax = plt.gca()

    df.plot(kind='line', y='TextBlob_Subjectivity', color='red', ax=ax)
    # set the x-spine
    ax.spines['left'].set_position('zero')

    # turn off the right spine/ticks
    ax.spines['right'].set_color('none')
    ax.yaxis.tick_left()

    #set the y-spine
    ax.spines['bottom'].set_position('zero')

    # turn off the top spine/ticks
    ax.spines['top'].set_color('none')
    ax.xaxis.tick_bottom()

    print("The X-Axis represents sentences within the document")
    return plt.show()

def plot_readability_by_sentence(self):
    blob = TextBlob(self.cleaned_text)
    split_text = blob.sentences
    df = pd.DataFrame((''.join(split_text[i]) for i in range(len(split_text))), columns=['Sentences'])
    df["Sentence Word Count"] = pd.DataFrame(len(df["Sentences"][i].split()) for i in range(len(df)))
    df["FS_GradeScore"] = pd.DataFrame((textstat.flesch_reading_ease(df["Sentences"][i]) for i in range(len(df))))
    # gca stands for 'get current axis'
    ax = plt.gca()

    df.plot(kind='line',y='FS_GradeScore',color='blue',ax=ax)
    df.plot(kind='line',y='Sentence_Word_Count', color='black', ax=ax)
    # set the x-spine
    ax.spines['left'].set_position('zero')

    # turn off the right spine/ticks
    ax.spines['right'].set_color('none')
    ax.yaxis.tick_left()

    # set the y-spine
    ax.spines['bottom'].set_position('zero')

    # turn off the top spine/ticks
    ax.spines['top'].set_color('none')
    ax.xaxis.tick_bottom()

    print("The X-Axis represents sentences within the document")
    return plt.show()



app = Flask(__name__,static_folder='./static',template_folder="templates")

@app.route('/') # To render Homepage
def home_page():
    return render_template('index.html')

url_returnerd_html = " "

@app.route('/single_page_URL', methods=['POST'])
def single_page_URL_results():
    url_returned = request.form['Single_URL']
    url_returnerd_html = url_returned
    htmlinput = get_html(url_returned)
    cleanedtext = cleaned_text(htmlinput)
    tokensalpha = tokens_alpha(htmlinput)
    tokenizeexclstopwords = tokenize_excl_stopwords(cleanedtext)
    sentiments = sentiment(cleanedtext)

    summary = {
        "Analysis Type": "Extracted Web-Page",
        "title": extract_title(htmlinput),
        "pub_date": get_date(url_returned),

        "url": url_returned,
        # Semantic Content
        "tags": extract_tags(htmlinput),
        "content_summary": content_summary(cleanedtext),
        "most_common_trigrams": common_trigrams(tokensalpha),

        # Lexical Content
        "title_word_count": (len(extract_title(htmlinput).replace(":", "").replace("'", "").replace("/", "").split())),
        "text_word_count": word_count(cleanedtext),
        "vocab_count_excl_commonwords": len(tokenizeexclstopwords),
        "most frequent words": tokenizeexclstopwords[-5:],

        "sentence_count": sentence_count(cleanedtext),
        "average_word_count_per_sentence": word_count(cleanedtext) / sentence_count(cleanedtext),

        # Readability
        "FS_GradeLevel": FS_ReadingEaseLevel(FS_ReadingEaseScore(cleanedtext)),
        "FS_GradeScore": FS_ReadingEaseScore(cleanedtext),

        # Metadata Metrics
        "image_count": image_count(htmlinput),
        "embedded_vids_count": embedded_vids_count(htmlinput),
        "other_embedded_items_count": other_embedded_items_count(htmlinput, embedded_vids_count(htmlinput)),
        "imgs_per_1000words": ((image_count(htmlinput)) / (word_count(cleanedtext))) * 1000,
        "vids_per_1000words": ((embedded_vids_count(htmlinput)) / (word_count(cleanedtext))) * 1000,

        # Sentiment Analysis

        "polarity": sentiments[0] * 100,
        "subjectivity": sentiments[1] * 100,
    }

    session.pop('summary', None)
    session['summary'] = summary

    url_df = pd.DataFrame.from_dict(summary, orient='index')
    pd.set_option('display.max_colwidth', 800)

    return render_template('single_page_URL.html',url_text= url_returned,urldetails_tables=[url_df.to_html(classes='data', header="false")])


@app.route('/detailed_analysis_URL', methods=['POST'])
def detailed_analysis_text():
    url_returned = request.form['URL_Text']
    htmlinput = get_html(url_returned)
    cleanedtext = cleaned_text(htmlinput)
    image_counter.addcounter()
    text_df = sentence_by_sentence_analysis(cleanedtext)

    pd.set_option('display.max_colwidth', 800)

    plot_sentence_by_sentence(cleanedtext, type='wordcount').savefig(f'static/wordcount{image_counter._counter}.png',
                                                                     bbox_inches="tight")

    plot_sentence_by_sentence(cleanedtext, type='readability').savefig(
        f'static/readability{image_counter._counter}.png', bbox_inches="tight")

    plot_sentence_by_sentence(cleanedtext, type='subjectivity').savefig(
        f'static/subjectivity{image_counter._counter}.png', bbox_inches="tight")

    plot_sentence_by_sentence(cleanedtext, type='polarity').savefig(f'static/polarity{image_counter._counter}.png',
                                                                    bbox_inches="tight")

    return render_template('detailed_analysis_URL.html', image_counter=str(image_counter._counter),
                           text_tables=[text_df.to_html(classes='data', header="false")])

@app.route('/summary', methods=['POST'])
def summary():
    url_returned = url_returnerd_html
    # url_returned = request.form['Single_URL']
    htmlinput = get_html(url_returned)
    cleanedtext = cleaned_text(htmlinput)
    image_counter.addcounter()
    #text_df = sentence_by_sentence_analysis(cleanedtext)
    wordcloud(cleanedtext).savefig(f'static/wordcloud{image_counter._counter}.png',bbox_inches="tight")
    #text_summarizer(url_returned)
    #text_summarizer_keyword(url_returned)
    return render_template('summary.html', image_counter=str(image_counter._counter))
#text_tables=[text_df.to_html(classes='data', header="false")

if __name__ == '__main__':
    app.secret_key = 'super secret key'
    app.config['SESSION_TYPE'] = 'filesystem'
    app.run(debug = True)