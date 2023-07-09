import numpy as np
import pandas as pd
import json
import re
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from transformers import pipeline
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

class ResearchPaperSummariser:
    
    def __init__(self, corpus):
        self.Pdf = corpus
    
    def preprocess_corpus(self, pdf):
        select_sections = []
        remove_sections = []
        key_list = list(pdf.keys())
        for i in range(len(key_list)):
            if key_list[i].isalpha():
                select_sections.append(key_list[i])
            else:
                pattern = r'^(\d+\.){1,3}(\s.*)$'
                matches = re.findall(pattern, key_list[i])
                if matches:
                    if int(matches[0][0].split('.')[0]) < 10:
                        matches[0][1].strip()
                        if pdf[key_list[i]] != '':
                            select_sections.append(key_list[i])
                        else:
                            remove_sections.append(key_list[i])
                    else:
                        pdf[key_list[i-1]] = pdf[key_list[i-1]] + \
                            key_list[i] + pdf[key_list[i]]
                        remove_sections.append(key_list[i])
                else:
                    remove_sections.append(key_list[i])
        [pdf.pop(i) for i in remove_sections]
    
        return pdf, select_sections


    
    def tokenize_sentences(self, text):
        stop_words = set(stopwords.words('english'))
        sentences = nltk.sent_tokenize(text)
        tokenized_sentences = []
        for sentence in sentences:
            words = nltk.word_tokenize(sentence)
            words = [word for word in words if word.isalpha() and word not in stop_words]
            tokenized_sentences.append(" ".join(words))
        return tokenized_sentences



    
    def generate_tfidf_matrix(self, tokenized_corpus):
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(
            [sentence for document in tokenized_corpus.values() for sentence in document])
        return tfidf_matrix



    
    def rank_sentences(self, corpus, tokenized_corpus, tfidf_matrix):
        final_paper_corpus = {}
        for document_key, sentences in tokenized_corpus.items():
            document_index = list(corpus.keys()).index(document_key)
            sentence_scores = cosine_similarity(
                tfidf_matrix[document_index], tfidf_matrix)[0]
            ranked_sentences = sorted(((score, sentence) for score, sentence in zip(
                sentence_scores, sentences)), reverse=True)
    
            num_sentences = int(len(ranked_sentences) * 0.4)
    
            final_paper_corpus[document_key] = ''
            if len(ranked_sentences) > 400:
                for score, sentence in ranked_sentences[:num_sentences]:
                    final_paper_corpus[document_key] += sentence
                print(f"Document: {document_key}")
                for score, sentence in ranked_sentences[:num_sentences]:
                    print(f"Score: {score:.3f}\tSentence: {sentence}")
                print("\n")
            else:
                for score, sentence in ranked_sentences:
                    final_paper_corpus[document_key] += sentence
                print(f"Document: {document_key}")
                for score, sentence in ranked_sentences:
                    print(f"Score: {score:.3f}\tSentence: {sentence}")
                print("\n")
    
        return final_paper_corpus

    
    def generate_processed_corpus(self):
        pdf = self.Pdf
        pdf, select_sections = self.preprocess_corpus(pdf)
    
        tokenized_corpus = {}
        for key, value in pdf.items():
            value = value.lower()
            tokenized_sentences = self.tokenize_sentences(value)
            tokenized_corpus[key] = tokenized_sentences
    
        tfidf_matrix = self.generate_tfidf_matrix(tokenized_corpus)

    final_paper_corpus = self.rank_sentences(pdf, tokenized_corpus, tfidf_matrix)

    return final_paper_corpus

def generate_summary(self, processed_corpus):
    return_summary = {}
    summarization = pipeline("summarization")
    for i in processed_corpus:
        original_text = processed_corpus[i]
        summary_text = summarization(original_text)[0]['summary_text']
        return_summary[i] = summary_text
    return return_summary
corpus = {} # Provide your corpus data here

Create an instance of the ResearchPaperSummariser class
summarizer = ResearchPaperSummariser(corpus)

Generate the processed corpus
processed_corpus = summarizer.generate_processed_corpus()

Generate the summary
summary = summarizer.generate_summary(processed_corpus)

Print the summary
print(summary)
