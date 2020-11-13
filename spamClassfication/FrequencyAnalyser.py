# Loading NLTK
import pprint

import nltk
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords
import matplotlib.pyplot as plt



def tokenize_word(text):
    tokenized_word = word_tokenize(text)
    # print(tokenized_word)
    return tokenized_word


def frequency(tokenized_word):
    fdist = FreqDist(tokenized_word)
    # print(fdist.pprint())
    return fdist


def plot(frequency_distribution):
    frequency_distribution.plot(30, cumulative=False)
    plt.show()


def remove_stop_words(tokenized_sentence):
    stop_words = set(stopwords.words("english"))
    filtered_sent = []
    for word in tokenized_sentence:
        if word not in stop_words:
            filtered_sent.append(word)
    # print("All stopwords:", stop_words)
    # print("Tokenized Sentence:", tokenized_sentence)
    # print("Filtered Sentence:", filtered_sent)
    return filtered_sent


def get_text_from_dictionary(input_array):
    mail_text = ""
    for input in input_array:
        mail_text += input['text']
    return mail_text


# if __name__ == '__main__':
#     ham, spam = CSVReader.get_spam_and_ham_from_csv()
#     text = get_text_from_dictionary(spam)
#
#     word_list = tokenize_word(text)
#     frequency_distribution = frequency(word_list)
#     plot(frequency_distribution)
#
#     filtered_sentence = remove_stop_words(word_list)
#     frequency_filtered_sentence = frequency(filtered_sentence)
#     plot(frequency_filtered_sentence)
