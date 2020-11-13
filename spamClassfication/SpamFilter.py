from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report
import pandas as pd
from spamClassfication import FrequencyAnalyser
import warnings
warnings.filterwarnings('ignore')


def main():
    df = pd.read_csv("/Users/daniel/PycharmProjects/KI/spamClassfication/enron.csv", sep=";")
    x = df["subject"].fillna(" ") + " " + df["text"].fillna(" ")
    y = df["ham yes or no"]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=0)
    train_baseline(x_test, x_train, y_test, y_train)

    cv = CountVectorizer(stop_words='english')
    train_vectorizer(cv, x_test, x_train, y_test, y_train, "Count-Vectorizer")

    tv = TfidfVectorizer(use_idf=True, smooth_idf=True, stop_words='english')
    train_vectorizer(tv, x_test, x_train, y_test, y_train, "Tfidf-Vectorizer")


def train_vectorizer(vectorizer, x_test, x_train, y_test, y_train, name):
    vectorizer.fit(x_train)
    x_train_analysis = x_train
    x_train = vectorizer.transform(x_train)
    x_test = vectorizer.transform(x_test)
    print("\n{}".format(name))
    y_pred = train_model_NB(x_train, y_train, x_test, y_test)
    wrong_classifications = get_wrong_classifications(x_train_analysis, y_pred, y_train)
    plot_frequency(wrong_classifications)


def train_model_NB(x_train, y_train, x_test, y_test):
    model = MultinomialNB()
    model.fit(x_train, y_train)
    k_fold = KFold(n_splits=10, shuffle=True, random_state=0)
    y_pred = cross_val_predict(model, x_train, y_train, cv=10)
    conf_mat = confusion_matrix(y_train, y_pred)
    print("Naive Bayes Model:")
    print("F-Score per fold", cross_val_score(model, x_train, y_train, cv=k_fold, n_jobs=1))
    print(classification_report(y_train, y_pred))
    print(conf_mat)
    model_score = model.score(x_test, y_test)
    y_pred_test = model.predict(x_test)
    print("Testdata:\nF-Score:", model_score)
    print(classification_report(y_test, y_pred_test))
    print(confusion_matrix(y_test, y_pred_test))
    return y_pred


def get_wrong_classifications(x_train1, y_pred, y_train):
    number_correct = 0
    i = 0
    wrong_classifications = []
    for i, y in enumerate(y_train):
        if y_pred[i] != y:
            wrong_classifications.append(str(x_train1.get(i)))
            number_correct += 1.0
    return wrong_classifications


def plot_frequency(wrong_classifications):
    text_to_tokenize = ""
    for text in wrong_classifications:
        text_to_tokenize += text
    # pprint.pprint(wrong_classifications)
    tokenized_word = FrequencyAnalyser.tokenize_word(text_to_tokenize)
    tokenized_words_cleaned = FrequencyAnalyser.remove_stop_words(tokenized_word)
    freq = FrequencyAnalyser.frequency(tokenized_words_cleaned)
    FrequencyAnalyser.plot(freq)


def train_baseline(x_test, x_train, y_test, y_train):
    # Create ZeroR model
    dummy_model = DummyClassifier(strategy='most_frequent')
    dummy_model.fit(x_train, y_train)
    k_fold = KFold(n_splits=10, shuffle=True, random_state=0)
    y_pred = cross_val_predict(dummy_model, x_train, y_train, cv=10)
    conf_mat = confusion_matrix(y_train, y_pred)
    print("ZeroR-Model:")
    print(cross_val_score(dummy_model, x_train, y_train, cv=k_fold, n_jobs=1))
    print(classification_report(y_train, y_pred))
    print(conf_mat)


if __name__ == '__main__':
    main()
