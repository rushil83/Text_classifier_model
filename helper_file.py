from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import numpy as np
lemmatizer = WordNetLemmatizer()
stop = set(stopwords.words("english"))



def load_files(path):   
    text_category = open(path,mode='r')
    category_list= []
    for lines in text_category:
        text_category = word_tokenize(lines.lower())
        for i in text_category:
            text_category = [lemmatizer.lemmatize(i)]

            category_list += list(text_category)

    temp=[]
    for w in category_list:
        if w not in stop:
            temp.append(w)

    return temp


def bag_of_words(text_file1,text_file2):
    bag of words = text_file1 + text_file2
    return bag_of_words


def load_data(text_file1,text_file2,word_bank):
    x = []
    y = []
    files = [text_file1,text_file2]
    label_dict = {'text_file1':0,'text_file2':1}
    for file in files:
        for word in file:
            if word.lower() in word_bank:
                index_value = word_bank.index(word.lower())
                x.append([index_value+1])
                y.append([label[str(file)]])
    return x,y
