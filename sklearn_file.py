from sklearn.naive_bayes import MultinomialNB

from sklearn.feature_extraction.text import CountVectorizer
import sklearn



def sklearn_model(algorithm, args, kwds):
	sklearn_classifier_model = algorithm()
	sklearn_classifier_model.fit(*args, **kwds)
	return sklearn_classifier_model()


def loading_file_sklearn(text_file_path):
	cvect = CountVectorizer(stop_words='english')
	text_file = open(text_file_path,mode='r').read()
	return text_file


def preprocessing_sklearn_model(algorithm,text_file1,text_file2):
	strach_classifier_model = algorithm()
	cvect = CountVectorizer(stop_words='english')
	bag_of_words = [text_file1,text_file2]
	classes = [0,1]
	train = cvect.fit_transform(bag_of_words).toarray()
	strach_classifier_model.fit(train,classes)
	return strach_classifier_model()
	


