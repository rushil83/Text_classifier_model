# Text Classifier Model


Helper file:
- load the large text file
- preprocessing of text file (`nlp stuff`)
    - lematizing
    - stemming
    - eradicating stop words
    - and etc..
for e.g. our hodor
![sfsd](data:image/jpeg;base64,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)
- creating bag of words and word vectore feature space
- loading all our model in x,y form


Sklearn file:
- Classifier our text model using sklearn available model such as logisticregression, naivebais and etc..
- here we have also created and alternative preprocesing function such that we can evaluate which library (sklearn     or nltk) would be better to preprocess in text related model


Numpy file:
- created a simple neural network model in numpy to test our classifier model
- main purpose was to get know the math behind neural network


Tensorflow file:
-  created tensorflow neural network model consisting of 3 hidden layers unit. 
![deeper](https://www.wired.com/wp-content/uploads/2015/06/1UZLj4yUiEPOE01ZJgqLrlw-1.jpeg)

