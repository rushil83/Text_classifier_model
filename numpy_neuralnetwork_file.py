import numpy as np

def numpy_classifier(x,y):
    x = np.array(x)
    y = np.array(y)
    weights1 = 2*np.random.random((1,len(x))) - 1
    weights2 = 2*np.random.random((len(x),1)) - 1
    bias1 = 2*np.random.random((1,len(x))) - 1
    bias2 = 2*np.random.random((len(x),1)) - 1


    for j in range(4):
        input_layer = x
        hidden_layer1 = 1/(1+np.exp(-((np.dot(input_layer,weights1)))))
        hidden_layer2 = 1/(1+np.exp(-((np.dot(hidden_layer1,weights2)))))
        error = y - hidden_layer2
        l2_delta = error  *(hidden_layer2/(1-hidden_layer2))
        l1_error = l2_delta.dot(weights2.T)
        l1_delta = l1_error * (hidden_layer1/(1-hidden_layer1))
        weights2 += hidden_layer1.T.dot(l2_delta)
        weights1 += input_layer.T.dot(l1_delta)
        print(str(np.mean(np.abs(error))))
