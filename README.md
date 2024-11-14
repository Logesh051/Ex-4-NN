
<H3>ENTER YOUR NAME: Logesh.N.A</H3>
<H3>ENTER YOUR REGISTER NO: 212223240078</H3>
<H3>EX. NO.4</H3>
<H3>DATE: 27.9.24</H3>
<H1 ALIGN =CENTER>Implementation of MLP with Backpropagation for Multiclassification</H1>
<H3>Aim:</H3>
To implement a Multilayer Perceptron for Multi classification
<H3>Theory</H3>
A Multilayer Perceptron (MLP) is a type of feedforward artificial neural network, consisting of multiple neuron layers organized in a directed graph, meaning signals flow in one direction. Each neuron, except those in the input layer, has a nonlinear activation function. MLPs use backpropagation, a supervised learning technique, to adjust weights based on error signals, making them effective for complex tasks like speech and image recognition and machine translation.  

![MLP Architecture](https://user-images.githubusercontent.com/112920679/198804559-5b28cbc4-d8f4-4074-804b-2ebc82d9eb4a.jpg)

MLPs adjust weights by applying the Error Correction Rule, the Least Mean Squares (LMS) method, and a backpropagation algorithm divided into two phases: a Feedforward Pass and a Backward Pass. During the Feedforward Pass, input data flows neuron by neuron to the output layer, while weights remain unchanged. The Backward Pass then uses the error signal to move back through the layers, adjusting weights according to the delta rule to minimize errors.
![MLP Neuron Connections](https://user-images.githubusercontent.com/112920679/198814300-0e5fccdf-d3ea-4fa0-b053-98ca3a7b0800.png)

The forward pass starts from the first hidden layer and ends at the output layer, where function signals are calculated for each neuron, and weights remain unchanged. In the backward pass, the error signal starts at the output layer and moves leftward, adjusting synaptic weights through the delta rule.

![Forward Pass](https://user-images.githubusercontent.com/112920679/198814349-a6aee083-d476-41c4-b662-8968b5fc9880.png)  
![Backward Pass](https://user-images.githubusercontent.com/112920679/198814362-05a251fd-fceb-43cd-867b-75e6339d870a.png)

To implement an MLP algorithm, start by importing the necessary libraries, then load and organize the dataset into features and labels, and split it into training and testing sets. Normalize the data, initialize the MLP classifier with chosen parameters such as hidden layer sizes, activation function, and max iterations, then predict values on test data. Finally, evaluate performance using confusion_matrix() and classification_report() functions.

<H3>Program:</H3> 

```py
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']
irisdata = pd.read_csv(url, names=names)

X = irisdata.iloc[:, 0:4]
y = irisdata['Class']

le = preprocessing.LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.20, random_state=42)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)
mlp.fit(X_train, y_train)

predictions = mlp.predict(X_test)

flower_predictions = le.inverse_transform(predictions)


print(flower_predictions)  
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
```

<H3>Output:</H3>

![Screenshot 2024-09-27 090101](https://github.com/user-attachments/assets/6e5dcf13-00ef-4b49-a09c-75cbf7de0ebd)



<H3>Result:</H3>
Thus, MLP is implemented for multi-classification using python.
