# train_models.py
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import  accuracy_score
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

def train_logistic_regression(X_train, y_train):
    # Initialize the Logistic Regression model
    logmodel = LogisticRegression(random_state=1)
    
    # Train the model
    logmodel.fit(X_train, y_train)
    
    return logmodel

def train_knn(X_train, y_train, X_test, y_test):
    # Define a range of values for K
    k_values = list(range(1, 50)) 

    # Initialize lists to store performance metrics
    error_rates = []

    # Iterate through different values of K
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)

        
        error_rates.append(np.mean(y_pred != y_test))

    # Plot the elbow graph
    plt.plot(k_values, error_rates, marker='o', markerfacecolor='red')
    plt.title('Elbow Method for Optimal K')
    plt.xlabel('Number of Neighbors (K)')
    plt.ylabel('Accuracy')
    plt.show()

    # Choose the number of neighbors (you can adjust as needed)
    n_neighbors = 13
    
    # Initialize the K-Nearest Neighbors model
    knn_model = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-1)
    
    # Train the model
    knn_model.fit(X_train, y_train)
    
    return knn_model

def train_random_forest(X_train, y_train):
    # Initialize the Random Forest model
    rfc_model = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=0)
    
    # Train the model
    rfc_model.fit(X_train, y_train)
    
    return rfc_model

def train_svm(X_train, y_train):
    # Initialize the Support Vector Machine model
    svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=0)
    
    # Train the model
    svm_model.fit(X_train, y_train)
    
    return svm_model

def train_deep_neural_network(X_train, y_train):
    # Convert y_train to integer type
    y_train = y_train.astype(int)
    
    # Build the DNN model
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(2, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

    return model
