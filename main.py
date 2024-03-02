# main.py
from data_utils import load_and_preprocess_data
from preprocessing import clean_and_transform_data
from feature_scaling import feature_scaling
from train_models import (
    train_logistic_regression, train_knn, train_random_forest, train_svm, train_deep_neural_network
)
from evaluation_metrics import evaluate_model
from cross_validation import k_fold_cross_validation
from sklearn.model_selection import train_test_split
import tensorflow as tf

def main():
    # Load and preprocess data
    dataset = load_and_preprocess_data('data/defaultofcreditcardclients.csv')

    # Clean and transform data
    clean_dataset = clean_and_transform_data(dataset)

    # Feature scaling
    scaled_dataset = feature_scaling(clean_dataset)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_dataset.iloc[:, :-1].values, scaled_dataset.iloc[:, -1].values,
        test_size=0.3, random_state=1
    )

    # Train Logistic Regression model
    logmodel = train_logistic_regression(X_train, y_train)
    y_pred_logistic = logmodel.predict(X_test)
    evaluate_model("Logistic Regression", y_test, y_pred_logistic)

    # Train K-Nearest Neighbors model
    knn_model = train_knn(X_train, y_train, X_test, y_test)
    y_pred_knn = knn_model.predict(X_test)
    evaluate_model("K-Nearest Neighbors", y_test, y_pred_knn)

    # Train Random Forest model
    rfc_model = train_random_forest(X_train, y_train)
    y_pred_rf = rfc_model.predict(X_test)
    evaluate_model("Random Forest", y_test, y_pred_rf)

    # Train Support Vector Machine model
    svm_model = train_svm(X_train, y_train)
    y_pred_svm = svm_model.predict(X_test)
    evaluate_model("Support Vector Machine", y_test, y_pred_svm)

    # Train Deep Neural Network model
    dnn_model = train_deep_neural_network(X_train, y_train)
    y_pred_dnn = dnn_model.predict(X_test)
    y_pred_dnn = tf.argmax(y_pred_dnn, axis=1)
    evaluate_model("Deep Neural Network", y_test, y_pred_dnn)

    # Perform k-fold cross-validation for K-Nearest Neighbors
    k_fold_cross_validation(knn_model, X_train, y_train)

if __name__ == "__main__":
    main()