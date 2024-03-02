# cross_validation.py
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def k_fold_cross_validation(model, X, y, k=10):
    k_fold = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    accuracy_list = []
    precision_list = []
    recall_list = []
    f1_score_list = []
    roc_auc_list = []

    for train_indices, test_indices in k_fold.split(X, y):
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        accuracy_list.append(accuracy_score(y_test, predictions))
        precision_list.append(precision_score(y_test, predictions))
        recall_list.append(recall_score(y_test, predictions))
        f1_score_list.append(f1_score(y_test, predictions))
        roc_auc_list.append(roc_auc_score(y_test, predictions))

    return {
        'accuracy': accuracy_list,
        'precision': precision_list,
        'recall': recall_list,
        'f1_score': f1_score_list,
        'roc_auc': roc_auc_list
    }
