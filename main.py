import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score





# Load train and test data from CSV files
X_train_data = pd.read_csv('/Users/sachinpranav/Downloads/Kannada_MNIST_datataset_paper/Kannada_MNIST_npz/Kannada_MNIST/X_kannada_MNIST_train.csv')
y_train_data = pd.read_csv('/Users/sachinpranav/Downloads/Kannada_MNIST_datataset_paper/Kannada_MNIST_npz/Kannada_MNIST/y_kannada_MNIST_train.csv')

X_test_data = pd.read_csv('/Users/sachinpranav/Downloads/Kannada_MNIST_datataset_paper/Kannada_MNIST_npz/Kannada_MNIST/X_kannada_MNIST_test.csv')
y_test_data = pd.read_csv('/Users/sachinpranav/Downloads/Kannada_MNIST_datataset_paper/Kannada_MNIST_npz/Kannada_MNIST/y_kannada_MNIST_test.csv')

y_train = y_train_data.values.ravel()
y_test = y_test_data.values.ravel()

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train_data, y_train_data, test_size=0.2, random_state=42)

# Initialize PCA with 10 components for training data
pca = PCA(n_components=10)
X_train_pca = pca.fit_transform(X_train)
X_val_pca = pca.transform(X_val)
print(X_val_pca)
# Initialize PCA with 10 components for test data
X_test_pca = pca.transform(X_test_data)
print(X_test_pca)

def evaluate_model(model, X_train, y_train, X_val, y_val):
    # Fit the model to the training data
    model.fit(X_train, y_train)

    # Make predictions on the validation data
    y_val_pred = model.predict(X_val)

    # Calculate evaluation metrics on validation data
    accuracy_val = accuracy_score(y_val, y_val_pred)
    precision_val = precision_score(y_val, y_val_pred, average='weighted')
    recall_val = recall_score(y_val, y_val_pred, average='weighted')
    f1_val = f1_score(y_val, y_val_pred, average='weighted')
    cm_val = confusion_matrix(y_val, y_val_pred)
    roc_auc_val = roc_auc_score(y_val, model.predict_proba(X_val), multi_class='ovr')

    return accuracy_val, precision_val, recall_val, f1_val, cm_val, roc_auc_val

decision_tree = DecisionTreeClassifier()
random_forest = RandomForestClassifier()
naive_bayes = GaussianNB()
knn_classifier = KNeighborsClassifier()
svm_classifier = SVC(probability=True)  # Use probability for ROC-AUC calculation

models = [decision_tree, random_forest, naive_bayes, knn_classifier, svm_classifier]
model_names = ['Decision Tree', 'Random Forest', 'Naive Bayes', 'K-NN Classifier', 'SVM']

# Dictionary to store the results for each model on the training data
train_results = {}

for model, name in zip(models, model_names):
    print(f"Training and evaluating {name} on the training data...")
    (accuracy_val, precision_val, recall_val, f1_val, cm_val, roc_auc_val) = evaluate_model(model, X_train_pca, y_train, X_val_pca, y_val)

    train_results[name] = {
        'accuracy_val': accuracy_val,
        'precision_val': precision_val,
        'recall_val': recall_val,
        'f1_val': f1_val,
        'cm_val': cm_val,
        'roc_auc_val': roc_auc_val,
    }

# Print results for training data
for name, metrics in train_results.items():
    print(f"Metrics for {name} on the training data:")
    print("Validation Accuracy:", metrics['accuracy_val'])
    print("Validation Precision:", metrics['precision_val'])
    print("Validation Recall:", metrics['recall_val'])
    print("Validation F1-Score:", metrics['f1_val'])
    print("Validation Confusion Matrix:\n", metrics['cm_val'])
    print("Validation ROC-AUC:", metrics['roc_auc_val'])
    print("\n")

def evaluate_model_test(model, X_train, y_train, X_test, y_test):
    # Fit the model to the entire training data
    model.fit(X_train, y_train)

    # Make predictions on the test data
    y_test_pred = model.predict(X_test)

    # Calculate evaluation metrics on test data
    accuracy_test = accuracy_score(y_test, y_test_pred)
    precision_test = precision_score(y_test, y_test_pred, average='weighted')
    recall_test = recall_score(y_test, y_test_pred, average='weighted')
    f1_test = f1_score(y_test, y_test_pred, average='weighted')
    cm_test = confusion_matrix(y_test, y_test_pred)
    roc_auc_test = roc_auc_score(y_test, model.predict_proba(X_test), multi_class='ovr')

    return accuracy_test, precision_test, recall_test, f1_test, cm_test, roc_auc_test

# Dictionary to store the results for each model on the test data
test_results = {}

for model, name in zip(models, model_names):
    print(f"Training and evaluating {name} on the test data...")
    (accuracy_test, precision_test, recall_test, f1_test, cm_test, roc_auc_test) = evaluate_model_test(model, X_train_pca, y_train, X_test_pca, y_test_data)

    test_results[name] = {
        'accuracy_test': accuracy_test,
        'precision_test': precision_test,
        'recall_test': recall_test,
        'f1_test': f1_test,
        'cm_test': cm_test,
        'roc_auc_test': roc_auc_test,
    }


print("sachin")

# Print results for test data
for name, metrics in test_results.items():
    print(f"Metrics for {name} on the test data:")
    print("Test Accuracy:", metrics['accuracy_test'])
    print("Test Precision:", metrics['precision_test'])
    print("Test Recall:", metrics['recall_test'])
    print("Test F1-Score:", metrics['f1_test'])
    print("Test Confusion Matrix:\n", metrics['cm_test'])
    print("Test ROC-AUC:", metrics['roc_auc_test'])
    print("\n")


component_sizes = [15, 20, 25, 30]

for n_components in component_sizes:
    # Initialize PCA with the current component size
    pca = PCA(n_components=n_components)

    # Fit and transform the training data
    X_train_pca = pca.fit_transform(X_train_data)

    # Transform the test data using the same PCA object
    X_test_pca = pca.transform(X_test_data)

    print(f"Metrics for {n_components} components:")
    for model, name in zip(models, model_names):
        accuracy, precision, recall, f1, cm, roc_auc = evaluate_model(model, X_train_pca, y_train_data, X_test_pca, y_test_data)
        print(f"Metrics for {name}:")
        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1-Score:", f1)
        print("Confusion Matrix:\n", cm)
        print("ROC-AUC:", roc_auc)
        print("\n")
