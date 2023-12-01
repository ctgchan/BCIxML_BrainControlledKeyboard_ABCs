
# Load libraries
import pandas as pd
# from pandas import read_csv
# from pandas.plotting import scatter_matrix
# from matplotlib import pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import StratifiedKFold
# from sklearn.metrics import classification_report
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import accuracy_score
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from sklearn.naive_bayes import GaussianNB
# from sklearn.svm import SVC
# import seaborn as sns

...
# Load dataset
data = pd.read_csv("ABCML_Data.csv")
data['Image'] = data['Image'].astype('category')
image_data = data.groupby("Image").mean().round(2)
print(image_data)


from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import pandas as pd

data = pd.read_csv("ABCML_Data.csv")
data['Image'] = data['Image'].astype('category')
image_data = data.groupby("Image").mean().round(2)
print(image_data)


def train_svmm_model():
    data = pd.read_csv("ABCML_Data.csv") 
    

    # Assuming 'X' contains your features (Beta and Alpha frequencies) and 'y' contains corresponding labels
    X = data[['Beta', 'Alpha']].values
    y = data['Image'].values

    # Splitting the data into training and validation sets (adjust test_size and random_state as needed)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalizing the features (scaling between 0 and 1)
    scaler = MinMaxScaler()
    X_train_normalized = scaler.fit_transform(X_train)
    X_val_normalized = scaler.transform(X_val)


    # Initializing SVM classifier (you can experiment with different kernels and parameters)
    svm_classifier = SVC(kernel='rbf', C=1.0, gamma='scale')  # Example with RBF kernel

    # Training the SVM model
    svm_classifier.fit(X_train_normalized, y_train)

    # Making predictions on the validation set
    predictions = svm_classifier.predict(X_val_normalized)

    # Calculating accuracy
    accuracy = accuracy_score(y_val, predictions)

    return svm_classifier, scaler

# Remember to replace X and y with your actual feature and label data. 
# Also, feel free to adjust the SVM parameters (kernel, C, etc.) and explore different kernels (e.g., 'linear', 'poly', 'rbf') to find the best model for your data.

# Assuming 'X_new' contains new samples of Beta and Alpha frequencies for prediction

def predict_with_svmm_model(svm_classifier, scaler, X_new):
#trial testing data
    # Normalize the new data using the same scaler used for training/validation data
    
    X_new_normalized = scaler.transform(X_new)

    # Making predictions on the new data
    new_predictions = svm_classifier.predict(X_new_normalized)
    return new_predictions
    # Displaying the predicted labels
    # print("Predicted Labels for New Data:")
    # for prediction in new_predictions:
    #     print(prediction)



