# -------------------------------------------------------------------------
# AUTHOR: Miro Abdalian
# FILENAME: title of the source file
# SPECIFICATION: description of the program
# FOR: CS 4210- Assignment #3
# TIME SPENT: 30 min
# -----------------------------------------------------------*/

# IMPORTANT NOTE: YOU HAVE TO WORK WITH THE PYTHON LIBRARIES numpy AND pandas to complete this code.

# importing some Python libraries
from sklearn import svm
import numpy as np
import pandas as pd

# defining the hyperparameter values
C = [1, 5, 10, 100]
degree = [1, 2, 3]
kernel = ["linear", "poly", "rbf"]
decision_function_shape = ["ovo", "ovr"]

# reading the training data by using Pandas library
df = pd.read_csv('./HW3-7/optdigits.tra', sep=',', header=None)

# getting the first 64 fields to create the feature training data and convert them to NumPy array
X_training = np.array(df.values)[:, :64]
# getting the last field to create the class training data and convert them to NumPy array
y_training = np.array(df.values)[:, -1]

# reading the training data by using Pandas library
df = pd.read_csv('./HW3-7/optdigits.tes', sep=',', header=None)

# getting the first 64 fields to create the feature testing data and convert them to NumPy array
X_test = np.array(df.values)[:, :64]
# getting the last field to create the class testing data and convert them to NumPy array
y_test = np.array(df.values)[:, -1]

# created 4 nested for loops that will iterate through the values of c, degree, kernel, and decision_function_shape
highest_SVM_accuracy = 0
for c in C:  # iterates over c
    for dg in degree:  # iterates over degree
        for knl in kernel:  # iterates kernel
            for shape in decision_function_shape:  # iterates over decision_function_shape
                # Create an SVM classifier that will test all combinations of c, degree, kernel, and decision_function_shape.
                # For instance svm.SVC(c=1, degree=1, kernel="linear", decision_function_shape = "ovo")
                clf = svm.SVC(C=c, degree=dg, kernel=knl, decision_function_shape=shape)
                # Fit SVM to the training data
                clf.fit(X_training, y_training)

                # make the SVM prediction for each test sample and start computing its accuracy
                # hint: to iterate over two collections simultaneously, use zip()
                # Example. for (x_testSample, y_testSample) in zip(X_test, y_test):
                # to make a prediction do: clf.predict([x_testSample])
                counter = 0
                for (x_testSample, y_testSample) in zip(X_test, y_test):
                    class_predicted = clf.predict([x_testSample])
                    if class_predicted == y_testSample:
                        counter += 1

                accuracy = counter/len(y_test)

                # check if the calculated accuracy is higher than the previously one calculated. If so, update the highest accuracy and print it together
                # with the SVM hyperparameters. Example: "Highest SVM accuracy so far: 0.92, Parameters: a=1, degree=2, kernel= poly, decision_function_shape = 'ovo'"
                if accuracy > highest_SVM_accuracy:
                    highest_SVM_accuracy = accuracy
                    print(f"Highest SVM accuracy so far: {highest_SVM_accuracy:.2%},Parameters: a={c}, degree={dg}, kernel={knl}, decision_function_shape={shape}")
