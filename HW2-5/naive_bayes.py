#-------------------------------------------------------------------------
# AUTHOR: Miro Abdalian
# FILENAME: title of the source file
# SPECIFICATION: description of the program
# FOR: CS 4210- Assignment #2
# TIME SPENT: 1 hour
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import csv
#reading the training data in a csv file
dbTraining = []
X = []
Y = []

#reading the data in a csv file
with open('./HW2-5/weather_training.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0: #skipping the header
            dbTraining.append (row)
        
#transform the original training features to numbers and add them to the 4D array X.
#For instance Sunny = 1, Overcast = 2, Rain = 3, so X = [[3, 1, 1, 2], [1, 3, 2, 2], ...]]
#--> add your Python code here
Outlook = {
    'Sunny': 1,
    'Overcast': 2,
    'Rain': 3,  
}
Temperature = {
    'Cool': 1,
    'Mild': 2,
    'Hot': 3,
}
Humidity = {
    'Normal': 1,
    'High': 2
}
Wind = {
    'Weak': 1,
    'Strong': 2
}

for data in dbTraining:
    X.append([Outlook[data[1]], Temperature[data[2]], Humidity[data[3]], Wind[data[4]]])

#transform the original training classes to numbers and add them to the vector Y.
#For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
#--> add your Python code here

PlayTennis = {
    'Yes': 1,
    'No': 2
}
for data in dbTraining:
    Y.append(PlayTennis[data[5]])
    
#fitting the naive bayes to the data
clf = GaussianNB()
clf.fit(X, Y)

#reading the test data in a csv file
dbTesting = []
#reading the data in a csv file
with open('./HW2-5/weather_test.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0:
            dbTesting.append (row)
        else: #printing the header os the solution
            print("Day".ljust(15) + "Outlook".ljust(15) + "Temperature".ljust(15) +\
                "Humidity".ljust(15) + "Wind".ljust(15) + "PlayTennis".ljust(15) + \
                "Confidence".ljust(15))

prediction = []      
for row in dbTesting:
    data = [Outlook[row[1]], Temperature[row[2]], Humidity[row[3]], Wind[row[4]]]
    tempPredication = clf.predict_proba([data])[0] #making probabilistic predictions
    if tempPredication[0] > tempPredication[1]:
        prediction.append(['Yes', tempPredication[0]])
    else:
        prediction.append(['No', tempPredication[1]])

for i in range(len(dbTesting)):
    if round(prediction[i][1], 2) >= 0.75: #classification confidence
        print(str(dbTesting[i][0]).ljust(15) + str(dbTesting[i][1]).ljust(15) + \
            str(dbTesting[i][2]).ljust(15) + str(dbTesting[i][3]).ljust(15) + \
            str(dbTesting[i][4]).ljust(15) + str(prediction[i][0]).ljust(15) + \
            str(round(prediction[i][1], 2)).ljust(15))
    