#-------------------------------------------------------------------------
# AUTHOR: Miro Abdalian
# FILENAME: title of the source file
# SPECIFICATION: description of the program
# FOR: CS 4210- Assignment #2
# TIME SPENT: 30min
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#importing some Python libraries
from sklearn.neighbors import KNeighborsClassifier
import csv

db = []

#reading the data in a csv file
with open('./HW2-3/binary_points.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         db.append (row)
    
groundTruth = {
    "-": 0,
    "+": 1,
}


wrongPredict = 0
for i, instance in enumerate(db):
    X = []
    Y = []
    testSample = [int(instance[0]), int(instance[1])]
    for j, data in enumerate(db):
        if data != instance:
            X.append([int(data[0]), int(data[1])])
            Y.append(groundTruth[data[2]])
        
    clf = KNeighborsClassifier(n_neighbors=1, p=2)
    clf = clf.fit(X, Y)    
    class_predicted = clf.predict([testSample])[0]
    if class_predicted != groundTruth[instance[2]] :
        wrongPredict += 1

print(f"error rate = {wrongPredict/len(db):.0%}")   






