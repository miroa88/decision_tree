#-------------------------------------------------------------------------
# AUTHOR: your name
# FILENAME: title of the source file
# SPECIFICATION: description of the program
# FOR: CS 4210- Assignment #1
# TIME SPENT: how long it took you to complete the assignment
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#importing some Python libraries
from sklearn import tree
import matplotlib.pyplot as plt
import csv
db = []
X = []
Y = []

#reading the data in a csv file
with open('contact_lens.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         db.append (row)
         print(row)
        
def transform(index, feature_class):
  if index == 0:
    if feature_class == 'Young':
      return 1
    elif feature_class == 'Presbyopic':
      return 2
    else:
      return 3
  elif index == 1:
    if feature_class == 'Myope':
      return 1
    else:
      return 2
  elif index == 2:
    if feature_class == 'Yes':
      return 1
    else:
      return 2 
  elif index == 3:
    if feature_class == 'Reduced':
      return 1
    else:
      return 2 
  elif index == 4:
    if feature_class == 'Yes':
      return 1
    else:
      return 2
    
for row in db:
  temp = []
  for i in range(0, 4):
    temp.append(transform(i, row[i]))
  X.append(temp)
  Y.append(transform(4, row[4]))

#fitting the decision tree to the data

clf = tree.DecisionTreeClassifier(criterion = 'entropy', )
clf = clf.fit(X, Y)

#plotting the decision tree
tree.plot_tree(clf, feature_names=['Age', 'Spectacle', 'Astigmatism', 'Tear'], class_names=['Yes','No'], filled=True, rounded=True)
plt.show()


