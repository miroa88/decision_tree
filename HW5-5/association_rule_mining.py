# -------------------------------------------------------------------------
# AUTHOR: Miro Abdalian
# FILENAME: title of the source file
# SPECIFICATION: description of the program
# FOR: CS 4210- Assignment #5
# TIME SPENT: 30 min
# -----------------------------------------------------------*/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules

# read the dataset using pandas
df = pd.read_csv('./HW5-5/retail_dataset.csv', sep=',')

# find the unique items all over the data an store them in the set below
itemset = set()
for i in range(0, len(df.columns)):
    items = (df[str(i)].unique())
    itemset = itemset.union(set(items))

# remove nan (empty) values by using:
itemset.remove(np.nan)

# Apriori module requires a dataframe that has either 0 and 1 or True and False as data.
# Example:

# Bread Wine Eggs
# 1     0    1
# 0     1    1
# 1     1    1

# To do that, create a dictionary (labels) for each transaction,
# store the corresponding values for each item (e.g., {'Bread': 0, 'Milk': 1}) in that transaction,
# and when is completed, append the dictionary to the list encoded_vals below
# (this is done for each transaction)

encoded_vals = []
for index, row in df.iterrows():
    labels = {'Bread': 0, 'Wine': 0, 'Eggs': 0, 'Meat': 0,
              'Cheese': 0, 'Pencil': 0, 'Diaper': 0, 'Milk': 0, 'Bagel': 0}
    for any in row:
        if any == any: # to make sure its not nan
            labels[any] = 1
    encoded_vals.append(labels)

# adding the populated list with multiple dictionaries to a data frame
ohe_df = pd.DataFrame(encoded_vals)

# calling the apriori algorithm informing some parameters
freq_items = apriori(ohe_df, min_support=0.2, use_colnames=True, verbose=1)
rules = association_rules(freq_items, metric="confidence", min_threshold=0.6)

# iterate the rules data frame and print the apriori algorithm results by using the following format:
for index, row in rules.iterrows():
    print("-------------------- " + str(index) + " ----------------------")
    print(", ".join(list(row['antecedents'])) +
          " -> " + ", ".join(list(row['consequents'])))
    print("Support:", row['support'])
    print("Confidence:", row['confidence'])
    suportCount = 0
    for label in encoded_vals:
        isAppeared = True
        for any in list(row['consequents']):
            if label[any] == 0:
                isAppeared = False
                break
        if isAppeared:
            suportCount += 1
    prior = suportCount/len(encoded_vals)       
    print("Prior:", prior)
    print("Gain in Confidence:", str(100*(row['confidence']-prior)/prior))

# To calculate the prior and gain in confidence, find in how many transactions the consequent
# of the rule appears (the supporCount below). Then,
# use the gain formula provided right after.
# prior = suportCount/len(encoded_vals) -> encoded_vals is the number of transactions
#print("Gain in Confidence: " + str(100*(rule_confidence-prior)/prior))

# Finally, plot support x confidence

plt.scatter(rules['support'], rules['confidence'], alpha=0.5)
plt.xlabel('support')
plt.ylabel('confidence')
plt.title('Support vs Confidence')
plt.show()
