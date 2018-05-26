# Play with the data set

import pandas as pd
import numpy as np

import os

test_labels = pd.read_csv('~/.kaggle/competitions/birds/labels.csv')
img_names = test_labels.iloc[:,0]
img_classes = test_labels.iloc[:,1]

print(img_names[0])
print(img_classes[0])
print(img_names[1])
print(img_classes[1])

classes = []
with open(os.path.expanduser('~/.kaggle/competitions/birds/names.txt')) as f:
	for line in f:
		classes.append(line.rstrip())
print(classes[475])