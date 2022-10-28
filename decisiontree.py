import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
import matplotlib.pyplot as plt

datapath = r'.\BRCA\BRCA.xlsx'
print(datapath)
df = pd.read_excel(datapath)

x = df.drop(columns='TCGAid')
x = x.drop(columns='type')
y = df['type']

model = DecisionTreeClassifier(max_depth=6,random_state=200)
model.fit(x, y)

features = x.columns
importances = model.feature_importances_

importances_df = pd.DataFrame([features,importances],index = ['ID','SCORE']).T
print(importances_df)
outpath = r'.\BRCA\BRCA_importances.csv'
importances_df.to_csv(outpath,encoding='utf_8_sig',index=False)