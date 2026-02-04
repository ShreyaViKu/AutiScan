import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

df = pd.read_csv("dataset/autism_child_data.csv")

df = df[['A1_Score','A2_Score','A3_Score','A4_Score','A5_Score',
         'A6_Score','A7_Score','A8_Score','A9_Score','A10_Score',
         'age','gender','jundice','austim','Class/ASD']]

df['gender'] = df['gender'].map({'m':1,'f':0})
df['jundice'] = df['jundice'].map({'yes':1,'no':0})
df['austim'] = df['austim'].map({'yes':1,'no':0})
df['Class/ASD'] = df['Class/ASD'].map({'YES':1,'NO':0})

df.dropna(inplace=True)

X = df.drop('Class/ASD', axis=1)
y = df['Class/ASD']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

pickle.dump(model, open("model/autism_model.pkl", "wb"))
print("Model trained successfully")
