import pandas as pd
import arff
import pickle

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# -----------------------------
# 1. Load ARFF dataset
# -----------------------------
with open("dataset/Autism-Child-Data.arff", "r") as f:
    dataset = arff.load(f)

df = pd.DataFrame(dataset["data"], columns=[a[0] for a in dataset["attributes"]])

# -----------------------------
# 2. Preprocessing
# -----------------------------
df.replace("?", pd.NA, inplace=True)
df.dropna(inplace=True)

df["gender"] = df["gender"].map({"m": 1, "f": 0})
df["jundice"] = df["jundice"].map({"yes": 1, "no": 0})
df["austim"] = df["austim"].map({"yes": 1, "no": 0})
df["Class/ASD"] = df["Class/ASD"].map({"YES": 1, "NO": 0})

features = [f"A{i}_Score" for i in range(1, 11)] + ["age", "gender", "jundice", "austim"]
X = df[features]
y = df["Class/ASD"]

# -----------------------------
# 3. Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# 4. Optimized Logistic Regression
# -----------------------------
lr_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(max_iter=1000))
])

lr_params = {
    "lr__C": [0.1, 1, 10],
    "lr__solver": ["liblinear", "lbfgs"]
}

lr_grid = GridSearchCV(lr_pipeline, lr_params, cv=5)
lr_grid.fit(X_train, y_train)
best_lr = lr_grid.best_estimator_

# -----------------------------
# 5. Neural Network (MLP)
# -----------------------------
mlp = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp", MLPClassifier(
        hidden_layer_sizes=(16, 8),
        activation="relu",
        max_iter=1000,
        random_state=42
    ))
])

mlp.fit(X_train, y_train)

# -----------------------------
# 6. Evaluation
# -----------------------------
lr_acc = accuracy_score(y_test, best_lr.predict(X_test))
mlp_acc = accuracy_score(y_test, mlp.predict(X_test))

print("Logistic Regression Accuracy:", lr_acc)
print("Neural Network Accuracy:", mlp_acc)

# -----------------------------
# 7. Save BOTH models (AI Ensemble)
# -----------------------------
with open("model/autism_model.pkl", "wb") as f:
    pickle.dump((best_lr, mlp), f)

print("✅ Final AI model saved successfully")
