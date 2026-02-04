import arff
import pandas as pd

with open("dataset/Autism-Child-Data.arff", "r") as f:
    data = arff.load(f)

df = pd.DataFrame(
    data['data'],
    columns=[attr[0] for attr in data['attributes']]
)

df.to_csv("dataset/autism_child_data.csv", index=False)
print("Dataset converted successfully")
