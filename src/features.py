import pandas as pd

# Extract titles from the name of the passengers
def extract_title(name):
    return name.split(",")[1].split(".")[0].strip()

# Create features needed to train the model
def create_features(df):
    df = df.copy()

    df["Title"] = df["Name"].apply(extract_title)

    # Replace uncommon titles with one common name 'Rare'
    rare_titles = ['Dr', 'Rev', 'Major', 'Mlle', 'Col', 'Don', 'Mme', 'Ms', 'Lady', 'Sir', 'Capt', 'Countess', 'Jonkheer', 'Dona']

    df["Title"] = df["Title"].replace(rare_titles, "Rare")
 
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1       # size of the family
    df["IsAlone"] = (df["FamilySize"]==1).astype(int)       # if someone is alone or with family
    df["CabinKnown"] = df["Cabin"].notnull().astype(int)    # whether the cabin is known

    return df
