"""
src/preprocess.py  —  Data cleaning & feature engineering
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib, os

FEATURE_COLUMNS = [
    "Pclass","Sex","Age","SibSp","Parch",
    "Fare","FamilySize","IsAlone","Title",
    "Embarked_Q","Embarked_S",
]
TARGET_COLUMN = "Survived"

def extract_title(name: str) -> int:
    title_map = {"Mr":1,"Miss":2,"Mrs":3,"Master":4,"Rare":5}
    rare = {"Dr","Rev","Col","Major","Mlle","Countess","Ms","Lady",
            "Jonkheer","Don","Dona","Capt","Sir"}
    title = name.split(",")[1].split(".")[0].strip()
    return title_map["Rare"] if title in rare else title_map.get(title, 5)

def clean_and_engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.drop(columns=["Ticket","Cabin","PassengerId","Name"], errors="ignore", inplace=True)
    df["Age"]      = df["Age"].fillna(df["Age"].median())
    df["Fare"]     = df["Fare"].fillna(df["Fare"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"]    = (df["FamilySize"] == 1).astype(int)
    df["Sex"]        = df["Sex"].map({"male":0,"female":1})
    embarked_dummies = pd.get_dummies(df["Embarked"], prefix="Embarked", drop_first=True)
    df = pd.concat([df, embarked_dummies], axis=1)
    df.drop(columns=["Embarked"], inplace=True)
    for col in ["Embarked_Q","Embarked_S"]:
        if col not in df.columns:
            df[col] = 0
    return df

def preprocess_for_training(df: pd.DataFrame):
    df = df.copy()
    df["Title"] = df["Name"].apply(extract_title)
    df = clean_and_engineer(df)
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()
    scaler = StandardScaler()
    X[["Age","Fare"]] = scaler.fit_transform(X[["Age","Fare"]])
    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(scaler, "artifacts/scaler.pkl")
    return X, y

def preprocess_for_inference(data: dict) -> pd.DataFrame:
    df = pd.DataFrame([{
        "Survived":0,"PassengerId":0,"Name":"Unknown, Mr. X",
        "Ticket":"000","Cabin":None,
        "Pclass":data["pclass"],"Sex":data["sex"],"Age":data["age"],
        "SibSp":data["sibsp"],"Parch":data["parch"],
        "Fare":data["fare"],"Embarked":data.get("embarked","S"),
    }])
    df["Title"] = df["Name"].apply(extract_title)
    df = clean_and_engineer(df)
    df = df[FEATURE_COLUMNS].copy()
    scaler = joblib.load("artifacts/scaler.pkl")
    df[["Age","Fare"]] = scaler.transform(df[["Age","Fare"]])
    return df
