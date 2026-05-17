"""tests/test_preprocess.py"""
import sys, os, pytest
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocess import extract_title, clean_and_engineer

def test_mr():    assert extract_title("Braund, Mr. Owen Harris") == 1
def test_miss():  assert extract_title("Heikkinen, Miss. Laina") == 2
def test_mrs():   assert extract_title("Futrelle, Mrs. Jacques") == 3
def test_master():assert extract_title("Palsson, Master. Gosta") == 4
def test_rare():  assert extract_title("Artagaveytia, Dr. Ramon") == 5

@pytest.fixture
def raw_df():
    return pd.DataFrame({
        "Survived":[1,0,1],"Pclass":[1,3,2],
        "Name":["A, Mrs. B","C, Mr. D","E, Miss. F"],
        "Sex":["female","male","female"],"Age":[29.0,None,22.0],
        "SibSp":[0,1,0],"Parch":[0,0,1],
        "Ticket":["x","y","z"],"Fare":[53.1,7.25,None],
        "Cabin":[None,None,None],"Embarked":["S","S",None],
    })

def test_no_nulls(raw_df):
    raw_df["Title"] = raw_df["Name"].apply(extract_title)
    assert clean_and_engineer(raw_df).isnull().sum().sum() == 0

def test_family_size(raw_df):
    raw_df["Title"] = raw_df["Name"].apply(extract_title)
    c = clean_and_engineer(raw_df)
    assert c["FamilySize"].iloc[0] == 1
    assert c["FamilySize"].iloc[1] == 2

def test_is_alone(raw_df):
    raw_df["Title"] = raw_df["Name"].apply(extract_title)
    c = clean_and_engineer(raw_df)
    assert c["IsAlone"].iloc[0] == 1
    assert c["IsAlone"].iloc[1] == 0

def test_sex_numeric(raw_df):
    raw_df["Title"] = raw_df["Name"].apply(extract_title)
    c = clean_and_engineer(raw_df)
    assert set(c["Sex"].unique()).issubset({0,1})
