import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import time as t
import sklearn.utils as u
import sklearn.preprocessing as pp
import sklearn.tree as tr
import sklearn.ensemble as es
import sklearn.metrics as m
import sklearn.linear_model as lm
import sklearn.neural_network as nn
import numpy as np
import warnings as w

# Suppress warnings
w.filterwarnings('ignore')

# Load and preprocess data
data = pd.read_csv("/AI-Data.csv")

# Define grade ID mapping
gradeID_dict = {
    "G-01": 1, "G-02": 2, "G-03": 3, "G-04": 4, "G-05": 5,
    "G-06": 6, "G-07": 7, "G-08": 8, "G-09": 9, "G-10": 10,
    "G-11": 11, "G-12": 12
}

# Drop columns not used in model
columns_to_drop = [
    "gender", "StageID", "GradeID", "NationalITy", "PlaceofBirth",
    "SectionID", "Topic", "Semester", "Relation", "ParentschoolSatisfaction",
    "ParentAnsweringSurvey", "AnnouncementsView"
]
data = data.drop(columns=columns_to_drop)

# Replace GradeID with numerical values
data = data.replace({"GradeID": gradeID_dict})

# Encode categorical columns
for column in data.columns:
    if data[column].dtype == object:
        le = pp.LabelEncoder()
        data[column] = le.fit_transform(data[column])

# Split data into training and test sets
ind = int(len(data) * 0.70)
feats = data.values[:, :-1]
lbls = data.values[:, -1]
feats_Train = feats[:ind]
feats_Test = feats[ind:]
lbls_Train = lbls[:ind]
lbls_Test = lbls[ind:]

# Define and evaluate models
models = {
    "Decision Tree": tr.DecisionTreeClassifier(),
    "Random Forest": es.RandomForestClassifier(),
    "Perceptron": lm.Perceptron(),
    "Logistic Regression": lm.LogisticRegression(),
    "MLP Classifier": nn.MLPClassifier(activation="logistic")
}

for name, model in models.items():
    model.fit(feats_Train, lbls_Train)
    lbls_pred = model.predict(feats_Test)
    acc = m.accuracy_score(lbls_Test, lbls_pred)
    print(f"\n{name} Accuracy: {round(acc, 3)}")
    print(m.classification_report(lbls_Test, lbls_pred))
    t.sleep(1)

# User input prediction
choice = input("Do you want to test specific input (y or n): ")
if choice.lower() == "y":
    # Collect user input
    inputs = {
        'Gender': {"M": 1, "F": 0},
        'Nationality': input("Enter Nationality: "),
        'Place of Birth': input("Place of Birth: "),
        'Grade ID': gradeID_dict.get(input("Grade ID as (G-<grade>): "), -1),
        'Section': input("Enter Section: "),
        'Topic': input("Enter Topic: "),
        'Semester': {"F": 0, "S": 1}.get(input("Enter Semester (F or S): ").upper(), -1),
        'Relation': {"Father": 0, "Mum": 1}.get(input("Enter Relation (Father or Mum): "), -1),
        'Raised Hands': int(input("Enter raised hands: ")),
        'Visited Resources': int(input("Enter Visited Resources: ")),
        'Announcements Viewed': int(input("Enter announcements viewed: ")),
        'Discussions': int(input("Enter no. of Discussions: ")),
        'Parent Answered Survey': {"Y": 1, "N": 0}.get(input("Enter Parent Answered Survey (Y or N): ").upper(), -1),
        'Parent School Satisfaction': {"Good": 1, "Bad": 0}.get(input("Enter Parent School Satisfaction (Good or Bad): "), -1),
        'Absent Days': {"Under-7": 1, "Above-7": 0}.get(input("Enter No. of Abscenes(Under-7 or Above-7): "), -1)
    }
    
    arr = np.array([
        inputs['Raised Hands'], inputs['Visited Resources'], 
        inputs['Discussions'], inputs['Absent Days']
    ])
    
    for name, model in models.items():
        pred = model.predict(arr.reshape(1, -1))[0]
        pred_class = ["H", "M", "L"][pred]  # Assuming classification labels are 0, 1, 2
        print(f"Using {name}: {pred_class}")
    
    print("\nExiting...")
    t.sleep(1)
else:
    print("Exiting..")
    t.sleep(1)
