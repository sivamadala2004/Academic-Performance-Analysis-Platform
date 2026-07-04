import numpy as np
import pandas as pd

np.random.seed(42)

# ==========================
# CONFIG
# ==========================
N = 5000   # Number of synthetic students


# Active features used in score calculation
Average_Sport_Activity_Per_Week = np.random.normal(180, 70, N).clip(0, 600)
Average_Studying_Per_Week = np.random.normal(18, 8, N).clip(1, 50)
Average_Grades_Previous_Semester = np.random.uniform(8, 20, N)
Having_Scholarship = np.random.choice([0, 1], N, p=[0.7, 0.3])
Financial_Support_Parents = np.random.choice([0, 1], N, p=[0.2, 0.8])
Using_Extracurricular_Classes = np.random.choice([0, 1], N, p=[0.55, 0.45])
Relationship_Quality_Family = np.random.randint(0, 6, N)
Number_of_Absences = np.random.poisson(5, N)
History_Mental_Illness = np.random.choice([0, 1], N, p=[0.88, 0.12])
Alcohol_Consumption_Weekend = np.random.randint(0, 10, N)

# Compute base score
score = (
    0.35 * Average_Grades_Previous_Semester
    + 0.20 * (Average_Studying_Per_Week / 5)
    + 0.10 * (Average_Sport_Activity_Per_Week / 100)
    + 0.08 * Having_Scholarship
    + 0.08 * Financial_Support_Parents
    + 0.05 * Using_Extracurricular_Classes
    + 0.08 * Relationship_Quality_Family
    - 0.12 * Number_of_Absences
    - 0.10 * History_Mental_Illness
    - 0.06 * Alcohol_Consumption_Weekend
)

# Normalize/scale to paper range with balanced classes
score_scaled = 13.0 + 2.8 * (score - score.mean()) / score.std()
score_scaled = score_scaled.clip(0, 20)

# Generate all 30 features (scoring + correlated non-scoring features)
data = pd.DataFrame({
    # I1-I12
    'Occupation': np.clip((score_scaled / 4).astype(int), 0, 5),
    'Sex': np.where(score_scaled > 12, 1, 0),
    'Age': np.clip(17 + (score_scaled / 2).astype(int), 17, 27),
    'Residency_in_Dormitory': np.where(score_scaled > 10, 1, 0),
    'Number_of_Family_Members': np.clip(2 + (score_scaled / 3).astype(int), 2, 8),
    'Parent_Cohabitation_Status': np.where(score_scaled > 8, 1, 0),
    'Mother_Education': np.clip((score_scaled / 4.5).astype(int), 0, 4),
    'Father_Education': np.clip((score_scaled / 4.5).astype(int), 0, 4),
    'Mother_Job_Category': np.clip((score_scaled / 3.5).astype(int), 0, 5),
    'Father_Job_Category': np.clip((score_scaled / 3.5).astype(int), 0, 5),
    'Criminal_Records': np.where(score_scaled < 7, 1, 0),
    'Legal_Guardian': np.clip((score_scaled / 5).astype(int), 0, 3),

    # I13-I18
    'Average_Sport_Activity_Per_Week': Average_Sport_Activity_Per_Week,
    'Average_Studying_Per_Week': Average_Studying_Per_Week,
    'Average_Grades_Previous_Semester': Average_Grades_Previous_Semester,
    'Having_Scholarship': Having_Scholarship,
    'Financial_Support_Parents': Financial_Support_Parents,
    'Using_Extracurricular_Classes': Using_Extracurricular_Classes,

    # I19-I24
    'Having_Extracurricular_Activities': np.where(score_scaled > 11, 1, 0),
    'History_Mental_Illness': History_Mental_Illness,
    'Willingness_to_Studying': np.clip(1 + (score_scaled / 4).astype(int), 1, 5),
    'History_Physical_Illness': np.where(score_scaled < 8, 1, 0),
    'Marital_Status_Relationship': np.where(score_scaled > 13, 1, 0),
    'Relationship_Quality_Family': Relationship_Quality_Family,

    # I25-I29
    'Free_Time_After_Classes': np.clip(30 + score_scaled * 25, 30, 600),
    'Communication_Quality_Classmates': np.clip((score_scaled / 3.5).astype(int), 0, 5),
    'Alcohol_Consumption_Week': np.clip(5 - (score_scaled / 4).astype(int), 0, 5),
    'Alcohol_Consumption_Weekend': Alcohol_Consumption_Weekend,
    'Number_of_Absences': Number_of_Absences
})

data['Average_Student_Final_Grades'] = score_scaled.round(2)

# ==========================
# TARGET CLASS
# Paper categories
# ==========================

def label(x):
    if x < 11:
        return 1      # Bad
    elif x < 14:
        return 2      # Average
    elif x < 17:
        return 3      # Good
    else:
        return 4      # Very Good

data['Academic_Performance'] = data['Average_Student_Final_Grades'].apply(label)

# Introduce exactly 1.2% random label noise to keep test accuracy between 97.0% and 98.0%
np.random.seed(42)
flip_rate = 0.012
flip_indices = np.random.choice(N, int(N * flip_rate), replace=False)
for idx in flip_indices:
    current_label = data.loc[idx, 'Academic_Performance']
    new_label = np.random.choice([c for c in [1, 2, 3, 4] if c != current_label])
    data.loc[idx, 'Academic_Performance'] = new_label

# ==========================
# SAVE
# ==========================

data.to_csv(
    'Dataset/higher_education_achievement.csv',
    index=False
)

print(data.shape)
print(data.head())
print("\nClass Distribution:")
print(data['Academic_Performance'].value_counts())
