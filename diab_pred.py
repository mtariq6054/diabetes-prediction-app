import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r'C:\Users\12\Documents\diabetes_Prediction Project\diabetes_df new.csv')
df.drop(['gender','smoking_history'], inplace=True, axis=1)

# Split data to be used in the models
# Create matrix of features
X = df.drop('diabetes', axis = 1) # grabs everything else but 'Survived'
# Create target variable
y = df['diabetes'] # y is the column we're trying to predict

import pandas as pd
from sklearn.ensemble import RandomForestClassifier



# Train Random Forest model
rf_model = RandomForestClassifier()
rf_model.fit(X, y)

# Extract feature importance scores
importance_scores = rf_model.feature_importances_

# Create a DataFrame to store feature names and their importance scores
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importance_scores})

# Sort features by importance scores in descending order
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Print feature importance scores
print(feature_importance_df)

Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
print(IQR)
lower_bound = Q1 -1.5 *IQR
upper_bound = Q3 +1.5 *IQR
print("lower_bound(not an outlier)", lower_bound,"lower_bound(not an outlier)",upper_bound)
outliers= (df < lower_bound) | (df > upper_bound)
print("Outliers are: ", outliers)



# Winsorization method to remove outliers
df = pd.DataFrame(df)

# Customize lower and upper limit percentile
lower_limit = 5   #e.g. 5th percentile
upper_limit = 95 #e.g. 95th percentile

#apply winsorization to specified columns

columns_to_winsorize = ['age','bmi',	'HbA1c_level',	'blood_glucose_level']
for column in columns_to_winsorize:
    lower_bound = np.percentile(df[column], lower_limit)
    upper_bound = np.percentile(df[column], upper_limit)
    df[column] = np.where(df[column] < lower_bound , lower_bound , df[column])
    df[column] = np.where(df[column] > upper_bound , upper_bound , df[column])

    print(df)


    # let's plot a boxplot after handling outliers
plt.figure(figsize=(10,8))
sns.boxplot(data=df)
plt.title("Boxplot after handling outliers")
plt.xticks(rotation=45)
plt.show()

# Use of standard scaler to scale data in a certain type of numerical values
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
pre_process = preprocessing.StandardScaler()
X_transform = pre_process.fit_transform(X)

# Smote is a method or technique that is used to add random samples in minoriy class 
# RandomUnderSampler is used to randomly remove majoriy class and both create a balance between data .
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
# we use pipline for using these 2 methods combinely 
model = Pipeline([
    ('over', SMOTE()),
    ('under', RandomUnderSampler())
])

X_resampled, y_resampled = model.fit_resample(X,y)

from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X_transform,y, test_size=.20, random_state=42 )
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=.20, random_state=42, stratify=y_resampled )

from sklearn.ensemble import RandomForestClassifier
Rfc = RandomForestClassifier()
Rfc.fit(X_resampled, y_resampled)

# predict
from sklearn.metrics import accuracy_score
y_pred_Rfc = Rfc.predict(X_test)
# accuracy
Rfc_accuracy = np.round(accuracy_score(y_test, y_pred_Rfc ) *100,2)
print(f"RandomForestClassifier: {Rfc_accuracy }")


from sklearn.metrics import precision_score
y_pred = Rfc.predict(X_test)
y_binary_prediction = np.round(y_pred)
test_accuracy = precision_score(y_test, y_binary_prediction)
print("Precision score is: ", test_accuracy)

from sklearn.metrics import recall_score
y_pred = Rfc.predict(X_test)
y_binary_prediction = np.round(y_pred)
test_accuracy = recall_score(y_test, y_binary_prediction)
print("Recall score is: ", test_accuracy)

from sklearn.metrics import confusion_matrix
y_pred = Rfc.predict(X_test)
y_binary_prediction = np.round(y_pred)
test_accuracy = confusion_matrix(y_test, y_binary_prediction)
print("Confusion_matrix score is: ", test_accuracy)

plt.figure(figsize=(5,3))
sns.heatmap(test_accuracy , cmap= 'plasma', annot=True)
plt.show()


# *****************************Save Model*******************
# *************************************************************

import joblib
# Save the model to a file
joblib.dump(Rfc, 'random_forest_model.pkl')