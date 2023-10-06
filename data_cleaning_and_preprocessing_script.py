from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor

# Read employee.csv file into a pandas dataframe
chatterbox = pd.read_csv('employees.csv', low_memory=False)

# ############################### Handling Data Inconsistency in Title, Gender, Marital_Status #######################
# Let's assume the 'Gender' is correct and that 'Title' may be wrong
chatterbox.loc[((chatterbox['Title'] == 'Ms') | (chatterbox['Title'] == 'Miss')) &
               (chatterbox['Gender'] == 'Male'), 'Title'] = 'Mr'
chatterbox.loc[(chatterbox['Title'] == 'Mr') & (chatterbox['Gender'] == 'Female') &
               (chatterbox['Marital_Status'] == 'Single'), 'Title'] = 'Miss'
chatterbox.loc[(chatterbox['Title'] == 'Mr') & (chatterbox['Gender'] == 'Female') &
               (chatterbox['Marital_Status'] == 'Married'), 'Title'] = 'Ms'
chatterbox.loc[(chatterbox['Title'] == 'Mr') & (chatterbox['Gender'] == 'Female') &
               ((chatterbox['Marital_Status'] != 'Married') |
                (chatterbox['Marital_Status'] != 'Single')), 'Title'] = 'Ms'
chatterbox.loc[(chatterbox['Title'] == 'Miss') & (chatterbox['Gender'] == 'Female') &
               (chatterbox['Marital_Status'] == 'Married'), 'Title'] = 'Ms'
######################################################################################################################

# ############################### Handling Missing Values ############################################################
# preparing a dataframe to use for building models to impute the missing values
_chatterbox = chatterbox.copy()
# As we are applying one-hot encoding for religion, designation there is no need for 'Religion_ID' and 'Designation_ID'
# Therefore we drop them
# We will not use Date_Resigned and 'Inactive_Date' as Active employees doesn't have a value for those fields
# 'Employee_Code' and 'Name' cannot be used as features for model and 'Reporting_emp_1&2' fields are mostly empty
_chatterbox = _chatterbox.drop(['Employee_Code', 'Name', 'Religion_ID', 'Designation_ID', 'Date_Resigned',
                                'Inactive_Date', 'Reporting_emp_1', 'Reporting_emp_2'], axis=1)
# Convert 'Date_Joined' filed to numerical by converting to number of days since epoch
_chatterbox['Date_Joined_Days'] = np.nan
for index, row in _chatterbox.iterrows():
    date = row['Date_Joined'].split('/')
    date = datetime(int(date[2]), int(date[0]), int(date[1]))
    _chatterbox.at[index, 'Date_Joined_Days'] = int(date.timestamp() / 86400)
# Apply one-hot encoding for 'Title', 'Gender', 'Status', 'Employment_Category', 'Employment_Type', 'Religion'
# and 'Designation'
ohe = OneHotEncoder()
feature_array = ohe.fit_transform(_chatterbox[['Title', 'Gender', 'Status', 'Employment_Category', 'Employment_Type',
                                               'Religion', 'Designation']]).toarray()
features = pd.DataFrame(feature_array, columns=ohe.get_feature_names_out())
_chatterbox = pd.concat([_chatterbox, features], axis=1)
# Then drop the fields: 'Title', 'Gender', 'Status', 'Employment_Category', 'Employment_Type', 'Religion'
# and 'Designation'
_chatterbox = _chatterbox.drop(['Title', 'Gender', 'Date_Joined',
                                'Status', 'Employment_Category', 'Employment_Type', 'Religion',
                                'Designation'], axis=1)
# create a copy of dataframe containing processed 'Date_Joined' and one-hot encoded columns
final = _chatterbox.copy()
# Label encode the 'Marital_Status' for classification
le = LabelEncoder()
original = final.copy()
mask = final.isnull()
final.Marital_Status = le.fit_transform(final.Marital_Status)
final = final.where(~mask, original)
# replace '0000' in 'Year_of_Birth' to '0' to convert to integer
final.loc[final['Year_of_Birth'] == "'0000'", 'Year_of_Birth'] = '0'
final['Year_of_Birth'] = final['Year_of_Birth'].astype('int64')
final['Marital_Status'] = final['Marital_Status'].astype('float64')
# Since the 'Year_of_Birth' has a skewed distribution, get the median to replace the value 0
median_of_Year_of_Birth = int(final['Year_of_Birth'].median())
# Split the final dataframe to Test and Training sets to build a classification model for 'Marital_Status' imputation
Train = final[final['Marital_Status'].notna()]
Train = Train[Train['Year_of_Birth'] > 0]
Train = Train.reset_index(drop=True)

Test = final[final['Marital_Status'].isna()]
Test.loc[Test['Year_of_Birth'] == 0, 'Year_of_Birth'] = median_of_Year_of_Birth
Test = Test.reset_index(drop=True)

# Let's use DecisionTreeClassifier for the model for imputing 'Marital_Status'
# After a random search with cross-validation, the following results were obtained:
# For 'Marital_Status' with using only the employee.csv for inference,
# The best decision tree params are : {'min_samples_split': 2, 'min_samples_leaf': 30, 'max_leaf_nodes': 1000,
# 'max_features': 90, 'max_depth': 20, 'criterion': 'entropy'}
# The cross-validation accuracy score is: 0.8637225548902195

# Then lets build the DecisionTreeClassifier with above hyperparameters
dtc = DecisionTreeClassifier(min_samples_split=1, min_samples_leaf=10, max_leaf_nodes=30, max_features=60,
                             max_depth=20, criterion='gini')
dtc.fit(Train.loc[:, ~Train.columns.isin(['Employee_No', 'Marital_Status'])], Train['Marital_Status'])
# Get predictions
predicted = dtc.predict(Test.loc[:, ~Test.columns.isin(['Employee_No', 'Marital_Status'])])
# Apply the predicted values to the Test dataframe
Test['Marital_Status'] = predicted
# Apply the predicted values to final dataframe
for index, row in Test.iterrows():
    series = final.loc[final['Employee_No'] == row['Employee_No'], 'Marital_Status']
    if series.size == 1:
        final.loc[final['Employee_No'] == row['Employee_No'], 'Marital_Status'] = row['Marital_Status']

# Now lets build a regression model to impute the 'Year_of_Birth'
# Make the 'Year_of_Birth' floats for regression
final['Year_of_Birth'] = final['Year_of_Birth'].astype('float64')
# Split the final dataframe to Test and Training sets to build a regression model for 'Year_of_Birth' imputation
Train = final[final['Year_of_Birth'] > 0]
Train = Train.reset_index(drop=True)

Test = final[final['Year_of_Birth'] <= 0]
Test = Test.reset_index(drop=True)

# Let's use DecisionTreeRegressor for the model for imputing 'Year_of_Birth'
# After a random search with cross-validation, the following results were obtained:
# For 'Year_of_Birth' with using only the employee.csv for inference,
# The best decision tree params are : {'min_samples_split': 2, 'min_samples_leaf': 1, 'max_leaf_nodes': 10,
# 'max_features': 90, 'max_depth': 30, 'criterion': 'squared_error'}
# The cross-validation root mean squared error: 10.17548159880632
dtc = DecisionTreeRegressor(min_samples_split=2, min_samples_leaf=2, max_leaf_nodes=None, max_features=70, max_depth=5,
                            criterion='squared_error')
dtc.fit(Train.loc[:, ~Train.columns.isin(['Employee_No', 'Year_of_Birth'])], Train['Year_of_Birth'])
# Get predictions
predicted = dtc.predict(Test.loc[:, ~Test.columns.isin(['Employee_No', 'Year_of_Birth'])])
# Apply the predicted values to the Test dataframe
Test['Year_of_Birth'] = np.round(predicted)
# Apply the predicted values to final dataframe
for index, row in Test.iterrows():
    series = final.loc[final['Employee_No'] == row['Employee_No'], 'Year_of_Birth']
    if series.size == 1:
        final.loc[final['Employee_No'] == row['Employee_No'], 'Year_of_Birth'] = row['Year_of_Birth']

# Remaking the chatterbox dataframe by applying the imputed values
# Reconvert 'Marital_Status' to an integer to apply inverse transformation of label encoding
_array = final['Marital_Status'].astype('int64')
_array = _array.to_numpy()
result = le.inverse_transform(_array)
# Apply the data in 'Marital_Status' data in final data frame to chatterbox
chatterbox['Marital_Status'] = result
# Reconvert the 'Year_of_Birth' to an integer and then an object
final['Year_of_Birth'] = final['Year_of_Birth'].astype('int64')
final['Year_of_Birth'] = final['Year_of_Birth'].astype('object')
# Apply the data in 'Year_of_Birth' data in final data frame to chatterbox
chatterbox['Year_of_Birth'] = final['Year_of_Birth']
#######################################################################################################################

# ####################### Other Data Cleaning #########################################################################
# There are no duplicated rows in the employee.csv
# There are also no duplicated Employee_No, Employee_Code in the employee.csv

# The only discrete value in the dataset is 'Year_of_Birth' and it has few to no outliers

# Since active employees doesn't have a 'Date_Resigned' or a 'Inactive_Date' they are mostly marked as '\N' while
# some other cells have '0000-00-00'
# Some of the inactive employees also have '\N' and '0000-00-00' in 'Date_Resigned' column
chatterbox.loc[chatterbox['Status'] == 'Active', 'Inactive_Date'] = '\\N'
chatterbox.loc[(chatterbox['Status'] == 'Inactive') & ((chatterbox['Date_Resigned'] == '\\N')
                                                       | (chatterbox['Date_Resigned'] == '0000-00-00'))
               & ((chatterbox['Inactive_Date'] != '\\N')
                  | (chatterbox['Inactive_Date'] != '0000-00-00')), 'Date_Resigned'] = chatterbox['Inactive_Date']
chatterbox.loc[chatterbox['Status'] == 'Active', 'Date_Resigned'] = '\\N'
# This makes the 'Date_Resigned' column an exact copy of 'Inactive_Date'
#######################################################################################################################

# Writing the output dataframe to employee_preprocess_200440C.csv file
chatterbox.to_csv('employee_preprocess_200440C.csv')
#%%
