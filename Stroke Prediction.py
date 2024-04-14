#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df=pd.read_csv(r'C:\Users\Sakib\Downloads\Bangladessh.project1\healthcare-dataset-stroke-data.csv')
df.sample(5)


# In[3]:


df.shape


# In[4]:


df.isnull().sum()


# In[5]:


df.columns


# In[6]:


df.bmi.describe()


# In[7]:


df.dropna()


# In[8]:


df.isnull().sum()


# In[9]:


df.dropna(subset='bmi',inplace=True)


# In[10]:


df.gender.unique()


# In[11]:


df['gender'].value_counts()


# In[12]:


df[df['gender']=='Other']


# In[13]:


df.drop(df[df['gender'] == 'Other'].index,inplace=True)


# In[14]:


sns.countplot(data=df, x='gender')
plt.title("Checking out Gender")
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()


# In[15]:


df.columns


# In[16]:


df.age.value_counts()


# In[17]:


sns.histplot(df['age'], bins=15, kde=True)  # Adjust the number of bins as needed
plt.title("Distribution of Age")
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()


# In[18]:


df['hypertension'].value_counts()


# In[19]:


sns.countplot(data=df, x='hypertension')
plt.title("Distribution of Hypertension")
plt.xlabel('Hypertension')
plt.ylabel('Count')
plt.show()


# In[20]:


df['heart_disease'].value_counts()


# In[21]:


sns.countplot(data=df, x='heart_disease')
plt.title("Distribution of Heart Disease")
plt.xlabel('Heart Disease')
plt.ylabel('Count')
plt.show()


# In[22]:


df.ever_married.unique()


# In[23]:


sns.countplot(data=df, x='ever_married')
plt.title("Distribution of Ever Married")
plt.xlabel('Ever Married')
plt.ylabel('Count')
plt.show()


# In[24]:


df.work_type.value_counts()


# In[25]:


sns.countplot(data=df, x='work_type')
plt.title("Distribution of Work Type")
plt.xlabel('Work Type')
plt.ylabel('Count')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability if needed
plt.show()


# In[26]:


df.Residence_type.value_counts()


# In[27]:


sns.countplot(data=df, x='Residence_type')
plt.title("Distribution of Residence Type")
plt.xlabel('Residence Type')
plt.ylabel('Count')
plt.show()


# In[28]:


df.avg_glucose_level.value_counts()


# In[29]:


sns.histplot(df['avg_glucose_level'], kde=True,bins=15)  
plt.title("Distribution of Average Glucose Level")
plt.xlabel('Average Glucose Level')
plt.ylabel('Frequency')
plt.show()


# In[30]:


df.bmi.value_counts()


# In[31]:


sns.histplot(df['bmi'], kde=True,bins=20)  
plt.title("Distribution of BMI")
plt.xlabel('BMI')
plt.ylabel('Frequency')
plt.show()


# In[32]:


df.smoking_status.value_counts()


# In[33]:


df.groupby('smoking_status')['hypertension'].mean()*100


# In[34]:


df.columns


# In[35]:


contingency_table = pd.crosstab(df['smoking_status'], df['heart_disease'])
print(contingency_table)


# In[36]:


# Define a function to apply the replacement logic
def replace_smoking_status(row):
    if row['smoking_status'] == 'Unknown':
        if row['heart_disease'] == 0:
            return 'never smoked'
        elif row['heart_disease'] == 1:
            return 'smokes'
    return row['smoking_status']

# Apply the function to create a new column or replace the existing one
df['smoking_status'] = df.apply(replace_smoking_status, axis=1)


# In[37]:


df.stroke.value_counts()


# In[38]:


sns.pairplot(data=df, hue="stroke")
#g.fig.set_size_inches(25, 8)  # Set the figure size to (5, 6)
#plt.show()


# In[39]:


sns.catplot(x='stroke', y='avg_glucose_level', hue='smoking_status', kind='box', data=df)
plt.show()


# In[40]:


df['smoking_status'].replace('formerly smoked', 'smokes', inplace=True)


# In[41]:


import seaborn as sns
import matplotlib.pyplot as plt

# Select numerical columns only
numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns

# Create box plots for each numerical column
for column in numerical_columns:
    plt.figure(figsize=(8, 6))  # Adjust the figure size as needed
    sns.boxplot(x=df[column])
    plt.title(f'Box Plot of {column}')
    plt.xlabel(column)
    plt.show()


# In[42]:


df.columns


# In[43]:


df['avg_glucose_level'].describe()


# In[44]:


df[df['avg_glucose_level']>=145]


# In[45]:


df['bmi'].describe()


# In[46]:


# Calculate quartiles
Q1 = df['avg_glucose_level'].quantile(0.25)
Q3 = df['avg_glucose_level'].quantile(0.75)

# Interquartile range (IQR)
IQR = Q3 - Q1

# Determine the upper and lower bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Identify outliers
outliers_index = df[(df['avg_glucose_level'] < lower_bound) | (df['avg_glucose_level'] > upper_bound)].index

# Drop outliers inplace
df.drop(outliers_index, inplace=True)


# In[47]:


# Calculate quartiles for 'bmi'
Q1_bmi = df['bmi'].quantile(0.25)
Q3_bmi = df['bmi'].quantile(0.75)

# Interquartile range (IQR) for 'bmi'
IQR_bmi = Q3_bmi - Q1_bmi

# Determine the upper and lower bounds for outliers for 'bmi'
lower_bound_bmi = Q1_bmi - 1.5 * IQR_bmi
upper_bound_bmi = Q3_bmi + 1.5 * IQR_bmi

# Identify outliers in 'bmi'
outliers_index_bmi = df[(df['bmi'] < lower_bound_bmi) | (df['bmi'] > upper_bound_bmi)].index

# Drop outliers inplace for 'bmi'
df.drop(outliers_index_bmi, inplace=True)


# In[48]:


df['bmi'].describe()


# In[49]:


df.info()


# In[50]:


df.sample(5)


# In[51]:


df.work_type.value_counts()


# In[52]:


df.work_type.replace('Never_worked','Self-employed',inplace=True)


# In[53]:


from scipy.stats import chi2_contingency


# In[54]:


df.columns


# In[55]:


import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

# Define the list of categorical variables
categorical_variables = ['gender', 'hypertension', 'heart_disease', 'ever_married',
                         'work_type', 'Residence_type', 'smoking_status']

# Create a DataFrame to store the p-values
p_values = pd.DataFrame(index=categorical_variables, columns=['p-value'])

# Perform chi-square test for each combination and store the p-values
for variable in categorical_variables:
    contingency_table = pd.crosstab(df[variable], df['stroke'])
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    p_values.loc[variable, 'p-value'] = p

# Convert the data type of 'p-value' column to float
p_values['p-value'] = p_values['p-value'].astype(float)

# Create a heatmap of p-values
plt.figure(figsize=(10, 8))
sns.heatmap(p_values, annot=True, cmap='coolwarm', fmt=".4f")
plt.title('Chi-square Test P-values for Stroke')
plt.xlabel('Stroke')
plt.ylabel('Categorical Variables')
plt.show()


# In[56]:


p_values['p-value'] 


# In[ ]:





# In[57]:


df.work_type.replace('children','Private',inplace=True)


# # Model Building

# In[58]:


from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# In[59]:


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier


# In[60]:


df.drop(['Residence_type', 'id'], axis=1, inplace=True)


# In[61]:


df.head(2)


# In[62]:


# X is all columns except the last one
X = df.iloc[:, :-1]

# y is the last column
y = df.iloc[:, -1]


# In[71]:


from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print the shapes of the resulting arrays to verify the split
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


# In[72]:


numeric_features = ['age', 'avg_glucose_level', 'bmi']
categorical_features = ['gender', 'hypertension', 'heart_disease', 'ever_married',
                        'work_type', 'Residence_type', 'smoking_status']


# In[ ]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline




# # LogisticRegression

# In[74]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline



# Define the numeric features and categorical features
numeric_features = ['age', 'avg_glucose_level', 'bmi']
categorical_features = ['gender', 'ever_married', 'work_type', 'smoking_status']

# Create the ColumnTransformer
column_transformer = ColumnTransformer(
    transformers=[
        ('numeric', StandardScaler(), numeric_features),
        ('categorical', OneHotEncoder(), categorical_features)
    ],
    remainder='drop'  # Drop any columns not specified in transformers
)

# Define the logistic regression model
logistic_regression = LogisticRegression()

# Create the pipeline
pipe = Pipeline([
    ('preprocessor', column_transformer),
    ('classifier', logistic_regression)
])

# Fit the pipeline on training data
pipe.fit(X_train, y_train)

# Predict on test data
y_pred = pipe.predict(X_test)

# Calculate R2 score
r2 = r2_score(y_test, y_pred)
print("R2 score:", r2)

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion matrix:\n", conf_matrix)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[84]:


# Create a DataFrame with the provided features
new_data = pd.DataFrame({
    'gender': ['Female'],
    'age': [50.0],
    'hypertension': [1],
    'heart_disease': [1],
    'ever_married': ['Yes'],
    'work_type': ['Private'],
    'avg_glucose_level': [150.65],
    'bmi': [54.0],
    'smoking_status': ['smokes']
})

# Predict using the trained pipeline
prediction = pipe.predict(new_data)

# Print the prediction
if prediction == 0:
    print("Stroke possibility is less.")
elif prediction == 1:
    print("Stroke possibility is High!")



# In[89]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import r2_score, confusion_matrix, accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


# Define the numeric features and categorical features
numeric_features = ['age', 'avg_glucose_level', 'bmi']
categorical_features = ['gender', 'ever_married', 'work_type', 'smoking_status']

# Create the ColumnTransformer
column_transformer = ColumnTransformer(
    transformers=[
        ('numeric', StandardScaler(), numeric_features),
        ('categorical', OneHotEncoder(), categorical_features)
    ],
    remainder='drop'  # Drop any columns not specified in transformers
)

# Create pipelines for each classifier
pipelines = {
    
    'K-Nearest Neighbors': make_pipeline(column_transformer, KNeighborsClassifier()),
    'Decision Tree': make_pipeline(column_transformer, DecisionTreeClassifier()),
    'Random Forest': make_pipeline(column_transformer, RandomForestClassifier()),
    'AdaBoost': make_pipeline(column_transformer, AdaBoostClassifier()),
    'Gradient Boosting': make_pipeline(column_transformer, GradientBoostingClassifier()),
    'SVM': make_pipeline(column_transformer, SVC())
}


# Train and evaluate each pipeline
for name, pipeline in pipelines.items():
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
       # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Print evaluation metrics
    print(f"\n{name} Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Confusion Matrix:\n", conf_matrix)


# In[ ]:





# In[ ]:




