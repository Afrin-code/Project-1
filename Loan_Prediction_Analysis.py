#Loan Prediction Analysis

#Importing Libraries
import pandas as pd
import matplotlib.pyplot as  plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler,RobustScaler
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import warnings
warnings.filterwarnings("ignore")

#1.Reading of Data
def load_data():
	
	df=pd.read_csv(r"C:\Users\Sony\Desktop\python\loan.csv")
	
	print(df.head())
	print(df.info())
	print(df.describe())

	#Remove 'month' from term
	df['term']=df['term'].replace({'36 months':36,'60 months':60})

	# Finding Categorical Columns
	categorical_feature_mask = df.dtypes==object

	# filter categorical columns using mask and turn it into a list
	categorical_cols = df.columns[categorical_feature_mask].tolist()
	print(categorical_cols)

	# Finding Numerical Columns
	numerical_feature_mask = df.dtypes!=object

	# filter categorical columns using mask and turn it into a list
	numerical_cols = df.columns[numerical_feature_mask].tolist()
	print(numerical_cols)
	return df

#2.Exploratory Data Analysis (EDA)
'''
def EDA(df):
	#Create copy of dataframe 
	df_copy=df.copy()
	df_copy.dropna(inplace=True)

	#Count of bad loan target feature
	print(df_copy['bad_loan'].value_counts())
	plt.figure(figsize=(5,5))
	sns.countplot(df_copy['bad_loan'],palette='plasma')
	plt.title("Count of bad_loan")
	plt.xticks(ticks=np.arange(2),labels=['Good Customer','Bad Customer'])
	plt.show()

	#Count of home_ownership w.r.s. bad_loan
	print(df_copy.groupby('bad_loan')['home_ownership'].value_counts())

	plt.figure(figsize=(10,8))
	sns.countplot(df_copy['home_ownership'], hue='bad_loan', data=df_copy)
	plt.title("Count of home_ownership w.r.s bad_loan")
	plt.show()

	#Countplot of term 
	print(df_copy['term'].value_counts())
	plt.figure(figsize=(8,5))
	plt.title("Countplot for term")
	sns.countplot(df_copy['term'],palette='plasma')
	plt.xticks(ticks=np.arange(2),labels=['36 Months','60 Months'])
	plt.show()

	#Countplot of term w.r.s bad_loan

	plt.figure(figsize=(10,8))
	sns.countplot(df_copy['term'], hue='bad_loan', data=df_copy)
	plt.title("Countplot for term w.r.s bad_loan")
	plt.show()

	#Countplot for emp_length Feature
	print(df_copy['emp_length'].value_counts())
	plt.figure(figsize=(5,5))
	plt.title("Countplot for Employee Length")
	sns.countplot(df_copy['emp_length'],palette='plasma')
	plt.xticks(ticks=np.arange(11))
	plt.show()

	#Countplot of emp_length w.r.s bad_loan
	plt.figure(figsize=(10,8))
	sns.countplot(df_copy['emp_length'], hue='bad_loan', data=df_copy)
	plt.title("Countplot for emp_length w.r.s bad_loan")
	plt.show()

	#Distribution plot of loan_amnt
	sns.distplot(df_copy['loan_amnt'],kde=False)
	plt.title("Distribution of Loan")
	plt.xlabel("Range of loan")
	plt.ylabel("count")
	plt.show()

	#Distribution plot of int_rates
	sns.distplot(df_copy['int_rate'],kde=False)
	plt.title("Distribution of int_rate")
	plt.xlabel("Range of Interest Rate")
	plt.ylabel("count")
	plt.show()

	#Distribution of Annual Income
	plt.figure(figsize=(5,5))
	plt.title("Distribution Annual Income")
	plt.distplot(df_copy['annual_inc'],kde=False)
	#plt.xticks(rotation=45)
	plt.xlabel("Annual Income")
	plt.ylabel("Count")
	plt.show()

	#As distribution is skewed apply log transformation on it
	#Distribution of Annual Income
	sns.distplot(np.log1p(df_copy['annual_inc'],kde=False)
	plt.xticks(rotation=45)
	plt.xlabel("Annual Income")
	plt.ylabel("Count")
	plt.show()

	#Relationship between Loan Amount and Annual Income
	plt.figure(figsize=(5,5))
	plt.scatter(df_copy['loan_amnt'],np.log(df_copy['annual_inc']))
	plt.title("Annual Income vs Loan Amount")
	plt.ylabel("Loan Amount")
	plt.xlabel("Annual Income")
	plt.show()

	#Count of loan purpose
	print(df_copy['purpose'].value_counts())
	plt.figure(figsize=(15,5))
	plt.title("Countplot for Loan Purpose")
	sns.countplot(df_copy['purpose'],palette='plasma')
	plt.xlabel("Loan Purpose")
	plt.xticks(rotation=45)
	plt.show()

	#Checking for outlier
	plt.figure(figsize=(5,5))
	sns.boxplot(y=np.log(df_copy['annual_inc']))
	plt.show()

	#Correlation of features
	plt.figure(figsize=(10,10))
	sns.heatmap(df_copy.corr(),annot=True,cbar=True,cmap='viridis',linewidth=0.2)
	plt.show()'''

#3. Data Preprocessing and Feature Engineering
def data_prec(df):
	#Finding Missing Values
	df.isnull().sum()

	#Handling Missing Values
	df['emp_length']=df['emp_length'].fillna(df['emp_length'].mean())
	df['annual_inc']=df['annual_inc'].fillna(df['annual_inc'].mean())
	df['delinq_2yrs']=df['delinq_2yrs'].fillna(df['delinq_2yrs'].mode()[0])
	df['longest_credit_length']=df['longest_credit_length'].fillna(df['longest_credit_length'].mode()[0])
	df['revol_util']=df['revol_util'].fillna(df['revol_util'].median())
	df['total_acc']=df['total_acc'].fillna(df['total_acc'].median())
	
	# Finding Categorical Columns
	categorical_feature_mask = df.dtypes==object
	# filter categorical columns using mask and turn it into a list
	categorical_cols = df.columns[categorical_feature_mask].tolist()
	print(categorical_cols)

	#Label Encoding to convert Categorical Features into Numerical Features
	for c in categorical_cols:
		lbl = LabelEncoder() 
		lbl.fit(list(df[c].values)) 
		df[c] = lbl.transform(list(df[c].values))
		
	#Apply log transformation to deal with skewness of annual income
	df['annual_inc']=np.log1p(df['annual_inc'])
	return df

#Classification Model Building

def class_model(df):
	#Data splitting train and test Data
	y = df["bad_loan"]
	x = df.drop(['bad_loan'],axis=1)
	x= RobustScaler().fit_transform(x)
	
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.3)

	#Feature Scaling
	'''PT=PowerTransformer()
	x_train=PT.fit_transform(x_train)
	x_test=PT.fit_transform(x_test)'''
	

	#LogisticRegression classification Model without cross Validation
	log = LogisticRegression()
	log.fit(x_train, y_train)
	log_pred = log.predict(x_test)
	log_accuracy = metrics.accuracy_score(y_test, log_pred)
	print("Accuracy: ",log_accuracy)

	log_precision=metrics.precision_score(y_test, log_pred,pos_label=0)
	print("Precision: ",log_precision)

	log_recall=metrics.recall_score(y_test, log_pred,pos_label=0)
	print("Recall: ",log_recall)

	log_f1_score= metrics.f1_score(y_test, log_pred,pos_label=0)
	print("F1 Score: ",log_f1_score)

	print("Confusion Matrix:\n",confusion_matrix(y_test,log_pred))
	print("Classification Report:\n",classification_report(y_test,log_pred))

	#LogisticRegression classification Model with cross Validation
	

	log_cross_val = cross_val_score(log, x, y, cv=10, scoring='accuracy')
	log_cv_accuracy = log_cross_val.mean()
	print("Accuracy: ",log_cv_accuracy)

	log_cross_val_pre = cross_val_score(log, x, y, cv=10, scoring='precision_macro')
	log_cv_precision = log_cross_val_pre.mean()
	print("Precision: ",log_cv_precision)

	log_cross_val_re = cross_val_score(log, x, y, cv=10, scoring='recall_macro')
	log_cv_recall = log_cross_val_re.mean()
	print("Recall: ",log_cv_recall)

	log_cross_val_f1 = cross_val_score(log, x, y, cv=10, scoring='f1_macro')
	log_cv_f1_score = log_cross_val_f1.mean()
	print("F1 Score: ",log_cv_f1_score)

#Driver Functions
#Calling of Data and EDA functions
df=load_data()
#EDA(df)

#Calling of Data Preprocessing, Feature Engineering and Model building
df=data_prec(df)
print(df.head(5))
class_model(df)
