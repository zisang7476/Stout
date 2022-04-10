#!/usr/bin/env python
# coding: utf-8

# In[45]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#pip install plotly
import plotly.graph_objects as go
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


# # Case 1

# ## Part 1 Data Overview

# In[398]:


#import data for case 1
loan = pd.read_csv("loans_full_schema.csv")
loan.head(5)


# **Description:**
# 
# This dataset contains massive information regarding loan applications and applicants' information. It has 10,000 observations with 55 varibales.
# 
# ***Variables:*** there're bascially 3 categories of variables: 
# 1. Personal information (from 'emp_title' to 'debt_to_income_joint' )
# 2. History of credit behaviors (from 'delinq_2y' to 'public_record_bankrupt' )
# 3. Characters for the loan application (from 'loan_purpose' to 'paid_late_fees' )
# 
# Summary for each numeric variable is showed below.
# 
# 

# In[12]:


print(loan.shape)
loan_des = loan.describe().round(2)
loan_des


# **Issue:**
# 
# There are quite many NA in some variables, we need to fillna or drop those observations before building model.

# ## Part 2 Visualizations

# In this part, we are going to take a close look on loan data by visualizations, step by step.
# 

# **First, let's get a overview of loans:**
# 
# There're 10 thousand loans issued during 2018 Q1, of which total amount is 163.52 million. The amount of loan issued for each month are 54.56, 49.98, 59.98 million, respectively

# In[389]:


# Overview of Loans
sumloan_month = loan.groupby(['issue_month'])['loan_amount'].sum().to_frame()
sumloan_month['loan_amount'] = sumloan_month['loan_amount']/10**6
df = sumloan_month.reindex(['Jan-2018', 'Feb-2018', 'Mar-2018'])
plt.figure(figsize=(6, 6))
plots = sns.barplot(x = df.index, y = 'loan_amount', data=df, color = 'deepskyblue')
for bar in plots.patches:
    plots.annotate(format(bar.get_height(), '.2f'),
                   (bar.get_x() + bar.get_width() / 2,
                    bar.get_height()), ha='center', va='center',
                   size=10, xytext=(0, 8),
                   textcoords='offset points')
plt.xlabel('')
plt.ylabel('Total Loans Issued in Millions')
plt.title("Loan Overview")
plt.show()
#sumloan_month.plot.bar()


# **Why people need loans?**
# 
# The data tells us that people borrow money for 12 reasons. As the pie chart shows below, most people (~51.4%) borrow money to consolidate thier current debt. And ~22.5% people need money to pay thier credit cards balance. Also, ~6.8% people use the loan to make improvment in thier home.

# In[198]:


# Main purpose of loans
loan_purpose = loan.groupby(['loan_purpose'])['loan_amount'].sum().to_frame()
loan_purpose['num'] = loan.groupby(['loan_purpose'])['loan_amount'].count()
loan_purpose['average_amount'] = loan_purpose['loan_amount']/loan_purpose['num']
loan_purpose

plt.figure(figsize=(15, 15))
lab=loan_purpose.index
my_color = ['#63b2ee','#76da91','#f8cb7f', '#f89588','#7cd6cf','#9192ab','#7898e1','#efa666','#eddd86','#9987ce','#63b2ee', '#76da91']
plt.pie(loan_purpose['num'], labels=lab, autopct='%1.1f%%', colors = my_color)
plt.title("Loan Purpose by Numbers")

plt.legend()
plt.show()


# **How they apply for a loan?**
# 
# As the pie chart shows below, most people (~85%) apply for a loan by themselves while the remaining ~15% people jointly apply for loans

# In[181]:


# application type loans
loan_type=loan.groupby(['application_type'])['loan_amount'].count().to_frame()
plt.pie(loan_type['loan_amount'], labels=loan_type.index, autopct='%1.1f%%', shadow=0, colors = ['skyblue','deepskyblue'])
plt.title("Application Type by Numbers")
plt.show()


# **Next, we're courious about WHO borrow the money:**
# 
# Using the following map, we can easily tell that people from California (Top 1 with over 20 mn loan), Texas (13+ mn), New York (12+ mn), and Florida (11+ mn) borrow most loan.
# 
# *Note: if the map doesn't show, please open another file 'map.html' in the same folder*

# In[397]:


# Map for loans
sumloan_state = loan.groupby(['state'])['loan_amount'].sum().to_frame()

fig = go.Figure(data=go.Choropleth( 
    locations=sumloan_state.index,  
    z = sumloan_state['loan_amount'].astype(float), 
    locationmode = 'USA-states',
    colorscale = 'Blues', 
    colorbar_title = "Total Loan Amount", 
)) 

fig.update_layout( 
    title_text = 'Loan Amount for States', 
    geo_scope='usa',  
    #scope: "world"，"usa"，"europe"，"asia"，"africa"，"north america"，"south america" 
)
fig.show()
fig.write_html("map.html") 


# **Are those people poor or rich?**
# 
# The following two histograms present the distribution of annual income and debt to income ratio.
# 1. Most borrowers have a medium income between 25K and 100K. However, the distribution does have a fat tail at the right end, showing there're more super rich people whose annual income is over 175K that also need a loan
# 2. Most borrowers have a healthy financial leverage as debt to income ratio below 0.4, indicating low default probability. Nevertheless, those people at fat tail end whose debt to income ratio over 1.0 need more attention.
# 
# *Note: There're many exterme large numbers in income data so we need to log them before putting those income into our predictive model.*

# In[175]:


# Who borrows
plt.figure(figsize=(10, 5))

# Annual Income Distribution
x = loan['annual_income'] /1000
p95 = np.percentile(x, 95)
x[x>p95] = p95

plt.subplot(1, 2, 1)
plt.hist(x, bins=25, range=[0, p95], density = 1, color = 'deepskyblue')
plt.title("Annual Income Distribution")
plt.xlabel('Annual income in thousand dollars')

# Debt/Income Distribution
y = loan['debt_to_income']/100
y[y>1] = 1
plt.subplot(1, 2, 2)
plt.hist(y, bins=25, density = 1, color = 'deepskyblue')
plt.title("Debt/Income Distribution")
plt.xlabel('Debt/Income')
plt.show()


# **Finally, let's look at interest rate**
# 
# According to common sense, each loan applicant would have a letter grade which indicate his/her credibility. I use box plot to compare the interest rates for different grades.
# 
# As showed below, 'A' refers to the highest credibility with lowest interest rate around 7% while G refers to the lowest credibility so that the interest rate is over 30% to compensate potential default risks.
# 
# I also find an outlier for grade D in which the applicant of grade D yet borrows loan with the same interest rate as grade A applicants.

# In[73]:


# interest rate for different grade
loan.boxplot(column='interest_rate',by='grade', grid = False, figsize=(8,6), fontsize=10)
plt.xlabel('Grade')
plt.ylabel('Interest rate (%)')
plt.suptitle('Interest rate grouped by grade')
plt.title('')
plt.show()


# ## Part 3 Predictive model

# In part 3, we need to buil predictive model to project interest rate for different applicants using related information. The first model is multiple linear regression model. After the most basic one, I also choose the random forest regression model as an alternative because I think rf model fit the decision process.
# 
# However, before actually building models, we need to first clean our data as there're many NA and outliers.
# 
# Above all, let's look at on our dataset again.

# In[399]:


df = loan.copy(deep = True)
df.head(10)


# In[392]:


df.columns


# ### 3.1 Data Cleaning and Features Picking

# **Variables**
# As said above, there're bascially 3 categories of variables: 
# 1. personal information (from 'emp_title' to 'debt_to_income_joint' )
# 2. history of credit behaviors (from 'delinq_2y' to 'public_record_bankrupt' )
# 3. characters for the loan application (from 'loan_purpose' to 'paid_late_fees' )
# 
# I need to pick variables from those 3 categories as features for predictive model
# feature_personal = ['emp_title', 'emp_length', 'state', 'homeownership', 'annual_income', 'verified_income', 'debt_to_income', 'annual_income_joint', 'verification_income_joint', 'debt_to_income_joint']
# 
# **Principle for data cleaning**
# 1. Fill na with reasonable numbers. For example, for employment length, fill 0 for no employment.
# 2. deal with extreme numbers, like income, use log
# 3. get dummies for category variables, like home ownership or grade
# 4. do some calculations to generate more proper variables, i.e, percentage of credit utilized
# 5. change date to period
# 6. Last step is to drop observations that still contains NA
# 
# **Apply those principles to all 3 categories of variables and pick all avaliable feature variables**

# **First, let's start with application variables as it may interact with the other two**
# 
# feature_application = ['joint', 'amount', 'term', 'grade_A', 'grade_B', 'grade_C', 'grade_D', 'grade_E', 'grade_F']
# 
# *Note: grade is a sepcial variable because it's highly related with interest rate. Therotically, adding grade into independet varibles could greatly enhance the explaining ability of the model. However, in real world, what we need to do is to determine each applicant's credibility based on his or her information and use the credibility to offer interest rate. Therefore, I think grade should not belong to independent variables. But I first take grade as one of the all avaliable variables, then I test my models with both features with grade and features without grade.*

# In[400]:


df = loan.copy(deep = True)
# First, let's start with application variables as it may interact with the other two
df = pd.get_dummies(df, columns=['application_type'], prefix = '', prefix_sep = '', drop_first= 0)
df.drop(columns = ['individual'], inplace = True)

#log loan amount
df['amount'] = np.log(df['loan_amount'])
df['Grade'] = df['grade']

df = pd.get_dummies(df, columns=['grade'], drop_first= 0)
feature_application = ['joint', 'amount', 'term', 'grade_A', 'grade_B', 'grade_C', 'grade_D', 'grade_E', 'grade_F']

df.head(10)


# **Second, variables for personal features**
# 
# feature_personal = ['emp_length', 'home_MORTGAGE', 'home_OWN', 'income_log', 'debt_to_income', 'income_joint_log', 'debt_to_income_joint']

# In[401]:


# variables for personal features
# job information
df['emp_length'] = df['emp_length'].fillna(0) # 0 refers to no job currently

# property information
#home_mapping = {"OWN":1, "MORTGAGE":0.5, "RENT":0}
#df['homeownership'] = df['homeownership'].map(home_mapping)
df = pd.get_dummies(df, columns=['homeownership'], prefix = 'home')

# log income
df['income_log'] = np.log(df['annual_income'])

# joint information
df['annual_income_joint'] = df['annual_income_joint'].fillna(0)
df['income_joint_log'] = np.log(df['annual_income_joint'] * df['joint'] + 1)

df['debt_to_income_joint'] = df['debt_to_income_joint'].fillna(0)
df['debt_to_income_joint'] = df['debt_to_income_joint'] * df['joint']


feature_personal = ['emp_length', 'home_MORTGAGE', 'home_OWN', 'income_log', 'debt_to_income', 'income_joint_log', 'debt_to_income_joint']
df.head(10)


# **Third, variables for credit behaviors**
# 
# feature_credit =['delinq_2y','months_since_last_delinq', 'credit_month','inquiries_last_12m', 'open_credit_lines',
#        'total_credit_limit', 'total_credit_utilized_pct', 'num_collections_last_12m', 'num_historical_failed_to_pay',
#        'months_since_90d_late', 'current_accounts_delinq', 'total_collection_amount_ever', 'num_accounts_120d_past_due',
#        'num_accounts_30d_past_due', 'num_active_debit_accounts', 'total_debit_limit', 'num_open_cc_accounts',
#        'num_mort_accounts', 'account_never_delinq_percent', 'tax_liens', 'public_record_bankrupt']

# In[402]:


# variables for credit behaviors


df['months_since_last_delinq'] = df['months_since_last_delinq'].fillna(100000)

df['credit_month'] = 12*(2022 - df['earliest_credit_line'])

df['total_credit_utilized_pct'] = df['total_credit_utilized'] / df['total_credit_limit']

df['months_since_90d_late'] = df['months_since_90d_late'].fillna(100000)
feature_credit =['delinq_2y','months_since_last_delinq', 'credit_month',
       'inquiries_last_12m', 'open_credit_lines',
       'total_credit_limit', 'total_credit_utilized_pct',
       'num_collections_last_12m', 'num_historical_failed_to_pay',
       'months_since_90d_late', 'current_accounts_delinq',
       'total_collection_amount_ever', 'num_accounts_120d_past_due',
       'num_accounts_30d_past_due', 'num_active_debit_accounts',
       'total_debit_limit', 'num_open_cc_accounts',
       'num_mort_accounts', 'account_never_delinq_percent', 'tax_liens', 'public_record_bankrupt']

df.head(10)


# **Finally, we combine all three feature subsets into one, and prepare the final dataset for model**

# In[403]:


# Pick all possible features
feature = feature_personal + feature_credit + feature_application
#drop unnecessary columns
feature_ext = feature+['interest_rate']
df_clean = df.loc[:,feature_ext]
df_clean.dropna(how = 'any', inplace = True)
df_clean.head(10)


# ### 3.2 Multi Linear Regression Model

# *Note: for each model, we seperate the whole dataset into training data (80%) and testing data (20%) by random sample*

# In[327]:


# Generate training set and testing set
df_train = df_clean.sample(frac = 0.8).copy(deep = True)
df_test = df_clean[~df_clean.index.isin(df_train.index)]
print(df_train.shape,df_test.shape)

from sklearn import linear_model
lm=linear_model.LinearRegression()

x_train = df_train[feature]
y_train = df_train['interest_rate']
model = lm.fit(x_train, y_train)
R2 = model.score(x_train, y_train)
print('R2 = %.3f' % R2)

# Result using out-of-sample-data
x_test = df_test[feature]
y_test = df_test['interest_rate']

score = model.score(x_test, y_test)
result = model.predict(x_test)
plt.figure()
plt.plot(np.arange(len(result)), y_test,label='true value')
plt.plot(np.arange(len(result)),result,label='predict value')

#'go-' 'ro-'
plt.title('score: %.3f'%score)
plt.legend()
plt.show()


# For MLR, I write a function 'mlr'(inputs are df_clean, and feature list) to generate R2 and out-of-sample test result. Function 'mlr_plot' can also generate plot to compare the predict value and true value in test data.

# In[382]:


def mlr(df_clean, feature):
    # Generate training set and testing set
    df_train = df_clean.sample(frac = 0.8).copy(deep = True)
    df_test = df_clean[~df_clean.index.isin(df_train.index)]
    #print(df_train.shape,df_test.shape)
    
    from sklearn import linear_model
    lm=linear_model.LinearRegression()

    x_train = df_train[feature]
    y_train = df_train['interest_rate']
    model = lm.fit(x_train, y_train)
    R2 = model.score(x_train, y_train)
    #print('R2 = %.3f' % R2)
    
    # Result using out-of-sample-data
    x_test = df_test[feature]
    y_test = df_test['interest_rate']

    score = model.score(x_test, y_test)
    result = model.predict(x_test)
    #plt.figure()
    #plt.plot(np.arange(len(result)), y_test,label='true value')
    #plt.plot(np.arange(len(result)),result,label='predict value')

    #plt.title('score: %.3f'%score)
    #plt.legend()
    #plt.show()
    
    r_pass = [R2, score]
    return r_pass

def mlr_plot(df_clean, feature):
    # Generate training set and testing set
    df_train = df_clean.sample(frac = 0.8).copy(deep = True)
    df_test = df_clean[~df_clean.index.isin(df_train.index)]
    print(df_train.shape,df_test.shape)
    
    from sklearn import linear_model
    lm=linear_model.LinearRegression()

    x_train = df_train[feature]
    y_train = df_train['interest_rate']
    model = lm.fit(x_train, y_train)
    R2 = model.score(x_train, y_train)
    print('R2 = %.3f' % R2)
    
    # Result using out-of-sample-data
    x_test = df_test[feature]
    y_test = df_test['interest_rate']

    score = model.score(x_test, y_test)
    result = model.predict(x_test)
    plt.figure()
    plt.plot(np.arange(len(result)), y_test,label='true value')
    plt.plot(np.arange(len(result)),result,label='predict value')

    plt.title('score: %.3f'%score)
    plt.legend()
    plt.show()
    
    r_pass = [R2, score]
    return r_pass


# **First, let's try feature with grade in MLR**

# In[394]:


mlr_plot(df_clean, feature)


# The above result shows that R2 is 0.952, and score for out-of-sample test is 0.952. This model has a very strong explaining capability which is not surprising as it contains grade.

# Then we repeat the model for 100 times to get the average score. As showed below, the avg. R2 is 0.952 and the avg. score is also 0.952.

# In[381]:


N = 100
test_result = pd.DataFrame(index = range(N), columns = ['R2','test_result'])

for i in range(N):
    re = mlr(df_clean, feature)
    test_result.loc[i, 'R2'] = re[0]
    test_result.loc[i, 'test_result'] = re[1]

print(test_result.mean())


# **Then, let's try feature without grade in MLR**

# In[383]:


# No grade in features
feature_nograde = feature[:-6]

mlr_plot(df_clean, feature_nograde)

N = 100
test_result = pd.DataFrame(index = range(N), columns = ['R2','test_result'])

for i in range(N):
    re = mlr(df_clean, feature_nograde)
    test_result.loc[i, 'R2'] = re[0]
    test_result.loc[i, 'test_result'] = re[1]

print(test_result.mean())


# Now the result is not as good as the model with grade. The R2 is only 0.342 while the score for out-of-sample test is 0.328. This model has less strong explaining capability than the previous one as the feature set drops 'grade'.
# 
# After repeating for 100 times, we get the avg.R2 of 0.339 and the avg. score of 0.331. 

# ### 3.2 Other predictive models (Random Forest as final model)

# In[ ]:


from sklearn import neighbors
model_KNeighborsRegressor = neighbors.KNeighborsRegressor()
from sklearn import svm
model_SVR = svm.SVR()

from sklearn import tree
model_DecisionTreeRegressor = tree.DecisionTreeRegressor()

from sklearn import ensemble
model_RandomForestRegressor = ensemble.RandomForestRegressor(n_estimators=20)


# As usual, after trying different models, I finally choose random forest regression. The function 'rf' with the same inputs of df_clean and feature is to return R2 and out-of-sample result for the model. The function 'rf_plot' can also provide visualization for comparison.

# In[384]:


# Random Forest Model
def rf_plot(df_clean, feature):
    # Generate training set and testing set
    df_train = df_clean.sample(frac = 0.8).copy(deep = True)
    df_test = df_clean[~df_clean.index.isin(df_train.index)]
    print(df_train.shape,df_test.shape)
    
    from sklearn import ensemble
    mod =ensemble.RandomForestRegressor(n_estimators=40)

    x_train = df_train[feature]
    y_train = df_train['interest_rate']
    model = mod.fit(x_train, y_train)
    R2 = model.score(x_train, y_train)
    print('R2 = %.3f' % R2)
    
    # Result using out-of-sample-data
    x_test = df_test[feature]
    y_test = df_test['interest_rate']

    score = model.score(x_test, y_test)
    result = model.predict(x_test)
    plt.figure()
    plt.plot(np.arange(len(result)), y_test,label='true value')
    plt.plot(np.arange(len(result)),result,label='predict value')

    plt.title('score: %.3f'%score)
    plt.legend()
    plt.show()
    
    r_pass = [R2, score]
    return r_pass

def rf(df_clean, feature):
    # Generate training set and testing set
    df_train = df_clean.sample(frac = 0.8).copy(deep = True)
    df_test = df_clean[~df_clean.index.isin(df_train.index)]
    #print(df_train.shape,df_test.shape)
    
    from sklearn import ensemble
    mod =ensemble.RandomForestRegressor(n_estimators=40)

    x_train = df_train[feature]
    y_train = df_train['interest_rate']
    model = mod.fit(x_train, y_train)
    R2 = model.score(x_train, y_train)
    #print('R2 = %.3f' % R2)
    
    # Result using out-of-sample-data
    x_test = df_test[feature]
    y_test = df_test['interest_rate']

    score = model.score(x_test, y_test)
    result = model.predict(x_test)
    #plt.figure()
    #plt.plot(np.arange(len(result)), y_test,label='true value')
    #plt.plot(np.arange(len(result)),result,label='predict value')

    #plt.title('score: %.3f'%score)
    #plt.legend()
    #plt.show()
    
    r_pass = [R2, score]
    return r_pass


# **First try random forest regression in which feature set contains grade. The result are showed in the plot below.**
# 
# We have 7725 observations for training and 1931 ones for testing. The R2 is 0.993, incredibly high, and the out-of-sample score is 0.952.

# In[385]:


rf_plot(df_clean, feature)


# **Next try random forest regression without 'grade'. The result are showed in the plot below.**
# 
# We have 7725 observations for training and 1931 ones for testing. The R2 is 0.908, while the out-of-sample score is only 0.371.

# In[386]:


# No grade in features
feature_nograde = feature[:-6]
rf_plot(df_clean, feature_nograde)


# By repeating this model without 'grade' for 20 times, we get the avg. R2 of 0.906 while the avg. out-of-sample score of 0.372

# In[368]:


# Replicate testing
N = 20
test_result = pd.DataFrame(index = range(N), columns = ['R2','test_result'])

for i in range(N):
    re = rf(df_clean, feature)
    test_result.loc[i, 'R2'] = re[0]
    test_result.loc[i, 'test_result'] = re[1]

print(test_result.mean())


# ## Part 4 Comments and Reflection

# If I have more time, I would definitely take a even closer check on every independent variables. I also need to check the correlations and collinearity between those variables.
# Clustering could also be applied for prediction.

# # Case 2 

# In[404]:


#import data for case 2
cs2 = pd.read_csv("casestudy2.csv", index_col= 0 )
cs2.head(5)


# In[405]:


cs2.describe()


# **Revenue for each year**
# 
# Revenue for 2015: 29.04 mn
# 
# Revenue for 2016: 25.73 mn
# 
# Revenue for 2017: 31.42 mn

# In[406]:


# Year Revnue
rev = pd.DataFrame()
rev['revenue'] = cs2.groupby(['year'])['net_revenue'].sum()
print(rev)
rev.plot.bar(rot=0,color = 'skyblue')


# In[407]:


cs2['last_year'] = cs2['year'] -1
cs2['rev_prev'] = cs2.groupby(['customer_email'])['net_revenue'].shift(1)
cs2.head(10)


# **Revenue breakdown for existing customers and new customers is showed in the following table**
# 
# rev_exist refers to Revenue from existing customers
# 
# rev_new refers to Revenue from new customers
# 
# *Note: for 2015, we can't seperate existing revenue and new revenue for 2015 due to lack of data of 2014.*

# In[268]:


rev['rev_exist'] = cs2.groupby(['year'])['rev_prev'].sum()
rev['rev_new'] = rev['revenue'] - rev['rev_exist']
#Due to lack of data of 2014, we can't seperate existing revenue and new revenue for 2015, so leave them blant
rev.iloc[0,1] = np.nan
rev.iloc[0,2] = np.nan
rev


# **Existing Customer Revenue and its change are showed in the following table**
# 
# rev_exist_prev refers to Existing Customer Revenue for previous year
# 
# rev_growth_exist refers to Existing Customer Growth

# In[269]:


# To get exisiting customer growth
rev['rev_exist_prev'] = rev['rev_exist'].shift(1)
rev['rev_growth_exist'] = rev['rev_exist'] -  rev['rev_exist_prev']
rev


# In[259]:


#Revenue lost from attrition
print("Revenue lost from attrition for 2017 are","%.2f" % -rev.iloc[2,4])


# **Number of Customers and its breakdown are showed in the following table**

# In[277]:


# Customers Count

# Total Customers Current Year
rev['num_total'] = cs2.groupby(['year'])['customer_email'].count()

# Total Customers Previous Year
rev['num_total_prev'] =rev['num_total'].shift(1)

# New Customers
rev['num_exist'] = cs2.groupby(['year'])['rev_prev'].count()
rev['num_new'] = rev['num_total'] - rev['num_exist']

#Due to lack of data of 2014, we can seperate existing or new customers for 2015, so leave them blant
rev.iloc[0,7] = np.nan
rev.iloc[0,8] = np.nan

rev


# In[282]:


# Visualization
labels = rev.index
barWidth = 0.25

r1 = np.arange(len(labels))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]

fig, ax = plt.subplots()
rects1 = ax.bar(r1, rev['revenue']/10**6, barWidth, label='Revenue')
rects2 = ax.bar(r2, rev['rev_exist']/10**6, barWidth, label='Revenue from Existing Customers')
rects3 = ax.bar(r3, rev['rev_new']/10**6, barWidth, label='Revenue from New Customers')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Revenue in million')
ax.set_title('Revenue for Each Year')
plt.xticks([r + barWidth for r in range(len(labels))], labels)
ax.legend()

fig.tight_layout()

plt.show()


# In[387]:


labels = rev.index
barWidth = 0.25

r1 = np.arange(len(labels))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]

fig, ax = plt.subplots()
rects1 = ax.bar(r1, rev['num_total']/10**3, barWidth, label='Total Customers')
rects2 = ax.bar(r2, rev['num_exist']/10**3, barWidth, label='Existing Customers')
rects3 = ax.bar(r3, rev['num_new']/10**3, barWidth, label='New Customers')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_title('Customer Numbers in thousand')
plt.xticks([r + barWidth for r in range(len(labels))], labels)
ax.legend()

fig.tight_layout()

plt.show()


# **Observation:**
# 1. Revenue from existing customers was decreasing in the past two years due to failure of customer retention.
# 2. However, new customers revenue shows robust growth momentum, which support the overall revenue growth.
# 3. For further observation, we could calculate the avg. revenue per customer and compare the avg. revenue per customer of existing customers with that of new customers.

# *Answered by Ziming (Gary) Sang in Apr.10th*
