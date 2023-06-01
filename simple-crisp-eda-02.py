#!/usr/bin/env python
# coding: utf-8

# # <p style="font-family: helvetica; letter-spacing: 3px; font-size: 30px; font-weight: bold; color:#1B2631; align:left;padding: 0px">Regression With A 🦀Crab🦀 Age Dataset<span class="emoji">📖🤓📖</span>
# </p>

# ![image.jpg](attachment:images.jpg)

# <p style="font-family: helvetica; letter-spacing: 1px; font-size: 20px; font-weight: bold; color:#1B2631; align:left;padding: 0px;border-bottom: 1px solid #003300"><span class="emoji">🗨️</span>Context
# </p>

# <div class="alert alert-block alert-info"> <b>NOTES TO THE READERS</b><br> This is a 2023 edition of Kaggle's Playground Series where the Kaggle Community hosts a variety of fairly light-weight challenges that can be used to learn and sharpen skills in different aspects of machine learning and data science</div>
# 

# <a id="1"></a>
# <p style="font-family: helvetica; letter-spacing: 1px; font-size: 20px; font-weight: bold; color:#1B2631; align:left;padding: 0px;border-bottom: 1px solid #003300"><span class="emoji">🔀</span>Install and Import
# </p>

# In[48]:


import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
import numpy as np
from IPython.display import display, HTML
import seaborn as sns
import random
import matplotlib.pyplot as plt
import plotly.express as px
import pandas_profiling as pp
from scipy import stats
from scipy.stats import norm
import h2o
from h2o.automl import H2OAutoML
get_ipython().run_line_magic('matplotlib', 'inline')


# <a id="1"></a>
# <p style="font-family: helvetica; letter-spacing: 1px; font-size: 20px; font-weight: bold; color:#1B2631; align:left;padding: 0px;border-bottom: 1px solid #003300"><span class="emoji">📈🔭</span>Data Overview
# </p>

# #### As per the competition, this is a fairly light-weight dataset that is synthetically generated from real-world data, and will provide an opportunity to quickly iterate through various models/feature engineering ideas. 
# #### Also, as given in the dataset description, both train & test dataset have been generated from a deep learning model trained on the Crab Age Prediction (link avaiable below). Feature distributions are close to, but not exactly the same, as the original.

# In[49]:


#train             = pd.read_csv('/kaggle/input/playground-series-s3e15/train.csv')
#test              = pd.read_csv('/kaggle/input/playground-series-s3e15/test.csv')
#sample_submission = pd.read_csv('/kaggle/input/playground-series-s3e15/sample_submission-02.csv')
train             = pd.read_csv('input/competition_playground_series/train.csv')
test              = pd.read_csv('input/competition_playground_series/test.csv')
sample_submission = pd.read_csv('input/competition_playground_series/sample_submission-02.csv')


# <div class="alert alert-block alert-info"> <b>NOTES TO THE READER</b><br> Use the link:- https://www.kaggle.com/datasets/sidhus/crab-age-prediction to take a look at the Crab Age Prediction Dataset from which our data has been obtained.</div>

# <a id="2"></a>
# <p style="font-family: helvetica; letter-spacing: 1px; font-size: 20px; font-weight: bold; color:#1B2631; align:left;padding: 0px;border-bottom: 1px solid #003300"><span class="emoji">🔍📊</span>Exploratory Data Analysis
# </p>

# In[50]:


def eda(df):
    print("==================================================================")
    print("1. Dataframe Shape: ",df.shape)
    print("==================================================================")
    print("2. Explore the Data: ")
    display(HTML(df.head(5).to_html()))
    print("==================================================================")
    print("3. Information on the Data: ")
    data_info_df                      = pd.DataFrame(df.dtypes, columns=['data type'])
    data_info_df['Duplicated_Values'] = df.duplicated().sum()
    data_info_df['Missing_Values']    = df.isnull().sum().values 
    data_info_df['%Missing']          = df.isnull().sum().values / len(df)* 100
    data_info_df['Unique_Values']     = df.nunique().values
    df_desc                           = df.describe(include='all').transpose()
    data_info_df['Count']             = df_desc['count'].values
    data_info_df['Mean']              = df_desc['mean'].values
    data_info_df['STD']               = df_desc['std'].values
    data_info_df['Min']               = df_desc['min'].values
    data_info_df['Max']               = df_desc['max'].values
    data_info_df                      = data_info_df[['Count','Mean','STD', 'Min', 'Max','Duplicated_Values','Missing_Values',
                                                     '%Missing','Unique_Values']]   
    display(HTML(data_info_df.to_html()))
    print("==================================================================")
    print("4. Correlation Matrix Heatmap - For Numeric Variables:")
    num_cols = df.select_dtypes(include = ['int64','float64']).columns.tolist()
    correlation_matrix = df[num_cols].corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, fmt='.2f')
    plt.show()
    print("==================================================================")
    print("5. Correlation with Target Variable :")
    target_corr = correlation_matrix['Age'].drop('Age')
    target_corr_sorted = target_corr.sort_values(ascending=False)
    sns.set(font_scale=0.8)
    sns.set_style("white")
    sns.set_palette("PuBuGn_d")
    sns.heatmap(target_corr_sorted.to_frame(), cmap="coolwarm", annot=True, fmt='.2f')
    plt.show()
    print("==================================================================")
    print("6. Distribution of Numerical Variables")
    for col in num_cols:
        sns.histplot(df[col], kde=True)
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.title('Distribution of {}'.format(col))
        plt.show()
    print("==================================================================")
    print("7. Distribution of Categorical Variables")
    cat_cols = df.select_dtypes(include = ['object']).columns.tolist()
    for col in cat_cols:
        value_counts = df[col].value_counts(normalize=True) * 100
        fig, ax = plt.subplots(figsize=(8, 3))
        #top_n = min(17, len(value_counts))
        #ax.barh(value_counts.index[:top_n], value_counts.values[:top_n])
        ax.barh(value_counts.index, value_counts.values)
        ax.set_xlabel('Percentage Distribution')
        ax.set_ylabel(f'{col}')
        plt.tight_layout()
        plt.show()
    print("==================================================================")


# In[51]:


train_new = train[['Sex', 'Length', 'Diameter', 'Height', 'Weight', 'Shucked Weight',
       'Viscera Weight', 'Shell Weight', 'Age']]


# In[52]:


eda(train_new)


# <a id="3"></a>
# <p style="font-family: helvetica; letter-spacing: 1px; font-size: 20px; font-weight: bold; color:#1B2631; align:left;padding: 0px;border-bottom: 1px solid #003300"><span class="emoji">📝📚</span>Analysis Summary
# </p>

# 1. There are 10 variables - 8 features, 1 target variable ('Age') & 1 primary key ('id')
# 
# 2. 9 variables are numeric and 1 - 'Sex' is categorical in nature
# 
# 3. No duplicates in the data & missing values in the dataset.
# 
# 4. Positive correlation observed between length, height, diameter & weight.
# 
# 5. Distribution of our Target Variable - 'Age' is primarily normal with no outliers.
# 
# 6. Shell Weight, as compared to the other features, has the highest correlation with our target variable.
# 
# 7. Among categorical variables, 'M' or Male has the highest distribution followed by 'I' - Indeterminate & 'F'- Female

# <a id="3"></a>
# <p style="font-family: helvetica; letter-spacing: 1px; font-size: 20px; font-weight: bold; color:#1B2631; align:left;padding: 0px;border-bottom: 1px solid #003300"><span class="emoji">✅</span>Multicollinearity Check
# </p>

# In[53]:


train_temp = train[['Sex', 'Length', 'Diameter', 'Height', 'Weight', 'Shucked Weight','Viscera Weight', 'Shell Weight']]
train_temp['Sex'] = train_temp['Sex'].map({'M':0, 'F':1, 'I':2})
vif_data = pd.DataFrame()
vif_data["feature"] = train_temp.columns
vif_data["VIF"] = [variance_inflation_factor(train_temp.values, i)
                          for i in range(len(train_temp.columns))] 
display(HTML(vif_data.to_html()))


# <div class="alert alert-block alert-info"> <b></b><br> It was evident from the Correlation Matrix and the Variance Inflation Factor (VIF) values that most of the predictor variables in this dataset are highly correlated with each other.<br><br>
# To address multicollinearity in the age prediction problem cases, you can consider the following techniques:<br><br>
# 
# 1. Feature selection: Identify and select a subset of the most important features that have the strongest relationship with the target variable. This can be done using statistical techniques (e.g., p-values, correlation analysis) or feature importance measures from models like random forests or gradient boosting machines.
# <br>
# 2. Dimensionality reduction: Apply dimensionality reduction techniques such as Principal Component Analysis (PCA) or Linear Discriminant Analysis (LDA) to transform the original set of correlated variables into a new set of uncorrelated variables (principal components or discriminant functions). These new variables capture the most important information while reducing multicollinearity.
# <br>
# 3. Combine correlated variables: Instead of using all correlated variables individually, consider creating composite variables that capture the essence of the correlated predictors. For example, you could create a single variable that represents the overall size by combining 'Length', 'Diameter', and 'Height' using a formula or a weighting scheme.
# <br>
# 4. Ridge regression: Ridge regression is a variant of linear regression that includes a regularization term to mitigate multicollinearity. It can help reduce the impact of correlated predictors by shrinking the coefficients. By penalizing large coefficient values, ridge regression encourages the model to distribute the impact of correlated variables more evenly.
# <br>
# 5. Collect more data: Increasing the sample size can sometimes alleviate multicollinearity issues by providing a more diverse and representative dataset. With more observations, the model can better estimate the relationships between predictors and the target variable, potentially reducing the impact of multicollinearity.</div>
# 

# <a id="4"></a>
# <p style="font-family: helvetica; letter-spacing: 1px; font-size: 20px; font-weight: bold; color:#1B2631; align:left;padding: 0px;border-bottom: 1px solid #003300"><span class="emoji"></span>📊Modelling
# </p>

# In[91]:


#!pip install h2o


# In[73]:


h2o.init(
    nthreads=-1,     # number of threads when launching a new H2O server
    max_mem_size=12  # in gigabytes
)


# In[74]:


train_h2o_frame = h2o.H2OFrame(train)
test_h2o_frame = h2o.H2OFrame(test)


# In[75]:


x = train_h2o_frame.columns
y = "Age"
x.remove(y)

aml = H2OAutoML(max_models=10, seed=1)
aml.train(x=x, y=y, training_frame=train_h2o_frame)


# In[80]:


lb = aml.leaderboard
lb.head(rows=lb.nrows)


# In[84]:


#preds = aml.predict(test)
preds = aml.leader.predict(test_h2o_frame)


# In[86]:


df = test_h2o_frame.cbind(preds)
df.head(2)


# In[92]:


res = df[:, ["id", "predict"]]
res.set_names(['id','Age'])
res.head(2)


# <div class="alert alert-block alert-info"> <span></span>If you use the top model on the AutoML Leaderboard just like I have, that will probably be a Stacked Ensemble and we do not yet have a function to extract feature importance for that type of model yet. (although there is a ticket open to add this).<br> Hence, The top two models are Stacked Ensembles, but the third is a GBM, so we can extract variable importance from that model.<span></span></div>

# In[94]:


# Get third model
m = h2o.get_model(lb[2,"model_id"])
m.varimp(use_pandas=True)


# In[90]:


m.varimp_plot() 


# <a id="4"></a>
# <p style="font-family: helvetica; letter-spacing: 1px; font-size: 20px; font-weight: bold; color:#1B2631; align:left;padding: 0px;border-bottom: 1px solid #003300"><span class="emoji">🔜🚀</span>Suggestive Next Steps
# </p>

#    This is a simple & plain example of analyzing a dataset and using H2O AutoML technique to predict "Age" here. 
#    However, you can choose your own model from the leaderboard or use this code to make further edits & iterations ! =))

# <div class="alert alert-block alert-info"> <span>🔄</span>This is currently a WIP (work in progress) version, check out the same notebook for my further approaches!<span>👀</span></div>
# 

# <p style="font-family:helvetica;font-size: 18px;letter-spacing:1px;color:#82E0AA; align:center;padding: 0px">
# <span class="emoji">😊</span>Thank you so much for taking the time to check out my notebook on Kaggle! <br> <span class="emoji">🚀</span>Your support means a lot to me. If you found it helpful, please consider giving it a thumbs up or leaving a comment. <br><span class="emoji">💌</span>Your encouragement and feedback are greatly appreciated. Thank you again!
# </p>
