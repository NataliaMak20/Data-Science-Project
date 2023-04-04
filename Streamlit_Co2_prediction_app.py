# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 11:09:45 2023

@author: A.Miri
"""
import numpy as np
from sklearn.metrics import mean_squared_error
import seaborn as sns
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, cross_val_predict
from sklearn import metrics
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from numpy import array 
from numpy import argmax 
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score
import seaborn as sns
from sklearn.svm import SVR
from sklearn.linear_model import Lasso, Ridge
import streamlit as st





st.title('Predicting Co2 Emission by Vehicles')


dataset_name = ('Co2 emission')



classifier_name = st.sidebar.selectbox(
    'Select a Regression Model',
    ('Random Forest','SVR' )
)
nav = st.sidebar.radio("Navigation Menu",["Purpose", "Data & Modelling", "Results"])
df = pd.read_csv("C:/Users/Admin/Downloads/CO2 Emissions_Canada.csv")

X = df[['Engine Size(L)', 'Cylinders', 'Fuel Consumption City (L/100 km)','Fuel Consumption Hwy (L/100 km)', 'Fuel Consumption Comb (L/100 km)']]
y = df['CO2 Emissions(g/km)']


def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == 'SVR':
        
        params['C'] = 1e3
        params['kernel']='rbf'
        params['gamma']= 'auto'        
    else:
        
        params['bootstrap'] = False
        params['max_features']=3
        params['n_estimators']=10
    return params

params = add_parameter_ui(classifier_name)

def get_classifier(clf_name, params):
    clf = None
    if clf_name == 'SVR':
        clf = SVR(C=params['C'],kernel=params['kernel'],gamma= params['gamma'])
    
    else:
        clf = RandomForestRegressor(bootstrap=False, max_features=3, n_estimators=10,
            random_state=42)
    return clf

clf = get_classifier(classifier_name, params)

#Identify the categorical features
cat_features = ['Fuel Type']
onehot_encoder = OneHotEncoder()
onehot_encoder.fit(df[cat_features])
encoded_features = onehot_encoder.transform(df[cat_features]).toarray()
processed_data = np.concatenate((df, encoded_features), axis=1)


# In[5]:


# Create a list of new column names for the encoded features
new_cols = onehot_encoder.get_feature_names(cat_features)


# In[6]:


# Convert the encoded features array to a dataframe
encoded_features_df = pd.DataFrame(encoded_features, columns=new_cols)


# In[7]:


numeric_df = pd.concat([encoded_features_df, X], axis=1, join='inner')

#### CLASSIFICATION ####

X_train, X_test, y_train, y_test = train_test_split(numeric_df, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float32))
X_test_scaled = scaler.transform(X_test.astype(np.float32))

clf.fit(X_train_scaled, y_train)
y_pred = clf.predict(X_train_scaled)

rmse = mean_squared_error(y_train, y_pred, squared=False)
score_r2 = r2_score(y_train, y_pred)



if nav == "Purpose":
    st.write("""
    #### Explore different regression models on Co2 emission from Canada

    """)
    st.header("Purpose")
    st.write("""* Utilising a Machine Learning approach to address the expected CO2 emissions resulting from vehicles.""")
    st.write("* The dataset we've used to predict the Co2 emission is from Canada in 2017.")
    st.write("* The purpose of the project is to help car manufacturers in predicting the amount of CO2 their cars will emit by utilising several factors such as engine size, number of cylinders, fuel consumption rates. This will aid the manufacturers in meeting future CO2 standards set globally, ensuring their cars are environmentally acceptable.")
    st.write("* In this presentation we've chosen to only show our top two performing models which is randomforest and SVR because they have the lowest rmse score of all the five models we train.")    
    
    st.write("""###### By: Amir Anissian, Emil Jesperssen, Natalia Makarova""")




if nav == "Data & Modelling":
    dataset_name = ('Co2 emission')

    st.write(f"## {dataset_name} Dataset")
    st.write('Shape of data:', X.shape)
    st.write('number of target:', y.shape)
    st.write('Shape of numeric data:', numeric_df.shape)
    numeric_df
    st.write(f'model = {classifier_name}')
    
    sns.catplot(data=df, x="Cylinders", y="CO2 Emissions(g/km)", hue="Fuel Type", kind="swarm",alpha=0.5)
    plt.title("CO2 Emissions by Cylinders and Fuel Type")
    plt.xlabel("Cylinders")
    plt.ylabel("CO2 Emissions (g/km)")
    plt.tight_layout()  # optional, to adjust the layout of the plot
    co2_emission_by_c_ft = plt.gcf()
    plt.show()  # add this line to display the plot in Streamlit
    st.pyplot(co2_emission_by_c_ft)
    
    

if nav == "Results":
    
    st.write(f'model = {classifier_name}')
    st.write('rmse =', rmse)
    st.write('R^2 =', score_r2)


    #### PLOT DATASET ####
    # Project the data onto the 2 primary principal components
        
    #Here we make the same plot with y_pred
    data_1 = y_pred
    data_2 = y_train
    data = [data_1, data_2]
    
    fig = plt.figure(figsize =(10, 7))
    ax = fig.add_subplot(111)
     
    # Creating axes instance
    bp = ax.boxplot(data, patch_artist = True,
                    notch ='True', vert = 0)
    colors = ['#0000FF', '#00FF00']
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        
    for whisker in bp['whiskers']:
        whisker.set(color ='#8B008B',
                    linewidth = 1.5,
                    linestyle =":")
    for cap in bp['caps']:
        cap.set(color ='#8B008B',
                linewidth = 2)
    
    for median in bp['medians']:
        median.set(color ='red',
                   linewidth = 3)
        
    for flier in bp['fliers']:
        flier.set(marker ='D',
                  color ='#e7298a',
                  alpha = 0.5)     
    # x-axis labels
    ax.set_yticklabels(['Predictions', 'True Value'])
     
    # Adding title
    plt.title("Boxplot of Predictions and True Values")
     
    # Removing top axes and right axes
    # ticks
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    
    
    # show plot
    st.pyplot(fig)
    
    
    plt.hist(y_pred, label='Predictions',)
    histogram = plt.gcf()
    plt.hist(y_train, alpha=0.5, label='True Values')
    plt.title("Prediction VS True values")
    plt.legend(loc='upper right')
    st.pyplot(histogram)
    
    
    