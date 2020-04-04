#!/usr/bin/env python
# coding: utf-8

# ![image.png](attachment:image.png)

# ### Table of content

#     1. Introduction
#     2. Downloading and Installing Prerequisites
#     3. Importing python libraries
#     4. Loading Datasets
#     5. Finding country wise information on COVID-19
#     6. Global cases reported till Date
#     7. Country wise reported COVID-19 cases
#     8. Top 10 countries with COVID-19 cases
#     9. Correlation Analysis of COVID-19 cases
#     10. Visualization of COVID-19 global cases using Folium map
#     11. Visualization of COVID-19 in India
#     12. Correlation Analysis of COVID-19 cases in India
#     13. Visualization using Folium map for statewise COVID-19 cases in India
#     14. Data Preprocessing for COVID-19 cases prediction
#     15. Predictions
#         1. Prediction cases for COVID-19 worldwide
#         2. Prediction cases for COVID-19 in India
#         3. Prediction cases for COVID-19 in US
#     16. Save the predicted results into excel file

# ### Introduction

# Coronavirus is a family of viruses that can cause illness, which can vary from common cold and cough to sometimes more severe disease. Middle East Respiratory Syndrome (MERS-CoV) and Severe Acute Respiratory Syndrome (SARS-CoV) were such severe cases with the world already has faced.
# SARS-CoV-2 (n-coronavirus) is the new virus of the coronavirus family, which first discovered in 2019, which has not been identified in humans before. It is a contiguous virus which started from Wuhan in December 2019. Which later declared as Pandemic by WHO due to high rate spreads throughout the world. Currently (on date 27 March 2020), this leads to a total of 24K+ Deaths across the globe, including 16K+ deaths alone in Europe.
# Pandemic is spreading all over the world; it becomes more important to understand about this spread. This NoteBook is an effort to analyze the cumulative data of confirmed, deaths, and recovered cases over time. In this notebook, the main focus is to analyze the spread trend of this virus all over the world.

# ### Downloading and Installing Prerequisites

# In[1]:


get_ipython().system('pip install pycountry_convert ')
get_ipython().system('pip install folium')
#!wget https://raw.githubusercontent.com/tarunk04/COVID-19-CaseStudy-and-Predictions/master/models/model_deaths.h5
#!wget https://raw.githubusercontent.com/tarunk04/COVID-19-CaseStudy-and-Predictions/master/models/model_confirmed.h5
#!wget https://raw.githubusercontent.com/tarunk04/COVID-19-CaseStudy-and-Predictions/master/models/model_usa_c.h5


# ### Importing python libraries

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import ticker
import pycountry_convert as pc
import folium
get_ipython().system('pip install branca')
import branca


# In[3]:


from datetime import datetime, timedelta, date
from scipy.interpolate import make_interp_spline, BSpline
get_ipython().system('pip install plotly')
import plotly.express as px
import json, requests


# In[4]:


from keras.layers import Input, Dense, Activation, LeakyReLU
from keras import models
from keras.optimizers import RMSprop, Adam


# ### Loading Datasets

# In[5]:


df_confirmed = pd.read_csv('global_covid_confirmed_daily_updates.csv')
print('Global Covid-19 Confirmed:')
df_confirmed.head(5)


# In[6]:


df_deaths = pd.read_csv('global_covid_deaths_daily_updates.csv')
print('Global Covid-19 Deaths:')
df_deaths.head(5)


# In[7]:


ts_confirmed = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
print('Time series of Covid-19 Confirmed: ')
ts_confirmed.head(5)


# In[8]:


ts_confirmed.shape


# In[9]:


ts_deaths = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')
print('Time series of Covid-19 Deaths: ')
ts_deaths.head(5)


# In[10]:


ts_deaths.shape


# In[11]:


df_covid19 = pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/web-data/data/cases_country.csv")
print('Covid 19 cases World Dataset:')
df_covid19.head()


# ## Finding country wise information on COVID-19

# In[12]:


df_countries_cases = df_covid19.copy().drop(['Lat','Long_','Last_Update'],axis =1)


# In[13]:


df_countries_cases.head()


# In[14]:


df_countries_cases.index = df_countries_cases["Country_Region"]


# In[15]:


df_countries_cases = df_countries_cases.drop("Country_Region", axis = 1)
df_countries_cases.head()


# ### Global cases reported till Date

# Total number of Confirmed, Deaths, Recovered and Active cases across the globe

# In[16]:


globalcases_report = pd.DataFrame(df_countries_cases.sum()).transpose().style.background_gradient(cmap='Wistia',axis=1)
globalcases_report


# ### Country wise reported COVID-19 cases

# In[17]:


df_countries_cases.sort_values('Confirmed', ascending= False).style.background_gradient(cmap='Wistia')


# ## Top 10 countries with COVID-19 cases

# In[18]:


f = plt.figure(figsize=(10,5))
f.add_subplot(111)

plt.axes(axisbelow=True)
plt.barh(df_countries_cases.sort_values('Confirmed')["Confirmed"].index[-10:],df_countries_cases.sort_values('Confirmed')["Confirmed"].values[-10:],color="orange")
plt.tick_params(size=5,labelsize = 13)
plt.xlabel("Confirmed Cases",fontsize=18)
plt.title("Top 10 Countries (Confirmed Cases)",fontsize=20)
plt.grid(alpha=0.3)


# In[19]:


f = plt.figure(figsize=(10,5))
f.add_subplot(111)

plt.axes(axisbelow=True)
plt.barh(df_countries_cases.sort_values('Deaths')["Deaths"].index[-10:],df_countries_cases.sort_values('Deaths')["Deaths"].values[-10:],color="brown")
plt.tick_params(size=5,labelsize = 13)
plt.xlabel("Death Cases",fontsize=18)
plt.title("Top 10 Countries (Death Cases)",fontsize=20)
plt.grid(alpha=0.3)


# In[20]:


f = plt.figure(figsize=(10,5))
f.add_subplot(111)

plt.axes(axisbelow=True)
plt.barh(df_countries_cases.sort_values('Recovered')["Recovered"].index[-10:],df_countries_cases.sort_values('Recovered')["Recovered"].values[-10:],color="green")
plt.tick_params(size=5,labelsize = 13)
plt.xlabel("Recovered Cases",fontsize=18)
plt.title("Top 10 Countries (Recovered Cases)",fontsize=20)
plt.grid(alpha=0.3)


# In[21]:


f = plt.figure(figsize=(10,5))
f.add_subplot(111)

plt.axes(axisbelow=True)
plt.barh(df_countries_cases.sort_values('Active')["Active"].index[-10:],df_countries_cases.sort_values('Active')["Active"].values[-10:],color="Darkcyan")
plt.tick_params(size=5,labelsize = 13)
plt.xlabel("Active Cases",fontsize=18)
plt.title("Top 10 Countries (Active Cases)",fontsize=20)
plt.grid(alpha=0.3)
#plt.savefig(out+'Top 10 Countries (Confirmed Cases).png')


# ### Correlation Analysis of COVID-19 cases

# In[22]:


df_countries_cases.corr().style.background_gradient(cmap='Reds')


# ## Visualization of COVID-19 global cases using Folium map

# In[23]:


world_map = folium.Map(location=[10,0], tiles="cartodbpositron", zoom_start=2,max_zoom=6,min_zoom=2)
for i in range(0,len(ts_confirmed)):
    folium.Circle(
        location=[ts_confirmed.iloc[i]['Lat'], ts_confirmed.iloc[i]['Long']],
        tooltip = "<h5 style='text-align:center;font-weight: bold'>"+ts_confirmed.iloc[i]['Country/Region']+"</h5>"+
                    "<div style='text-align:center;'>"+str(np.nan_to_num(ts_confirmed.iloc[i]['Province/State']))+"</div>"+
                    "<hr style='margin:10px;'>"+
                    "<ul style='color: #444;list-style-type:circle;align-item:left;padding-left:20px;padding-right:20px'>"+
        "<li>Confirmed: "+str(ts_confirmed.iloc[i,-1])+"</li>"+
        "<li>Deaths:   "+str(ts_deaths.iloc[i,-1])+"</li>"+
        "<li>Mortality Rate:   "+str(np.round(ts_deaths.iloc[i,-1]/(ts_confirmed.iloc[i,-1]+1.00001)*100,2))+"</li>"+
        "</ul>"
        ,
        radius=(int((np.log(ts_confirmed.iloc[i,-1]+1.00001)))+0.2)*50000,
        color='#ff6600',
        fill_color='#ff8533',
        fill=True).add_to(world_map)

world_map


# In[24]:


temp_df = pd.DataFrame(df_countries_cases['Confirmed'])
temp_df = temp_df.reset_index()
fig = px.choropleth(temp_df, locations="Country_Region",
                    color=np.log10(temp_df.iloc[:,-1]), # lifeExp is a column of gapminder
                    hover_name="Country_Region", # column to add to hover information
                    hover_data=["Confirmed"],
                    color_continuous_scale=px.colors.sequential.Plasma,locationmode="country names")
fig.update_geos(fitbounds="locations", visible=False)
fig.update_layout(title_text="Confirmed Cases across globe")
fig.update_coloraxes(colorbar_title="Confirmed Cases across globe",colorscale="Reds")
# fig.to_image("Global Heat Map confirmed.png")
fig.show()


# In[25]:


temp_df = pd.DataFrame(df_countries_cases['Deaths'])
temp_df = temp_df.reset_index()
fig = px.choropleth(temp_df, locations="Country_Region",
                    color=np.log10(temp_df.iloc[:,-1]), # lifeExp is a column of gapminder
                    hover_name="Country_Region", # column to add to hover information
                    hover_data=["Deaths"],
                    color_continuous_scale=px.colors.sequential.Plasma,locationmode="country names")
fig.update_geos(fitbounds="locations", visible=False)
fig.update_layout(title_text="Death Cases across globe")
fig.update_coloraxes(colorbar_title="Death Cases across globe",colorscale="Reds")
# fig.to_image("Global Heat Map confirmed.png")
fig.show()


# ## Visualization of COVID-19 in India

# In[26]:


india_data_json = requests.get('https://api.rootnet.in/covid19-in/unofficial/covid19india.org/statewise').json()
df_india_cases = pd.io.json.json_normalize(india_data_json['data']['statewise'])
df_india_cases = df_india_cases.set_index("state")
df_india_cases.head()


# In[27]:


total = df_india_cases.sum()
total.name = "Total"
pd.DataFrame(total).transpose().style.background_gradient(cmap='Wistia',axis=1)


# In[28]:


df_india_cases.style.background_gradient(cmap='Wistia')


# In[29]:


f = plt.figure(figsize=(10,5))
f.add_subplot(111)

plt.axes(axisbelow=True)
plt.barh(df_india_cases.sort_values('confirmed')["confirmed"].index[-10:],df_india_cases.sort_values('confirmed')["confirmed"].values[-10:],color="orange")
plt.tick_params(size=5,labelsize = 13)
plt.xlabel("Confirmed Cases",fontsize=18)
plt.title("Top 10 States in India (Confirmed Cases)",fontsize=20)
plt.grid(alpha=0.3)
#plt.savefig(out+'Top 10 States_India (Confirmed Cases).png')


# In[30]:


f = plt.figure(figsize=(10,5))
f.add_subplot(111)

plt.axes(axisbelow=True)
plt.barh(df_india_cases.sort_values('deaths')["deaths"].index[-10:],df_india_cases.sort_values('deaths')["deaths"].values[-10:],color="brown")
plt.tick_params(size=5,labelsize = 13)
plt.xlabel("Death Cases",fontsize=18)
plt.title("Top 10 States in India (Deaths Cases)",fontsize=20)
plt.grid(alpha=0.3)
#plt.savefig(out+'Top 10 States_India (Confirmed Cases).png')


# ### Correlation Analysis of COVID-19 cases in India

# In[31]:


df_india_cases.corr().style.background_gradient(cmap='Reds')


# ### Visualization using Folium map for statewise COVID-19 cases in India

# In[32]:


# Adding states geolocation(Latitude,Longitude) data to India dataset
geolocations = {
    "Kerala" : [10.8505,76.2711],
    "Maharashtra" : [19.7515,75.7139],
    "Karnataka": [15.3173,75.7139],
    "Telangana": [18.1124,79.0193],
    "Uttar Pradesh": [26.8467,80.9462],
    "Rajasthan": [27.0238,74.2179],
    "Gujarat":[22.2587,71.1924],
    "Delhi" : [28.7041,77.1025],
    "Punjab":[31.1471,75.3412],
    "Tamil Nadu": [11.1271,78.6569],
    "Haryana": [29.0588,76.0856],
    "Madhya Pradesh":[22.9734,78.6569],
    "Jammu and Kashmir":[33.7782,76.5762],
    "Ladakh": [34.1526,77.5770],
    "Andhra Pradesh":[15.9129,79.7400],
    "West Bengal": [22.9868,87.8550],
    "Bihar": [25.0961,85.3131],
    "Chhattisgarh":[21.2787,81.8661],
    "Chandigarh":[30.7333,76.7794],
    "Uttarakhand":[30.0668,79.0193],
    "Himachal Pradesh":[31.1048,77.1734],
    "Goa": [15.2993,74.1240],
    "Odisha":[20.9517,85.0985],
    "Andaman and Nicobar Islands": [11.7401,92.6586],
    "Puducherry":[11.9416,79.8083],
    "Manipur":[24.6637,93.9063],
    "Mizoram":[23.1645,92.9376],
    "Assam":[26.2006,92.9376],
    "Meghalaya":[25.4670,91.3662],
    "Tripura":[23.9408,91.9882],
    "Arunachal Pradesh":[28.2180,94.7278],
    "Jharkhand" : [23.6102,85.2799],
    "Nagaland": [26.1584,94.5624],
    "Sikkim": [27.5330,88.5122],
    "Dadra and Nagar Haveli":[20.1809,73.0169],
    "Lakshadweep":[10.5667,72.6417],
    "Daman and Diu":[20.4283,72.8397]    
}
df_india_cases["Lat"] = ""
df_india_cases["Long"] = ""
for index in df_india_cases.index :
    df_india_cases.loc[df_india_cases.index == index,"Lat"] = geolocations[index][0]
    df_india_cases.loc[df_india_cases.index == index,"Long"] = geolocations[index][1]


# In[33]:


# url = "https://raw.githubusercontent.com/Subhash9325/GeoJson-Data-of-Indian-States/master/Indian_States"
# state_json = requests.get(url).json()
india = folium.Map(location=[23,80], zoom_start=4,max_zoom=10, tiles='OpenStreetMap', attr= "INDIA" ,min_zoom=4,height=500,width="80%")
for i in range(0,len(df_india_cases[df_india_cases['confirmed']>0].index)):
    folium.Circle(
        location=[df_india_cases.iloc[i]['Lat'], df_india_cases.iloc[i]['Long']],
        tooltip = "<h5 style='text-align:center;font-weight: bold'>"+df_india_cases.iloc[i].name+"</h5>"+
                    "<hr style='margin:10px;'>"+
                    "<ul style='color: #444;list-style-type:circle;align-item:left;padding-left:20px;padding-right:20px'>"+
        "<li>Confirmed: "+str(df_india_cases.iloc[i]['confirmed'])+"</li>"+
        "<li>Active:   "+str(df_india_cases.iloc[i]['active'])+"</li>"+
        "<li>Recovered:   "+str(df_india_cases.iloc[i]['recovered'])+"</li>"+
        "<li>Deaths:   "+str(df_india_cases.iloc[i]['deaths'])+"</li>"+
        
        "<li>Mortality Rate:   "+str(np.round(df_india_cases.iloc[i]['deaths']/(df_india_cases.iloc[i]['confirmed']+1)*100,2))+"</li>"+
        "</ul>"
        ,
        radius=(int(np.log2(df_india_cases.iloc[i]['confirmed']+1)))*15000,
        color='#ff6600',
        fill_color='#ff8533',
        fill=True).add_to(india)

india


# ### Data Preprocessing for COVID-19 cases prediction

# In[34]:


grp_confirm_data_df = ts_confirmed.groupby('Country/Region', as_index=False).sum()
grp_confirm_data_df.reset_index(inplace=True, drop=True)
grp_confirm_data_df.shape


# In[35]:


grp_death_data_df = ts_deaths.groupby('Country/Region', as_index=False).sum()
grp_death_data_df.reset_index(inplace=True, drop=True)
grp_death_data_df.shape


# In[36]:


grp_confirm_data_df.head()


# In[37]:


grp_death_data_df.head()


# In[38]:


country_df = grp_confirm_data_df["Country/Region"]
country_df.shape


# # Predictions

# ## 1. Prediction cases for COVID-19 worldwide

# ### Use Moving Average and ExponentiallyWeightedMovingAverage models¶

# In[39]:


def cal_moving_avg(input_df, window=2):
    """Calculates Moving avg using the window size."""
    _df = input_df.rolling(window, axis=1).median()['4/1/20']
    
    return _df

def cal_ewma(input_df, comm=0.3):
    """Calculates the exp weighted moving average using the window size."""
    _df = input_df.ewm(com=comm).mean()['4/1/20']
    
    return _df


# In[40]:


all_pred_df = pd.DataFrame()
all_pred_df_ew = pd.DataFrame()

for df in [grp_confirm_data_df, grp_death_data_df]:
    _df = cal_moving_avg(df, window=2)
    ew_df = cal_ewma(df, comm=0.4)
    all_pred_df = pd.concat([all_pred_df, _df], axis=1)
    all_pred_df_ew = pd.concat([all_pred_df_ew, ew_df], axis=1)
    
all_pred_df.columns =['Confirmed', 'Deaths']    
all_pred_df_ew.columns =['Confirmed', 'Deaths']   


# In[41]:


print('\n')
print('COVID-19 cases worldwide till 1/4/2020 using MA:')
all_pred_df


# In[42]:


print('\n')
print('COVID-19 cases worldwide till 1/4/2020 using EWMA:')
all_pred_df_ew


# In[43]:


pred_1_df = pd.merge(country_df, all_pred_df, left_index= True, right_index = True)
pred_1_df.head()


# In[44]:


pred_2_df = pd.merge(country_df, all_pred_df_ew, left_index= True, right_index = True)
pred_2_df.head()


# ### 2. Prediction cases for COVID-19 in India

# In[45]:


ind_conf = grp_confirm_data_df[grp_confirm_data_df['Country/Region'] == 'India']
ind_conf


# In[46]:


ind_deaths = grp_death_data_df[grp_death_data_df['Country/Region'] == 'India']
ind_deaths


# #### Use Moving Average and EWMA models¶

# In[47]:


def cal_moving_avg(input_df, window=2):
    """Calculates Moving avg using the window size."""
    _df = input_df.rolling(window, axis=1).median()['4/1/20']
    
    return _df

def cal_ewma(input_df, comm=0.3):
    """Calculates the exp weighted moving average using the window size."""
    _df = input_df.ewm(com=comm).mean()['4/1/20']
    
    return _df


# In[48]:


ind_all_pred_df = pd.DataFrame()
ind_all_pred_df_ew = pd.DataFrame()

for df in [ind_conf, ind_deaths]:
    _df = cal_moving_avg(df, window=2)
    ew_df = cal_ewma(df, comm=0.4)
    ind_all_pred_df = pd.concat([ind_all_pred_df, _df], axis=1)
    ind_all_pred_df_ew = pd.concat([ind_all_pred_df_ew, ew_df], axis=1)
    
ind_all_pred_df.columns =['Confirmed', 'Deaths']    
ind_all_pred_df_ew.columns =['Confirmed', 'Deaths'] 


# In[49]:


print('\n')
print('COVID-19 cases in India till 1/4/2020 using MA:')
ind_all_pred_df


# In[50]:


print('\n')
print('COVID-19 cases in US till 1/4/2020 using EWMA:')
ind_all_pred_df_ew


# ### 3. Prediction cases for COVID-19 in US

# In[51]:


us_conf = grp_confirm_data_df[grp_confirm_data_df['Country/Region'] == 'US']
us_conf


# In[52]:


us_deaths = grp_death_data_df[grp_death_data_df['Country/Region'] == 'US']
us_deaths


# ### Use Moving Average and EWMA models

# In[53]:


def cal_moving_avg(input_df, window=2):
    """Calculates Moving avg using the window size."""
    _df = input_df.rolling(window, axis=1).median()['4/1/20']
    
    return _df

def cal_ewma(input_df, comm=0.3):
    """Calculates the exp wighted moving average using the window size."""
    _df = input_df.ewm(com=comm).mean()['4/1/20']
    
    return _df


# In[54]:


us_all_pred_df = pd.DataFrame()
us_all_pred_df_ew = pd.DataFrame()

for df in [us_conf, us_deaths]:
    _df = cal_moving_avg(df, window=2)
    ew_df = cal_ewma(df, comm=0.4)
    us_all_pred_df = pd.concat([us_all_pred_df, _df], axis=1)
    us_all_pred_df_ew = pd.concat([us_all_pred_df_ew, ew_df], axis=1)
    
us_all_pred_df.columns =['Confirmed', 'Deaths']    
us_all_pred_df_ew.columns =['Confirmed', 'Deaths']    


# In[55]:


print('\n')
print('COVID-19 cases in US till 1/4/2020 using MA:')
us_all_pred_df


# In[56]:


print('\n')
print('COVID-19 cases in US till 1/4/2020 using EWMA:')
us_all_pred_df_ew


# ## Save the predicted results into excel file

# In[57]:


get_ipython().system('pip install openpyxl')
pred_1_df.to_excel("Submission_COVID19_Globalcases_1.xlsx", index = False)
pred_2_df.to_excel("Submission_COVID19_Globalcases_2.xlsx", index = False)


# In[ ]:




