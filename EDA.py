# %% [code]
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %% [markdown]
# # **Store Sales - Time Series Forecasting**

# %% [markdown]
# # About the data

# %% [markdown]
# **<code>Train data</code>** The training data, comprising time series of features store_nbr, family, and onpromotion as well as the target sales.store_nbr identifies the store at which the products are sold.family identifies the type of product sold.sales gives the total sales for a product family at a particular store at a given date. Fractional values are possible since products can be sold in fractional units (1.5 kg of cheese, for instance, as opposed to 1 bag of chips).onpromotion gives the total number of items in a product family that were being promoted at a store at a given date.
# 
# **<code>test data</code>**
# The test data, having the same features as the training data. You will predict the target sales for the dates in this file.The dates in the test data are for the 15 days after the last date in the training data.
# 
# **<code>stores data</code>**
# Store metadata, including city, state, type, and cluster.
# cluster is a grouping of similar stores.
# 
# **<code>oil data</code>**
# Daily oil price. Includes values during both the train and test data timeframes. (Ecuador is an oil-dependent country and it's economical health is highly vulnerable to shocks in oil prices.)
# 
# **<code>holidays_events data</code>**
# Holidays and Events, with metadata
# NOTE: Pay special attention to the transferred column. A holiday that is transferred officially falls on that calendar day, but was moved to another date by the government. A transferred day is more like a normal day than a holiday. To find the day that it was actually celebrated, look for the corresponding row where type is Transfer. For example, the holiday Independencia de Guayaquil was transferred from 2012-10-09 to 2012-10-12, which means it was celebrated on 2012-10-12. Days that are type Bridge are extra days that are added to a holiday (e.g., to extend the break across a long weekend). These are frequently made up by the type Work Day which is a day not normally scheduled for work (e.g., Saturday) that is meant to payback the Bridge.Additional holidays are days added a regular calendar holiday, for example, as typically happens around Christmas (making Christmas Eve a holiday).

# %% [markdown]
# # 1. Modules

# %% [code] {"execution":{"iopub.status.busy":"2024-08-01T15:25:34.110249Z","iopub.execute_input":"2024-08-01T15:25:34.110659Z","iopub.status.idle":"2024-08-01T15:25:50.552278Z","shell.execute_reply.started":"2024-08-01T15:25:34.110626Z","shell.execute_reply":"2024-08-01T15:25:50.551052Z"}}
pip install patchworklib


# %% [code] {"execution":{"iopub.status.busy":"2024-08-01T15:25:50.554491Z","iopub.execute_input":"2024-08-01T15:25:50.554868Z","iopub.status.idle":"2024-08-01T15:26:04.042428Z","shell.execute_reply.started":"2024-08-01T15:25:50.554835Z","shell.execute_reply":"2024-08-01T15:26:04.040829Z"}}
pip install scikit-misc


# %% [code] {"execution":{"iopub.status.busy":"2024-08-01T15:26:04.044259Z","iopub.execute_input":"2024-08-01T15:26:04.044687Z","iopub.status.idle":"2024-08-01T15:26:17.196176Z","shell.execute_reply.started":"2024-08-01T15:26:04.044652Z","shell.execute_reply":"2024-08-01T15:26:17.194680Z"}}
pip install joypy

# %% [markdown]
# # 2. Pacakages

# %% [code] {"execution":{"iopub.status.busy":"2024-08-01T15:26:17.199432Z","iopub.execute_input":"2024-08-01T15:26:17.200574Z","iopub.status.idle":"2024-08-01T15:26:19.180854Z","shell.execute_reply.started":"2024-08-01T15:26:17.200514Z","shell.execute_reply":"2024-08-01T15:26:19.179619Z"}}
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sn
import warnings
warnings.filterwarnings("ignore")
%matplotlib inline
import plotnine as p9
import plotly.express as px
import patchworklib as pw
pd.set_option('display.max_rows', 100)
from joypy import joyplot


# %% [markdown]
# # 3. Importing Data

# %% [code] {"execution":{"iopub.status.busy":"2024-08-01T15:26:19.182607Z","iopub.execute_input":"2024-08-01T15:26:19.183333Z","iopub.status.idle":"2024-08-01T15:26:23.465869Z","shell.execute_reply.started":"2024-08-01T15:26:19.183288Z","shell.execute_reply":"2024-08-01T15:26:23.464615Z"}}
#Read available data sets
train = pd.read_csv(r"/kaggle/input/store-sales-time-series-forecasting/train.csv")
test = pd.read_csv(r"/kaggle/input/store-sales-time-series-forecasting/test.csv")
stores = pd.read_csv(r"/kaggle/input/store-sales-time-series-forecasting/stores.csv")
transaction = pd.read_csv(r"/kaggle/input/store-sales-time-series-forecasting/transactions.csv")
holidays_events = pd.read_csv(r"/kaggle/input/store-sales-time-series-forecasting/holidays_events.csv")
oil = pd.read_csv(r"/kaggle/input/store-sales-time-series-forecasting/oil.csv")

train['date']=pd.to_datetime(train['date'])
train['year']=train['date'].dt.year
train['month']=train['date'].dt.month
train['weekday']=train['date'].dt.weekday

from plotnine import (
    ggplot,
    aes,
    after_stat,
    geom_density,
    geom_histogram,
    geom_vline,
    geom_rect,
    labs,
    annotate,
    theme_tufte,
    geom_freqpoly,
    coord_cartesian,
    geom_text,
    theme,
    geom_bar,
    scale_fill_manual,
    geom_line,
    scale_y_log10
    
)

train['store_nbr']=train['store_nbr'].astype('category')
train['family']=train['family'].astype('category')


# %% [markdown]
# # 4. Sales

# %% [code] {"execution":{"iopub.status.busy":"2024-08-01T15:26:23.467443Z","iopub.execute_input":"2024-08-01T15:26:23.467947Z"}}

g=pd.DataFrame(train.groupby(['date','year','month'])['sales'].sum()).reset_index()

p1 = \
(
    ggplot(g,aes(x='date',y='sales'))+
    geom_line()+
    p9.scale_y_log10()+
    p9.theme(axis_text_x=p9.element_text(angle=90))
 
)

g=pd.DataFrame(train['onpromotion'].apply(lambda z:'True' if z!=0 else 'False'))

p2= \
(ggplot(g,aes('onpromotion',fill='onpromotion'))
 + p9.geom_bar()
 + geom_text(
     aes(label=after_stat('count')),
     stat='count',
     nudge_x=-0.14,
     nudge_y=0.125,
     va='bottom'
 )
 + geom_text(
     aes(label=after_stat('prop*100'), group=1),
     stat='count',
     nudge_x=0.14,
     nudge_y=0.125,
     va='bottom',
     format_string='({:.1f}%)'
 )
 + p9.theme(legend_position='none')
 + p9.scale_y_log10()
 
)

g=pd.DataFrame(train[train.sales>0]['sales'],columns=['sales'])

p3 = \
(ggplot(train,aes('sales'))
+ p9.geom_histogram(fill="red",binwidth=0.1)
+ p9.scale_x_log10()
+ p9.scale_y_log10() 
+ labs(x = "Positive unit sales")
)

p4 = \
(ggplot(train,aes('store_nbr',fill='store_nbr'))
 + geom_bar()
 + p9.theme(legend_position='none',axis_text_x=p9.element_text(angle= 90))
 
)

p5 = \
(ggplot(train,aes('family',fill='family'))
 + geom_bar()
 + p9.theme(legend_position='none',axis_text_x=p9.element_text(angle=90))
 
)


p1=pw.load_ggplot(p1,figsize=(3,3))
p2=pw.load_ggplot(p2,figsize=(3,3))
p3=pw.load_ggplot(p3,figsize=(3,3))
p4=pw.load_ggplot(p4,figsize=(5,1))
p5=pw.load_ggplot(p5,figsize=(5,1))

p12345=(p1|p2|p3)/p4/p5
p12345.savefig()

# %% [markdown]
# We find:
# 
# * There is an increasing trend over time, indicating rising sales over time different units that are being sold..
# 
# * Only a small fraction of items are on promotion; for longer fraction this information is not known. i.e the majority of items are not on promotion.
# 
# * The distribution of *store_nbr* and *family* shows uniformity. This means that the data per family and store_nbr is equal without any small number of entries.

# %% [markdown]
# The graph below shows the monthly aggregated sales by store_nbr

# %% [code] {"execution":{"iopub.status.idle":"2024-08-01T15:30:50.805461Z","shell.execute_reply.started":"2024-08-01T15:30:33.941650Z","shell.execute_reply":"2024-08-01T15:30:50.803656Z"}}
g=pd.DataFrame(train.groupby(['date','store_nbr','month'])['sales'].sum()).reset_index()
px.line(g, x="date", y="sales", title='Monthly aggregated sales',color='store_nbr')
 

# %% [markdown]
# We find:
# 
# * If you look at the time series of the stores one by one , we realize there are some unnecessary rows in the data.If you select the stores from above , some of them have no sales at the beginning of 2013. Those stores are 20,21,22,29,36,42,52 and 53.
# 

# %% [code]


# %% [code] {"execution":{"iopub.status.busy":"2024-08-01T15:30:50.807769Z","iopub.execute_input":"2024-08-01T15:30:50.808187Z","iopub.status.idle":"2024-08-01T15:30:53.287213Z","shell.execute_reply.started":"2024-08-01T15:30:50.808152Z","shell.execute_reply":"2024-08-01T15:30:53.286108Z"}}
p=train[train['sales']>0]
p=pd.DataFrame(p.groupby(['store_nbr','family']).agg({'date':['min','max']}))
p.columns=p.columns.droplevel(0)

p['min']=pd.to_datetime(p['min'])
p['max']=pd.to_datetime(p['max'])


p['minmonth']=p['min'].dt.month
p['minyear']=p['min'].dt.year
p['maxmonth']=p['max'].dt.month
p['maxyear']=p['max'].dt.year

min=pd.DataFrame(p.groupby(['minmonth','minyear'])['minmonth'].value_counts()).reset_index()

max=pd.DataFrame(p.groupby(['maxmonth','maxyear'])['maxmonth'].value_counts()).reset_index()

p1 = \
(
    ggplot(min,aes('minmonth','minyear',fill='count'))+
    p9.geom_tile()+    
    p9.scale_y_reverse() +
    p9.scale_fill_distiller(palette =1, trans = "log10") +
    p9.theme_tufte() +
    theme(legend_position = "bottom")+  
    labs(x = "", y = "", fill = "# First Non-Zero Entries", title = "Effective start of sample time series",
    subtitle = "Counting the Year & Month of the first non-zero sales entry. Colour scale is logarithmic.")+
    geom_text(aes(label='count'))
)

p2 = \
(
    ggplot(max,aes('maxmonth','maxyear',fill='count'))+
    p9.geom_tile()+    
    p9.scale_y_reverse() +
    p9.scale_fill_distiller(palette =1, trans = "log10") +
    p9.theme_tufte() +
    theme(legend_position = "bottom")+  
    labs(x = "", y = "", fill = "# Last Non-Zero Entries", title = "Effective end of sample time series",
    subtitle = "Counting the Year & Month of the last non-zero sales entry. Colour scale is logarithmic.")+
    geom_text(aes(label='count'))
)


p1=pw.load_ggplot(p1,figsize=(7,3))
p2=pw.load_ggplot(p2,figsize=(7,3))
p12=(p1|p2)
p12.savefig()


# %% [markdown]
# We find:
# 
# * Figure 1 shows the effective start of all time series. Out of 1729, 1482(85%) of time series  starts before 2015.
# 
# * Similarly ,Figure 2 shows the end of these time series. Only 72 time series end;rather have data upto 2017 july ,rest of them end in August. It is not neccessarily true that the time series in 2017 has ended due to no sales, several reasons may affect the sales , such as out of stock items , natural disasters  etc.

# %% [markdown]
# 

# %% [code] {"execution":{"iopub.status.busy":"2024-08-01T15:30:53.289290Z","iopub.execute_input":"2024-08-01T15:30:53.290207Z","iopub.status.idle":"2024-08-01T15:31:09.281715Z","shell.execute_reply.started":"2024-08-01T15:30:53.290160Z","shell.execute_reply":"2024-08-01T15:31:09.280495Z"}}
g=train[train['sales']>0]
#g=g.reindex(columns=['store_nbr','sales'])
g['median']=g['store_nbr'].copy()
f=g.groupby(['store_nbr'])['sales'].median()
g['median']=g['median'].map(f)
g['store_nbr']=g['store_nbr'].astype('category')

(
    ggplot(g,aes(x='reorder(store_nbr,median)',y='sales',color='store_nbr'))+
    p9.geom_boxplot()+
    scale_y_log10() +
    theme(legend_position = "none") +
    labs(x = "Store number (reordered)")+
    p9.theme(legend_position='none',axis_text_x=p9.element_text(angle=90))
)

# %% [markdown]
# We find:
# 
# * The difference in sales appear to be small in size

# %% [code] {"execution":{"iopub.status.busy":"2024-08-01T15:31:09.286091Z","iopub.execute_input":"2024-08-01T15:31:09.286482Z","iopub.status.idle":"2024-08-01T15:31:09.466648Z","shell.execute_reply.started":"2024-08-01T15:31:09.286449Z","shell.execute_reply":"2024-08-01T15:31:09.465081Z"}}
g=train.groupby(['weekday','month'])['sales'].mean().reset_index()

# %% [code] {"execution":{"iopub.status.busy":"2024-08-01T15:31:09.468012Z","iopub.execute_input":"2024-08-01T15:31:09.468368Z","iopub.status.idle":"2024-08-01T15:31:09.885628Z","shell.execute_reply.started":"2024-08-01T15:31:09.468329Z","shell.execute_reply":"2024-08-01T15:31:09.884423Z"}}
(
    ggplot(g,aes('month','weekday',fill='sales'))+
    p9.geom_tile() +
    labs(x = "Month of the year", y = "Day of the week") 
    #+p9.scale_fill_distiller(palette = "Spectral")

    
)

# %% [markdown]
# We find:
# 
# * The weekends have significantly higher average sales
# 
# * Thrusdays in particular have consistantly lower sales
#  
# * The higher level for December is most likely due to Christmas

# %% [markdown]
# # 5. Family Product

# %% [code] {"execution":{"iopub.status.busy":"2024-08-01T15:31:09.886976Z","iopub.execute_input":"2024-08-01T15:31:09.888241Z","iopub.status.idle":"2024-08-01T15:31:18.997894Z","shell.execute_reply.started":"2024-08-01T15:31:09.888192Z","shell.execute_reply":"2024-08-01T15:31:18.996667Z"}}
g=train.groupby(['date','family','month'])['sales'].sum().reset_index()
px.line(g, x="date", y="sales", title='Monthly aggregated sales',color='family')


# %% [markdown]
# We find:
#     
# * Some series are inconsistent through time line
# 
# * The above figure shows the monthly aggregated values for Family
# 
# * Some family series starts after certain time and not from start. This will likely explain the No sales figure in stores  

# %% [markdown]
# for store in g.store_nbr.unique():
#     g.loc[g['store_nbr']==store,'norm']=g[g['store_nbr']==store]['sales']/g[g['store_nbr']==store]['sales'].max()
# 
# px.line(g, x="family", y="norm", title='Monthly aggregated sales',color='store_nbr')
# 

# %% [markdown]
# We find:
#     
# * Above figure disaplays normalized  sales througout stroe_nbr

# %% [code] {"execution":{"iopub.status.busy":"2024-08-01T15:31:18.999396Z","iopub.execute_input":"2024-08-01T15:31:19.000309Z","iopub.status.idle":"2024-08-01T15:31:19.200211Z","shell.execute_reply.started":"2024-08-01T15:31:19.000265Z","shell.execute_reply":"2024-08-01T15:31:19.199095Z"}}
c = train.groupby(["store_nbr", "family"]).sales.sum().reset_index().sort_values(["family","store_nbr"])
c = c[c.sales == 0]
c

# %% [markdown]
# We find:
# 
# * The above family has almost no sales. These examples are rare ,However some product families depends on seasonality. Some of them might not be active for last 90 days , doesnt mean they are passive

# %% [code] {"execution":{"iopub.status.busy":"2024-08-01T15:31:19.202193Z","iopub.execute_input":"2024-08-01T15:31:19.203102Z","iopub.status.idle":"2024-08-01T15:31:19.504247Z","shell.execute_reply.started":"2024-08-01T15:31:19.203052Z","shell.execute_reply":"2024-08-01T15:31:19.503048Z"}}
a = train.groupby("family").sales.mean().sort_values(ascending = False).reset_index()
px.bar(a, y = "family", x="sales", color = "family", title = "Which product family preferred more?")

# %% [markdown]
# We find:
# 
# * Grocery and Beverages are top sellling products

# %% [markdown]
# # 6. Store

# %% [code] {"execution":{"iopub.status.busy":"2024-08-01T15:31:19.505526Z","iopub.execute_input":"2024-08-01T15:31:19.505905Z","iopub.status.idle":"2024-08-01T15:31:25.840322Z","shell.execute_reply.started":"2024-08-01T15:31:19.505877Z","shell.execute_reply":"2024-08-01T15:31:25.839105Z"}}
stores=stores.astype('category')
p1 = \
(
    ggplot(stores,aes(x='city',fill='city'))+
    geom_bar()+
    p9.theme(legend_position='none',axis_text_x=p9.element_text(angle=90))

)

p2 = \
(
    ggplot(stores,aes(x='state',fill='state'))+
    geom_bar()+
    p9.theme(legend_position='none',axis_text_x=p9.element_text(angle=90))

)

p3 = \
(
    ggplot(stores,aes(x='type',fill='type'))+
    geom_bar()+
    p9.theme(legend_position='none',axis_text_x=p9.element_text(angle=90))+
     geom_text(
     aes(label=after_stat('count')),
     stat='count',
     va='bottom'
 )    
)


p4 = \
(
    ggplot(stores,aes(x='cluster',fill='cluster'))+
    geom_bar()+
    p9.theme(legend_position='none',axis_text_x=p9.element_text(angle=90))

)

p1=pw.load_ggplot(p1,figsize=(4,2))
p2=pw.load_ggplot(p2,figsize=(4,2))
p3=pw.load_ggplot(p3,figsize=(4,5))
p4=pw.load_ggplot(p4,figsize=(4,2))


p1234=(p1|p2)/(p3|p4)
p1234.savefig()

# %% [markdown]
# We find:
# 
# * The cities fall into certain groups, Six cities have 2 or 3 stores. “Guayaquil” and  “Quito” are a group with 8 and 18 stores, and rest of them having less than 1 store
# 
# * The city distribution is reflected in the state distribution as well, with “Pichincha” having 19 stores, “Guayas” 11, and the rest between 1 and 3.
# 
# * In the types, we see that D and “C” are the most frequent, with “A” and “B” having similar medium frequency and “E” only accounting for 4 stores.
# 
# * The cluster feature shows a range from 1 to 7.
# 

# %% [code] {"execution":{"iopub.status.busy":"2024-08-01T15:31:25.841742Z","iopub.execute_input":"2024-08-01T15:31:25.842077Z","iopub.status.idle":"2024-08-01T15:31:26.486592Z","shell.execute_reply.started":"2024-08-01T15:31:25.842048Z","shell.execute_reply":"2024-08-01T15:31:26.485466Z"}}
d = pd.merge(train, stores)
g=d.groupby(['date','type'])['sales'].mean().reset_index()
px.line(g, x = "date", y= "sales", color = "type")


# %% [markdown]
# We find:
#     
# * Above figure shows sales as per Type
# 

# %% [code] {"execution":{"iopub.status.busy":"2024-08-01T15:31:26.487940Z","iopub.execute_input":"2024-08-01T15:31:26.488288Z","iopub.status.idle":"2024-08-01T15:31:26.692776Z","shell.execute_reply.started":"2024-08-01T15:31:26.488259Z","shell.execute_reply":"2024-08-01T15:31:26.691553Z"}}
for typee in g.type.unique():
    g.loc[g['type']==typee,'norm']=g[g['type']==typee]['sales']/g[g['type']==typee]['sales'].max()

px.line(g, x="date", y="norm", title='Monthly aggregated sales',color='type')


# %% [markdown]
# We find:
# 
# * Above figure is the normalized sales for Type
# 
# * There is a almost stable pattern apart from sales volume 
# 
# * We find that the sales volume for types “A” and “D” is very similar most of the times. The same is true for types “B” and “C” on a lower level. Stores of type “E” have notably the fewest overall sales.

# %% [code] {"execution":{"iopub.status.busy":"2024-08-01T15:31:26.694239Z","iopub.execute_input":"2024-08-01T15:31:26.694684Z","iopub.status.idle":"2024-08-01T15:31:32.427984Z","shell.execute_reply.started":"2024-08-01T15:31:26.694646Z","shell.execute_reply":"2024-08-01T15:31:32.426613Z"}}
g=d.groupby(['date','state'])['sales'].sum().reset_index()
(
ggplot(g,aes('date', 'sales')) +

  geom_line(color = "blue") +

  scale_y_log10() +

  p9.facet_wrap('state')+
    p9.theme(axis_text_x=p9.element_text(angle=90))

)

# %% [markdown]
# We find:
# 
# * Above figure shows sales by state 
# 
# * Manabi shows notable increase in sales over time
# 
# * The store in “Pastaza” was only opened in late 2015. This needs to be taken into account when making predictions based on store-aggregates before 2015.
# 

# %% [markdown]
# 

# %% [code] {"execution":{"iopub.status.busy":"2024-08-01T15:31:32.430184Z","iopub.execute_input":"2024-08-01T15:31:32.430746Z","iopub.status.idle":"2024-08-01T15:31:32.728219Z","shell.execute_reply.started":"2024-08-01T15:31:32.430698Z","shell.execute_reply":"2024-08-01T15:31:32.727159Z"}}
px.line(d.groupby(["city", "year"]).sales.mean().reset_index(), x = "year", y = "sales", color = "city")

# %% [markdown]
# We find:
#     
# * Sales by city

# %% [markdown]
# # 7. Transactions

# %% [code] {"execution":{"iopub.status.busy":"2024-08-01T15:31:32.729495Z","iopub.execute_input":"2024-08-01T15:31:32.729860Z","iopub.status.idle":"2024-08-01T15:31:32.751429Z","shell.execute_reply.started":"2024-08-01T15:31:32.729832Z","shell.execute_reply":"2024-08-01T15:31:32.750124Z"}}
transaction['date']=pd.to_datetime(transaction['date'])
g=transaction.groupby(['date'])['transactions'].sum().reset_index()

# %% [code] {"execution":{"iopub.status.busy":"2024-08-01T15:31:32.752694Z","iopub.execute_input":"2024-08-01T15:31:32.753022Z","iopub.status.idle":"2024-08-01T15:31:33.288962Z","shell.execute_reply.started":"2024-08-01T15:31:32.752994Z","shell.execute_reply":"2024-08-01T15:31:33.287483Z"}}

(
    ggplot(g,aes('date','transactions'))+
    geom_line()+
    p9.geom_smooth(method = "loess", color = "red", span = 1/5)

)

# %% [markdown]
# We find:
# 
# * There is a spike before christmas , followed by corresponding drops when stores are closed.
# 
# * Sales appear to be stable throughout this time range

# %% [code] {"execution":{"iopub.status.busy":"2024-08-01T15:31:33.290707Z","iopub.execute_input":"2024-08-01T15:31:33.291095Z","iopub.status.idle":"2024-08-01T15:31:33.466696Z","shell.execute_reply.started":"2024-08-01T15:31:33.291063Z","shell.execute_reply":"2024-08-01T15:31:33.465238Z"}}
gg=pd.merge(train.groupby(['date','store_nbr']).sales.sum().reset_index(),transaction,how='left')

# %% [code] {"execution":{"iopub.status.busy":"2024-08-01T15:31:33.468136Z","iopub.execute_input":"2024-08-01T15:31:33.468508Z","iopub.status.idle":"2024-08-01T15:31:34.904757Z","shell.execute_reply.started":"2024-08-01T15:31:33.468476Z","shell.execute_reply":"2024-08-01T15:31:34.903570Z"}}
px.line(gg,x='date', y='transactions', color='store_nbr',title = "Transactions" )

# %% [markdown]
# We find:
# 
# * There is a stable pattern in Transaction. 
# 
# * All months are similar except December from 2013 to 2017 

# %% [code] {"execution":{"iopub.status.busy":"2024-08-01T15:31:34.906217Z","iopub.execute_input":"2024-08-01T15:31:34.906622Z","iopub.status.idle":"2024-08-01T15:31:35.009129Z","shell.execute_reply.started":"2024-08-01T15:31:34.906589Z","shell.execute_reply":"2024-08-01T15:31:35.007832Z"}}
g = transaction.set_index("date").resample("M").transactions.mean().reset_index()
g["year"] = g.date.dt.year
px.line(g, x='date', y='transactions', color='year',title = "Monthly Average Transactions" )

# %% [code] {"execution":{"iopub.status.busy":"2024-08-01T15:31:35.010835Z","iopub.execute_input":"2024-08-01T15:31:35.011271Z","iopub.status.idle":"2024-08-01T15:31:35.108049Z","shell.execute_reply.started":"2024-08-01T15:31:35.011232Z","shell.execute_reply":"2024-08-01T15:31:35.106648Z"}}

g = transaction.copy()
g["year"] = g.date.dt.year
g["dayofweek"] = g.date.dt.dayofweek+1
g = g.groupby(["year", "dayofweek"]).transactions.mean().reset_index()
px.line(g, x="dayofweek", y="transactions" , color = "year", title = "Transactions")

# %% [markdown]
# we can see that there is a highly correlation between total sales and transactions also.

# %% [markdown]
# g['norm']=g['sales'].min()+ \
# (g['transactions']-g['transactions'].min()) /(g['transactions'].max()-g['transactions'].min()) \
# *(g['sales'].max()-g['sales'].min())
# 
# (
#     ggplot(g,aes('date','sales'))+
#     geom_line()+
#     geom_line(aes('date','norm'),color='blue')
# )

# %% [markdown]
# # 8. Oil Price

# %% [code] {"execution":{"iopub.status.busy":"2024-08-01T15:31:35.109579Z","iopub.execute_input":"2024-08-01T15:31:35.110011Z","iopub.status.idle":"2024-08-01T15:31:37.429724Z","shell.execute_reply.started":"2024-08-01T15:31:35.109968Z","shell.execute_reply":"2024-08-01T15:31:37.428528Z"}}
oil['date']=pd.to_datetime(oil['date'])
oil.rename(columns={'dcoilwtico':'oilprice'},inplace=True)
oil['diff7']=oil['oilprice'].diff(7).dropna()


p1 = \
(
    ggplot(oil,aes(x='date',y='oilprice'))+
    geom_line(color='black')+
    p9.geom_smooth(method='loess',color='red',span=1/5)
)

p2 = \
(
    ggplot(oil,aes('date',y='diff7'))+
    geom_line(color='black')+
    p9.geom_smooth(method = "loess", color = "red", span = 1/5) +
    labs(y = "Weekly variations in Oil price")
)


p1=pw.load_ggplot(p1,figsize=(10,2))
p2=pw.load_ggplot(p2,figsize=(10,2))

p1

# %% [code] {"execution":{"iopub.status.busy":"2024-08-01T15:31:37.431050Z","iopub.execute_input":"2024-08-01T15:31:37.431372Z","iopub.status.idle":"2024-08-01T15:31:38.009776Z","shell.execute_reply.started":"2024-08-01T15:31:37.431346Z","shell.execute_reply":"2024-08-01T15:31:38.008342Z"}}
p2

# %% [code] {"execution":{"iopub.status.busy":"2024-08-01T15:31:38.015809Z","iopub.execute_input":"2024-08-01T15:31:38.016298Z","iopub.status.idle":"2024-08-01T15:31:38.095728Z","shell.execute_reply.started":"2024-08-01T15:31:38.016251Z","shell.execute_reply":"2024-08-01T15:31:38.094383Z"}}
oil['date']=pd.to_datetime(oil.date)
oilXsales=pd.merge(train.groupby(['date'])['sales'].mean().reset_index(),oil,how='left')

# %% [code] {"execution":{"iopub.status.busy":"2024-08-01T15:31:38.097184Z","iopub.execute_input":"2024-08-01T15:31:38.097644Z","iopub.status.idle":"2024-08-01T15:31:38.586684Z","shell.execute_reply.started":"2024-08-01T15:31:38.097593Z","shell.execute_reply":"2024-08-01T15:31:38.585490Z"}}
oilXsales['norm2']=oilXsales['sales'].min()+ \
(oilXsales['oilprice']-oilXsales['oilprice'].min()) /(oilXsales['oilprice'].max()-oilXsales['oilprice'].min()) \
*(oilXsales['sales'].max()-oilXsales['sales'].min())

(
    ggplot(oilXsales,aes('date','sales'))+
    geom_line()+
    geom_line(aes('date','norm2'),color='blue')
)

# %% [markdown]
# We find:
# 
# * Over time sales increases and the oil price drops
# 
# * Over the last year, the oil price was relatively stable.
# 
# * The large oil price decrease during the 2nd half of 2014 might have affected the lower sales numbers in the 1st half of 2015

# %% [markdown]
# # 9. Holidays and Events

# %% [code] {"execution":{"iopub.status.busy":"2024-08-01T15:31:38.588018Z","iopub.execute_input":"2024-08-01T15:31:38.588386Z","iopub.status.idle":"2024-08-01T15:31:45.581861Z","shell.execute_reply.started":"2024-08-01T15:31:38.588354Z","shell.execute_reply":"2024-08-01T15:31:45.580730Z"}}
holidays_events=holidays_events.astype('category')

p1 = \
(
    ggplot(holidays_events,aes('type',fill='type'))+
    geom_bar()+
    p9.theme(legend_position='none',axis_text_x=p9.element_text(angle=90))
)

p2 = \
(
    ggplot(holidays_events,aes('locale',fill='locale'))+
    geom_bar()+
    p9.theme(legend_position='none',axis_text_x=p9.element_text(angle=90))
)

g=pd.DataFrame(holidays_events.description.value_counts(),columns=['count']).reset_index()

p3 = \
(
    ggplot(g.head(15),mapping=aes('description','count'))+
    p9.geom_col(fill='blue')+
    p9.theme(legend_position='none')+
    p9.coord_flip()+
    labs(x='Most Frequent',y='Frequency')
)

p4 = \
(
    ggplot(holidays_events,aes('transferred',fill='transferred'))+
    geom_bar()+
    theme(legend_position='none')
)

p5 = \
(
    ggplot(holidays_events,aes('locale_name',fill='locale_name'))+
    geom_bar()+
    theme(legend_position='none',axis_text_x=p9.element_text(angle=45,hjust=1,vjust=0.9))
)

p1=pw.load_ggplot(p1,figsize=(3,3))
p2=pw.load_ggplot(p2,figsize=(3,3))
p3=pw.load_ggplot(p3,figsize=(3,3))
p4=pw.load_ggplot(p4,figsize=(3,3))
p5=pw.load_ggplot(p5,figsize=(10,2))

p12345=(p1|p2)/(p3|p4)/p5
p12345.savefig()

# %% [markdown]
# We find:
# 
# * Mosts special days are of the type “Holiday” and are either of the locale “Local” or “National”. Relatively few “Regional” holidays are present in our data.
# 
# * The large number of national holidays is emphasised in the second plot
# 
# * The lower left plot lists a few of the most frequent holiday descriptions ( Carnival is clearly important.
# 
# * The majority of days off is not transferred.

# %% [code]


# %% [code]
