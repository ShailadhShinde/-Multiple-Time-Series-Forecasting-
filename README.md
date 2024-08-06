
<div align="center">

![logo](https://github.com/ShailadhShinde/Time_series/blob/main/assets/header.png)  
<h1 align="center"><strong>Store Sales <h6 align="center">A Time Series Forecasting (CV) project</h6></strong></h1>

![Python - Version](https://img.shields.io/badge/PYTHON-3.9+-blue?style=for-the-badge&logo=python&logoColor=white)

</div>
This project focuses on:

- Exploratory Data Analysis
- 
-
-

#### -- Project Status: [Completed]

#### -- time_series.py / time_Series.ipynb - Contains code for the project

#### -- eda-time-series.ipynb / EDA.py - Exploratory Data Analysis [Click to view](https://www.kaggle.com/code/shailadh/eda-time-series?scriptVersionId=190759981)

----

## [📚 Project Documentation 📚](http://smp.readthedocs.io/)

### 📋 Table of Contents
- [Overview](#overview)
  - [About the dataset](#atd)
  - [Sample Selection](#ss)
  - [Preprocessing](#pp)
  - [Feature Engineering](#fe)
  - [Evaluation](#eval)
  - [Model](#model)
- [Results](#results)
- [Getting Started](#gs)
  - [Prerequisites](#pr)


###  📌 Project Overview  <a name="overview"></a>

This project is a Notebook about time series forcasting for store sales.The purpose is to predict sales for 1000s of products sold at favourite stores located in South America’s west coast Ecuador.

- ### About the dataset  <a name="atd"></a>

  `The train data` contains time series of the stores and the product families combination. The sales column gives the total sales for a product family at a particular store at a given date. Fractional values are possible since products can be sold in fractional units (1.5 kg of cheese, for instance, as opposed to 1 bag of chips).The onpromotion column gives the total number of items in a product family that were being promoted at a store at a given date.

  `Stores` data gives some information about stores such as city, state, type, cluster.

  `Transaction` data is highly correlated with train's sales column. You can understand the sales patterns of the stores.

  `Holidays and events` data is a meta data. This data is quite valuable to understand past sales, trend and seasonality components. However, it needs to be arranged. You are going to find a comprehensive data manipulation for this data. That part will be one of the most important chapter in this notebook.

  `Daily Oil Price data` is another data which will help us. Ecuador is an oil-dependent country and it's economical health is highly vulnerable to shocks in oil prices. That's why, it will help us to understand which product families affected in positive or negative way by oil price.

- ### Sample Selection  <a name="ss"></a>

  Used 2017 data to extract and construct samples

  train date: 20170228 - 20170830

  validation: 20170725 - 20170808
- ### Preprocessing  <a name="pp"></a>
  Filled missing or negtive promotion and target values with 0.

- ### Feature Engineering  <a name="fe"></a>
 1. #### Basic features
    * category features: store, family, city, state, type
    * promotion

 2. #### Statitical features:
    we use some methods to stat some targets for different keys in different time windows
    * time windows
      * nearest days: [1,3,5,7,14,30,60,140]
      * key：store, store x family 
    * target: promotion, unit_sales, zeros
    * method
      * mean, median, max, min, std
  
- ### Evaluation  <a name="eval"></a>
  The evaluation metric used is Root Mean Squared Logarithmic Error. RMSLE = $\sqrt{\frac{1}{n} \sum_{i=1}^n (\log(p_i + 1) - \log(a_i+1))^2 }$

- ### Model <a name="model"></a>
    16 Light Gradient Boosting Model trained for each day

----

## 💫 Results <a name="results"></a>

  Got a good score resulting in top 1 % of the kaggle leader board
  
   <p align="center">
  <img width="60%" src="https://github.com/ShailadhShinde/Time_series/blob/main/assets/score.JPG">
 </p>

  
---

## 🚀 Getting Started <a name="gs"></a>

### ✅ Prerequisites <a name="pr"></a>
 
 - <b>Dataset prerequisite for training</b>:
 
 Before starting to train a model, make sure to download the dataset from <a href="https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data" target="_blank">here </a> or add it to your notebook
 ### 🐳 Setting up and Running the project

 Just download/copy the files `time_series.py / time_Series.ipynb ` and `EDA.ipynb / EDA.py ` and run them

  
