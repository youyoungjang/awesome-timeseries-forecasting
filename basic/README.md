<h1 align="center">Basic Knowledge</h1>
<h5 align="center"> </h5>

## Contents  
- [Introduction](#introduction)
- [ARIMA](#arima)
- [SOURCE](#source)

## Introduction  

### components of time series
- Non-stationary Time Series = trend + seasonality + residual  
- Stationary Time Series = only residual

### characteristics of time series data
- sometimes time series data have a cycle which means rising and falls without any fixed frequency.  
- some might return to mean value. (mean reverting data)  
- time series data with momentum continue in the same direction over time until the momentum dissapears.  

### basic steps in time series forecasting  
- define problems & collect data  
- eda and stationary check(graphs, seasonality, trend, business cycle ...)  
- choose & fit & evaluate model  

### how to check stationarity  
- visual test  
- statistical test

examples of statistical test are as follows  

**ADF**: Augmented Dickey Fuller Test)  
**KPSS**: Kwiatkowsky-Phillips-Schmidt-Shin  

### how to make data as stationary  
If data is not stationary we have to make it as stationary data. If we compute differences between observations that are consecutive then we may remove most of non-stationary part in data. Or we can take log transformation to remove non-stationarity.   

### classical time series methods  
- Autoregression  
- Moving Average  
- Autoregressive Moving Average (ARMA)  
- Autoregressive Integrated Moving Average (ARIMA)  
- Seasonal Autoregressive Integrated Moving Average (SARIMA)  
- Vector Autoregression (VAR)  
- Vector Autoregression Moving Average (VARMA)  

### challenges of classifical methods  
- how do we decide differencing order?  
- how do we decide model order (AR? MA?)  

To do so, we have to understand autocorrelation plot and partial autocorrelation plot.  

- how do we decide seasonality order?  

We also have to what method we should choose to remove seasonality.  

In conclusion, it is very complicated to integrate all these factors in simple way when we use classical methods.  


## ARIMA  



---
## Source  
This contents are originated from [this course](https://fastcampus.app/course-detail/210867). You may take this course if you want more details.  
