<h1 align="center">Basic Knowledge of Classical Algorithms</h1>
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

### components  
**AR** part actually estimates the value of time series based on regressor given the past time series values. $n$ th AR model is described as follows.    

$$ y_t = a_1 y_{t-1} + a_2 y_{t-2} + ... + a_n y_{t-n} + \epsilon $$  

**MA** part is independent of AR part and does not rely on the past values. It uses past errors.  

$$ y_t = m_1 \epsilon_{t-1} + m_2 \epsilon_{t-2} + m_3 \epsilon_{t-3} + ... + \epsilon $$  

One way to compute the 1st error is `Levenberg-Marquardt Arglrithm`.  

If we combine AR and MA then we get `ARMA` model which only fits stationary time series data. In previous chapters diffrencing is an effective way to make non-stationary data as stationay. We can check stationarity by ADF test which sets the null hypothesis that if the time series data have unit root then they are not stationary.  

To convert data back to the original one, we compute cumulative sum of difference series. Cumulative sum can be interpreted as Integration. So this is the 3rd part of `ARIMA` model which can be called as **Integrated**.  

If we tell the model the order of differencing then model can perform conversion and reversion automatically.  

---
## Source  
This contents are originated from [this course](https://fastcampus.app/course-detail/210867). You may take this course if you want more details.  
