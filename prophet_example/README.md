<h1 align="center">Facebook Prophet</h1>

## Contents  
- [Properties](#properties)
- [References](#references)

## Properties  

### overview  
- trend + seasonality + holidays  
- can adjust the effect of holidays or other one-time events  
- can handle missing values and outliers  
- can adjust dramatic trend changes/trends that show asymptotic behavior  

### how it works  
- trend: piecewise linear/logistic growth curve trend with changepoint detection  
- seasonality: modeled using fourier series  
- holidays: user-provided  

$$ y(t) = g(t) + s(t) + h(t) + \epsilon(t) $$  


## References  
- [Prophet Document](https://facebook.github.io/prophet/docs/quick_start.html#python-api)  
- [Fastcampus Course](https://fastcampus.app/course-detail/210867)  
- [Use Case](https://github.com/aldente0630/mofc-demand-forecast)  
