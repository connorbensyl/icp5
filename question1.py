import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import scale
from sklearn.neighbors import KernelDensity
import sklearn.datasets
train = pd.read_csv('data.csv')



data = train.select_dtypes(include=[np.number]).interpolate().dropna()#only get numerical columns
scatterplot = plt.scatter(data.GarageArea, data.SalePrice).get_array()#get scatterplot and convert data points to array

mean = np.mean(data["SalePrice"])#calculate mean and standard deviation
std = np.std(data["SalePrice"])


for x in scatterplot: #iterate through scatterplot array
    for y in scatterplot:
        z_score = (np.abs((y - mean)/std)) #if z score > 3 then it is an anomaly
        if(z_score > 3):
            data.drop(np.where(data["GarageArea"] == x) and np.where(data["SalePrice"] == y)) # then we drop that point from the dataset

