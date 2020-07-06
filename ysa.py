# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 23:47:46 2020

@author: DELL
"""
import random
import numpy as np
import pandas as pd
import math
from sklearn import preprocessing 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import r2_score, mean_squared_error
min_max_scaler = preprocessing.MinMaxScaler()

class collect_data:
  
    def dataseti(self) :
        dataframe = pd.read_excel('C:\\Users\\DELL\\Desktop\\Pyhton\\Energy.xlsx')
        df = dataframe.values
        X = df[:,0:8]
        Y = df[:,8:9]
        
        """Veri setini train ve test olarak bölüyorum."""   
        X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.20,random_state=5)
    
        """Data setindeki veriler arasında farklılığın fazla olduğundan hepsini tek bir düzeyde tutmak için normalizasyon uyguluyorum. """
        x_scaled = min_max_scaler.fit_transform(X_train)
        X_train_normed = pd.DataFrame(x_scaled)
        
        Y_scaled = min_max_scaler.fit_transform(Y_train)
        Y_train_normed = pd.DataFrame(Y_scaled)
        
        X_train_normed = X_train_normed.values
        Y_train_normed = Y_train_normed.values
        
        x_scaledt = min_max_scaler.fit_transform(X_test)
        X_test_normed = pd.DataFrame(x_scaledt)
        
        Y_scaledt = min_max_scaler.fit_transform(Y_test)
        Y_test_normed = pd.DataFrame(Y_scaledt)
        
        X_test_normed = X_test_normed.values
        Y_test_normed = Y_test_normed.values
      
        return X_train_normed,X_test_normed,Y_train_normed,Y_test_normed
        
    
    
class my_ANN:
    
    
    def __init__(self, n_inp, n_hid,n_out, iterr, ogr_oran):
       
        self.n_inp = n_inp
        self.n_hid = n_hid
        self.n_out = n_out
        self.iterr = iterr 
        self.ogr_oran = ogr_oran
    
    
    def sigmoid(self, value):
        return (1 / (1 + np.exp(-value))) 
    
    
    def mse(self,y,y_tahmin):
        
        unscaled_Y = min_max_scaler.inverse_transform(y)
        unscaled_tahmin = min_max_scaler.inverse_transform(y_tahmin)
        error = mean_squared_error(unscaled_Y, unscaled_tahmin)
        return error
    
    def r2_score(self,y,y_tahmin):
        
        unscaled_Y = min_max_scaler.inverse_transform(y)
        unscaled_tahmin = min_max_scaler.inverse_transform(y_tahmin)
        r2 = r2_score(unscaled_Y,unscaled_tahmin)
        return r2
        
    def test(self,X_test,Y_test,matris_in,matris_out):
        """Ağırlık matrisleri kullanılarak çıkışlar tahmin edilir.""" 
        hid = np.dot(X_test, matris_in)   
        hid_output = self.sigmoid(hid) 
        out = np.dot(hid_output, matris_out) 
        output = self.sigmoid(out)
        error = self.mse(Y_test,output) #MSE fonksiyonu ile tahmin edilen çıkışlar ile gerçek çıkışlar arasındaki hata analiz edilir.
        r2 = self.r2_score(Y_test,output) # Gerçek çıkış(Y_test) ile tahmin çıkış(output) arasındaki ilişki r2_score fonksiyonu ile bulunur.
        
    def train(self,X, Y):
        
        counter = 0
        matris_in = np.random.rand(self.n_inp, self.n_hid) 
        matris_out = np.random.rand(self.n_hid, self.n_out) 
        
        while counter <= self.iterr:
            
            matrisOut = matris_out
            hid = np.dot(X, matris_in)
            bias = np.ones((1, self.n_hid)) #Bias ile outputun hiçbir zaman sıfır olmayacağını garantilemiş oluyorum. 
            hid = np.add(hid, bias)    
            hid_output = self.sigmoid(hid) 
            out = np.dot(hid_output, matris_out) 
            output = self.sigmoid(out)
            #GİZLİ-ÇIKIŞ KATMAN
            delta_k = output * (1 - output) * (Y - output) 
            k = self.ogr_oran * delta_k
            delta_w = np.dot(k.T,hid_output)
            matris_out += delta_w.T 
            #GİRİŞ-GİZLİ KATMAN                 
            delta_h_1 = hid_output * (1 - hid_output) 
            delta_h_2 = np.dot(delta_k,matrisOut.T) 
            delta_h = delta_h_1 * delta_h_2
            delta_w2 = np.dot(X.T,(self.ogr_oran * delta_h)) 
            matris_in += delta_w2
            
            error = self.mse(Y,output)
            counter = counter+1
            
        return matris_in,matris_out  
            
        
DataFrame = collect_data()

X_train_normed,X_test_normed,Y_train_normed,Y_test_normed = DataFrame.dataseti()

"""Belirlenen iterasyon sayısı kadar ileri yayılım ve geri yayılım yapılarak ağ eğitilir."""
network = my_ANN(8, 8, 1, 5000, 0.001)

matris_in,matris_out = network.train(X_train_normed, Y_train_normed)

network.test(X_test_normed,Y_test_normed,matris_in,matris_out)






