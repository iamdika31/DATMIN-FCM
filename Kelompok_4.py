# -*- coding: utf-8 -*-
"""
Created on Sun May 19 20:16:31 2019

@author: DIKA(Edited By Faiz)
"""

import numpy as np
import pandas as pd
#atribut = np.array(([12,7,9],[5,4,5],[8,11,4],[10,3,8],[9,1,3]))
df = pd.read_csv("iris_data.csv")

cluster = 3 
w = 2 
max_iter = 5
e = 0.01
p0 = 0

sl = df['Sepal Length'].values
sw = df['Sepal Width'].values
pl = df['Petal Length'].values
pw = df['Petal Width'].values

sl_array = np.array(sl)
sw_array = np.array(sw)
pl_array = np.array(pl)
pw_array = np.array(pw)
atribut = np.vstack((sl_array, sw_array, pl_array, pw_array)).T

miu = np.array(([0.3,0.7,0.7],[0.2,0.8,0.8],[0.4,0.6,0.6],
                [0.8,0.2,0.2],[0.4,0.6,0.6],[0.3,0.7,0.4],
                [0.6,0.6,0.8],[0.1,0.6,0.5],[0.2,0.4,0.7],
                [0.2,0.9,0.3],[0.6,0.5,0.3],[0.2,0.2,0.2],
                [0.9,0.6,0.6],[0.5,0.1,0.2],[0.8,0.5,0.3],
                [0.5,0.1,0.3],[0.4,0.2,0.4],[0.5,0.3,0.2],
                [0.4,0.7,0.3],[0.6,0.7,0.1],[0.8,0.2,0.9],
                [0.8,0.7,0.5],[0.7,0.3,0.7],[0.1,0.9,0.9],
                [0.8,0.9,0.6],[0.3,0.9,0.7],[0.2,0.4,0.5],
                [0.7,0.1,0.9],[0.2,0.1,0.5],[0.9,0.1,0.2]))
for i in range(max_iter):
    #untuk hitung centroid
    #untuk cluster 1
    miuC1 = np.zeros((len(atribut),len(atribut[0])+1))
    for i in range (len(atribut)):
        miuC1[i][0] = miu[i][0] ** 2
        miuC1[i][1] = miuC1[i][0] * atribut[i][0]
        miuC1[i][2] = miuC1[i][0] * atribut[i][1]
        miuC1[i][3] = miuC1[i][0] * atribut[i][2]
        miuC1[i][4] = miuC1[i][0] * atribut[i][3]
        
    
    #untuk cluster 2
    miuC2= np.zeros((len(atribut),len(atribut[0])+1))
    
    for i in range (len(atribut)):
        miuC2[i][0] = miu[i][1]**2
        miuC2[i][1] = miuC2[i][0] * atribut[i][0]
        miuC2[i][2] = miuC2[i][0] * atribut[i][1]
        miuC2[i][3] = miuC2[i][0] * atribut[i][2]
        miuC2[i][4] = miuC2[i][0] * atribut[i][3]
        
    #untuk cluster 3
    miuC3= np.zeros((len(atribut),len(atribut[0])+1))
    
    for i in range (len(atribut)):
        miuC3[i][0] = miu[i][2]**2
        miuC3[i][1] = miuC3[i][0] * atribut[i][0]
        miuC3[i][2] = miuC3[i][0] * atribut[i][1]
        miuC3[i][3] = miuC3[i][0] * atribut[i][2]
        miuC3[i][4] = miuC3[i][0] * atribut[i][3]
        
    
    sum_miuC1 = np.sum(miuC1,axis = 0)  
    sum_miuC2 = np.sum(miuC2,axis = 0)
    sum_miuC3 = np.sum(miuC3,axis = 0)
    
    
    jarak_centroid = np.zeros((cluster,len(atribut[0])))
    for i in range(len(jarak_centroid)):
       for j in range(len(jarak_centroid[0])):
           if(j != len(jarak_centroid[0])):
               if(i == 0):
                   jarak_centroid[i][j] = sum_miuC1[j+1]/sum_miuC1[0]
               elif(i == 1) :
                   jarak_centroid[i][j] = sum_miuC2[j+1]/sum_miuC2[0]
               else :
                   jarak_centroid[i][j] = sum_miuC3[j+1]/sum_miuC3[0]
                 
    #print(jarak_centroid)              
    
    #hitung fungsi objective c1
    fungsiObj_C1 = np.zeros((len(atribut),len(atribut[0])+1))
    for i in range(len(atribut)):
        temp = 0
        for j in range(len(atribut[0])):
            fungsiObj_C1[i][j] = (atribut[i][j] - jarak_centroid[0][j])**2
            temp = temp + fungsiObj_C1[i][j]
        fungsiObj_C1[i][len(atribut[0]-1)] = temp 
        
        
    #hitung fungsi objective c2
    fungsiObj_C2 = np.zeros((len(atribut),len(atribut[0])+1))
    for i in range(len(atribut)):
        temp = 0
        for j in range(len(atribut[0])):
            fungsiObj_C2[i][j] = (atribut[i][j] - jarak_centroid[1][j])**2
            temp = temp + fungsiObj_C2[i][j]
        fungsiObj_C2[i][len(atribut[0]-1)] = temp 
        
    #hitung fungsi objective c3
    fungsiObj_C3 = np.zeros((len(atribut),len(atribut[0])+1))
    for i in range(len(atribut)):
        temp = 0
        for j in range(len(atribut[0])):
            fungsiObj_C3[i][j] = (atribut[i][j] - jarak_centroid[1][j])**2
            temp = temp + fungsiObj_C3[i][j]
        fungsiObj_C3[i][len(atribut[0]-1)] = temp 
    
    p_cluster = np.zeros((len(atribut),1))
    P1 = np.zeros((len(atribut),1))
    P2 = np.zeros((len(atribut),1))
    P3 = np.zeros((len(atribut),1))
    #P_Cluster
    for i in range(len(atribut)):
        P1[i][0] = miuC1[i][0] * fungsiObj_C1[i][4]  
        P2[i][0] = miuC2[i][0] * fungsiObj_C2[i][4]
        P3[i][0] = miuC3[i][0] * fungsiObj_C3[i][4]
        p_cluster[i][0] = P1[i][0] + P2[i][0] + P3[i][0]
        
        
    #update miu
    total = np.zeros((len(atribut),1))
    cluster_baru = np.zeros((len(atribut),1))
    for i in range(len(atribut)):
        total[i][0] = fungsiObj_C1[i][4] + fungsiObj_C2[i][4] + fungsiObj_C3[i][4]
        miu[i][0] = fungsiObj_C1[i][4] / total[i][0] 
        miu[i][1] = fungsiObj_C2[i][4] / total[i][0]
        miu[i][2] = fungsiObj_C3[i][4] / total[i][0]
        if(miu[i][0] > miu[i][1] and miu[i][0] > miu[i][2] ):
            cluster_baru[i][0] = 1
        elif(miu[i][1] > miu[i][0] and miu[i][1] > miu[i][2] ):
            cluster_baru[i][0] = 2
        elif(miu[i][2] > miu[i][0] and miu[i][2] > miu[i][1] ):
            cluster_baru[i][0] = 3
        else :
            cluster_baru[i][0] = 2 or 3
        
print(jarak_centroid)#history fungsi objective tiap iterasi
print(p_cluster)
print('CLuster akhir adalah: \n',cluster_baru)

#ini blm tak tambah fungsi P kalo dia berhenti apa lanjut
          

 