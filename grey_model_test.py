# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 00:02:01 2017

@author: albertnava
"""

import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt

# Data to test the programm
dates = ['1-15', '1-16', '1-17', '1-18', '1-19', '1-20', '1-21', '1-22', '1-23', '1-24', '1-25', '1-26', \
'1-27', '1-28', '1-29', '1-30', '1-31']
cabbage = [194859.9, 238733.2, 195361.4, 177969.8, 231074.6, 149686.9, 139618.5, \
142804.2, 130070.9, 157811.5, 260115.4, 351227.5, 302161.2, 373370.5, 296045.4, 178621.0, 251443.0]
tomato = [40020.8, 48106.1, 55545.4, 51588.2, 47523.3, 38189.2, 32854.4, 62849.4, 48717.8, \
59526.6, 81282.2, 64255.6, 51868.5, 37608.4, 20800.5, 25297.7, 47885.5]


def GreyModel_11(serie_t) :
    
    PE = []
    Z1 = []
    X1 = []
    Xp1 = []
    Xp0 = []
    tmp = 0
    tam_s = len(serie_t)
        
    for i in range (0, tam_s):
        val= serie_t[i]
        tmp += val
        #print tmp
        X1.append(tmp)
    print len(X1)
    for j in range (1,len(X1)):
        z = 0.5*X1[j]+0.5*X1[j-1]
        Z1.append(z)
    
    #print len(Z1)
    Bp = np.ones((len(Z1),2))
    Bp[:,0] = -1*np.asarray(Z1)
    B = Bp
    B = np.asarray(B)
    Y = np.asarray( serie_t[1:tam_s])
    BT = np.transpose(B)
    Bt = BT.dot(B)
    dBT = np.linalg.det(Bt)
    if (dBT==0):
        print "The matrix Bt is singular..."
    else:
        A = np.linalg.inv(Bt).dot(BT)
        Au = A.dot(Y)

    for k in range (0,tam_s):
        xp1 = (serie_t[0]-Au[1]/Au[0])* math.exp(-Au[0]*k)+Au[1]/Au[0]
        xp0 = (serie_t[0]-Au[1]/Au[0])* math.exp(-Au[0]*k)*(1-math.exp(Au[0]))
        Xp1.append(xp1)
        Xp0.append(xp0)
    
    # Forescating the new data after all past values 
    xp = (serie_t[0]-Au[1]/Au[0])* math.exp(-Au[0]*tam_s)*(1-math.exp(Au[0]))    
    print ''    
    print Au 
    
    E = (np.asarray(serie_t)-np.asarray(Xp0))
    E = E[1:tam_s]
    PE = ((np.asarray(Xp0)-np.asarray(serie_t))/np.asarray(serie_t))*100 # If there is a zero in the values will div/0
    MAPE = np.mean(np.abs(PE))
    print MAPE
    
    # For Fourier series modified residual GM(1,1) model
    
    Xf0 = np.empty(np.shape(Xp0))
    T = tam_s-1
    w = (2*np.pi)/T
    Z = (tam_s-1)/2 -1
    print ("Longitud de lista Z = ", Z)    
    k = len(E)
    m = 2*Z+1
    A = np.ones((k,m))
    B = np.ones((k,m))
    n = 1
    
      
    for i in range (k):
        A[i,:]=i+2
    
    for j in range(1,m):
        B[:,j]=n
        if(j%2==0):
            n=n+1
    B = B*w    
    AB = A*B
    
        
    for l in range(1,m):
        if (l%2==0):
            AB[:,l]=(np.pi/2)-AB[:,l]
    
    AB[:,0]=np.pi/3
    P = np.cos(AB)
    Pt = np.transpose(P)
    Pb = Pt.dot(P)
    dPb = np.linalg.det(Pb)
    if (dPb==0):
        print "The matrix Pb is singular..."
    else:
        Pn = np.linalg.inv(Pb).dot(Pt)
        C = Pn.dot(E)
        #print C
        En0 = P.dot(C) 
        Xf0 = Xp0[1:]+En0
    
    #print np.shape(Xf0)
    
    return Xp0 , Xf0, xp

#x = np.linspace(0, len(tomato))    
[list_p, Xfou, val] = GreyModel_11(tomato)     
Errort = ((np.asarray(Xfou)-np.asarray(tomato[1:]))/np.asarray(tomato[1:]))*100
Errort2 = ((np.asarray(list_p[1:])-np.asarray(tomato[1:]))/np.asarray(tomato[1:]))*100
print (list_p)

plt.figure()
plt.plot(tomato, 'rs' )
plt.plot(list_p, 'b^' )
plt.xlabel('Dates')
plt.ylabel('Production')
plt.title('Simple GM for production')
plt.xticks(range(len(dates)), dates)
plt.xticks(range(len(dates)), dates, rotation=45) #writes strings with 45 degree angle
plt.grid()
plt.show()

    