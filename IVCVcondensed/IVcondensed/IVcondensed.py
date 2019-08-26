# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 11:31:59 2019

This is a module to perform condensed IV analysis across many sensors and detectors. 
@author: Sneha Ramshanker 
"""

import matplotlib.pyplot as plt
from  .. import Functions as fun 
import pandas as pd 
import seaborn as sns 
import numpy as np
from optparse import OptionParser
from matplotlib.lines import Line2D
import os


"""
Optional Parsing
"""

parser = OptionParser()
parser.add_option("--file", type="string",
                  help="Path of input file",
                  dest="file", default='Testdata\Breakdown.csv')

parser.add_option("--impathIV",type = "string", help = "Path to store IV plot", dest = "impathIV", default = "IV_plots/Combined_Plots/Type-vs-SEn.png")
options, arguments = parser.parse_args()

file = options.file
impathIV = options.impathIV


"""
Function definitions
"""

def plotbycat(xaxis, yaxis, marker_cat, markers, impath, df, condid = "No", title = 0, jitter = "Default"):
    """
    Parameters:
        xaxis: String - Name of xaxis column in dataframe df 
        yaxis: String - Name of yaxis column in dataframe df 
        marker_cat: String - Name of marker column in dataframe df (category of data that the different markers represent ex. SEn)
        markers: List: List of name for each marker 
        df: Pandas Dataframe
        impath: String - path where image will be saved 
    Optional Parameters:
        condid: String ("No" or "Cond") - An ID indicating whether values presented in the marker category should be averaged on not (condensed id). 
                Default is "No"
        title: String - Name of title of plot. By default there is no title 
        jitter: Float ("Default" or float value between range 0 and 1 ) - Separation of the markers.  Max Jitter: 1. Min Jitter: 0. By default, jitter = 0.2 
        
         
    Note:
        This is a function that produces condensed plots of data extracted from several detectors and wafers. 
    """
    
    
    
    
    meanarr = []
    
    for i in range(0, len(markers)):
        dftype = (df[df[marker_cat].str.contains(markers[i])])
        mean = dftype.groupby(xaxis, as_index = False).mean()
        mean[marker_cat] = markers[i]
        #print(mean)
        meanarr.append(mean)
    print(meanarr)
    
    
    marker_vis = ["X", "o","v", "D" ]

    label = markers 
    
    if condid == "Cond":
        conddf = pd.DataFrame() #Condensed mean array 
    
        for i in range(0, len(meanarr)):
            conddf = pd.concat([conddf, meanarr[i]])
        condmean = conddf.groupby(xaxis, as_index = False).mean()
        condstd = conddf.groupby(xaxis, as_index = False).agg(np.std, ddof=0)
        yerr = list(condstd[condstd.columns[1]])
    
        x = list(condmean[xaxis])
        y = list(condmean[yaxis])
        plt.errorbar(x, y, yerr, fmt = 'X', markersize='15', elinewidth=2)
        if title != 0:
            plt.title(title)
        plt.ylabel("Breakdown Voltage" +" (V)")
        plt.xlabel(xaxis)
        plt.savefig(impath)
        plt.close()
    else:
        for i in range(0, len(meanarr)):
            if (jitter == "Default"):
                ax = sns.stripplot(x = xaxis, y = yaxis, hue = marker_cat, color = 'b',  marker=marker_vis[i], jitter = 0.2, size = 12,  data = meanarr[i], label = marker_cat +str(label[i]))
            else:
                ax = sns.stripplot(x = xaxis, y = yaxis, hue = marker_cat, color = 'b',  marker=marker_vis[i], jitter = jitter, size = 12,  data = meanarr[i], label = marker_cat +str(label[i]))

            ax.grid("Both")
        
        legend_elements = [Line2D([0], [0], marker='X', color='w', label= markers[0],
                              markerfacecolor='k', markersize=15),
                       Line2D([0], [0], marker='o', color='w', label= markers[1],
                              markerfacecolor='k', markersize=15),
                       Line2D([0], [0], marker='v', color='w', label= markers[2],
                              markerfacecolor='k', markersize=15),
                       Line2D([0], [0], marker='D', color='w', label= markers[3],
                             markerfacecolor='k', markersize=15)]
        
        lgd = ax.legend( handles=legend_elements, loc='upper center', bbox_to_anchor=(1.15, 0.9),
              ncol=1, fancybox=True, shadow=True)
        
        
        if title != 0:
            plt.title(title)
        plt.ylabel("Breakdown Voltage"+" (V)")
        plt.savefig(impath, bbox_extra_artists=(lgd,), bbox_inches='tight')
        plt.close()

"""
Rest of the Program
"""

#Setting plot style
plt.rcParams['figure.figsize'] = (12, 10)

df = fun.simpletextread(file, ',')



PIN = df[df['Detector '].str.contains("PIN")] 
df.drop(PIN.index, axis=0, inplace = True) #Not including PIN values 
NM =  df[df['Detector '].str.contains("NM")]
df.drop(NM.index, axis=0, inplace = True) #Not including NM values
df['Detector '] = df['Detector '].astype(str).str[:-6]

#Assigning Type to the wafers 
conditions = [df['Wafer '].str.contains('27'), df['Wafer '].str.contains('28'), df['Wafer '].str.contains('29'), df['Wafer '].str.contains('30')]
choices = ['1.1','1.2', '2.0', '3.2']
df['Type'] = np.select(conditions, choices, default='0')
    



"""
xaxis = "Type"
yaxis = 'Breakdown Voltage (Thresh 0.9995)'
marker_cat = "Detector " #Marker category
markers = ["SE2", "SE3", "SE5"]

impathIVcond = "IV_plots/Combined_Plots/Cond-Type-vs-SEn.png" #MODIFY BY USER
plotbycat(xaxis, yaxis, marker_cat, markers, df, impathIV)
plotbycat(xaxis, yaxis, marker_cat, markers, df, impathIVcond, condid = "Cond")

"""
xaxis = "Detector "
yaxis = 'Breakdown Voltage (Thresh 0.9995)'
marker_cat = "Type" #Marker category
markers = ["1.1", "1.2", "2.0", "3.2"]

impath = "IV_plots/Combined_Plots/SEn-vs-Type-2.png" #MODIFY BY USER 
plotbycat(xaxis, yaxis, marker_cat, markers,  df,  impath, jitter = 0)

"""
#Only plotting Type 3-2
df = df[df['Type'].str.contains("3.2")] 
conditions = [df['Wafer '].str.contains('11'), df['Wafer '].str.contains('12'), df['Wafer '].str.contains('13'), df['Wafer '].str.contains('14')]
choices = ['WNo11','WNo12', 'WNo13', 'WNo14']
df['Wafer '] = np.select(conditions, choices, default = "NaN")
clean =  df[df['Wafer '].str.contains("NaN")]
df.drop(clean.index, axis=0, inplace = True)
xaxis = "Wafer "
yaxis =  df.columns[2]
marker_cat = "Detector " #Marker category
markers = ["SE2", "SE3", "SE5"]
impath32 = "IV_plots/Combined_Plots/Type3-2-vs-SEn.png" #MODIFY BY USER 
impath32cond = "IV_plots/Combined_Plots/Cond-Type3-2-vs-SEn.png" #MODIFY BY USER 
plotbycat(xaxis, yaxis, marker_cat, markers, df, impath32, title = "Type 3.2")
plotbycat(xaxis, yaxis, marker_cat, markers, df, impath32cond, condid = "Cond", title = "Type 3.2")
"""
