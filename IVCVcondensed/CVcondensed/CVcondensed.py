# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 11:31:59 2019

This is a module to perform condensed CV analysis across many sensors and detectors. 
@author: Sneha Ramshanker 
"""
import Functions as fun
import matplotlib.pyplot as plt 
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
                  dest="file", default='Testdata\CVdata.csv')



options, arguments = parser.parse_args()

file = options.file

"""
Function Definitions
"""
def plotbycat(xaxis, yaxis, marker_cat, markers, df, impath, condid = "No", title = 0):
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
        title: String - Name of title of plot. By default it is "Type 3-2"
         
    Note:
        This is a function that produces condensed plots of data extracted from several detectors and wafers. 
    """
    
    meanarr = []
    
    
    for i in range(0, len(markers)):
        dftype = (df[df[marker_cat].str.contains(markers[i])])
        mean = dftype.groupby(xaxis, as_index = False).mean()
        mean[marker_cat] = markers[i]  
        meanarr.append(mean)
    
    
    marker_vis = ["X", "o","v" ]
    label = markers 
    
    if condid == "Cond":
        conddf = pd.DataFrame() #Condensed mean array 
    
        for i in range(0, len(meanarr)):
            conddf = pd.concat([conddf, meanarr[i]])
        
        condmean = conddf.groupby(xaxis, as_index = False).mean()
        
        
        condstd = conddf.groupby(xaxis, as_index = False).agg(np.std, ddof=0)
        yerr = list(condstd[yaxis])
        
    
        x = list(condmean[xaxis])
        y = list(condmean[yaxis])
        print(x)
        
        print(yerr)
        plt.errorbar(x, y, yerr, fmt = 'X', markersize='15', elinewidth=2)
        plt.grid('Both')
        plt.ylabel(yaxis)
        plt.xlabel(xaxis)
        if title != 0:
            plt.title(title)
        else:
            plt.title("Type 3-2" )
       
        plt.savefig(impath)
        plt.close()
    else:
        for i in range(0, len(meanarr)):
            ax = sns.stripplot(x = xaxis, y = yaxis, hue = marker_cat, color = 'b', jitter = 0.15,    marker=marker_vis[i], size = 12,  data = meanarr[i], label = marker_cat +str(label[i]))
            ax.grid("Both")
            
        
        legend_elements = [Line2D([0], [0], marker='X', color='w', label= markers[0],
                                  markerfacecolor='k', markersize=15),
                           Line2D([0], [0], marker='o', color='w', label= markers[1],
                                  markerfacecolor='k', markersize=15),
                           Line2D([0], [0], marker='v', color='w', label= markers[2],
                                  markerfacecolor='k', markersize=15)]
            
        lgd =  ax.legend( handles=legend_elements, loc='upper center', bbox_to_anchor=(1.15, 0.9),
                  ncol=1, fancybox=True, shadow=True)
        if title != 0:
            print("Testing")
            plt.title(title)
        else:
            plt.title("Type 3-2" )
        plt.savefig(impath, bbox_extra_artists=(lgd,), bbox_inches='tight')
        plt.close()

"""
Rest of the Program
"""

#Setting plot style
fun.setPlotStyle()
plt.rcParams['figure.figsize'] = (12, 10)

#Storing file in dataframe 
df = fun.simpletextread(file, ',')


PIN = df[df['Detector '].str.contains("PIN")] 
df.drop(PIN.index, axis=0,inplace=True) #Not including PIN values 
df['Detector '] = df['Detector '].astype(str).str[:-6] #Removing .txt 



#conditions = [df['Wafer'].str.contains('11'), df['Wafer'].str.contains('12'), df['Wafer'].str.contains('13'), df['Wafer'].str.contains('14')]
conditions = [df['Wafer'].str.contains('95'), df['Wafer'].str.contains('30')]

#choices = ['WNo11', 'WNo12', 'WNo13', 'WNo14']
choices = ['3.1', '3.2']
df['Type'] = np.select(conditions, choices, default='0')
    


"""
#Generating Regular CV Plots 
for i in range(0, len (df.columns[2:-1])):
    
    xaxis = "Wafer "
    yaxis =  df.columns[2:-1][i]
    marker_cat = "Detector " #Marker category
    markers = ["SE2", "SE3", "SE5"]
    if yaxis[-1] == " ":
        impath = "CV_plots/Combined_Plots/" + str(yaxis[:-1]) + "/Type-vs-SEn.png"  #MODIFY FOR USER
    else:
        impath = "CV_plots/Combined_Plots/" + str(yaxis) + "/Type-vs-SEn.png"   #MODIFY FOR USER
    if (os.path.isdir("CV_plots/Combined_Plots/" + str(yaxis))== False):   #MODIFY FOR USER
        os.mkdir("CV_plots/Combined_Plots/" + str(yaxis))   #MODIFY FOR USER
    if yaxis == 'Peak Depth (um)' or yaxis ==' Peak Value (cm^-3)' or yaxis =='Doping Width FWHM  (um)':
        plotbycat(xaxis, yaxis, marker_cat, markers, df, impath, title = "Gain Doping Profile - Type 3.2")
    else:
        plotbycat(xaxis, yaxis, marker_cat, markers, df,  impath)

#Generating Condensed CV Plots 

for i in range(0, len (df.columns[2:-1])):
    
    xaxis = "Wafer "
    yaxis =  df.columns[2:-1][i]
    marker_cat = "Detector " #Marker category
    markers = ["SE2", "SE3", "SE5"]
    if yaxis[-1] == " ":
        impath = "CV_plots/Combined_Plots/" + str(yaxis[:-1]) + "/Cond-Type-vs-SEn.png"  #MODIFY FOR USER
    else:
        impath = "CV_plots/Combined_Plots/" + str(yaxis) + "/Cond-Type-vs-SEn.png"    #MODIFY FOR USER
    if (os.path.isdir("CV_plots/Combined_Plots/" + str(yaxis))== False):    #MODIFY FOR USER
        os.mkdir("CV_plots/Combined_Plots/" + str(yaxis))      #MODIFY FOR USER
        
    if yaxis == 'Peak Depth (um)' or yaxis ==' Peak Value (cm^-3)' or yaxis =='Doping Width FWHM  (um)':
        plotbycat(xaxis, yaxis, marker_cat, markers, df,  impath, condid = "Cond", title = "Gain Doping Profile - Type 3.2")
    else:
        plotbycat(xaxis, yaxis, marker_cat, markers, df, impath, condid = "Cond")

"""

#Generating regular plots for type by type comparision
for i in range(0, len (df.columns[-4:-1])):
    
    xaxis = "Type"
    yaxis =  df.columns[-4:-1][i]
    marker_cat = "Detector " #Marker category
    markers = ["SE2", "SE3", "SE5"]
    if yaxis[-1] == " ":
        impath = "CV_plots/Combined_Plots/" + str(yaxis[:-1]) + "/Type-vs-SEn.png" #MODIFY FOR USER
    else:
        impath = "CV_plots/Combined_Plots/" + str(yaxis) + "/Type-vs-SEn.png" #MODIFY FOR USER
    if (os.path.isdir("CV_plots/Combined_Plots/" + str(yaxis))== False):       #MODIFY FOR USER 
        os.mkdir("CV_plots/Combined_Plots/" + str(yaxis))     #MODIFY FOR USER 
        

    plotbycat(xaxis, yaxis, marker_cat, markers, df,  impath, title = "Type 3-2 vs Type 3-1")

#Generating Condensed plots for type by type comparision
for i in range(0, len (df.columns[-4:-1])):
    
    xaxis = "Type"
    yaxis =  df.columns[-4:-1][i]
    marker_cat = "Detector " #Marker category
    markers = ["SE2", "SE3", "SE5"]
    if yaxis[-1] == " ":
        impath = "CV_plots/Combined_Plots/" + str(yaxis[:-1]) + "/Cond-Type-vs-SEn.png"  #MODIFY FOR USER
    else:
        impath = "CV_plots/Combined_Plots/" + str(yaxis) + "/Cond-Type-vs-SEn.png"   #MODIFY FOR USER
    if (os.path.isdir("CV_plots/Combined_Plots/" + str(yaxis))== False):    #MODIFY FOR USER
        os.mkdir("CV_plots/Combined_Plots/" + str(yaxis))   #MODIFY FOR USER
        
    plotbycat(xaxis, yaxis, marker_cat, markers, df, impath, condid = "Cond", title = "Type 3-2 vs Type 3-1")