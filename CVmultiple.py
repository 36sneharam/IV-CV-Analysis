# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 09:38:27 2019

This is the steering module for performing CV analysis on multiple wafers. It should be noted that the wafer and file directories are user specific and should be modified for the each user. The points to modify as commented below with the tag "MODIFY FOR USER"
@author: sneha
"""
import os 
import Functions as fun


#wafers = ['EXX30330-WNo14-Type3-2-NewMeas','EXX30330-WNo14-Type3-2' ] #, 'EXX30330-WNo12-Type3-2', 'EXX30330-WNo13-Type3-2', 'EXX30330-WNo14-Type3-2', 'SE5-redo', 'EXX30330-WNo14-Type3-2-NewMeas']
wafers = [ 'EXX28995-WNo4-Type3-1']

#Optional parsers 
"""
file = options.file
impathCV = options.impathCV
impathDPW = options.impathDPW
delim = options.delim
"""
xlpath = 'Testdata/CV31.xlsx' #MODIFY FOR USER
id = "Path 1" #Specifying path of the file 
for n in range(0, len(wafers)):
    if (os.path.isdir("CV_plots/"+wafers[n])== False): #MODIFY FOR USER
        os.mkdir("CV_plots/"+wafers[n]) #MODIFY FOR USER
    if (os.path.isdir("DPW_plots/"+wafers[n])== False): #MODIFY FOR USER
        os.mkdir("DPW_plots/"+wafers[n])#MODIFY FOR USER
    if (os.path.isdir("CVinv_plots/"+wafers[n])== False): #MODIFY FOR USER
        os.mkdir("CVinv_plots/"+wafers[n]) #MODIFY FOR USER

    waferdir = "Datasets"+ "/" + "CV"+ "/" + wafers[n] #MODIFY FOR USER
    
    #waferdir = "Datasets"+ "/" + "CV"+ "/" +"SE2-redo"+"/"+ wafers[n] #MODIFY FOR USER

    filelist = os.listdir(waferdir)
    for i in range (0, len(filelist)):
        filedir = waferdir + "/"+filelist[i]
        
        impathCV = "CV_plots/"+wafers[n] + "/" + filelist[i][:-4] + ".png" #MODIFY FOR USER
        impathDPW = "DPW_plots/"+ wafers[n] + "/" + filelist[i][:-4] + ".png" #MODIFY FOR USER
        impathCVinv = "CVinv_plots/" + wafers[n] + "/" + filelist[i][:-4]+".png" #MODIFY FOR USER
        impathDPWcut = "DPW_plots/"+ wafers[n] + "/" + filelist[i][:-4] + "_cut"+".png" #MODIFY FOR USER
        
        fun.tablebvol(xlpath,wafers[n],'A', 'CV')
        fun.tablebvol(xlpath, filelist[i],'B', 'CV')
        
        
        print(impathCV)
        os.system("python CV.py --file " + filedir +  " --impathDPW "+ impathDPW + " --impathCV " + impathCV + " --impathCVinv " + impathCVinv + " --impathDPWcut " + impathDPWcut) 
        
input("Press enter to close")
          
