# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 09:30:06 2019

This is the steering module for performing IV analysis on multiple wafers. It should be noted that the wafer and file directories are user specific and should be modified for the each user. The points to modify as commented below with the tag "MODIFY FOR USER"

@author: Sneha Ramshanker
"""
import os
import Functions as fun


wafers = ['EDX30329-WNo10', 'EDX30329-WNo11', 'EDX30329-WNo12', 'EDX30329-WNo9', 'EXX30327-WNo1', 'EXX30327-WNo3', 'EXX30327-WNo4', 'EXX30327-WNo5', 'EXX30327-WNo6', 'EXX30328-WNo1', 'EXX30328-WNo2', 'EXX30328-WNo3', 'EXX30328-WNo4', 'EXX30328-WNo5', 'EXX30330-WNo11', 'EXX30330-WNo12', 'EXX30330-WNo13', 'EXX30330-WNo14', 'EXX30327-WNo1', 'EDX30327-WNo3', 'EDX30327-WNo4',  'EDX30327-WNo5',  'EDX30327-WNo6',  'EDX30328-WNo1', 'EDX30328-WNo2',  'EDX30328-WNo3', 'EDX30328-WNo4', 'EDX30328-WNo5',   'EDX30330-WNo11',  'EDX30330-WNo12',  'EDX30330-WNo13', 'EDX30330-WNo14']
xlpath = 'Testdata/Breakdown.xlsx' #MODIFY FOR USER (path to the excel file)





for n in range(0, len(wafers)):
    #Making a new folder for the wafer 
    if (os.path.isdir("IV_plots/"+wafers[n])== False):
        os.mkdir("IV_plots/"+wafers[n]) #MODIFY FOR USER     
    waferdir = "Datasets"+ "/" + "IV"+ "/" + wafers[n] + "/" + "IV" #MODIFY FOR USER (where do you want to store the files)
    filelist = os.listdir(waferdir) 
    for i in range (1, len(filelist)):
        filedir = waferdir + "/"+filelist[i]
        impathIV = "IV_plots/"+wafers[n] + "/" + filelist[i][:-4] + ".png" #MODIFY FOR USER (where do you want to store the files)

        fun.tablebvol(xlpath,wafers[n],'A', 'Breakdown Voltage')
        fun.tablebvol(xlpath, filelist[i],'B', 'Breakdown Voltage')



        print(impathIV)
        os.system("python IV.py --file " + filedir +  " --impathIV "+ impathIV+ " --xlpath "+xlpath) 

input("Press enter to close")