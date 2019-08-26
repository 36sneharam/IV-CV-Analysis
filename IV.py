# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 15:11:15 2019

This is a module for plotting IV curves and determining the breakdown voltage. All user specific values (ex. file path, image path etc.) can be specified using optional parsers.  
@author: Sneha Ramshanker 

"""

import Functions as fun
import Constants as con
from optparse import OptionParser
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


"""
Optional Parsing
"""

parser = OptionParser()
parser.add_option("--file", type="string",
                  help="Path of file",
                  dest="file", default="Datasets\IV\EDX30329-WNo10\IV\LG1-SE5-3.txt")

parser.add_option("--impathIV",type = "string", help = "Path to store IV plot", dest = "impathIV", default = "IV_plots/test.png")
parser.add_option("--xlrow", type = "int", help = "What row of the excel file should this data be stored in", dest = "xlrow", default = 2)
parser.add_option("--xlpath", type = "string", help = "Path of excel file to store results", dest = "xlpath", default = 'Testdata/Breakdown.xlsx')

options, arguments = parser.parse_args()

file = options.file
impathIV = options.impathIV
xlrow = options.xlrow 
xlpath = options.xlpath

"""
Rest of the program
"""
fun.setPlotStyle() #Setting format of the graph


df = fun.storedataIV(file)
df = fun.cleanupIV(df, "Sweep Voltage", "pad")
bvol = fun.bvol(df, "Sweep Voltage", "pad")
fun.dataplot(df, 'Sweep Voltage', 'pad', impathIV, 'ylog', 'Voltage [V]', 'Current [A]', iden = "IV", bvol = bvol)

#Storing data in an excel file 
fun.tablebvol(xlpath, bvol ,'C', 'Breakdown Voltage')


"""
Determining Uncertainities - threshold 

# If i = 1, columns [0, 1] 2(1)-1
# If i = 2, columns [2, 3] 2(2) - 1
# If i = 3, columns [4, 5] 2(3) - 1
for i in range (1, 7):
    column_name = ['D', 'E', 'F', 'G', 'H', 'I']
    bvol = fun.bvol(df, "Sweep Voltage", "pad", thresh = 0.9995-0.005*i)
    fun.dataplot(df, 'Sweep Voltage', 'pad', impathIV[:-4]+str(i)+".png", 'ylog', 'Voltage [V]', 'Current [A]', iden = "IV", bvol = bvol)
    fun.tablebvol(xlpath, bvol ,column_name[i-1], 'Breakdown Voltage')
"""
