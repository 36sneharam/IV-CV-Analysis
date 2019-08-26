# IV/CV Analysis 
This program performs analysis on IV data and CV data. There are 6 modules for this program: 
1. **IV.py** : Plots and saves the IV curves and determines the breakdown voltage for a single datafile 
*Required Local Modules*: Functions.py, Constants.py 
2. **IVmultiple.py**: Steering the macro for saving IV curves and breakdown voltages for many datafiles. This macro calls IV.py for each input datafile. 
*Required Local Modules*: Functions.py, Constants.py, IV.py 
3. **CV.py**: Plots and saves the CV curves, Inverse Squared CV curves (InvCV), and Doping Profile vs width curves (DPW) for a single datafile 
*Required Local Modules*: Functions.py, Constants.py 
4. **CVmultiple.py**: Steering the macro for saving CV curves, Inverse Squared CV curves (InvCV), and Doping Profile vs width curves (DPW) for multiple datafiles. This macro calls CV.py for each input datafile.
*Required Local Modules*: Functions.py, Constants.py, CV.py 
5. Functions.py: Main functions library that contains all the functions used by IV.py and CV.py 
6. Constants.py: Contains values of all constants used by IV.py and CV.py (example: vacuum permittivity, charge of an electron, area of the detector etc.)
*Required Local Modules*: Constants.py 


Note: It is recommended to save all these modules in the same directory 

### Functions Library (Functions.py)
**Required python libraries**: numpy, pandas, matplotlib.pyplot, csv, scipy (stats, optimize.fsolve, interpolate.UnivariateSpline), seaborn, openpyxl

Most of these libraries should come built-in with python 
To install openpyxl: 
```bash
$ pip install openpyxl
```
To install seaborn
```bash
$ pip install seaborn
```
To install scipy 
```bash
$ pip install scipy
```

This is the functions library that contains all the functions used by IV.py, IVmultiple.py, CV.py, and CVmultiple.py. The functions are listed in alphabetical order and descriptions of the function and all the input parameters, optional input paramaters, return parameters are provided in the module. 

Example 1: This is a function found in Functions.py to determine the breakdown voltage 

```python 
def bvol(df, x, y, thresh = 0.9995):
    """
    Parameters:
        df: Pandas dataframe 
        x: String - name of x column in df (voltage)
        y: String - name of y column in df (capacitance)
    Optional Parameters: 
        thresh: Float - r squared threshold value
    Returns:
        bvol: Float - Breakdown Voltage 
    Note: 
        This function determines the breakdown by calculation the point where the relationship between the first and second derivative becomes extremely linear. 
    """
    frst_der = manual_der(df[x], df[y])
    scnd_der = manual_der(df[x], frst_der)
    
    r_value = 1
    N = 2
    while round(r_value,4) >= thresh: #Threshold for linearity
        X = np.log(frst_der[-N:])
        Y = np.log(scnd_der[-N:])
        slope, intercept, r_value, p_value, std_err = stats.linregress(X,Y)
        N = N+1
          
    voltage = df[x].as_matrix()
    bvol = voltage[-N+1]
    return bvol
```

As seen by the comments in the function, this function takes a dataframe, and two strings (x and y column names) as inputs and returns the breakdown voltage as a float. It has an optional parameter that can be left to its default value of 0.9995 or changed by passing an input. For example, if we want to change the threshold voltage to 0.8, the function can be called in the following way:
```python 
bvol(df, "Sweep Voltage", "pad", thresh = 0.8 )
```

### Constants library (Constants.py)

**Constants listed in this module**: vacuum permittivity (eps0), silicon permittivity factor (eps_si), electric charge of an electron (q), several scaling parameters, area of detector 


### IV Analysis (IV.py and IVmultiple.py)
**Required python libraries**: optparse, numpy, pandas,  seaborn, matplotlib.pyplot 

#### IV.py
This module performs IV analysis for a single datafile. It plots the IV curves, determines the breakdown voltage and has the capability of storing the breakdown voltage in an excel file. 

The input datafile (.txt file) should have the following format. 
```text 
,conc,
Sweep Voltage,pad,ring,v,NTC1  NTC2  SHT_T   Hum%
  22.27  22.65  22.26  54.46
0.0E+0,468.7600E-12,7.9998E-12,75.1162E-12
-2.0E+0,20.9990E-12,282.0000E-12,-318.7424E-12
-4.0E+0,34.4990E-12,305.4900E-12,-353.9816E-12
-6.0E+0,42.9990E-12,307.4900E-12,-368.7789E-12
-8.0E+0,48.4980E-12,309.9900E-12,-392.4886E-12
-10.0E+0,51.4990E-12,312.9900E-12,-368.4350E-12
-12.0E+0,53.4990E-12,316.4900E-12,-416.4912E-12
-14.0E+0,54.4990E-12,319.4900E-12,-419.4127E-12
-16.0E+0,56.0000E-12,323.0000E-12,-450.1472E-12
-18.0E+0,56.4990E-12,325.5000E-12,-469.7482E-12
-20.0E+0,56.9980E-12,327.9900E-12,-478.8545E-12
-22.0E+0,57.4990E-12,329.9900E-12,-394.6660E-12
```
If this is not the format of the file, the function cleanupIV in the Functions.py module needs to be modified to ensure that the data is loaded properly into the pandas dataframe.

This module takes four input parameters:
1. *file*: path of the input file (can be absolute or relative path)  
For example:
```bash 
"Datasets\IV\EDX30329-WNo10\IV\LG1-SE5-3.txt"
```
2. *impathIV*: path where you want to store the IV plots generated by the code
For example:
```bash 
"IV_plots/test.png"
```
3. *xlrow*: What row of the excel file should this data be stored in
For example:
```bash 
2
```
4. *xlpath*: Path of excel file to store breakdown voltage results 
For example:
```bash 
"Testdata/Breakdown.xlsx"
```
All these parameters can be parsed using option parsing (--file, --impathIV, --xlrow, --xlpath) 

Note: If you don't want to save the breakdown voltage in the excel file comment line 51. 

The output should be a saved plot that looks similar to this:

#### IVmultiple.py 
**Required python libraries**: os
This module calls IV.py for each input datafile. This macro is very user-specific and should be modified by the user accordingly. The points to modify are marked by the comment:
```python 
#MODIFY FOR USER" 
```
The macro uses the os library to parse the datafile path and the image path as an optional parser for IV.py. 

Example: The datafiles are stored in two folders whose paths are 
```bash 
"Datasets/IV/EDX30329/IV"
"Datasets/IV/EDX30329-WNo11/IV"
``` 
If we want to save the IV plots in two folder whose paths are 
```bash 
"IV_plots/EDX30329/"
"IV_plots/EDX30329-WNo11/"
``` 
with the same name as the .txt datafile (so IV plots from LG1-SE5-1.txt will be saved as LG1-SE5-1.png), we can run the following code:
```python 
import os
import Functions as fun
wafers = ['EDX30329-WNo10']
xlpath = 'Testdata/Breakdown.xlsx' #MODIFY FOR USER (path to the excel file)
for n in range(0, len(wafers)):
    #Making a new folder for the wafer 
    if (os.path.isdir("IV_plots/"+wafers[n])== False):
        os.mkdir("IV_plots/"+wafers[n]) #MODIFY FOR USER     
    waferdir = "Datasets"+ "/" + "IV"+ "/" + wafers[n] + "/" + "IV" #MODIFY FOR USER (where do you want to store the files)
    filelist = os.listdir(waferdir) 
    for i in range (1, len(filelist)):
        filedir = waferdir + "/"+filelist[i]
        impathIV = "IV_plots/"+wafers[n] + "/" + filelist[i][:-4] + ".png" 

        os.system("python IV.py --file " + filedir +  " --impathIV "+ impathIV) 
input("Press enter to close")
```

There is also a built in capability to save the wafer number, detector type and breakdown voltage as columns in an excel file. xlpath specifies the path of the excel file and the following code stores the wafer number and detector type in columns A and B of the excel file (breakdown voltage is stored in IV.py) 
```python 
fun.tablebvol(xlpath,wafers[n],'A', 'Breakdown Voltage')
fun.tablebvol(xlpath, filelist[i],'B', 'Breakdown Voltage')
os.system("python IV.py --file " + filedir +  " --impathIV "+ impathIV+ " --xlpath "+xlpath) 
```

### CV Analysis (CV.py and CVmultiple.py)
**Required python libraries**: optparse, numpy, pandas,  seaborn, matplotlib.pyplot , astropy.modeling

#### CV.py
This module performs CV analysis for a single datafile. It plots the CV curves, plots inverse capacitance vs voltage, and plots the Doping profile vs width (determined using the CV data). All these plots are saved as .png images. The program also determines the following parameters:
1. *Depletion Voltage of the Gain Layer*: Determined by fitting two lines and determining the point of intersection for the invCV graph 
2. *Total Depletion Voltage* :  Determined by fitting two lines and determining the point of intersection for the invCV graph
3. *Gain Doping Peak*: Determining by finding the maximum value in a segment of the data (segment is arbitrary selected)
4. *Width(FWHM) of the gain peak*: Determined by fitting a gaussian on the doping profile and finding the full width half maximum.  

All of these parameters can be stored in an excel file using the function tablebvol (described in Functions.py). At the moment, the first three parameters are being saved in the excel file. 

The input datafile (.txt file) should have the following format. 
```text 

NTC1  NTC2  SHT_T   Hum%
  23.96  25.02  23.86  53.00
Voltage (V), Current (A), Capacitance (F), D
0.000000,1.258115E-8,1.368250E-10,-0.003756
-1.000000,-3.218570E-7,1.134470E-10,0.002710
-2.000000,-2.975935E-7,1.060750E-10,0.003777
-3.000000,-4.110802E-8,1.007440E-10,0.005310
-4.000000,-1.907544E-6,9.787290E-11,0.005562
-5.000000,-8.710705E-8,9.571450E-11,0.005708
-6.000000,-1.983463E-6,9.400800E-11,0.005775
```
If this is not the format of the file, the function cleanupCV in the Functions.py module needs to be modified to ensure that the data is loaded properly into the pandas dataframe.

This module takes 7 input parameters:
1. *file*: path of the input file (can be absolute or relative path)  
For example:
```bash 
"Datasets\CV\EXX30330-WNo12-Type3-2\LG1-SE3-1.txt"
```
2. *impathCV*: path where you want to store the CV plots generated by the code
For example:
```bash 
"CV_plots/a1.png"
```
3. *impathDPW*: path where you want to store Doping Profile vs Width plot
For example:
```bash 
"DPW_plots/a1.png"
```
4. *impathCVinv*: path where you want to store inverse squared CV plot
For example:
```bash 
"CVinv_plots/b1.png"
```
5. *impathDPWcut*: path to store DPW plots with log scale. Note this is not super important and can be removed if this parameter is not needed. 
```bash 
"DPW_plots/a2.png"
```
6. *xlpath*: Path of excel file to store all the results 
For example:
```bash 
"Testdata/Breakdown.xlsx"
```
7. *delim*: What king of delimiter is separating your data
```bash 
"\t"
```

All these parameters can be parsed using option parsing (--file, --impathCV, --impathDPW, --impathCVinv, --impathDPWcut, --delim, --xlpath) 



#### CVmultiple.py 
**Required python libraries**: os
This module calls CV.py for each input datafile. This macro is very user-specific and should be modified by the user accordingly. The points to modify are marked by the comment:
```python 
#MODIFY FOR USER" 
```
The macro uses the os library to parse the datafile path and the image path as an optional parser for CV.py. 

Example: The datafiles are stored in two folders whose paths are 
```bash 
"Datasets/CV/EXX30330-WNo11-Type3-2"
"Datasets/CV/EXX30330-WNo12-Type3-2"
``` 
If we want to save the CV plots, DPW plots, and CVinv_plots in the following folders whose paths are 
```bash 
"CV_plots/EXX30330-WNo11-Type3-2/"
"CV_plots/EXX30330-WNo12-Type3-2/"
"DPW_plots/EXX30330-WNo11-Type3-2/"
"DPW_plots/EXX30330-WNo12-Type3-2/"
"CVinv_plots/EXX30330-WNo11-Type3-2/"
"CVinv_plots/EXX30330-WNo12-Type3-2/"
``` 
with the same name as the .txt datafile (so IV plots from LG1-SE5-1.txt will be saved as LG1-SE5-1.png), we can run the following code:
```python 
import os 
import Functions as fun

wafers = ['EXX30330-WNo11-Type3-2', 'EXX30330-WNo12-Type3-2']
xlpath = 'Testdata/result.xlsx' #MODIFY FOR USER

for n in range(0, len(wafers)):
    if (os.path.isdir("CV_plots/"+wafers[n])== False): #MODIFY FOR USER
        os.mkdir("CV_plots/"+wafers[n]) #MODIFY FOR USER
    if (os.path.isdir("DPW_plots/"+wafers[n])== False): #MODIFY FOR USER
        os.mkdir("DPW_plots/"+wafers[n])#MODIFY FOR USER
    if (os.path.isdir("CVinv_plots/"+wafers[n])== False): #MODIFY FOR USER
        os.mkdir("CVinv_plots/"+wafers[n]) #MODIFY FOR USER

    waferdir = "Datasets"+ "/" + "CV"+ "/" + wafers[n] #MODIFY FOR USER
    filelist = os.listdir(waferdir)
    for i in range (1, len(filelist)):
        filedir = waferdir + "/"+filelist[i]
        
        impathCV = "CV_plots/"+wafers[n] + "/" + filelist[i][:-4] + ".png" #MODIFY FOR USER
        impathDPW = "DPW_plots/"+ wafers[n] + "/" + filelist[i][:-4] + ".png" #MODIFY FOR USER
        impathCVinv = "CVinv_plots/" + wafers[n] + "/" + filelist[i][:-4]+".png" #MODIFY FOR USER
        impathDPWcut = "DPW_plots/"+ wafers[n] + "/" + filelist[i][:-4] + "_cut"+".png" #MODIFY FOR USER
        
        
        print(impathCV)
        os.system("python CV.py --file " + filedir +  " --impathDPW "+ impathDPW + " --impathCV " + impathCV + " --impathCVinv " + impathCVinv + " --impathDPWcut " + impathDPWcut) 
        
input("Press enter to close")
          

```

There is also a built in capability to save all the calculated parameters as columns in an excel file. xlpath specifies the path of the excel file and the following code stores the wafer number and detector type in columns A and B of sheets 'Gain Width' and 'Depletion Voltage' of the excel file (All the specific data is stored in CV.py)

```python 
fun.tablebvol(xlpath,wafers[n],'A', 'Gain Width')
fun.tablebvol(xlpath, filelist[i],'B', 'Gain Width')
fun.tablebvol(xlpath, wafers[n],'A', 'Depletion Voltage')
fun.tablebvol(xlpath, filelist[i],'B', 'Depletion Voltage')
os.system("python IV.py --file " + filedir +  " --impathIV "+ impathIV+ " --xlpath "+xlpath) 
```

***
For further questions email sneha.ramshanker@pmb.ox.ac.uk










