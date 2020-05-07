import sys, os
import urllib.parse
import urllib.request
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
from cdo_api_py import Client
from datetime import datetime
import scipy as si
from scipy import stats
import pylab as py
from tkinter import *
from tkinter import ttk
import urllib.parse
import urllib.request
import urllib.parse
import urllib.request

# Daily Flow Data Download and Checking Function:
def DownloadData(station_number, begin_date, end_date):
    # Create Directory to store files
    if not os.path.exists('./Results/DailyFlow'):
        os.makedirs('./Results/DailyFlow')
    # 1.# Define a function for obtaining the daily flow data from USGS Surface Data Portal
    ## Parameters - Station number and folder name
    def GetPeakFlowYear(station_number, FolderName, begin_date, end_date):
        ## Building URLs
        var1 = {'site_no': station_number}
        var2 = {'begin_date': begin_date}
        var3 = {'end_date': end_date}
        part1 = 'https://nwis.waterdata.usgs.gov/nwis/dv?'
        part2 = 'cb_00060=on&format=rdb&'
        part3 = '&referred_module=sw&period=&'
        part4 = '&'
        link = (part1 + part2 + urllib.parse.urlencode(var1) + part3 + urllib.parse.urlencode(var2) + part4 + urllib.parse.urlencode(var3))
        print('The USGS Link is: \n',link)

        ## Opening the link & retrieving data
        response = urllib.request.urlopen(link)
        page_data = response.read()

        ## File name assigning & storing the rav data as text file
        with open(FolderName+'Data_' + station_number + '_raw' + '.csv', 'wb') as f1:
            f1.write(page_data)
        f1.close

    # define and initialize the missing data dictionary
    ReplacedValuesDF = pd.DataFrame(0, index=["1. No Data","2. Gross Error"],
                                    columns=['Discharge'])

    ## Main Code
    #station_number=input('Enter UHC8 Number of the required Station (USGS Station Number/site_no) \t') #04180000
    #begin_date=input('Enter begin date (format:yyyy-mm-dd) \t') #2019-01-01
    #end_date=input('Enter end date (format:yyyy-mm-dd) \t') #2019-12-31
    #print('\t')

    ## Assigning the location for storing the data
    ## First Method
    FolderName='./Results/DailyFlow/'
    if os.path.exists(FolderName) == False:
        os.mkdir(FolderName)

    dailyflow_list_wb=GetPeakFlowYear(station_number,FolderName,begin_date,end_date)

    # read discharge data and give name for needed columns 
    data = pd.read_csv(FolderName+'Data_' + station_number + '_raw' + '.csv',skiprows=30,
    header=None,sep='\t',usecols=[2,3],names=['Timestamp','Discharge'])
    data.Timestamp=pd.to_datetime(data.Timestamp)
    data = data.set_index('Timestamp')

    # Check01 Remove No Data values
    # record the number of values with NaN and delete them

    ReplacedValuesDF['Discharge'][0]= data.isna().sum()

    data = data.dropna()

    # Check02 Check for Gross Errors
    # find the index of the gross errors for Discharge and record the number
    # the threshold is Q ≥ 0

    index=(data>=0)

    ReplacedValuesDF['Discharge'][1]= len(index)-index.sum()

    # delete the gross errors 

    data = data[index].dropna()
    
    # Calculate Tqmean values
    def CalcTqmean(Qvalues):    
        # calculate the number of values bigger than yearly mean value
        Tqmean =((Qvalues>Qvalues.mean()).sum())/len(Qvalues)
        return ( Tqmean )
    
    def CalcRBindex(Qvalues):
        # Drop None values
        Qvalues=Qvalues.dropna()
        dif=Qvalues.diff()
        dif=dif.dropna()
        # Calculate the sum of absolute values of day-to-day discharge change
        Total_abs=abs(dif).sum()
        # Total yearly discharge
        Total_dis=Qvalues.sum()
        # R-B index
        RBindex=Total_abs/Total_dis
        return ( RBindex )

    def Calc7Q(Qvalues):
        # Drop None values
        Qvalues=Qvalues.dropna()
        # Calculate the rolling 7-day minimum valuefor a year
        val7Q=Qvalues.rolling(window=7).mean().min()
        return ( val7Q )
    
    def CalcExceed3TimesMedian(Qvalues):
        Qvalues=Qvalues.dropna()
        median_year=Qvalues.median() #calculate median for each year
        median3x=(Qvalues>(3*median_year)).sum() #determine number of days where flow was 3 times the median
        return ( median3x )
    
    def GetAnnualStatistics(DataDF):
        global WYDataDF
        #create empty dataframe to fill with values
        colnames=['Mean Flow', 'Peak Flow', 'Median Flow', 'Coeff Var', 'Skew', 'Tqmean', 'R-B Index', '7Q', '3xMedian']
        year_data=DataDF.resample('AS-OCT').mean()
        WYDataDF = pd.DataFrame(0,index=year_data.index, columns=colnames)

        #fill data frame with annual values for the water year, water yer starts on october 1
        WYDataDF['Mean Flow']= DataDF.resample("AS-OCT")['Discharge'].mean()
        WYDataDF['Peak Flow']=DataDF.resample("AS-OCT")['Discharge'].max()
        WYDataDF['Median Flow']=DataDF.resample("AS-OCT")['Discharge'].median()
        WYDataDF['Coeff Var']=(DataDF.resample("AS-OCT")['Discharge'].std()/DataDF.resample("AS-OCT")['Discharge'].mean())*100.0
        WYDataDF['Skew']=DataDF.resample("AS-OCT")['Discharge'].apply(stats.skew)
        WYDataDF['Tqmean']=DataDF.resample("AS-OCT").apply({'Discharge':lambda x:CalcTqmean(x)}) #use .apply and lambda when using a custom function
        WYDataDF['3xMedian']=DataDF.resample("AS-OCT").apply({'Discharge':lambda x:CalcExceed3TimesMedian(x)})
        WYDataDF['7Q']=DataDF['Discharge'].resample("AS-OCT").apply({lambda x:Calc7Q(x)})
        WYDataDF['R-B Index']=DataDF['Discharge'].resample("AS-OCT").apply({lambda x:CalcRBindex(x)})
        #WYDataDF['site_no']=DataDF.resample('AS-OCT')['site_no'].mean()   

        return ( WYDataDF )

    ## Graphical analysis after data quality checking
    # plot daily streamflow hydrograph
    data.plot(figsize=(15,7))
    plt.xlabel('Time')
    plt.ylabel('Discharge (cfs)')
    plt.title('Daily Discharge Hydrograph')
    plt.legend(["Discharge"])
    plt.savefig(FolderName + "Daily Discharge.png")
    

    ## QQplot 
    plt.figure(figsize=(15,7))
    data_points=data.Discharge
    si.stats.probplot(np.log(data_points), dist='norm', plot=plt)
    plt.savefig(FolderName + "Q-Q Plot of Discharge.png")
    

    ## Boxplot
    plt.figure(figsize=(15,7))
    plt.boxplot(data.Discharge,whis=3)
    plt.savefig(FolderName + "Boxplot of Discharge.png")
    
    
    # Save the data quality checking results into a file
    ReplacedValuesDF.to_csv("Results/DailyFlow/Checked_Summary(Dailyflow).txt", sep="\t")

    # calculate descriptive statistics for each water year
    WYDataDF = GetAnnualStatistics(data)
    # Write data into annual metrics csv file
    WYDataDF.to_csv('Results/DailyFlow/Annual_Metrics.csv',sep=',', index=True)
    
    print("Discharge data processing is done!")
# DownloadData('03335500', '2019-01-01', '2019-12-31')

# Peak Flow Data Download and Checking Function:
def GetPeakFlow(station_number):
    ## Plot the return period by peak flow data
    ## Define a function for obtaining the peak flow data from USGS Surface Data Portal
    ## Parameters - Station number and folder name
    # Create Directory to store files
    if not os.path.exists('./Results/PeakFlow'):
        os.makedirs('./Results/PeakFlow')

    def GetPeakFlowYear(station_number, FolderName):
        ## Building URLs
        var1 = {'site_no': station_number}
        part1 = 'https://nwis.waterdata.usgs.gov/nwis/peak?'
        part2 = '&agency_cd=USGS&format=rdb'
        link = (part1 + urllib.parse.urlencode(var1) + part2)
        print('The USGS Link is: \n',link)

        ## Opening the link & retrieving data
        response = urllib.request.urlopen(link)
        page_data = response.read()

        ## File name assigning & storing the rav data as text file
        with open(FolderName+'Data_' + station_number + '_raw' + '.csv', 'wb') as f1:
            f1.write(page_data)
        f1.close

    ## Main Code
    #station_number=input('Enter UHC8 Number of the required Station (USGS Station Number/site_no) \t')
    #print('\t')
    ## Assigning the location for storing the data
    FolderName='./Results/PeakFlow/'
    peakflow_list_wb=GetPeakFlowYear(station_number,FolderName)

    # read discharge data and give name for needed columns 
    data = pd.read_csv(FolderName+'Data_' + station_number + '_raw' + '.csv',skiprows=74,
    header=None,sep='\t',usecols=[2,4],na_filter=True,names=['Timestamp','Peak_Discharge'])
    
    # delete empty data
    data = data.dropna()
    data = data.reset_index(drop=True)

    # get the values of mean and standard derivation of peak discharge data 
    discharge_mean=np.mean(data['Peak_Discharge'])
    discharge_sd=np.std(data['Peak_Discharge'])

    # calculate the streamflow of different return years 
    ReturnPeriod = [10, 25, 50, 100, 500]
    StreamFlow = []
    for i in ReturnPeriod:
        a = discharge_mean - (math.sqrt(6) / math.pi) * (0.5772 + math.log (math.log ( i / (i-1)))) * discharge_sd
        StreamFlow.append(a)

    # plot streamflow hydrograph
    plt.figure(figsize=(15,7))
    plt.plot(ReturnPeriod,StreamFlow)
    plt.xlabel('Return Period (yrs)')
    plt.ylabel('Discharge (cfs)')
    plt.title('Extreme Discharge Hydrograph')
    plt.savefig(FolderName + "Extreme Discharge Hydrograph.png")

    print("Return period discharge data processing is done!")


    ## create a new output file to store the checked results
    #ReplacedValuesDF.to_csv("Checked_Summary(PeakFlow).txt", sep="\t")

    print("All data processing is done!")

def GETPCP(token, NSEW, start_date, end_date):
    # Extract rainfall data
    # the Client object helps you acess the NCDC database with your token
    # Create Directory to store files
    if not os.path.exists('./Results/Precipitation'):
        os.makedirs('./Results/Precipitation')

    my_client = Client(token, default_units='None', default_limit=1000)

    # The extend is the lat, long of the target region.
    extent = dict()
    Dirs = ['north','south','east','west']
    #NSEW = input('Enter the extent, format:"N,S,E,W":')
    temp = NSEW.split(',')
    for i in range(len(Dirs)):
        extent[Dirs[i]] = float(temp[i])

    # Displaying the dictionary
    for key, value in extent.items():
        print(str(key)+':'+str(value)) #extent = 41.53,41.21,-84.90,-85.33

    # input of start data, end date, type of dataset, and name of gauge
    #start_date = input('Enter begin date (format:yyyy-mm-dd) \t') # 2019-01-01
    startdate = pd.to_datetime(start_date)
    #end_date = input('Enter end date (format:yyyy-mm-dd) \t') #2019-12-31
    enddate = pd.to_datetime(end_date)

    datasetid = 'GHCND'

    FolderName='./Results/Precipitation/'

    #The find_station function returns the dataframe containing stations' info within the input extent.
    stations = my_client.find_stations(
               datasetid = datasetid,
               extent = extent,
               startdate = startdate,
               enddate = enddate,
               return_dataframe = True)

    # download data from all stations specified by their id
    for i in stations.id:
        Rainfall_data = my_client.get_data_by_station(datasetid = datasetid, stationid = i,
                   startdate = startdate, enddate = enddate, return_dataframe = True,
                   include_station_meta = True)
        station_id = i.split(":")[1]
        filename = datasetid + '_' + station_id + '.csv'
        Rainfall_data.to_csv(FolderName+filename)

    # Get the daily average values of  all stations
    P=[]
    for i in stations.id:
        station_id = i.split(":")[1]
        filename = datasetid + '_' + station_id + '.csv'
        df = pd.read_csv(FolderName+filename)
        df = df[['date','PRCP']]
        P.append(df)
    data=pd.concat(P,axis=0,ignore_index=True)
    data.to_csv(FolderName+'P.csv')
    data = data.set_index('date')

    # Check01 Remove No Data values
    # define and initialize the missing data dictionary
    ReplacedValuesDF = pd.DataFrame(0, index=["1. No Data","2. Gross Error"],
                                    columns=['Precipitation'])
    # record the number of values with NaN and replace with zero

    ReplacedValuesDF['Precipitation'][0]= data.isna().sum()

    data = data.fillna(0)

    # Check02 Check for Gross Errors
    # find the index of the gross errors for Discharge and record the number
    # the threshold is 0≤ P ≤ 1870mm (the max daily rainfall in the world)

    # unit of raw data is 0.1mm
    index=(data>=0) & (data<=18700)

    ReplacedValuesDF['Precipitation'][1]= len(index)-index.sum()

    # delete the gross errors 

    data = data[index].dropna()

    data=data.groupby('date')['PRCP'].mean()

    # cover the unit into inch
    data=data/10/25.4
    data.index = pd.to_datetime(data.index)
    
    def GetAnnualStatistics(DataDF):
        global WYDataDF
        #create empty dataframe to fill with values
        colnames=['Mean Precip', 'Peak Precip']
        year_data=DataDF.resample('AS-OCT').mean()
        WYDataDF = pd.DataFrame(0,index=year_data.index, columns=colnames)

        #fill data frame with annual values for the water year, water yer starts on october 1
        WYDataDF['Mean Precip']= DataDF.resample("AS-OCT").mean()
        WYDataDF['Peak Precip']=DataDF.resample("AS-OCT").max()
        
        return ( WYDataDF )
    
    def GetMonthlyStatistics(DataDF):
        """This function calculates monthly descriptive statistics and metrics 
        for the given streamflow time series.  Values are returned as a dataframe
        of monthly values for each year."""

        # Define the name of columns and create a new dataframe
        cols=['Mean Precip','Max Precip']

        # Devide the dataset into monthly data
        Mon_Data=DataDF.resample('MS').mean()
        MoDataDF=pd.DataFrame(0,index=Mon_Data.index,columns=cols)
        GroupD=DataDF.resample('MS')
    
        #Calculate descriptive values
        MoDataDF['Mean Precip']=GroupD.mean()
        MoDataDF['Max Precip']=GroupD.max()
    
        return ( MoDataDF )

    # plot daily pricipitation hydrograph
    data.index=data.index.strftime('%Y-%m-%d')
    data.plot.bar(figsize=(15,7))
    plt.ylabel('Precipitation (in)')
    plt.xlabel('Time')
    plt.title('Hyetograph')
    plt.savefig(FolderName + "Daily Hyetograph.jpeg")

    # plot monthly pricipitation hydrograph
    data.index = pd.to_datetime(data.index)
    monthly_data=data.resample('M').sum()
    monthly_data.index = monthly_data.index.strftime('%Y-%m')
    monthly_data.plot.bar(figsize=(15,7))
    plt.ylabel('Precipitation (in)')
    plt.xlabel('Time')
    plt.title('Hyetograph')
    plt.savefig(FolderName + "Monthly Hyetograph.jpeg")
    
    # plot yearly pricipitation hydrograph
    yearly_data=data.resample('Y').sum()
    yearly_data.index = yearly_data.index.strftime('%Y')
    yearly_data.plot.bar(figsize=(15,7))
    plt.ylabel('Precipitation (in)')
    plt.xlabel('Time')
    plt.title('Yearly Hyetograph')
    plt.savefig(FolderName + "Yearly Hyetograph.jpeg")
    
    # Save the data quality checking results into a file
    ReplacedValuesDF.to_csv("./Results/Precipitation/Checked_Summary(Precip).txt", sep="\t")
    
    # calculate descriptive statistics for each water year
    WYDataDF = GetAnnualStatistics(data)
    # Write data into annual metrics csv file
    WYDataDF.to_csv('./Results/Precipitation/Annual_Metrics(Precip).csv',sep=',', index=True)
    
    # calculate descriptive statistics for each month
    MonthlyWYDataDF = GetMonthlyStatistics(data)
    # Write data into annual metrics csv file
    MonthlyWYDataDF.to_csv('./Results/Precipitation/Monthly_Metrics(Precip).csv',sep=',', index=True)

    print("Precipitation data processing is done!")

# GUI Functions

def Button_df():
    """
    Upon click, it will obtain values from textbox and parce it to the DownloadData(the function for daily flow data download) and run the function.
    """
    print("Processing daily flow request...")
    station_number=Station_df.get("1.0","end-1c")
    begin_date=Startdate_df.get("1.0","end-1c")
    end_date=Enddate_df.get("1.0","end-1c")
    DownloadData(station_number, begin_date, end_date)
    print("Daily flow request complete!")

def Button_pf():
    """
    Upon click, it will obtain values from textbox and parce it to the GetPeakFlow(the function for peak flow data download) and run the function.
    """
    print("Processing peak flow request...")
    station_number=Station_pf.get("1.0","end-1c")
    GetPeakFlow(str(station_number))
    print("Peak flow request complete!")

def Button_pcp():
    """
    Upon click, it will obtain values from textbox and parce it to the GetPCP(the function for precipitation data download) and run the function.
    """
    print("Processing peak flow request...")
    token=str(token_pcp.get("1.0","end-1c"))
    NSEW=str(N_pcp.get("1.0","end-1c")+","+S_pcp.get("1.0","end-1c")+","+E_pcp.get("1.0","end-1c")+","+W_pcp.get("1.0","end-1c"))
    start_date=str(startdate_pcp.get("1.0","end-1c"))
    end_date=str(enddate_pcp.get("1.0","end-1c"))
    GETPCP(token, NSEW, start_date, end_date)
    print("Peak flow request complete!")

# GUI Orgnizing
## Create Main Windwow
root = Tk()
root.title("Hydrologic Modeling Data Acquiring & Initial Analyzing")
root.geometry('400x400')

rows=0
while rows<50:
    root.rowconfigure(rows,weight=1)
    root.columnconfigure(rows,weight=1)
    rows+=1
# Creat Tabs For All Functions
nb=ttk.Notebook(root)
nb.grid(row=1,column=0,columnspan=50,rowspan=49,sticky='NESW')
page_df=ttk.Frame(nb)
nb.add(page_df,text='Daily Flow')
page_pf=ttk.Frame(nb)
nb.add(page_pf,text='Peak Flow')
page_pcp=ttk.Frame(nb)
nb.add(page_pcp,text='Precipitation')

# Adding widgets (labels,text input boxs, and a run button) to the "Daily Flow" tab.
Label(page_df,text="USGS Station Number:").pack()
Station_df=Text(page_df, height=1, width=10)
Station_df.pack()
Label(page_df,text="Start Date YYYY-MM-DD:").pack()
Startdate_df=Text(page_df, height=1, width=10)
Startdate_df.pack()
Label(page_df,text="End Date YYYY-MM-DD:").pack()
Enddate_df=Text(page_df, height=1, width=10)
Enddate_df.pack()
buttonCommit=Button(page_df, height=1, width=10, text="Run", command=lambda: Button_df()).pack()

# Adding widgets to the "Peak Flow" tab.
Label(page_pf,text="USGS Station Number:").pack()
Station_pf=Text(page_pf, height=1, width=10)
Station_pf.pack()
buttonCommit=Button(page_pf, height=1, width=10, text="Run", command=lambda: Button_pf()).pack()

# Adding widgets to the "Precipitation" tab.
# GETPCP(token, NSEW, start_date, end_date)
# GETDATA('lsANjWwoJQegJhKZtKNJPVDGWIGhBSJN', '41.53,41.21,-84.90,-85.33', '2018-01-01', '2019-12-31')
Label(page_pcp,text="NCDC Data Access Token").pack()
token_pcp=Text(page_pcp, height=1, width=40)
token_pcp.pack()
Label(page_pcp,text="North Extend in degree").pack()
N_pcp=Text(page_pcp, height=1, width=10)
N_pcp.pack()
Label(page_pcp,text="South Extend in degree").pack()
S_pcp=Text(page_pcp, height=1, width=10)
S_pcp.pack()
Label(page_pcp,text="East Extend in degree").pack()
E_pcp=Text(page_pcp, height=1, width=10)
E_pcp.pack()
Label(page_pcp,text="West Extend in degree").pack()
W_pcp=Text(page_pcp, height=1, width=10)
W_pcp.pack()
Label(page_pcp,text="Start Date YYYY-MM-DD:").pack()
startdate_pcp=Text(page_pcp, height=1, width=10)
startdate_pcp.pack()
Label(page_pcp,text="End Date YYYY-MM-DD:").pack()
enddate_pcp=Text(page_pcp, height=1, width=10)
enddate_pcp.pack()
buttonCommit=Button(page_pcp, height=1, width=10, text="Run", command=lambda: Button_pcp()).pack()

root.mainloop()