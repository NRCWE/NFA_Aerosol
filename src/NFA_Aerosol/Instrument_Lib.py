# -*- coding: utf-8 -*-
"""
This is the NFA library for importing data from our aerosol instruments and
some of our sensors as well. All concentration data returned from the functions
are in #/cm3 unless otherwise specified.

The current version of this document is:
    
Instrument_Lib v 1.1
The most recent change is the Load_Grimm_new function

@author: B279683
"""

import numpy as np
import pandas as pd
import datetime as datetime
import os
import Utility_Lib as UL
from collections import Counter

###############################################################################

###############################################################################
def detect_delimiter(file_path, encodings=['latin-1','utf-8','UTF-16', 'iso-8859-1', 'windows-1252'],
                     delimiters=[',', ';', '\t', '|'], sample_lines=5,
                     min_count_threshold=3, tolerance=1):
    """
    Detects the most likely delimiter from a file, allowing for fuzzy match.
    
    Params:
        - file_path: str
        - encodings: list of encodings to try
        - delimiters: list of delimiters to check
        - sample_lines: number of lines to sample
        - min_count_threshold: minimum number of consistent lines required
        - tolerance: max deviation in delimiter count across lines

    Returns: (encoding, delimiter)
    """
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                lines = f.readlines()
            break
        except UnicodeDecodeError:
            continue
    else:
        raise UnicodeDecodeError("Could not decode file with given encodings.")
    
    # Filter non-empty, non-comment lines from the bottom
    valid_lines = [line for line in reversed(lines) if line.strip() and not line.strip().startswith('#')]
    lines = list(reversed(valid_lines[:sample_lines]))  # Keep original order
    
    if not lines:
        raise ValueError("No valid lines found to analyze.")

    best_delim = None
    best_score = 0

    for delim in delimiters:
        counts = [line.count(delim) for line in lines]
        if not counts:
            continue
        mode = Counter(counts).most_common(1)[0][0]

        consistent = [c for c in counts if abs(c - mode) <= tolerance and c > 0]

        if len(consistent) >= min_count_threshold:
            score = len(consistent)
            if score > best_score:
                best_score = score
                best_delim = delim

    if best_delim:
        return encoding, best_delim
    else:
        raise ValueError("Could not reliably detect a delimiter.")
####
# def detect_delimiter(file_path, encodings=['latin-1','utf-8','UTF-16', 'iso-8859-1', 'windows-1252'],
#                      delimiters=[',', ';', '\t', '|'], sample_lines=5,
#                      min_count_threshold=3, tolerance=1):
#     """
#     Detects the most likely delimiter from a file, allowing for fuzzy match.
    
#     Params:
#         - file_path: str
#         - encodings: list of encodings to try
#         - delimiters: list of delimiters to check
#         - sample_lines: number of lines to sample
#         - min_count_threshold: minimum number of consistent lines required
#         - tolerance: max deviation in delimiter count across lines

#     Returns: (encoding, delimiter)
#     """
#     for encoding in encodings:
#         try:
#             with open(file_path, 'r', encoding=encoding) as f:
#                 lines = f.readlines()
                
#                 # Filter non-empty, non-comment lines from the bottom
#                 valid_lines = [line for line in reversed(lines) if line.strip() and not line.strip().startswith('#')]
#                 lines = list(reversed(valid_lines[:sample_lines]))  # Keep original order
                
#                 if not lines:
#                     raise ValueError("No valid lines found to analyze.")

#                 best_delim = None
#                 best_score = 0

#                 for delim in delimiters:
#                     counts = [line.count(delim) for line in lines]
#                     if not counts:
#                         continue
#                     mode = Counter(counts).most_common(1)[0][0]

#                     consistent = [c for c in counts if abs(c - mode) <= tolerance and c > 0]

#                     if len(consistent) >= min_count_threshold:
#                         score = len(consistent)
#                         if score > best_score:
#                             best_score = score
#                             best_delim = delim

#                 if best_delim:
#                     return encoding, best_delim
#                 else:
#                     continue
#             break
#         except UnicodeDecodeError:
#             continue
#     else:
#         raise UnicodeDecodeError("Could not decode file with given encodings.")
    
   
###############################################################################
###############################################################################
# def detect_delimiter(file_path, encodings=['utf-8', 'iso-8859-1', 'windows-1252'],
#                      delimiters=[',', ';', '\t', '|'], sample_lines=10,
#                      min_count_threshold=3, tolerance=1):
#     """
#     Detects the most likely delimiter from a file, allowing for fuzzy match.
    
#     Params:
#         - file_path: str
#         - encodings: list of encodings to try
#         - delimiters: list of delimiters to check
#         - sample_lines: number of lines to sample
#         - min_count_threshold: minimum number of consistent lines required
#         - tolerance: max deviation in delimiter count across lines

#     Returns: (encoding, delimiter)
#     """
#     for encoding in encodings:
#         try:
#             with open(file_path, 'r', encoding=encoding) as f:
#                 lines = [line for _, line in zip(range(sample_lines), f)
#                          if line.strip() and not line.startswith('#')]
#             break
#         except UnicodeDecodeError:
#             continue
#     else:
#         raise UnicodeDecodeError("Could not decode file with given encodings.")
    
#     # Filter non-empty, non-comment lines from the bottom
#     valid_lines = [line for line in reversed(lines) if line.strip() and not line.strip().startswith('#')]
#     lines = list(reversed(valid_lines[:sample_lines]))  # Keep original order
    
#     if not lines:
#         raise ValueError("No valid lines found to analyze.")

#     best_delim = None
#     best_score = 0

#     for delim in delimiters:
#         counts = [line.count(delim) for line in lines]
#         if not counts:
#             continue
#         mode = Counter(counts).most_common(1)[0][0]

#         consistent = [c for c in counts if abs(c - mode) <= tolerance and c > 0]

#         if len(consistent) >= min_count_threshold:
#             score = len(consistent)
#             if score > best_score:
#                 best_score = score
#                 best_delim = delim

#     if best_delim:
#         return encoding, best_delim
#     else:
#         raise ValueError("Could not reliably detect a delimiter.")
###############################################################################
###############################################################################

def Load_Aethalometer(file, start=0, end=0):
    """
    Function to load data from a .txt datafile generated by the CPC and extract
    relevant data for plotting.
    
    Parameters
    ----------
    file : str
        Path to a .csv file generated by the Aethalometer
    start : datetime, optional
        Specify starting time to plot by setting a datetime value e.g.
        datetime.datetime(2023,01,28,15,00,00). The default is 0.
    end : datetime, optional
        Specify end time to plot by setting a datetime value e.g.
        datetime.datetime(2023,01,28,15,00,00). The default is 0.
    Returns
    -------
    Data_return : numpy.array
        Array containing all data from the Aethalometer including a column
        of Datetime values for easy plotting.
    Header : list
        A list containing the header for all columns in the Data_return variable
    """
 
    encod,delim=detect_delimiter(file)
    
    Aethalometer_Data = pd.read_csv(file, delimiter=delim,encoding=encod,header=0,decimal=".")

    Aethalometer_Datetimes = np.array(Aethalometer_Data["Date / time local"])

    Aethalometer_Datetime = np.array([datetime.datetime.strptime(i, "%Y-%m-%dT%H:%M:%S") 
                                      for i in Aethalometer_Datetimes])
    
    # Select data within specified time interval if specified
    if (start != 0) or (end != 0):
        if (start != 0) & (end == 0):
            index = Aethalometer_Datetime>=start
        elif (start == 0) & (end != 0):
            index = Aethalometer_Datetime<=end
        else:
            index = (Aethalometer_Datetime>=start) & (Aethalometer_Datetime<=end)
        Aethalometer_Datetime = Aethalometer_Datetime[index]
        Aethalometer_Data = Aethalometer_Data[index]
    
    Aethalometer_Data.insert(0, "Datetime", Aethalometer_Datetime, True)
    
    Header = list(Aethalometer_Data)
    Data_return = np.array(Aethalometer_Data)
    
    return Data_return, Header

###############################################################################
###############################################################################
###############################################################################

def Load_APS(file, start=0, end=0):
    """
    Function to load data from a comma seperated .txt datafile generated by 
    the APS and extract relevant data for plotting.
    
    Parameters
    ----------
    file : str
        Path to a .txt file generated by the APS
    start : datetime, optional
        Specify starting time to plot by setting a datetime value e.g.
        datetime.datetime(2023,01,28,15,00,00). The default is 0.
    end : datetime, optional
        Specify end time to plot by setting a datetime value e.g.
        datetime.datetime(2023,01,28,15,00,00). The default is 0.
    Returns
    -------
    Data_return_aero : numpy.array
        An array containing the imported aerodynamic data from the APS. The 
        data is organized in columns of: time, total number concentration, 
        and sizbin data.
    bin_edges_aero : numpy.array
        Array of bin edges for the aerodynamic sizebins of the APS in nm
    Header_aero : list
        List of all the headers for the columns in the Data_return_aero array
    Data_return_optic : numpy.array
        An array containing the imported optical data from the APS. The 
        data is organized in columns of: time, total number concentration, 
        and sizebin data. NOTE the APS names the optic sizebins as channels 1-16
        without specifying the size of the bins. According to the manual, the 
        channels are equally spaced and sized, covering the measurement range of
        the spectrometer. It is therefore assumed that they cover the same range
        as the aerodynamic sizes.
    bin_edges_optic : numpy.array
        Array of bin edges for the optical sizebins of the APS in nm
    Header_optic : list
        List of all the headers for the columns in the Data_return_optic array.

    """
    # Load all data from the txt file
    # for seperator in [',','\t',';']:
    #     APS = pd.read_csv(file, sep=seperator, header=6, encoding='latin-1')
    #     if len(np.array(APS)[0,:])==1:
    #         pass
    #     else:
    #         break

    encod,delim=detect_delimiter(file)
    APS = pd.read_csv(file, delimiter=delim,encoding=encod, header=6)
    
    # Store the total concentration reported by the instrument
    APS_total = np.array([float(i.split("(")[0]) for i in APS['Total Conc.']])
    
    # Get headers for the aerodynamic and optical size bins
    APS_sizes_header_aero = list(APS)[4:56]
    APS_sizes_header_optic = list(APS)[57:73]
    
    # Store the aerodynamic and optical size bins 
    APS_data_aero = np.array(APS[APS_sizes_header_aero])
    APS_data_optic = np.array(APS[APS_sizes_header_optic])
    
    # Read the lower and upper bounds of the aerodynamic bin range
    Lower_bound = np.round(np.genfromtxt(file,delimiter=delim,encoding=encod,skip_header=4,skip_footer=APS_total.shape[0]+2)[1]*1000)
    Upper_bound = np.round(np.genfromtxt(file,delimiter=delim,encoding=encod,skip_header=5,skip_footer=APS_total.shape[0]+1)[1]*1000)
    
    # Calculate the midpoints of all the aerodynamic size bins
    Bin_mids = np.array(APS_sizes_header_aero[1:],dtype="float")*1000
    lowest_mid = np.round((Bin_mids[0]-Lower_bound)/2 + Lower_bound )
    Bin_mids = np.append(lowest_mid,Bin_mids)
    
    # Calculate the edges of all the aerodynamic size bins
    bin_edges = (Bin_mids[1:]-Bin_mids[:-1])/2+Bin_mids[:-1]
    bin_edges = np.append(Lower_bound,bin_edges)
    bin_edges_aero = np.append(bin_edges,Upper_bound)
    
    # Calculate the optical size bins, which according to the manual are 16 
    # equally sized and spaced bins over the range of the spectrometer 
    bin_edges_optic = np.linspace(bin_edges_aero[0],bin_edges_aero[-1],17,dtype="int")
    Optical_mids = (bin_edges_optic[1:]-bin_edges_optic[:-1])/2 + bin_edges_optic[:-1]
    
    # Convert dates and times from the APS data to python datetime objects
    APS_datetime = np.array([datetime.datetime.strptime(i, '%m/%d/%y %H:%M:%S') 
                              for i in np.array(APS["Date"]+" "+APS["Start Time"])])
    
    # Select data within specified time interval - if specified
    if (start != 0) or (end != 0):
        if (start != 0) & (end == 0):
            index = APS_datetime>=start
        elif (start == 0) & (end != 0):
            index = APS_datetime<=end
        else:
            index = (APS_datetime>=start) & (APS_datetime<=end)
        APS_datetime = APS_datetime[index]
        APS_total = APS_total[index]
        APS_data_aero = APS_data_aero[index]
        APS_data_optic = APS_data_optic[index]
    
    # Collect all aerodynamic data 
    Data_return_aero = np.column_stack([APS_datetime,APS_total,APS_data_aero])
    Header_aero = ["Datetime","Total"] + list(Bin_mids)
    
    # Collect all optical data
    Data_return_optic = np.column_stack([APS_datetime,APS_total,APS_data_optic])
    Header_optic = ["Datetime","Total"] + list(Optical_mids)
    
    return Data_return_aero,bin_edges_aero,Header_aero, Data_return_optic, bin_edges_optic, Header_optic

###############################################################################
def Load_APS_full(file, start=0, end=0, Returned_data=["Aero"]):
    
    """
    Function to load data from a comma seperated .txt datafile generated by 
    the APS and extract relevant data for plotting.
    
    Parameters
    ----------
    file : str
        Path to a .txt file generated by the APS
    start : datetime, optional
        Specify starting time to plot by setting a datetime value e.g.
        datetime.datetime(2023,01,28,15,00,00). The default is 0.
    end : datetime, optional
        Specify end time to plot by setting a datetime value e.g.
        datetime.datetime(2023,01,28,15,00,00). The default is 0.
    Returned_data: list, (optional)
        Specify the desired the returned data from the list of options:
            'Aero': return the data as sorted by aerodynamic bin sizes.
            'Optic': return the data as sorted by the optic bin sizes.
            'Cor': return 
        Default is 'Aero' for aerodynamic data
    Returns
    -------
    Return_list: dictionary
        Depending on the values chosen it will return the header, data and 
        bin_edges for each chosen data type; Aero, Optic, Cor.
        Header: list
            List of all the headers for the columns in the returned data,
            Time, Total, bin_mids (nm)
        Data: numpy.array
            An array containing the imported data from the APS. The 
            data is organized in columns of: time, total number concentration, 
            and sizbin data/ an array of correlated data
        Bin_edges: numpy.array
            Array of bin edges for the sizebins of the APS in nm
    """
    
    Return_list={}
    
    # Load all data from the txt file
    # for seperator in [',','\t',';']:
    #     APS = pd.read_csv(file, sep=seperator, header=6, encoding='latin-1')
    #     if len(np.array(APS)[0,:])==1:
    #         pass
    #     else:
    #         break
    encod,delim=detect_delimiter(file)
    APS = pd.read_csv(file, delimiter=delim,encoding=encod, header=6)
        
    # Get headers for the aerodynamic and optical size bins
    APS_sizes_header_aero = list(APS)[4:56]
    # Read the lower and upper bounds of the aerodynamic bin range
    Lower_bound = 523 #np.round(np.genfromtxt(file,delim=seperator,skip_header=4,skip_footer=APS.shape[0]+2)[1]*1000)
    Upper_bound = np.round(np.genfromtxt(file,delimiter=delim,encoding=encod,skip_header=5,skip_footer=APS.shape[0]+1)[1]*1000)
    
    # Calculate the midpoints of all the aerodynamic size bins
    Bin_mids = np.array(APS_sizes_header_aero[2:],dtype="float")*1000
    lowest_mid = np.round((Bin_mids[0]+Lower_bound)/2 )
    Bin_mids = np.append(lowest_mid,Bin_mids)
    
    # Calculate the edges of all the aerodynamic size bins
    bin_edges = (Bin_mids[1:]-Bin_mids[:-1])/2+Bin_mids[:-1]
    bin_edges = np.append(Lower_bound,bin_edges)
    bin_edges = np.append(bin_edges,Upper_bound)

    # Calculate the optical size bins, which according to the manual are 16 
    # equally sized and spaced bins over the range of 0.37-20  
    bin_edges_optic = np.round(10**np.linspace(np.log10(370),np.log10(bin_edges[-1]),17,dtype="float"),2)
    Optical_mids = (bin_edges_optic[1:]-bin_edges_optic[:-1])/2 + bin_edges_optic[:-1]
    
    
    #Checks whether the loaded data is correlated
    if np.max(APS['Sample #'])*16==len(APS):
        #If true the data is stored as correlated:
            
        APS_datetime = np.array([datetime.datetime.strptime(d, '%m/%d/%y %H:%M:%S') 
                                  for d in np.array(APS["Date"][0::16]+" "+APS["Start Time"][0::16])])
                        
        #Extracts the total concentration
        APS_total=np.array(APS["Total Conc."][0::16])
        APS_total=np.array([float(i[:-7]) for i in APS_total])
        
        #Sequences the array of correlated aerodynamic and optic concentrations
        # into 
        Cor_data=np.column_stack((APS_datetime.copy(),APS_total,APS_datetime.copy()))
        # for i in range(0,int((len(APS)+1)/16)):
        #     Cor_data[i,2]=np.array(np.array(APS)[i*16:(i+1)*16,5:56]).astype(float)
        #     Cor_data[i,1]=np.sum(Cor_data[i,2])
        Cor_data[:,2]=np.array(APS)[:,5:56].astype("float").reshape(int(59200/16),16,51)
        # Select data within specified time interval - if specified
        if (start != 0) or (end != 0):
            if (start != 0) & (end == 0):
                index = APS_datetime>=start
            elif (start == 0) & (end != 0):
                index = APS_datetime<=end
            else:
                index = (APS_datetime>=start) & (APS_datetime<=end)
        else: index=[True]*len(APS_datetime)
        
        APS_datetime=APS_datetime[index]
        Cor_data=Cor_data[index]
        APS_total=APS_total[index]
        
        #Differentiate between desired return values
        if 'Aero' in Returned_data:
            #Sum up the correlated data based on aerodynaic size bins
            Aero_data=[]
            for y in range(0,len(Cor_data)):
                Aero=[]
                for i in range(0,51):
                    Aero=np.append(Aero,np.sum(Cor_data[y,2][:,i]))
                if len(Aero_data)==0:
                    Aero_data=Aero.copy()
                else:
                    Aero_data=np.row_stack((Aero_data,Aero.copy()))
            
            # Collect all aerodynamic data 
            Data_return_aero = np.column_stack([APS_datetime,APS_total,Aero_data])
            Aero_header = ["Datetime","Total"] + list(Bin_mids)
            #Adds the aerodynamic data to the list of returned values.
            Return_list['Aero']={'Data':Data_return_aero,'Bin_edges':bin_edges,'Header': Aero_header}
            
        if 'Optic' in Returned_data:
            #Sum up the correlated data based on optic size bins
            Optic_data=[]
            for y in range(0,len(Cor_data)):
                Optic=[]
                for i in range(0,16):
                    Optic=np.append(Optic,np.sum(Cor_data[y,2][i,:]))
                if len(Optic_data)==0:
                    Optic_data=Optic.copy()
                else:
                    Optic_data=np.row_stack((Optic_data,Optic.copy()))
                    
            # Collect all optical data
            Data_return_optic = np.column_stack([APS_datetime,APS_total,Optic_data])
            Optic_header = ["Datetime","Total"] + list(Optical_mids)
            #Adds the optic data to the list of returned values.
            Return_list['Optic']={'Data':Data_return_optic,'Bin_edges':bin_edges_optic,'Header': Optic_header}
            
        if 'Cor' in Returned_data:
            #Returns the combined unit
            Cor_header=['Datetime','Total','Array of correlated data']+['Aero mids']+list(Bin_mids)+['Optic mids']+list(Optical_mids)
            #Adds the optic data to the list of returned values.
            Return_list['Cor']={'Data':Cor_data, 'Bin_edges':{'Aero':bin_edges,'Optic':bin_edges_optic},'Header':Cor_header}
        
    else: #In case the data is not correlated load as usual
    
        # Store the total concentration reported by the instrument
        APS_total=np.array([float(i[:-7]) for i in APS['Total Conc.']])
        #APS_total = np.array([float(i.split("(")[0]) for i in APS['Total Conc.']])
        
        # Get headers for the aerodynamic and optical size bins
        APS_sizes_header_optic = list(APS)[57:73]
        
        # Store the aerodynamic and optical size bins 
        Aero_data = np.array(APS[APS_sizes_header_aero])
        Optic_data = np.array(APS[APS_sizes_header_optic])
        
        # Convert dates and times from the APS data to python datetime objects
        APS_datetime = np.array([datetime.datetime.strptime(i, '%m/%d/%y %H:%M:%S') 
                                  for i in np.array(APS["Date"]+" "+APS["Start Time"])])
        
        # Select data within specified time interval - if specified
        if (start != 0) or (end != 0):
            if (start != 0) & (end == 0):
                index = APS_datetime>=start
            elif (start == 0) & (end != 0):
                index = APS_datetime<=end
            else:
                index = (APS_datetime>=start) & (APS_datetime<=end)
            APS_datetime = APS_datetime[index]
            APS_total = APS_total[index]
            Aero_data = Aero_data[index]
            Optic_data = Optic_data[index]
        
        
        #Differentiate between desired return values
        if 'Aero' in Returned_data:

            # Collect all aerodynamic data 
            Data_return_aero = np.column_stack([APS_datetime,APS_total,Aero_data])
            Aero_header = ["Datetime","Total"] + list(Bin_mids)
            #Adds the aerodynamic data to the list of returned values.
            Return_list['Aero']={'Data':Data_return_aero,'Bin_edges':bin_edges,'Header': Aero_header}
            
        if 'Optic' in Returned_data:
           
            # Collect all optical data
            Data_return_optic = np.column_stack([APS_datetime,APS_total,Optic_data])
            Optic_header = ["Datetime","Total"] + list(Optical_mids)
            #Adds the optic data to the list of returned values.
            Return_list['Optic']={'Data':Data_return_optic,'Bin_edges':bin_edges_optic,'Header': Optic_header}
            
        if 'Cor' in Returned_data:
            return "Data not correlated"

    return Return_list
###############################################################################
###############################################################################
###############################################################################

def Load_CPC(file, start=0, end=0):
    """
    Function to load data from a "," separated .txt datafile generated by the 
    CPC and extract relevant data for plotting.
    
    Parameters
    ----------
    file : str
        Path to a .txt file generated by the CPC
    start : datetime, optional
        Specify starting time to plot by setting a datetime value e.g.
        datetime.datetime(2023,01,28,15,00,00). The default is 0.
    end : datetime, optional
        Specify end time to plot by setting a datetime value e.g.
        datetime.datetime(2023,01,28,15,00,00). The default is 0.
    Returns
    -------
    Data_return : numpy.array
        Array containing datetimes from each measurement along with the 
        particle concentration reported by the CPC
    Header : list
        List of all the headers for the columns in the Data_return array
    """
    encod,delim=detect_delimiter(file)

    # Load the entire CPC file
    CPC_data = np.genfromtxt(file, delimiter=delim,encoding=encod,skip_header=18,skip_footer=4,usecols=1)
    CPC_time = np.genfromtxt(file, delimiter=delim,encoding=encod,skip_header=18,skip_footer=4,usecols=0,dtype=str)
   
    # # Convert the loaded times to datetime format
    CPC_Start_date = np.genfromtxt(file, delimiter=delim,encoding=encod,skip_header=4,
                                   skip_footer=CPC_data.shape[0]+15,dtype=str)[1]

    try:
        CPC_datetimes = np.array([datetime.datetime.strptime(CPC_Start_date + " " + i, "%m/%d/%y %H:%M:%S") 
                                  for i in CPC_time])

    except:
        CPC_Start_time = np.genfromtxt(file, delimiter=delim,encoding=encod,skip_header=4,usecols=1,dtype=str)[1]
        CPC_datetimes = np.array([datetime.datetime.strptime(
            CPC_Start_date + " " + CPC_Start_time, "%m/%d/%y %H:%M:%S") + datetime.timedelta(0,int(i))
                                  for i in CPC_time])
    
    # Select data within specified time interval if specified
    if (start != 0) or (end != 0):
        if (start != 0) & (end == 0):
            index = CPC_datetimes>=start
        elif (start == 0) & (end != 0):
            index = CPC_datetimes<=end
        else:
            index = (CPC_datetimes>=start) & (CPC_datetimes<=end)
        CPC_datetimes = CPC_datetimes[index]
        CPC_data = CPC_data[index]
    
    # Collect all data in a pandas DataFrame for easy handling
    Data_return = np.column_stack([CPC_datetimes,CPC_data])
    Header = ["Datetime", "Concentration"]
    return Data_return, Header

###############################################################################
###############################################################################
###############################################################################

def Load_data_from_folder(folder_path, load_function, keyword="", file_extension="", max_subfolder=0,**kwargs):
    """
    Generic function to load data from a folder.

    Parameters
    ----------
    folder_path : str
        Path to the folder containing the data files. The function will NOT search
        subfolders, so make sure the relevant data is in the specified folder.
    load_function : function
        speficiy which function should be used for treating the data.
        Remember to call it with 'IL.' in front. 
    keyword : str, optional
        If the relevant files have a specific keyword in their name, a str can 
        be added here to specify that the function must only concatenate 
        data from file names containing the keyword. The default is empty.
    file_extension : str, optional
        If the relevant files has a defining marker in the end of their name,
        e.g. data format ".txt", a str can be added here to specify that the
        function must only concatenate data begining with this.
        This restriction can be combined with file_begining to help sort through data.
        The default is empty.
    max_subfolder : int, optional
        Specify the number of subfolder levels that the function is supposed to
        include. A value of 1 means that the function will go into the first
        layer of subfolders of the specified path (if there are any). The 
        parameter can be used to make the function load datafiles from several
        subfolders and tie then together into one data array with chronological
        data.
    **kwargs: dict, optional
        Arbitrary keyword arguments passed to the load_function. Keywords should 
        match parameter names of the load_function, and their values specify the 
        arguments to be passed. e.g.
        
        Load_data_from_folder(r"C:\folderpath", Load_CPC, "data_to_load", start = datetime(2025,1,1,0,0,0))
        
        Here, all data files in the C:\folderpath directory, which contain the
        keyword "data_to_load" in their filename are loaded with the Load_CPC
        function, but only including datapoint from January 1st 2025, which was
        passed as the start parameter of the Load_CPC function. 

    Returns
    -------
    sorted_data: numpy
        The concantenated data from the folder as returned from the load_function,
        with column [0] = datetime, column [1:] = data
    bin_edges: list, optional
        If the data returns size-bins for bin_edges, they are returned here
    Header: list of str
        Header for the sorted_data

    """
    
    all_data = []
    if max_subfolder !=0:
        file_paths = UL.File_list(folder_path,keyword, max_subfolder)
        for file_path in file_paths:
            if file_path.endswith(file_extension):
                print("Loading: {0}".format(file_path.split("\\")[-1]))
                # print("Loading: {0}".format(file_path.split("\\")[-1]))
                if load_function==Load_OPS:
                    try:all_data.append(load_function(file_path, **kwargs))
                    except:all_data.append(Load_OPS_Direct(file_path, **kwargs))
                else:
                    all_data.append(load_function(file_path, **kwargs))
    else:
        for file_name in os.listdir(folder_path):
            if (keyword in file_name) and (file_name.endswith(file_extension)):
                print("Loading: {0}".format(file_name))
                # print("Loading: {0}".format(file_path.split("\\")[-1]))
                file_path = os.path.join(folder_path, file_name)
                
                if load_function==Load_OPS:
                    try:all_data.append(load_function(file_path, **kwargs))
                    except:all_data.append(Load_OPS_Direct(file_path, **kwargs))
                else:
                    all_data.append(load_function(file_path, **kwargs))
    
    data_list = [dataset[0] for dataset in all_data]

    combined_data = np.concatenate(data_list)
    sorted_data = combined_data[np.argsort(combined_data[:, 0])]
    Header=all_data[0][-1]
    
    #Checks the output format from the load function and returns the relevant data
    if len(all_data[0])==2:
        return sorted_data, Header
    elif len(all_data[0])==3:
        # Care should be taken to ensure that the bin_edges are the same across the data, as it only calls the first
        bin_edges=all_data[0][1]
        return sorted_data, bin_edges, Header
    else:
        return sorted_data, all_data[0][1:]


###############################################################################
###############################################################################

#
def Load_Discmini(file, start=0, end=0):
    """
    Function to load data from a .txt datafile generated by the DiscMini and 
    extract relevant data for plotting.
    
    Parameters
    ----------
    file : str
        Path to a .dat file generated by the DiscMini
    start : datetime, optional
        Specify starting time to plot by setting a datetime value e.g.
        datetime.datetime(2023,01,28,15,00,00). The default is 0.
    end : datetime, optional
        Specify end time to plot by setting a datetime value e.g.
        datetime.datetime(2023,01,28,15,00,00). The default is 0.
    Returns
    -------
    data_return : numpy.array
        An array with all the imported data from the discmini. Data
        is organized in columns of: time, data, filter, temperature. The header
        for each column is listed in the Header variable
    Header : list
        List of all the headers for the columns in the Data_return array:
        1. Particle number concentration, cm-3
        2. Average particle size, nm (only covers from 10-300 nm)
        3. Estimated lung deposited surface area, cm2/cm3
    
    """
    encod,delim=detect_delimiter(file)
    delim='\t'
    try:
        Discmini_time = np.genfromtxt(file,delimiter=delim,encoding=encod,skip_header=6,usecols=0,dtype=str)
        Discmini_datetimes = np.array([datetime.datetime.strptime(i, "%d-%b-%Y %H:%M:%S")
                                      for i in Discmini_time])
        Discmini_data = np.genfromtxt(file,delimiter=delim,encoding=encod,skip_header=6,usecols=[2,3,4],dtype=str)
        Discmini_data = np.char.replace(Discmini_data, ',', '.').astype(float)
    except:
        try:
            Discmini_time = np.genfromtxt(file,delimiter=delim,encoding=encod,skip_header=6,usecols=0,dtype=str)
            Discmini_datetimes = np.array([datetime.datetime.strptime(i, "%d-%m-%Y %H:%M:%S")
                                          for i in Discmini_time])
            Discmini_data = np.genfromtxt(file,delimiter=delim,encoding=encod,skip_header=6,usecols=[2,3,4],dtype=str)
            Discmini_data = np.char.replace(Discmini_data, ',', '.').astype(float)
        except:
            try:
                Discmini_data = np.genfromtxt(file,delimiter=delim,encoding=encod,skip_header=6,usecols=[1,2,3],dtype=str)
                Discmini_data = np.char.replace(Discmini_data, ',', '.').astype(float)
                
                # Read the starting time from the partector file
                tst=np.genfromtxt(file,delimiter=delim,encoding=encod,usecols=[0],dtype=str)
                #Locate and strip the starting date from string
                Date = datetime.datetime.strptime(tst[2].split("start date: ")[1].split("]")[0],"%Y.%m.%d")
                #Locate and strip the starting time from string
                Time = datetime.datetime.strptime(tst[3].split("start time: ")[1].split("]")[0],"%H:%M:%S")
                Time=Time.hour*3600 + Time.minute*60 + Time.second
                Start_time = Date+datetime.timedelta(0,Time)
                
                #Generate time array based on one second per measurement
                Discmini_datetimes=[]   
                for idx in range(0,len(Discmini_data)):
                    Discmini_datetimes.append(Start_time+datetime.timedelta(0,idx))
            except Exception as e:
                print(e)
                print("Time format not covered by the function!?")
                return
    
    # Select data within specified time interval if specified
    if (start != 0) or (end != 0):
        if (start != 0) & (end == 0):
            index = Discmini_datetimes>=start
        elif (start == 0) & (end != 0):
            index = Discmini_datetimes<=end
        else:
            index = (Discmini_datetimes>=start) & (Discmini_datetimes<=end)
        Discmini_datetimes = Discmini_datetimes[index]
        Discmini_data = Discmini_data[index]
        
    # Collect all data in a pandas DataFrame for easy handling
    Data_return = np.column_stack([Discmini_datetimes,Discmini_data])
    Header = ["Datetime","Number","Size","LDSA"]

    return Data_return, Header
   
###############################################################################
###############################################################################
###############################################################################

def Load_ELPI(file, start=0, end=0):
    """
    Function to load data from a .dat datafile generated by the ELPI and extract
    relevant data for plotting.
    
    Parameters
    ----------
    file : str
        Path to a .dat file generated by the ELPI
    start : datetime, optional
        Specify starting time to plot by setting a datetime value e.g.
        datetime.datetime(2023,01,28,15,00,00). The default is 0.
    end : datetime, optional
        Specify end time to plot by setting a datetime value e.g.
        datetime.datetime(2023,01,28,15,00,00). The default is 0.
    Returns
    -------
    data_return : numpy.array
        An array containing the imported data from the ELPI. The data is
        organized in columns of: time, total number concentration, data. Where
        data refers to all the concentrations measured at each impactor stage.
    Bin_edges : numpy.array
        Array of D50 values for the ELPI stages in nm
    Header : list
        List of all the headers for the columns in the Data_return array
    """
    encod,delim=detect_delimiter(file,sample_lines=100)

    # Load the entire ELPI file but excluding the header rows
    ELPI = np.genfromtxt(file,delimiter=delim,encoding=encod,skip_header=42,dtype=float)
    
    # Store the total concentration measured by the ELPI
    ELPI_Total = ELPI[:,49]
   
    # Load data from all of the ELPI stages
    ELPI_data = ELPI[:,34:48]
    
    # Convert the loaded times to datetime format
    ELPI_Rawtime = np.genfromtxt(file,delimiter=delim,encoding=encod,skip_header=42,usecols=0,dtype="str")
    ELPI_time = np.array([datetime.datetime.strptime(i, '%Y/%m/%d %H:%M:%S') 
                          for i in ELPI_Rawtime])
    
    # Select data within specified time interval if specified
    if (start != 0) or (end != 0):
        if (start != 0) & (end == 0):
            index = ELPI_time>=start
        elif (start == 0) & (end != 0):
            index = ELPI_time<=end
        else:
            index = (ELPI_time>=start) & (ELPI_time<=end)
        ELPI_time = ELPI_time[index]
        ELPI_Total = ELPI_Total[index]
        ELPI_data = ELPI_data[index]
        
    # Load D50 values for the ELPi stages:
    D50_vals = np.genfromtxt(file,delimiter=delim,encoding=encod,skip_header=12,skip_footer=ELPI.shape[0]+26,dtype="str")
    
    # Remove text from the first entry in the loaded row
    D50_vals[0] = D50_vals[0].split("=")[1]
    
    # Convert D50 values from strings to floats
    D50_vals = np.array(D50_vals,dtype=float)*1000
    Bin_mids = np.round((D50_vals[1:]+D50_vals[:-1])/2,0)
    
    # Collect all data in a pandas DataFrame for easy handling
    Data_return = np.column_stack([ELPI_time,ELPI_Total,ELPI_data])
    Header = ["Datetime", "Total Concentration"] + list(Bin_mids)
    
    return Data_return, D50_vals, Header

###############################################################################
###############################################################################
###############################################################################

def Load_FMPS(file, start=0, end=0):
    """
    Function to load data from "," seperated .txt datafile generated by the FMPS 
    and extract relevant data for plotting.
    Note that the "Date/Time Start:" of the file should have the format e.g.:
    "17. januar 2023 09:42:56" otherwise the file was exported incorretly from
    the raw TSI file e.g. by using a PC with a wrong time format/language setting.
    
    Parameters
    ----------
    file : str
        Path to a .txt file generated by the FMPS
    start : datetime, optional
        Specify starting time to plot by setting a datetime value e.g.
        datetime.datetime(2023,01,28,15,00,00). The default is 0.
    end : datetime, optional
        Specify end time to plot by setting a datetime value e.g.
        datetime.datetime(2023,01,28,15,00,00). The default is 0.
    Returns
    -------
    data_return : numpy.array
        An array containing the imported data from the FMPS. The data is
        organized in columns of: time, total number concentration, data. Where
        data refers to all the concentrations measured at each electrometer.
    Bin_edges : numpy.array
        Array of edges for the FMPS sizebins in nm
    Header : list
        List of all the headers for the columns in the Data_return array
    """
    encod,delim=detect_delimiter(file)

    # Load particle data from the txt file
    FMPS_data = np.genfromtxt(file,delimiter=delim,encoding=encod,skip_header=15)[:,1:-11]
    
    # Calculate total number concentrations
    FMPS_Total = np.genfromtxt(file,delimiter=delim,encoding=encod,skip_header=15)[:,35]
    
    # Load sizebin values from the data file
    Bin_mids = np.genfromtxt(file,delimiter=delim,encoding=encod,skip_header=13,skip_footer=FMPS_data.shape[0]+1)[1:-11]
    Bin_edges = (Bin_mids[1:]+Bin_mids[:-1])/2
    Bin_edges = np.append(5.6, Bin_edges)
    Bin_edges = np.append(Bin_edges,560)
    
    try:
        # Load the date and format it to a datetime value
        FMPS_date = np.genfromtxt(file,delimiter=delim,encoding=encod,skip_header=0,skip_footer=FMPS_data.shape[0]+9,dtype=str)[1]
        FMPS_date = FMPS_date.split("\"")[1]
        FMPS_date = FMPS_date.split(" ")[:3]
        
        # Dictionary to convert from written danish months to a value for datetime
        months = {
          "januar": 1,
          "februar": 2,
          "marts": 3,
          "april":4,
          "maj":5,
          "juni":6,
          "juli":7,
          "august":8,
          "september":9,
          "oktober":10,
          "november":11,
          "december":12
        }
        
        # Load the times
        FMPS_time = np.genfromtxt(file,delimiter=delim,encoding=encod,skip_header=15,dtype=str)[:,0]
        
        # Convert time and dates to a datetime
        FMPS_datetime = np.array([datetime.datetime(int(FMPS_date[2]),
                                                    months[FMPS_date[1]],
                                                    int(FMPS_date[0].split(".")[0]),
                                                    int(i.split(":")[0]),
                                                    int(i.split(":")[1]),
                                                    int(i.split(":")[2]))  for i in FMPS_time])
    except:
        # Load the date and format it to a datetime value
        FMPS_date = np.genfromtxt(file,delimiter=delim,encoding=encod,skip_header=0,skip_footer=FMPS_data.shape[0]+9,dtype=str)[2:]
        
        Year = int(FMPS_date[1].split(" ")[1])
        Month= FMPS_date[0].split(" ")[1]
        Day = int(FMPS_date[0].split(" ")[2])
        Time = FMPS_date[1].split(" ")[3] + " " + FMPS_date[1].split(" ")[4][:2]
        Time=datetime.datetime.strptime(Time, "%I:%M:%S %p") 
        # Dictionary to convert from written danish months to a value for datetime
        months = { "Jan": 1, "Feb": 2, "Mar": 3, "Apr":4, "May":5, "Jun":6,
          "Jul":7, "Aug":8, "Sep":9, "Oct":10, "Nov":11, "Dec":12        }
        
        Month=months[Month[:3]]
        
        start_datetime=datetime.datetime(Year,Month,Day,Time.hour,Time.minute,Time.second) 
        
        # Load the times
        FMPS_datetime = np.genfromtxt(file,delimiter=delim,encoding=encod,skip_header=15,dtype=str)[:,0]
        FMPS_datetime = np.array([datetime.datetime.strptime(i, "%I:%M:%S %p") 
                                          for i in FMPS_datetime])
        Elapsed_time=FMPS_datetime[1]-FMPS_datetime[0]
        FMPS_datetime = np.array([start_datetime + Elapsed_time*(1+i)
                                          for i in range(0,len(FMPS_datetime))])

    # Select data within specified time interval if specified
    if (start != 0) or (end != 0):
        if (start != 0) & (end == 0):
            index = FMPS_datetime>=start
        elif (start == 0) & (end != 0):
            index = FMPS_datetime<=end
        else:
            index = (FMPS_datetime>=start) & (FMPS_datetime<=end)
        FMPS_datetime = FMPS_datetime[index]
        FMPS_Total = FMPS_Total[index]
        FMPS_data = FMPS_data[index]
    
    # Collect all data in a pandas DataFrame for easy handling
    Data_return = np.column_stack([FMPS_datetime,FMPS_Total,FMPS_data])
    Header = ["Datetime","Total"] + list(Bin_mids)
    
    return Data_return, Bin_edges, Header

###############################################################################
###############################################################################
###############################################################################

def Load_Fourtect(file, start=0, end=0):
    """
    Function to load data from a .csv datafile generated by the Fourtec
    bluefish sensor and extract relevant data for plotting.
    
    Parameters
    ----------
    file : str
        Path to a .csv file generated by the Fourtec Datasuite program
    start : datetime, optional
        Specify starting time to plot by setting a datetime value e.g.
        datetime.datetime(2023,01,28,15,00,00). The default is 0.
    end : datetime, optional
        Specify end time to plot by setting a datetime value e.g.
        datetime.datetime(2023,01,28,15,00,00). The default is 0.
    Returns
    -------
    Data_return : numpy.array
        Array containing Temperature and relative humidity including a column
        of Datetime values for easy plotting.
    Header : list
        A list containing the header for all columns in the Data_return variable
    """
    encod,delim=detect_delimiter(file)

    
    Fourtec_Data = pd.read_csv(file, delimiter=delim,encoding=encod,skiprows=8,header=0)
    #Fourtec_Data = pd.read_csv(file, delimiter=',',skiprows=8,header=0,encoding="UTF-16")
    Fourtec_Datetime = Fourtec_Data["Date"].str.cat(Fourtec_Data["Time"],sep="\t")
    
    Fourtec_Datetime = np.array([datetime.datetime.strptime(i, "%d/%m/%Y\t %H:%M:%S") 
                                      for i in Fourtec_Datetime])
    
    # Select data within specified time interval if specified
    if (start != 0) or (end != 0):
        if (start != 0) & (end == 0):
            index = Fourtec_Datetime>=start
        elif (start == 0) & (end != 0):
            index = Fourtec_Datetime<=end
        else:
            index = (Fourtec_Datetime>=start) & (Fourtec_Datetime<=end)
        Fourtec_Datetime = Fourtec_Datetime[index]
        Fourtec_Data = Fourtec_Data[index]
        
    Fourtec_Data=Fourtec_Data[["Internal Digital Temperature","Internal RH"]]
    Fourtec_Data.insert(0, "Datetime", Fourtec_Datetime, True)
    Header = ["Datetime","Temperature","Relative humidity"]
    Data_return = np.array(Fourtec_Data)

    return Data_return, Header
###############################################################################
def Load_Fourtec(file):
    """
    Function to load data from a .csv datafile generated by the Fourtec
    bluefish sensor and extract relevant data for plotting.
    
    Parameters
    ----------
    file : str
        Path to a .csv file generated by the Fourtec Datasuite program

    Returns
    -------
    Fourtec : Classes.AerosolAlt
      Array containing datetimes from each measurement along with the 
      particle concentration reported by the DM
    """ 
    if file[-4:]=='.csv':
        encoding,delimiter=detect_delimiter(file)
        Fourtec_df = pd.read_csv(file, delimiter=delimiter,encoding=encoding,skiprows=8,usecols=[0,1,2,4])
        Fourtec_df.rename(columns={'Date':'Datetime','Internal Digital Temperature':'Temperature', 'Internal RH':'RH'},inplace=True)
        # Fourtec_df["Datetime"] = Fourtec_df["Datetime"].str.cat(Fourtec_df["Time"],sep="\t")
        Fourtec_df['Datetime'] = Fourtec_df['Datetime']+" "+Fourtec_df['Time']
        Fourtec_df["Datetime"] = pd.to_datetime(Fourtec_df['Datetime'],format="%d-%m-%Y %H:%M:%S")
        
    else:
        Fourtec_df = pd.read_excel(file,skiprows=8,usecols=[0,1,2,4])
        Fourtec_df['Date'] = [i.date() for i in Fourtec_df['Date']]
        Fourtec_df['Date'] = [datetime.datetime.combine(Fourtec_df['Date'][i],Fourtec_df['Time'][i]) for i in range(0,len(Fourtec_df['Date']))]        
        Fourtec_df.rename(columns={'Date':'Datetime','Internal Digital Temperature':'Temperature', 'Internal RH':'RH'},inplace=True)

    Fourtec_df=Fourtec_df.drop(columns='Time')
    return np.array(Fourtec_df),list(Fourtec_df)
###############################################################################
###############################################################################
###############################################################################

def Load_Grimm_new(file, start=0, end=0):
    """
    Function to load data from a raw .dat datafile generated by the new Grimm 
    software and extract relevant data for plotting. The provided file should 
    be the file nanmed xxx-C.dat in order to get the number concentration data.
    
    Parameters
    ----------
    file : str
        Path to a .txt file generated by the Grimm
    start : datetime, optional
        Specify starting time to plot by setting a datetime value e.g.
        datetime.datetime(2023,01,28,15,00,00). The default is 0.
    end : datetime, optional
        Specify end time to plot by setting a datetime value e.g.
        datetime.datetime(2023,01,28,15,00,00). The default is 0.
    Returns
    -------
    data_return : numpy.array
        An array containing the imported data from the Grimm. The data is
        organized in columns of: time, total number concentration, data. Where
        data refers to all the concentrations measured at each sizebin.
    Bin_edges : numpy.array
        Array of bin edges for the Grimm in nm
    Header : list
        List of all the headers for the columns in the Data_return array
    """
    encod,delim=detect_delimiter(file)

    # Load particle data from the txt file
    Grimm_data = np.genfromtxt(file,delimiter=delim,encoding=encod,skip_header=14)[:,1:-1]
    
    # Convert the concentration data to #/cm3 rather than #/liter as is reported by the Grimm
    Grimm_data = Grimm_data * 1e-3
    
    # Calculate the total particle concentration by summing all size bin concentrations
    Grimm_total = np.sum(Grimm_data,axis=1)
    
    # Store the edges og all the size bins. These are constant, and are therefore
    # not read in from the datafile, but just typed once
    file_heading = np.genfromtxt(file,delimiter=delim,encoding=encod,skip_header=13,skip_footer=Grimm_data.shape[0],dtype="str")[1:]
    bin_edges = np.array([float(i.split(" ")[0]) for i in file_heading])*1000
    
    # Calculate the midpoint of all sizebins
    Bin_mids = bin_edges[:-1] + (bin_edges[1:]-bin_edges[:-1])/2
    
    # Import the times reported by the Grimm
    Grimm_time = np.genfromtxt(file,delim="\t",skip_header=14,dtype=str)[:,0]
   
    Grimm_datetime = np.array([datetime.datetime.strptime(i, '%d/%m/%Y %H:%M:%S') 
                          for i in Grimm_time]) 
    
    # Select data within specified time interval if specified
    if (start != 0) or (end != 0):
        if (start != 0) & (end == 0):
            index = Grimm_datetime>=start
        elif (start == 0) & (end != 0):
            index = Grimm_datetime<=end
        else:
            index = (Grimm_datetime>=start) & (Grimm_datetime<=end)
        Grimm_datetime = Grimm_datetime[index]
        Grimm_total = Grimm_total[index]
        Grimm_data = Grimm_data[index]
    
    # Collect all data in a pandas DataFrame for easy handling
    Data_return = np.column_stack([Grimm_datetime,Grimm_total,Grimm_data])
    Header = ["Datetime","Total"] + list(Bin_mids)
    
    return Data_return,bin_edges,Header

###############################################################################
###############################################################################
###############################################################################

def Load_Grimm_old(file, start=0, end=0):
    """
    Function to load data from a comma seperated .txt datafile generated by 
    the old Grimm software and extract relevant data for plotting.
    
    Parameters
    ----------
    file : str
        Path to a .txt file generated by the Grimm
    start : datetime, optional
        Specify starting time to plot by setting a datetime value e.g.
        datetime.datetime(2023,01,28,15,00,00). The default is 0.
    end : datetime, optional
        Specify end time to plot by setting a datetime value e.g.
        datetime.datetime(2023,01,28,15,00,00). The default is 0.
    Returns
    -------
    data_return : numpy.array
        An array containing the imported data from the Grimm. The data is
        organized in columns of: time, total number concentration, data. Where
        data refers to all the concentrations measured at each sizebin.
    Bin_edges : numpy.array
        Array of bin edges for the Grimm in nm
    Header : list
        List of all the headers for the columns in the Data_return array
    """
    encod,delim=detect_delimiter(file)

    # Load particle data from the txt file
    Grimm_data = np.genfromtxt(file,delimiter=delim,encoding=encod,skip_header=2)[:,1:]
    
    # Calculate the total particle concentration by summing all size bin concentrations
    Grimm_total = np.sum(Grimm_data,axis=1)
    
    # Store the edges og all the size bins. These are constant, and are therefore
    # not read in from the datafile, but just typed once
    bin_edges = np.array([250,280,300,350,400,450,500,580,650,700,800,1000,1300,1600,
                          2000,2500,3000,3500,4000,5000,6500,7500,8500,10000,12500,
                          15000,17500,20000,25000,30000,32000,34000]) 
    
    # Calculate the midpoint of all sizebins
    Bin_mids = bin_edges[:-1] + (bin_edges[1:]-bin_edges[:-1])/2
    
    # Import the times reported by the Grimm
    Grimm_time = np.genfromtxt(file,delimiter=delim,encoding=encod,skip_header=2,dtype=str)[:,0]
    
    # Deal with an issue, where the Grimm will only report the date whenever a
    # new date begins. Here it only writes the date and not the additional time 
    # stamp of 12:00:00 AM, which is added now and converted to a datetime object.
    timer = []
    for i in Grimm_time:
        try:
            temp_time = datetime.datetime.strptime(i, '%m/%d/%Y %I:%M:%S %p')
        except:
            temp_str = i + " 12:00:00 AM"
            temp_time = datetime.datetime.strptime(temp_str, '%m/%d/%Y %I:%M:%S %p')
        timer += [temp_time]
    Grimm_datetime = np.array(timer)
    
    # Select data within specified time interval if specified
    if (start != 0) or (end != 0):
        if (start != 0) & (end == 0):
            index = Grimm_datetime>=start
        elif (start == 0) & (end != 0):
            index = Grimm_datetime<=end
        else:
            index = (Grimm_datetime>=start) & (Grimm_datetime<=end)
        Grimm_datetime = Grimm_datetime[index]
        Grimm_total = Grimm_total[index]
        Grimm_data = Grimm_data[index]
    
    # Collect all data in a pandas DataFrame for easy handling
    Data_return = np.column_stack([Grimm_datetime,Grimm_total,Grimm_data])
    Header = ["Datetime","Total"] + list(Bin_mids)
    
    return Data_return,bin_edges,Header

###############################################################################
###############################################################################
###############################################################################

def Load_Nanoscan(file, start=0, end=0):
    """
    Function to load and format data from a .csv datafile generated by the 
    Nanoscan and extract relevant data for plotting.
    
    Parameters
    ----------
    file : str
        Path to a .csv file generated by the Nanoscan
    start : datetime, optional
        Specify starting time to plot by setting a datetime value e.g.
        datetime.datetime(2023,01,28,15,00,00). The default is 0.
    end : datetime, optional
        Specify end time to plot by setting a datetime value e.g.
        datetime.datetime(2023,01,28,15,00,00). The default is 0.
    Returns
    -------
    data_return : numpy.array
        An array containing the imported data from the Nanoscan. The data is
        organized in columns of: time, total number concentration, data. Where
        data refers to all the concentrations measured at each sizebin.
    Bin_edges : numpy.array
        Array of Nanoscan size bins edges in nm
    Header : list
        List of all the headers for the columns in the Data_return array
    """
    encod,delim=detect_delimiter(file)

    # Load the entire Nanoscan file
    Nanoscan_data = pd.read_csv(file, delimiter=delim,encoding=encod, decimal='.', header=5,usecols=range(3,16))
    Nanoscan_data = np.array(Nanoscan_data)
    # Nanoscan_data = np.genfromtxt(file,delim=",",skip_header=9)[:,3:16]
    
    # Store bin sizes
    Bin_mids = np.genfromtxt(file,delimiter=delim,encoding=encod,skip_header=8,skip_footer=
                             Nanoscan_data.shape[0],dtype=float,comments="!!!!",
                             usecols=range(3,16))
    
    Bin_edges = (Bin_mids[1:]*Bin_mids[:-1])**0.5
    Bin_edges = np.append(10, Bin_edges)
    Bin_edges = np.append(Bin_edges,420)
    
    # Store the total concentration measured by the Nanoscan
    Nanoscan_Total = np.sum(Nanoscan_data,axis=1)
    
    # Convert the loaded times to datetime format
    Nanoscan_datetime = np.genfromtxt(file,delimiter=delim,encoding=encod,skip_header=9,dtype=str,usecols=2)
    
    Nanoscan_time = np.array([datetime.datetime.strptime(i, '%Y/%m/%d %H:%M:%S') 
                          for i in Nanoscan_datetime])    
    
    # Select data within specified time interval if specified
    if (start != 0) or (end != 0):
        if (start != 0) & (end == 0):
            index = Nanoscan_time>=start
        elif (start == 0) & (end != 0):
            index = Nanoscan_time<=end
        else:
            index = (Nanoscan_time>=start) & (Nanoscan_time<=end)
        Nanoscan_time = Nanoscan_time[index]
        Nanoscan_Total = Nanoscan_Total[index]
        Nanoscan_data = Nanoscan_data[index]
    
    # Collect all data in a pandas DataFrame for easy handling
    Data_return = np.column_stack([Nanoscan_time,Nanoscan_Total,Nanoscan_data])
    Header = ["Datetime","Total"] + list(Bin_mids)
    
    return Data_return, Bin_edges, Header

###############################################################################
###############################################################################
###############################################################################

def Load_OPCN3_Bin(file, start=0, end=0):
    """
    Function to load data from a .txt datafile generated by the OPCN3 and 
    extract relevant data for plotting.
    
    Parameters
    ----------
    file : str
        Path to a .txt file generated by the OPCN3
    start : datetime, optional
        Specify starting time to plot by setting a datetime value e.g.
        datetime.datetime(2023,01,28,15,00,00). The default is 0.
    end : datetime, optional
        Specify end time to plot by setting a datetime value e.g.
        datetime.datetime(2023,01,28,15,00,00). The default is 0.
    
    Returns
    -------
    Data_return : numpy.array
        Array containing datetime, total concentration in cm-3 and binned particle
        concentration data from the OPCN3.
    Bin_edges : numpy.array
        Array of the bin edges of the OPCN3 in nm
    Header : list
        A list containing the header for all columns in the Data_return variable
    
    Creator ABL
    """
    encod,delim=detect_delimiter(file)

    # Load the data from the OPCN3 .txt file
    Raw_data = np.genfromtxt(file,delimiter=delim,encoding=encod,skip_header=1,dtype="str")
    
    # Extract the count data related to particle bins
    Pbin_data = Raw_data[:,1:-9].astype("float") # counts
    
    # Extract the sample period and flowrate
    Period = Raw_data[:,26].astype("float")/100. # s 
    Flowrate = Raw_data[:,27].astype("float")/100. # cm3/s
      
    # Determine sample volume for each timestep
    Sample_volume = Period * Flowrate # cm3
    
    # Convert count data into concentration
    Pbin_data = np.true_divide(Pbin_data,Sample_volume[:,None]) # cm-3
    
    # Calculate total concentration
    Total = Pbin_data.sum(axis=1)
    
    
    # Convert the specifed dates and times into datetime objects. 
    # An hour is added due to time difference
    # Excess spaces in some time cells are corrected by the [-19:] indexing
    DateTime=np.array([datetime.datetime.strptime(i.split(".")[0][-19:], '%Y-%m-%dT%H:%M:%S') for i in Raw_data[:,0]])
    
    # Bin edges of the OPCN3
    Bin_edges = np.array([0.35, 0.46, 0.66, 1.0, 1.3, 1.7, 2.3, 3.0, 4.0, 5.2, 6.5, 
                          8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 25.0, 28.0, 
                          31.0, 34.0, 37.0, 40.0]) * 1000
    
    # Calculate bin mids in nm
    Bin_mids = (Bin_edges[:-1] + (Bin_edges[1:]-Bin_edges[:-1])/2) * 1000
    
    # Generate header
    Header = ["Datetime","Total"] + list(Bin_mids)
        
    # Select data within specified time interval if specified
    if (start != 0) or (end != 0):
        if (start != 0) & (end == 0):
            index = DateTime>=start
        elif (start == 0) & (end != 0):
            index = DateTime<=end
        else:
            index = (DateTime>=start) & (DateTime<=end)
        DateTime = DateTime[index]
        Pbin_data = Pbin_data[index]
    
    # Combine relevant data to one array before returning it
    Data_return = np.column_stack((DateTime,Total,Pbin_data))
    
    return Data_return, Bin_edges, Header

###############################################################################
###############################################################################
###############################################################################

def Load_OPCN3_PM(file, start=0, end=0):
    """
    Function to load data from a .csv datafile generated by the OPCN3 and 
    extract relevant data for plotting.
    
    Parameters
    ----------
    file : str
        Path to a .csv file generated by the OPCN3
    start : datetime, optional
        Specify starting time to plot by setting a datetime value e.g.
        datetime.datetime(2023,01,28,15,00,00). The default is 0.
    end : datetime, optional
        Specify end time to plot by setting a datetime value e.g.
        datetime.datetime(2023,01,28,15,00,00). The default is 0.

    Returns
    -------
    Data_return : numpy.array
        Array containing PM1, PM2.5 and PM10 data from the OPC including a column
        of Datetime values for easy plotting.
    Header : list
        A list containing the header for all columns in the Data_return variable

    Creators JBL, JOT, and ABL
    """
    encod,delim=detect_delimiter(file)

    # Load data
    PM_data=np.genfromtxt(file,delimiter=delim,encoding=encod,skip_header=1)[:,-4:-1]
    time_s=np.genfromtxt(file,delimiter=delim,encoding=encod,skip_header=1, dtype=str)[:,0]
    
    # Define header
    Header=["Time","PM1(ug/m3)","PM2.5(ug/m3)","PM10(ug/m3)"]
    
    # Convert the specifed dates and times into datetime objects. 
    # An hour is added due to time difference
    # Excess spaces in some time cells are corrected by the [-19:] indexing
    DateTime=np.array([datetime.datetime.strptime(i.split(".")[0][-19:], '%Y-%m-%dT%H:%M:%S') for i in time_s])
        
    # Select data within specified time interval if specified
    if (start != 0) or (end != 0):
        if (start != 0) & (end == 0):
            index = DateTime>=start
        elif (start == 0) & (end != 0):
            index = DateTime<=end
        else:
            index = (DateTime>=start) & (DateTime<=end)
        DateTime = DateTime[index]
        PM_data = PM_data[index]
    
    # Combine time with PM data
    PM_Data=np.column_stack((DateTime,PM_data))

    return PM_Data, Header

###############################################################################
###############################################################################
###############################################################################

def Load_OPS(file, start=0, end=0,sep=","):
        """
        Function to load data from comma seperated .txt datafile generated by 
        the OPS and extract relevant data for plotting.
        
        Parameters
        ----------
        file : str
            Path to a .txt file generated by the OPS
        start : datetime, optional
            Specify starting time to plot by setting a datetime value e.g.
            datetime.datetime(2023,01,28,15,00,00). The default is 0.
        end : datetime, optional
            Specify end time to plot by setting a datetime value e.g.
            datetime.datetime(2023,01,28,15,00,00). The default is 0.
        sep : str
            Specify the separator of the csv file columns. The default is ",". 
        Returns
        -------
        data_return : numpy.array
            An array containing the imported data from the OPS. The data is
            organized in columns of: time, total number concentration, data. Where
            data refers to all the concentrations measured at each sizebin.
        Bin_edges : numpy.array
            Array of bin edges for the OPS in nm
        Header : list
            List of all the headers for the columns in the Data_return array
        """
        encod,delim=detect_delimiter(file)

        # for seperator in [',','\t',';']:
        #     OPS = pd.read_csv(file, sep=seperator, header=13, encoding='latin1')
        #     if len(np.array(OPS)[0,:])==1:
        #         pass
        #     else:
        #         sep=seperator
        #         break
        OPS = pd.read_csv(file, delimiter=delim,encoding=encod, header=13)
        # Load particle data from the txt file
        # OPS = pd.read_csv(file, sep=sep, header=13, encoding= 'latin1')
        
        # Store OPS particle data
        OPS_data = np.array(OPS[list(OPS)[17:-2]])
        
        # # Store total number concentrations
        OPS_Total = np.array(OPS['Total Conc. (#/cm³)'])
        
        # Store sizebin values from the header
        Bin_mids = np.array(list(OPS)[17:-2],dtype=float)*1000 # nm
        
        # Load the lower and upper bin boundaries and combine them to get the
        # edges of all bins
        Bin_edges_LB = np.genfromtxt(file,delimiter=delim,encoding=encod,skip_header=10,skip_footer=OPS_data.shape[0]+4)[17:-1]
        Bin_edges_UB = np.genfromtxt(file,delimiter=delim,encoding=encod,skip_header=11,skip_footer=OPS_data.shape[0]+3,usecols=-2)
        Bin_edges = np.append(Bin_edges_LB,[Bin_edges_UB])
        
        #Checks unit format (dW, dW/dP, dW/dlogDp)
        Unit = np.genfromtxt(file,delimiter=delim,encoding=encod,skip_header=6,max_rows=1,dtype="str")[1]
        #Applies correction to report back the value in 1/cm^3
        if 'dW/dlogDp' in Unit:
            # Calculate the normalization vector
            dlogDp = np.log10(Bin_edges[1:])-np.log10(Bin_edges[:-1])
            OPS_data= OPS_data*dlogDp
        elif 'dW/dDp' in Unit:
            # Calculate the normalization vector
            dDp = Bin_edges[1:]-Bin_edges[:-1]
            OPS_data= OPS_data*dDp
        elif 'dW' in Unit:
            pass
        else:
            return print("The reported unit doesn't match either: dW, dW/dDp, dW/dlogDp")
        
        Bin_edges=Bin_edges*1000# nm 
        
        # # Combine and format the time and date to a datetime value
        OPS_date_and_time = np.array(OPS["Date"] + " " + OPS["Start Time"])
        OPS_datetime = np.array([datetime.datetime.strptime(i, '%m/%d/%Y %H:%M:%S') 
                              for i in OPS_date_and_time])  
        
        # Select data within specified time interval if specified
        if (start != 0) or (end != 0):
            if (start != 0) & (end == 0):
                index = OPS_datetime>=start
            elif (start == 0) & (end != 0):
                index = OPS_datetime<=end
            else:
                index = (OPS_datetime>=start) & (OPS_datetime<=end)
            OPS_datetime = OPS_datetime[index]
            OPS_Total = OPS_Total[index]
            OPS_data = OPS_data[index]
        
        # Collect all data in a pandas DataFrame for easy handling
        Data_return = np.column_stack([OPS_datetime,OPS_Total,OPS_data])
        Header = ["Datetime","Total"] + list(Bin_mids)

        return Data_return, Bin_edges, Header

###############################################################################
###############################################################################
###############################################################################

def Load_OPS_Direct(file, start=0, end=0):
        """
        Function to load data from a .csv file, which has been generated by the
        OPS without a PC connected. This type of export gives an additional bin
        (bin 17), which may contain the concentration for particles > 10 um, but
        this is not mentioned in the manual, and the bin is therefore excluded
        here.
        Be aware, that the data has the unit specified when the measurement was
        started! If other units are needed, you can convert with the functions
        in the NFA_utility library.
        
        Parameters
        ----------
        file : str
            Path to a .csv file generated by the OPS
        start : datetime, optional
            Specify starting time to plot by setting a datetime value e.g.
            datetime.datetime(2023,01,28,15,00,00). The default is 0.
        end : datetime, optional
            Specify end time to plot by setting a datetime value e.g.
            datetime.datetime(2023,01,28,15,00,00). The default is 0.
        Returns
        -------
        data_return : numpy.array
            An array containing the imported data from the OPS. The data is
            organized in columns of: time, total number concentration, data. Where
            data refers to all the concentrations measured at each sizebin.
        Bin_edges : numpy.array
            Array of bin edges for the OPS in nm
        Header : list
            List of all the headers for the columns in the Data_return array
        """
        encod,delim=detect_delimiter(file)

        # Load particle data from the csv file. Here the column named bin 17 is
        # excluded, as it is not mentioned in the TSI manual. Supposedly it 
        # gives concentration of particles > 10 um.
        OPS = np.genfromtxt(file, delimiter=delim,encoding=encod,skip_header=38)[:,:17]
        
        #Finds the deadtime to use for conversion from counts to concentration 
        Deadtime = np.genfromtxt(file,delimiter=delim,encoding=encod,skip_header=38)[:,18] # s
        
        # Store OPS particle data
        OPS_data = OPS[:,1:] # counts
        
        #Finds the sample length in seconds
        SL= np.genfromtxt(file, delimiter=delim,encoding=encod,skip_header=9,skip_footer=OPS.shape[0]+28,dtype=str)[1]
        SL= np.array(SL.split(":"))
        Sample_length=3600*int(SL[0])+60*int(SL[1])+int(SL[2])
        
        # Convert data from counts to particle concentrations
        OPS_data=np.true_divide(OPS_data,16.67*(Sample_length-Deadtime[:, np.newaxis])) # counts / (16.67 cm3/s *(60 s - Deadtime)
            
        # Calculate total number concentrations
        OPS_Total = np.nansum(OPS_data,axis=1)
        
        # Store sizebin values, which are constant and therefore not imported
        Bin_edges = np.genfromtxt(file, delimiter=delim,encoding=encod,skip_header=11,skip_footer=OPS.shape[0]+10,usecols=1)
        
        # Calculate mid points of the sizebins
        Bin_mids = Bin_edges[:-1] + (Bin_edges[1:]-Bin_edges[:-1])/2
        
        # Load the start date and time from the datafile
        OPS_date = np.genfromtxt(file, delimiter=delim,encoding=encod,skip_header=7,skip_footer=OPS.shape[0]+30,dtype="str")[1]
        OPS_starttime = np.genfromtxt(file, delimiter=delim,encoding=encod,skip_header=6,skip_footer=OPS.shape[0]+31,dtype="str")[1]
        
        # Convert start date and time into a datetime value
        start_datetime = datetime.datetime.strptime(OPS_date + " " + OPS_starttime, '%Y/%m/%d %H:%M:%S')
        
        # Store the elapsed time for each measurement in the datafile
        OPS_elapsedtime = OPS[:,0]
        
        # Add the elapsed time to the starting time, in order to get datetime
        # values for all measurements
        OPS_datetime = np.array([start_datetime + datetime.timedelta(0,i) for i in OPS_elapsedtime])
               
        # Select data within specified time interval if specified
        if (start != 0) or (end != 0):
            if (start != 0) & (end == 0):
                index = OPS_datetime>=start
            elif (start == 0) & (end != 0):
                index = OPS_datetime<=end
            else:
                index = (OPS_datetime>=start) & (OPS_datetime<=end)
            OPS_datetime = OPS_datetime[index]
            OPS_Total = OPS_Total[index]
            OPS_data = OPS_data[index]
        
        # Collect all data in a pandas DataFrame for easy handling
        Data_return = np.column_stack([OPS_datetime,OPS_Total,OPS_data])
        Header = ["Datetime","Total"] + list(Bin_mids)

        return Data_return, Bin_edges, Header
    
###############################################################################
###############################################################################
###############################################################################

def Load_Partector(file, start=0, end=0):
    """
    Function to load data from a .csv datafile generated by the Partector and 
    extract relevant data for plotting.
    
    Parameters
    ----------
    file : str
        Path to a .csv file generated by the Partector
    start : datetime, optional
        Specify starting time to plot by setting a datetime value e.g.
        datetime.datetime(2023,01,28,15,00,00). The default is 0.
    end : datetime, optional
        Specify end time to plot by setting a datetime value e.g.
        datetime.datetime(2023,01,28,15,00,00). The default is 0.
    Returns
    -------
    Data_return : numpy.array
        Array containing all data from the Partector including a column
        of Datetime values for easy plotting.
    Header : list
        A list containing the header for all columns in the Data_return variable
    """

    # Load partector file
    Partector_Data = pd.read_csv(file,  delimiter="\t",header=10)
    
    # Extract the relative times of the file
    Partector_times = np.array(Partector_Data["time"])
    
    # Read the starting time from the partector file
    Partector_start = pd.read_csv(file, delimiter="\t", skiprows=4, nrows=1,header=None)
    
    # Convert the starting time from a string to a datetime value
    start_str = str(Partector_start.loc[0]).split("Start: ")[1].split("\n")[0]
    start_datetime = datetime.datetime.strptime(start_str,"%d.%m.%Y %H:%M:%S")
    
    # Add the times of the file to the starting time
    Partector_Datetime = np.array([start_datetime + datetime.timedelta(0,i) for i in Partector_times])
    
    # Select data within specified time interval if specified
    if (start != 0) or (end != 0):
        if (start != 0) & (end == 0):
            index = Partector_Datetime>=start
        elif (start == 0) & (end != 0):
            index = Partector_Datetime<=end
        else:
            index = (Partector_Datetime>=start) & (Partector_Datetime<=end)
        Partector_Datetime = Partector_Datetime[index]
        Partector_Data = Partector_Data[index]
    
    # insert the datetime values to the data cube
    Partector_Data.insert(0, "Datetime", Partector_Datetime, True)
  
    # Get the header for each column and convert the DataFrame to an array
    Header = list(Partector_Data)
    Data_return = np.array(Partector_Data)
    
    return Data_return, Header

###############################################################################
###############################################################################
###############################################################################

def Load_SMPS(file,start=0,end=0):
    """
    Function to load data from a "," separated .txt datafile generated by the 
    SMPS and extract relevant data for plotting.
    
    Parameters
    ----------
    file : str
        Path to a .txt file generated by the SMPS
    start : datetime, optional
        Specify starting time to plot by setting a datetime value e.g.
        datetime.datetime(2023,01,28,15,00,00). The default is 0.
    end : datetime, optional
        Specify end time to plot by setting a datetime value e.g.
        datetime.datetime(2023,01,28,15,00,00). The default is 0.
    Returns
    -------
    data_return : numpy.array
        An array containing the imported data from the SMPS. The data is
        organized in columns of: time, total number concentration, data. Where
        data refers to all the concentrations measured at each sizebin.
    Bin_edges : numpy.array
        Array of the limits of the SMPS size bins
    Header : list
        List of all the headers for the columns in the Data_return array
    """
    encod,delim=detect_delimiter(file)

    # Load SMPS file
    SMPS = pd.read_csv(file, delimiter=delim,encoding=encod, header=25)
    
    # Get headers for the size bins
    SMPS_sizes_index = list(SMPS)[9:105]
    
    # Get the total concentration
    SMPS_total = np.array(SMPS["Total Conc. (#/cm³)"])
    
    # Store data for each size bin using the headers
    SMPS_data = np.array(SMPS[SMPS_sizes_index])
    
    # Get the size bins as numerical data rather than header format
    Bin_mids = np.genfromtxt(file,delimiter=delim,encoding=encod,skip_header=25,skip_footer=26,
                             dtype=float,comments="!!!!")[0,9:105]
    
    # Calculate all possible bin sizes of the SMPS (varies on flow settings)
    Bin_Mid_log = np.around(np.array([10**((i-0.5)/64.) for i in range(192)]),1)
    
    # Find the bins limits that match the current settings
    Size_index = np.nonzero(np.in1d(Bin_Mid_log, Bin_mids))[0]
    
    # include one additional upper index to get n+1 bin edges
    bin_index = np.append(Size_index,Size_index[-1]+1)
        
    # Calculate lower limits of all size bin and select the relevant ones
    Bin_edges = np.array([(10**((i-1)/64.)) for i in range(192)])[bin_index]
    
    # Format time data
    SMPS_time = np.array([datetime.datetime.strptime(i, '%d/%m/%Y %H:%M:%S') 
                          for i in np.array(SMPS["Date"]+" "+SMPS["Start Time"])])
    
    # Select data within specified time interval if specified
    if (start != 0) or (end != 0):
        if (start != 0) & (end == 0):
            index = SMPS_time>=start
        elif (start == 0) & (end != 0):
            index = SMPS_time<=end
        else:
            index = (SMPS_time>=start) & (SMPS_time<=end)
        SMPS_time = SMPS_time[index]
        SMPS_total = SMPS_total[index]
        SMPS_data = SMPS_data[index]
    
    # Collect all data in a pandas DataFrame for easy handling
    Data_return = np.column_stack([SMPS_time,SMPS_total,SMPS_data])
    Header = ["Datetime","Total"] + list(Bin_mids)
    
    return Data_return, Bin_edges, Header

###############################################################################
###############################################################################
###############################################################################

def Load_FMPS_1(file, year, month, start=0, end=0):
    """
    Function to load FMPS data similar to the one in Instrument_Lib.py but
    that one needed a different format 
    
    Parameters
    ----------
    file : str
        Path to a .txt file generated by the FMPS
    start : datetime, optional
        Specify starting time to plot by setting a datetime value e.g.
        datetime.datetime(2023,01,28,15,00,00). The default is 0.
    end : datetime, optional
        Specify end time to plot by setting a datetime value e.g.
        datetime.datetime(2023,01,28,15,00,00). The default is 0.

    Returns
    -------
    data_return : numpy.array
        An array containing the imported data from the FMPS. The data is
        organized in columns of: time, total number concentration, data. Where
        data refers to all the concentrations measured at each electrometer.
    Bin_edges : numpy.array
        Array of edges for the FMPS sizebins in nm
    Header : list
        List of all the headers for the columns in the Data_return array
        
    Creator: PLF

    """
    encod,delim=detect_delimiter(file)

    # Load particle data from the txt file
    FMPS_data = np.genfromtxt(file,delimiter=delim,encoding=encod,skip_header=15)[:,1:-11]
    
    # Calculate total number concentrations
    FMPS_Total = np.genfromtxt(file,delimiter=delim,encoding=encod,skip_header=15)[:,35]
    
    # Load sizebin values from the data file
    Bin_mids = np.genfromtxt(file,delimiter=delim,encoding=encod,skip_header=13,skip_footer=FMPS_data.shape[0]+1)[1:-11]
    Bin_edges = (Bin_mids[1:]+Bin_mids[:-1])/2
    Bin_edges = np.append(5.6, Bin_edges)
    Bin_edges = np.append(Bin_edges,560)
    
    # Load the times
    FMPS_time = np.genfromtxt(file,delimiter=delim,encoding=encod,skip_header=15,dtype=str)[:,0]
    
    # Convert time and dates to a datetime
    with open(file, 'r') as file:
        for line in file:
            if line.startswith('Date/Time Start:'):
                # Extract the date and time part from the line
                date_time_str = line.split('Date/Time Start:,"')[1].split('"')[0]
    
                # Parsing the string into a datetime object
                parsed_date = datetime.datetime.strptime(date_time_str, "%A, %B %d, %Y %I:%M:%S %p")
    
                # Extracting the day
                day_start = parsed_date.day
                break
    FMPS_datetime = []
    prev_hour_24 = None
    day_offset = 0
    
    for i in FMPS_time:
        parts = i.split(":")
        hour = int(parts[0])
        minute = int(parts[1])
        seconds, am_pm = parts[2].split(" ")
        second = int(seconds)
    
        # Convert to 24-hour format
        if am_pm == 'PM' and hour != 12:
            hour += 12
        elif am_pm == 'AM' and hour == 12:
            hour = 0
    
        # Determine if the day has rolled over
        current_hour_24 = hour
        if prev_hour_24 is not None and current_hour_24 < prev_hour_24:
            day_offset += 1
        prev_hour_24 = current_hour_24
    
        current_datetime = datetime.datetime(year, month, day_start, hour, minute, second) + datetime.timedelta(days=day_offset)
        FMPS_datetime.append(current_datetime)

    FMPS_datetime = np.array(FMPS_datetime)
    
    # Select data within specified time interval if specified
    if (start != 0) or (end != 0):
        if (start != 0) & (end == 0):
            index = FMPS_datetime>=start
        elif (start == 0) & (end != 0):
            index = FMPS_datetime<=end
        else:
            index = (FMPS_datetime>=start) & (FMPS_datetime<=end)
        FMPS_datetime = FMPS_datetime[index]
        FMPS_Total = FMPS_Total[index]
        FMPS_data = FMPS_data[index]
    
    # Collect all data in a pandas DataFrame for easy handling
    Data_return = np.column_stack([FMPS_datetime,FMPS_Total,FMPS_data])
    Header = ["Datetime","Total"] + list(Bin_mids)
    
    return Data_return, Bin_edges, Header

###############################################################################
###############################################################################
###############################################################################

def Load_filter_file(file: str,sheet_name : str = "Summary Filters"):
    """
    Load excel sheet of filter data and convert it into a dataframe.

    This function reads ELPI exports (usually .txt files), extracts datetime and
    particle size distribution information, applies unit conversions (e.g., from dW/dlogDp),
    and calculates total concentration and size-resolved particle data in cm⁻³.

    Parameters
    ----------
    file : str
        Path to the ELPI-exported .dat file.
    extra_data : bool, optional
        If True, retains and returns all non-distribution data in `.extra_data`. Default is False.

    Returns
    -------
    Data : dataframe
        Object containing parsed size distribution data, total concentration,
        and instrument metadata.

    Raises
    ------
    Exception
        If unit or weight format cannot be determined, or parsing fails.

    Notes
    -----
    - Bin mids and edges are stored in nanometers.
    - Normalization is done to convert to number concentration `dN`.
    - Supports dynamic density-aware edge recomputation when density ≠ 1.
    """

    # Load metadata and bin descriptors
        
        
    # Load the file first — only the first column, to find where data begins
    temp = pd.read_excel(file, sheet_name=sheet_name, usecols=[0], header=None)
    
    # Find the row index where the first column equals 'Filter'
    start_row = temp[temp[0] == "Filter diameter"].index[0]
    
    # Now load the full data, using that row as the header
    data = pd.read_excel(file, sheet_name=sheet_name, header=start_row)

    # data = pd.read_excel(file,sheet_name=sheet_name,skiprows=10)#usecols=[0,1,2,4])
    units = np.array(data.iloc[0], dtype=str)
    columns = np.array(data.columns, dtype=str)
    rows = {columns[i]: f"{columns[i]} {units[i]}" if "[" in units[i] else f"{columns[i]}"
            for i in range(len(units))}
    
    # Clean data for empty rows and standardize start and stop time
    for i in range(0,len(data)):
        time_check=0
        if type(data['Filter type'][i])!=str:
            data.drop(index=i,inplace=True)
        else:
            #Try and parse the time for non blank samples
            if data.loc[i,'Filter type']!='BLANK':
                
                #Determine start time
                if type(data.loc[i,"Start time"] ) == datetime.datetime:
                    data.loc[i,"Start time"] = datetime.datetime.combine(data.loc[i,"Measurement date"].date(),data.loc[i,"Start time"].time())

                elif type(data.loc[i,"Start time"] ) == datetime.time:
                    data.loc[i,"Start time"] = datetime.datetime.combine(data.loc[i,"Measurement date"].date(),data.loc[i,"Start time"])
                
                else: #In case of nan value for start
                    time_check=1
                    
                #Determine stop time
                if type(data.loc[i,"Stop time"] ) == datetime.datetime:
                    data.loc[i,"Stop time"]  = datetime.datetime.combine(data.loc[i,"Measurement date"].date(),data.loc[i,'Stop time'].time())
  
                elif type(data.loc[i,"Stop time"] ) == datetime.time:
                    data.loc[i,"Stop time"]  = datetime.datetime.combine(data.loc[i,"Measurement date"].date(),data.loc[i,'Stop time'])
                
                else: #In case of nan value for stop
                    time_check+=2
                    
                if time_check==0:
                    if data.loc[i,"Stop time"] < data.loc[i,"Start time"]:
                        #Incase the measurement extends across midnight
                        days=1+int(data.loc[i,'Sampled time']/24)
                        data.loc[i,'Stop time']=data.loc[i,"Stop time"] + datetime.timedelta(days)
                
                elif data.loc[i,"Sampled time"]>0:
                    if time_check==1:
                        data.loc[i,"Start time"] = data.loc[i,"Stop time"] - datetime.timedelta(data.loc[i,"Sampled time"]/24)
                    elif time_check==2:
                        data.loc[i,"Stop time"] = data.loc[i,"Start time"] + datetime.timedelta(data.loc[i,"Sampled time"]/24)

                else:
                    data.loc[i,'Sampled time'] = round(data.loc[i,"Sampled volume"]/data.loc[i,"Flow"]/60,2)
                    
                    if time_check==1:
                        data.loc[i,"Start time"] = data.loc[i,"Stop time"] - datetime.timedelta(0,int(data.loc[i,"Sampled volume"]/data.loc[i,"Flow"]*60))
                    elif time_check==2:
                        data.loc[i,"Stop time"] = data.loc[i,"Start time"] + datetime.timedelta(0,int(data.loc[i,"Sampled volume"]/data.loc[i,"Flow"]*60))
        
    data['Detection Limit'][data['Detection Limit'].isnull()]=0
    # Check whether the data is below the detection limit
    data['Mean concentration']=data['Mean concentration']*1000
    data['Mean concentration'][data['Mean concentration']<data['Detection Limit']*1000] ='BDL'

    # Final formalizing
    data.reset_index(inplace=True,drop=True)
    data=data.rename(columns=rows)
    data=data[['Filter diameter', 'Filter type', 'Filter number (reference)',
           'Measurement location (working area, personal or outdoors)',
           'Start time', 'Stop time', 'Sampled time [h]', 'Sampled volume [L]',
           'Mean concentration [µg/m3]']]

    # data.drop(columns=data.columns[19:],inplace=True) 
    
    # [data.drop(columns=data.columns[i],inplace=True) for i in [18,16,14,13,9,7,6,5,4,3]]
    
    # Split into two DataFrames
    data_blank = data[data["Filter type"] == "BLANK"]
    data_blank.reset_index(inplace=True,drop=True)

    data = data[data["Filter type"] != "BLANK"]
    data.reset_index(inplace=True,drop=True)

    return data, data_blank
