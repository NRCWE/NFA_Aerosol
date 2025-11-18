# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 14:54:00 2023

@author: B279683
"""

import numpy as np
import datetime as datetime
import pandas as pd
from scipy.optimize import curve_fit, least_squares
import os as os
from matplotlib.dates import date2num, num2date
from math import erf

#%% Library inventory
"""This library contains the following functions:
    
    APS_cor_average:
        Used for calculating the average of the correlated plots generated from APS
    
    APS_cor_summation: 
        A function to seperate the correlated APS data into its aerodynamic and
        optical data using the regular data structure. 
        
    Combine_NS_OPS:
        Function to combine Nanoscan and OPS data. 
        
    Diffusion_loss:
        Determine the diffusion loss of particles in tubing. 
        
    File_list
        Function to generate a list of all files in a folder.
        
    Fit_lognormal:
        A function to fit one or multiple peaks following lognormal distribution,
        with the option for tethering values to set values. 
        
    Fit_lognormal_multivariate:
        A function to fit one or multiple peaks following double lognormal distribution,
        with the option for tethering values to set values.
        
    Normal_multivariate   
        The mathmatical expression of a lognormal distribution along two variables.
        
    ICRP_fraction:    
        Function to calculate the lung deposited fraction for an array of bin sizes
        based on the ICRP model.
        
    fit_linear:
        Function to detmine the correlation between two sets of values, which have been
        aligned so as to have sensible comparison points. 
    
    Lognormal:
        The mathmatical expression of a log10 (lognormal distribution). The function can be
        used to genereate a theoretical lognormal distribution and is also used by
        the Fit_lognormal function.
        
    MPS_Ceff:
        Function to estimate the collection efficiency of the mini particle sampler (MPS)
        as a function of particle size and flowrate.
    
    Normal:
        The mathmatical expression of a lognormal distribution.
    
    Normalize_dndlogdp:
        Function to normalize an array of particle xxx concentrations (xxx can be
        mass, volume, surface area, and number), going from dxxx values to 
        dxxx/dlogDp.
    
    num2mass:
        Function to convert from number concentration to mass
        concentration, assuming spherical particles with a diameter equal to the
        size bin mid points, and a density as specified.
    
    num2surface:
        Function to convert from number concentration to surface area
        concentration, assuming spherical particles with a diameter equal to the
        size bin mid points.
        
    num2vol:
        Function to convert from number concentration to volume
        concentration, assuming spherical particles with a diameter equal to the
        size bin mid points.
    
    Partector_Ceff:
        Function to estimate the collection efficiency of the partectorTEM at the
        specified particle size in nm.
    
    Partector_sample_vol:
        Function for calculating the volume of air sampled through the TEM filter
    
    PM_calc:
        Function to calculate PM from size bins from an array similar to those
        returned from Load_xxx functions.
    
    Rolling_window:
        A function used to take the median or mean value within a rolling window, 
        in order to adjust the value of the central data point. 
    
    Save_2_Excel:
        Function to store data to an excel file.
        
    Save_plot:
        Function to save a figure at the specified path and with the given filename 
        and extension.
    
    segment_dataset:
        Function to segment a dataset into different activities e.g. background and
        different workplace processes.
    
    time_crop:
        Function to crop a dataset, by setting a start and/or an end time.
    
    time_rebin:
        Function to calculate the average values across time with a set time step.
        
    time_shift:
        Function to adjust the times of a dataset.
    
    Unnormalize_dndlogdp:
        Function to unnormalize an array of particle xxx concentrations (xxx can be
        mass, volume, surface area, and number), going from dxxx/dlogDp to dx values.

###############################################################################
In addition this library also includes the function Peak_fitter and its associated
functions.

    Peak_fitter:
        Find the location of the peak by identifying the maximum value,
        and fits one of three functions to the peak, as well as provide emission calculations
    
    """
#%% Group of base functions
def APS_cor_average(data,start=0,end=0):
    """
    Used for calculating the average of the correlated plots generated from APS
    
    Parameters
    ----------
    data : numpy.array
        Data format as returned from Load_APS_full(Returned_data=['Cor'])
    start : datetime, optional
        Optional start time to designate from which the averaging should be started.
        Default is 0, which starts from the beginning.
    end : datetime, optional
        Optional start time to designate from which the averaging should be ended.
        Default is 0, which starts from the beginning.

    Returns
    -------
    Average : numpy.array
        The average value for each Optic vs Aerodynamic bin across the period.
        Dimensions are [16,51]

    """
    data = data.copy()
    Rep_avg = data[0,2].copy()
    Average = data[0,:].copy()
    # Determines if the average should be found within a time frame.
    if (start != 0) or (end != 0):
        if (start != 0) & (end == 0):
            index = data[:,0]>=start
        elif (start == 0) & (end != 0):
            index = data[:,0]<=end
        else:
            index = (data[:,0]>=start) & (data[:,0]<=end)            
        data = data[index]
    
    datetime.timedelta()
    for A in range(0,len(data[0,2][0,:])):
        for O in range(0,len(data[0,2][:,0])):
            avg = []
            for t in range(0,len(data)):
                avg = np.append(avg,data[t,2][O,A])
            Rep_avg[O,A] = np.mean(avg)
            
    Average[0] = num2date((date2num(data[-1,0])+date2num(data[0,0]))/2)    
    Average[1] = np.sum(Rep_avg)
    Average[2] = Rep_avg
    return Average
###############################################################################
###############################################################################
def APS_cor_summation(data):
    """
    A function to seperate the correlated APS data into its aerodynamic and
    optical data using the regular data structure. 

    Parameters
    ----------
    data : array
        Correlated APS data as generated from Load_APS_full with Returned_data=["Cor"]

    Returns
    -------
    Aero_data : array
        Data structured as gained from Load_APS_full with Returned_data=["Aero"]
    Optic_data : array
        Data structured as gained from Load_APS_full with Returned_data=["Optic"]
    """
    Cor_data=data.copy()
    #Sum up the correlated data based on aerodynaic size bins
    APS_datetime=[]
    APS_total=[]
    
    Aero_data=[]
    Optic_data=[]
    
    for y in range(0,len(Cor_data)):
        Aero=[]
        APS_datetime=np.append(APS_datetime,Cor_data[y,0])
        APS_total=np.append(APS_total,Cor_data[y,1])
        for i in range(0,len(Cor_data[y,2][0,:])):
            Aero=np.append(Aero,np.sum(Cor_data[y,2][:,i]))
        if len(Aero_data)==0:
            Aero_data=Aero.copy()
        else:
            Aero_data=np.row_stack((Aero_data,Aero.copy()))
    
    # Collect all aerodynamic data 
    Aero_data = np.column_stack([APS_datetime,APS_total,Aero_data])

    #Sum up the correlated data based on optic size bins
    for y in range(0,len(Cor_data)):
        Optic=[]
        for i in range(0,16):
            Optic=np.append(Optic,np.sum(Cor_data[y,2][i,:]))
        if len(Optic_data)==0:
            Optic_data=Optic.copy()
        else:
            Optic_data=np.row_stack((Optic_data,Optic.copy()))
        
    # Collect all optical data
    Optic_data = np.column_stack([APS_datetime,APS_total,Optic_data])
    return Aero_data, Optic_data
###############################################################################
###############################################################################
def Combine_NS_OPS(data_NS,data_OPS,OPS_bin=[],starttime=None,endtime=None):
    """
    Function to combine Nanoscan and OPS data. If no start or end time are specified
    then the function will use the first and last datapoints that exist for both
    instruments. 
    It is assumed that the NS and OPS use their standard sizebins, meaning that
    the size bin edges are:
        
    NS: [ 10.  ,  13.45,  17.95,  23.95,  31.95,  42.6 ,  56.8 ,  75.75,
           101.05, 134.75, 179.7 , 239.6 , 319.5 , 420.  ]
    
    OPS: [  300.,   374.,   465.,   579.,   721.,   897.,  1117.,  1391.,
            1732.,  2156.,  2685.,  3343.,  4162.,  5182.,  6451.,  8031.,
           10000.]
    
    Parameters
    ----------
    data_NS : numpy.array
        An array of data from the Nanoscan as returned by the IL.Load_Nanoscan
        function.
    data_OPS : numpy.array
        An array of data from the OPS as returned by the IL.Load_OPS
        function.
    starttime : datetime, optional
        The starting time for combining the two datasets. If not speicifed, a 
        starting point is found as the first shared datetime. The default is None.
    endtime : datetime, optional
        The end time for combining the two datasets. If not speicifed, an end
        point is found as the last shared datetime. The default is None.

    Returns
    -------
    Combined_NS_OPS : numpy.array
        An array of the combined NS and OPS datasets, with newly calculated total
        concentrations. The last bin of the NS has been ignored and the second
        to last has been shortened and its number reduced accordingly.
    New_bin_edges : numpy.array
        Array of new size bin edges in nm.
    Header : list
        Header of all data columns

    """
    
    
    # Round all the datetime values, so they do not have seconds, in order to ease the alignment process
    data_OPS[:,0]= np.array([(dt+ datetime.timedelta(0,30)).replace(second=0) if dt.second>30 
                            else dt.replace(second=0) for dt in data_OPS[:,0]])
    data_NS[:,0]= np.array([(dt+ datetime.timedelta(0,30)).replace(second=0) if dt.second>30 
                            else dt.replace(second=0) for dt in data_NS[:,0]])
    
    # Determine the start time for combining the data either from the specified time
    # or from the first datapoint which exist for both instruments
    if starttime:
        starttime = starttime.replace(second=0)
    else:
        starttime = max(data_OPS[0,0],data_NS[0,0])
    
    # Determine the end time for combining the data either from the specified time
    # or from the last datapoint which exist for both instruments
    if endtime:
        endtime = endtime.replace(second=0)
    else:
        endtime = min(data_OPS[-1,0],data_NS[-1,0])
    
    # Grab the NS data within the specifed start and end times
    NS_time_matched = data_NS[(data_NS[:,0]>=starttime) & (data_NS[:,0]<=endtime)]
    
    # Grab the OPS data within the specifed start and end times
    OPS_time_matched = data_OPS[(data_OPS[:,0]>=starttime) & (data_OPS[:,0]<=endtime)]
    
    # Drop the total concentration column of the OPS data, as it is recalculated after combining with NS
    OPS_time_matched = np.delete(OPS_time_matched, 1, 1)
   
    # the penultimate NS sizebin has its upper limit reduced from 319.5 to 300, 
    # so the particle number is corrected accordingly
    Factor_reduction = (300-239.6)/(319.5-239.6) 
    New_concentration = NS_time_matched[:,-2].astype("float")*Factor_reduction
    NS_time_matched[:,-2] = New_concentration
    
    DF_NS = pd.DataFrame(NS_time_matched[:,:-1])
    DF_NS[0] = pd.to_datetime(DF_NS[0])
    
    DF_OPS = pd.DataFrame(OPS_time_matched)
    DF_OPS[0] = pd.to_datetime(DF_OPS[0])
    
    # Merge on 'datetime' column, using an outer join to keep all datetime values
    merged_df = pd.merge(DF_NS, DF_OPS, on=0, how='outer')

    # Sort by datetime (optional)
    merged_df = merged_df.sort_values(by=0).reset_index(drop=True)
    
    # The OPS and NS data are combined, but excluding the final bin of the NS
    Combined_NS_OPS = merged_df.to_numpy()
    
    # A new total concentration is calculated based on the combined size bin data
    Combined_NS_OPS[:,1] = np.round(Combined_NS_OPS[:,2:].sum(axis=1).astype("float"),0)
    
    # As it is assumed that the standard bins are used for both the NS and OPS, the new bins will always be:
    if len(OPS_bin)==0:
        New_bin_edges = [  10.  ,  13.45,  17.95,  23.95,  31.95,  42.6 ,  56.8 ,  75.75,
        101.05, 134.75, 179.7 , 239.6, 300.,   374.,   465.,   579.,   721.,   897.,  1117.,  1391.,
        1732.,  2156.,  2685.,  3343.,  4162.,  5182.,  6451.,  8031., 10000.]
    else:
        New_bin_edges = [  10.  ,  13.45,  17.95,  23.95,  31.95,  42.6 ,  56.8 ,  75.75,
        101.05, 134.75, 179.7 , 239.6]
        for i in OPS_bin:
            New_bin_edges.append(i)
        
    New_bin_edges = np.array(New_bin_edges)
    New_bin_mids = New_bin_edges[:-1]+(New_bin_edges[1:]-New_bin_edges[:-1])/2
    Header = ["Datetime","Total"] + list(New_bin_mids)
    
    return Combined_NS_OPS, New_bin_edges, Header

###############################################################################
###############################################################################
###############################################################################
def Diffusion_loss(Data_in, bin_mids, D_tube, L, Q, T = 293, P = 101.3e3):
    """
    Determine the diffusion loss of particles in tubing. The results are reported
    as the fraction of particles going through the tubing relative to the 
    concentration at the inlet for each particle size in the bins parameter. 
    
    In order to correct for the diffusion loss, the particle data should be 
    divided by this vector to represent the actual particle concentration of the
    sampled aerosol.
    
    All equations are from and labeled according to the book: 
    Willeke K, Baron PA. Aerosol measurement: principles, techniques, and 
    applications. 1st edition New York, NY, USA: Van Nostrand Reinhold; 1993.

    Parameters
    ----------
    Data_in : numpy.array
        An array of data as returned by the Load_xxx functions with columns
        of datetime, total conc, and size bin data
    bin_mids : numpy.array
        Array of midpoints for all particle size bins. Sizes should be in nm.
    D_tube : float
        Diameter of the tubing used for sampling. Diameter should be in meters
    L : float
        Length of the tubing used for sampling. Length should be in meters.
    Q : float
        Flow through the tubing in questions. The flow should be in L/min
    T : Temperature, optional
        Temperature of the aerosol in kelvin. The default is 293 K.
    P : Pressure, optional
        Pressure of the sampled aerosol in Pa. The default is 101.3e3 Pa.

    Returns
    -------
    Data_return : numpy.array
        An array of diffusion loss corrected data equivalent to the data_in shape 
        with columns of datetime, total conc, and size bin concentrations.
    eff : numpy.array
        Ratio of particle concentration before and after diffusion losses for
        each particle size in the bins parameter. To apply the results, the
        reported particle concentrations should be divided by this vector.
    
    """
    # Boltzmann's constant
    k = 1.380649e-23
    
    # Convert particle sizes to m
    Dp = np.array(bin_mids) * 1e-9 # m
    
    # Convert flow from L/min to m3/s
    Q = Q/(1000*60) # m3/s
    
    # Calculate flow velocity in the tubing
    V = Q/(1/4*np.pi*D_tube**2) # m/s
    
    # Mean free path at 20 C and atm pressure with correction for pressure and 
    # temperature variations. B&W (3-6)
    mfp_standard = 66.5*1e-9 # m
    mfp = mfp_standard*(101e3/P)*(T/293.15)*((1+110/293.15)/(1+110/T)) # m
    
    # Gas viscosity at room temperature and pressure with correction for pressure 
    # and temperature variations. B&W (3-10)
    eta_standard = 1.708e-5 # kg/(m*s)
    eta = eta_standard*((T/273.15)**1.5)*((393.396)/(T+120.246)) # kg/(m*s)
    
    # Gas density at room temperature and pressure with correction for pressure
    # and temperature variations
    rho = 1.293*(273.15/T)*(P/101.3e3) # kg/m^3
    
    # calculate Knudsen number. B&W (3-7)
    Kn = 2*mfp/Dp
    
    # Cunningham slip correction factor by Allen and Raabe (1985). B&W (3-8)
    Cc = 1+Kn*(1.142+0.558*np.exp(-0.999/Kn))
    
    # Reynolds number. B&W (6-3)
    Re = rho*V*D_tube/eta
    
    # Diffusion coefficient for particles in gas. B&W (3-13)
    Dc = k*T*Cc/(3*np.pi*eta*Dp)
    
    # Schmidt number. B&W (3-17)
    Sc = eta/(rho*Dc);
    
    # Convenient parameter when calculating the Sherwood number. B&W (6-44)
    xi = np.pi*Dc*L/Q
    
    # The flow is laminar if Re < ~2000
    if Re < 2000:
        # Sherwood number for laminar flow, Holman (1972). B&W (6-43)
        Sh = 3.66+0.2672/(xi+0.10079*xi**(1/3))
    # If Re > ~4000 the flow is turbulent. In the transition region it is difficult 
    # to determine, so here we assume the flow is turbulent if it is not laminar
    else:
        # Sherwood number for turbulent flow, Friedlander (1977). B&W (6-45)
        Sh = 0.0118*Re**(7/8)*Sc**(1/3)
    
    # Transport efficiency with diffusive particle loss. B&W (6-42)
    eff = np.exp(-Sh*xi)
    
    # Apply the diffusion loss correction
    Data_return = Data_in.copy()
    Data_return[:,2:] = Data_return[:,2:]/eff
    Data_return[:,1] = Data_return[:,2:].sum(axis=1)
    
    return Data_return, eff

###############################################################################
###############################################################################
###############################################################################
def File_list(path, search = 0,max_subfolder=0,nested_list=0):
    """
    Function to generate a list of all files in a folder. If the search 
    parameter is set to a string, only files containing the string keyword are 
    in the returned list.

    Parameters
    ----------
    path : str
        The path to the desired folder location.
    search : str, optional
        A string keyword so that only files with the keyword in their name will
        be included in the returned list. The default is 0.
    max_subfolder : int
        Specifies the depth/number of subfolders to traverse. If set to e.g. 2, 
        the function will include subfolders of subfolders compared to the 
        original directory path. The default is 0.
    nested_list : Boolean
        If set to true, the function will return a list of lists, where each
        sublist contains the files of a given subfolder. This can be relevant
        if files from each subfolder should not be mixed, allowing the user
        to loop through each list of lists. If false, the function returns a
        list of filenames without any division between them. The defauls is 0.        

    Returns
    -------
    files : list
        List of all the files in the specified folder, which also contain the
        specified keyword if defined.

    """
    # Make a list to hold the file names
    files = []
    
    # Walk through the directories of the initial path
    root_depth = path.rstrip(os.sep).count(os.sep)
    
    for root, _, filenames in os.walk(path):
        current_depth = root.count(os.sep)
        if current_depth-root_depth > max_subfolder:
            continue
        else:
            # If specified to use nested lists, join all files from subfolders within a list of lists
            if nested_list:
                files.append([os.path.join(root,file) for file in filenames])
            # Otherwise, run through all file names seperately
            else:
                for f in filenames:
                    files.append(os.path.join(root, f))
                    
    # Check if a specific keyword is needed within the file names
    if search:
        final_list = []
        # Run through all filenames (or list of filenames if nested lists are used)
        for item in files:
            if isinstance(item, list):  # Check if the item is a sublist
                final_list.append([file for file in item if search in file]) # remove items without the keyword
            else:
                if search in item: # check if the keyword (search) is within the filename (item)
                    final_list.append(item)
        final_list = [sublist for sublist in final_list if sublist] # remove empty lists
    else:
        final_list = files
    return final_list

###############################################################################
###############################################################################
###############################################################################
def Fit_lognormal(bin_pop,bin_mid,mu=[150],sigma=[2], factor=[1000],log_scaling=True,sort="Diameter", binding=[], tolerance=10.0):
    """
    A function to fit one or multiple peaks following lognormal distribution,
    with the option for tethering values to set values. 
    
    An example would be an OPS dataset, with a pronounced shoulder from a mode 
    below its diameter range. A guess for mu1 could then be 100nm,
    which can be bound by providing the "binding" list of [True]
    
    Parameters
    ----------
    bin_pop : numpy.array
        An array of the particle size number distribution used as the y values
        of the lognormal fit. If composing a combination of instruments must
        be inserted into one array. Additionally the data should be
        !NORMALIZED!
        for the resulting factors to be compareble to the total concentration
        reported in the load function. 
    bin_mids : numpy.array
        An array of size bin midpoints used as the x values of the lognormal fit.
        If composing a combination of instruments must be inserted into one array.
    mu : list of floats, optinonal
        If specified, acts as the initial guess of the particle modes, meaning
        the size where the particle size distribution peaks.
        The default is 150, but more modes can be added to the list.
    sigma : list of floats, optional
        Initial guess for the geometric standard deviation factor. A good guess
        is the size at peak height divided by the size at 2/3 peak height in
        the decending direction. E.g. the PSD peaks at 200 nm and is at 2/3 
        height at 140 nm, so the sigma_guess parameter should be 200/140 = 1.4.
        The default is 2, but more modes can be added to the list. 
    factor : list of floats, optional
        Initial guess for the parameter used to scale the lognormal distribution.
        Getting a good estimate can be difficult, but a guess in the same order
        of magnitude as the peak height, is a good start. 
        The default is 1000, but more modes can be added to the list.
    log_scaling: boolean, optional
        Value to designate whether the fit should be done against log10 data, 
        or the regular values. Using true values run the risk of larger modes
        dominating the fit, potentially lossing structure for low populated 
        modes. Default is True.
    sort: str, optional
        Value to designate whether the reported modes should be structured from
        smallest to largest diameter or from most to least populated mode,
        with the designations "Diameter" or "Number" respectively. Default is 
        "Diameter" 
    binding: bool list, optional
        A boolean list for each parameter whether they must be bound or not.
        If True the tolerance limit is put  on the bound value(s).
        The list has the following association:
        (mu1, sigma1, factor1, mu2, sigma2, factor2....factorN)
        The list only needs to be filled up, to the last True value.
        Default is 0 with no bound values.        
    tolerance: float, optional
        Percentage value around which the bound values can be fitted
        
    Returns
    -------
    popt : list
        A list containing the sorted fitted parameters (mu1, sigma1, factor1, mu2...)
        either sorted by mode diameter or from most to least populated mode
    perr : list
        Error estimates for the fitted parameters in the same order as the fit.
    """
    
    # Specify x and y data to fit
    xdata = np.array(bin_mid,dtype='float64')
    ydata = np.array(bin_pop,dtype='float64')
    
    if len(xdata)!=len(ydata):
        return print("Discrepency between number of bins and data")
    
    # Removes values of 0 or below from fitting
    mask=ydata>0

    xdata=xdata[mask]
    ydata=ydata[mask]
    if len(ydata)==0:
        return print("Empty dataset")
    
    mu=mu.copy()
    sigma=sigma.copy()
    factor=factor.copy()
    
    peak_number=len(mu)
    
    if peak_number*3>=len(ydata):
        return print("Peak number will lead to overfitting")
    
    if peak_number!=len(sigma):
        return print("Missing input for initial guess")

    # Gather all the initial guesses for parameters to fit in a list
    init_guess=[]
    for i in range(0,peak_number):
        init_guess.append(mu[i])
        init_guess.append(sigma[i])
        init_guess.append(factor[i])
    
    # Generate bounds to reduce the risk of producing impossible or irrelevant modes
    low_bounds=[0.1*min(xdata),1.15,0]*peak_number
    up_bounds=[10*max(xdata),5,max(max(ydata),max(factor))*2.5]*peak_number

    if tolerance>0:
      tolerance=tolerance/100
        
    # binding=number_to_bool_list(binding,peak_number*3)
    for i in range(0,len(binding)):
        if binding[i]==True:
            low_bounds[i]=init_guess[i]*(1-tolerance)
            up_bounds[i]=init_guess[i]*(1+tolerance)
            
    # Do the fit scaling by log 10 to give comparable fitting weight across different population sizes
    if log_scaling==True:
        popt, pcov   = curve_fit(Lognormal, xdata ,np.log10(ydata),p0 = init_guess,bounds=(low_bounds,up_bounds))
    elif log_scaling==False:    # Fit according to the true values to best fit the main mode
        popt, pcov   = curve_fit(Normal, xdata ,ydata,p0 = init_guess,bounds=(low_bounds,up_bounds))
    else:
        return print("log_scaling not set")
    
    # Get error estimates of the fits
    perr = np.sqrt(np.diag(pcov))
    
    #The next line of code sorts the data according to the desired focus; mode or population
    if len(popt)>3:
        # Step 1: Restructure both lists into triplets
        mode=[popt[i:i+3] for i in range(0,len(popt),3)]
        mode_perr=[perr[i:i+3] for i in range(0,len(perr),3)]
        
        # Step 2: Zip the corresponding elements from both lists
        zipped = list(zip(mode, mode_perr))
        
        # Step 3: Sort based on the 'a' values from the first element of the triplet in 'data'
        # Uses the sort definition to determine the order of returned modes
        if sort=="Diameter":
           sorted_zipped = sorted(zipped, key=lambda x: x[0][0])
        elif sort=="Number":
           sorted_zipped = sorted(zipped, key=lambda x: x[0][2], reverse=True)
        # Step 4: Unzip and flatten the sorted results back into separate lists
        sorted_mode, sorted_mode_perr = zip(*sorted_zipped)
        Popt = [item for sublist in sorted_mode for item in sublist]
        Perr = [item for sublist in sorted_mode_perr for item in sublist]
    else:
        Popt=popt
        Perr=perr
    # Returns the sorted fitted modes and their uncertainty.
    return Popt, Perr

###############################################################################
def Fit_lognormal_multivariate(bin_pop,bin_edges,mu=[(1500,1000)],sigma=[(1.5,1.7)],
                               factor=[100], rho=[0.25],log_fit=False, 
                               binding=[], tolerance=10.0):
    """
    A function to fit one or multiple peaks following double lognormal distribution,
    with the option for tethering values to set values. 
    
    An example would be a correlated APS dataset, where there is data for both 
    optical and aerodynamically ascribed diameter.
    This function will with a pronounced shoulder from a mode 
    below its diameter range. A guess for mu1 could then be 100nm,
    which can be bound by providing the "binding" list of [True]
    
    Parameters
    ----------
    bin_pop : numpy.array
        An array of the particle size number distribution used as the z values
        of the lognormal fit. If composing a combination of instruments must
        be inserted into one array. Additionally the data must not be normalized,
        as the multivariate requires double normalization, which is done in this
        function.
    bin_mids : numpy.array
        An array of size bin midpoints used as the x values of the lognormal fit.
        If composing a combination of instruments must be inserted into one array.
    mu : list of tuples of floats, optinonal
        If specified, acts as the initial guess of the particle modes, meaning
        the size where the particle size distribution peaks.
        The default is 1500 along the first axis and 1000 along the second,
        but more modes can be added to the list.
    sigma : list of tuples of floats, optional
        Initial guess for the geometric standard deviation factor. A good guess
        is the size at peak height divided by the size at 2/3 peak height in
        the decending direction. E.g. the PSD peaks at 200 nm and is at 2/3 
        height at 140 nm, so the sigma_guess parameter should be 200/140 = 1.4.
        The default is 2, but more modes can be added to the list. 
    factor : list of floats, optional
        Initial guess for the parameter used to scale the lognormal distribution.
        Getting a good estimate can be difficult, but a guess in the same order
        of magnitude as the peak height, is a good start. 
        The default is 1000, but more modes can be added to the list.
    rho : list of floats, optional
        
    log_scaling: boolean, optional
        Value to designate whether the fit should be done against log10 data, 
        or the regular values. Using true values run the risk of larger modes
        dominating the fit, potentially lossing structure for low populated 
        modes. Default is True.
    sort: str, optional
        Value to designate whether the reported modes should be structured from
        smallest to largest diameter or from most to least populated mode,
        with the designations "Diameter" or "Number" respectively. Default is 
        "Diameter" 
    binding: bool list, optional
        A boolean list for each parameter whether they must be bound or not.
        If True the tolerance limit is put  on the bound value(s).
        The list has the following association:
        (mu1, sigma1, factor1, mu2, sigma2, factor2....factorN)
        The list only needs to be filled up, to the last True value.
        Default is 0 with no bound values.        
    tolerance: float, optional
        Percentage value around which the bound values can be fitted
        
    Returns
    -------
    popt : list
        A list containing the sorted fitted parameters (mu1, sigma1, factor1, mu2...)
        either sorted by mode diameter or from most to least populated mode
    perr : list
        Error estimates for the fitted parameters in the same order as the fit.
    """
    
    xdata = np.array(bin_edges[0],dtype='float64')
    ydata = np.array(bin_edges[1],dtype='float64')
    zdata = np.array(bin_pop,dtype='float64')
    
    #Double normalization of the z-data
    for X in range(0,len(xdata)-1):
        for Y in range(0,len(ydata)-1):
            dlogDpx=np.log10(xdata[X+1]/xdata[X])
            dlogDpy=np.log10(ydata[Y+1]/ydata[Y])
            zdata[Y,X]=zdata[Y,X]/(dlogDpx*dlogDpy)
            
    # Specify x, y and z data to fit
    xdata = np.round((xdata[1:]+xdata[:-1])/2,1)
    ydata = np.round((ydata[1:]+ydata[:-1])/2,1)
    # zdata = np.array(bin_pop,dtype='float64')
    
    if (len(ydata),len(xdata))!=zdata.shape:
        return print("Discrepency between number of bins and data")
      
    ### test grid
    XX, YY = np.meshgrid(  xdata, ydata )

    #
    peak_number = len(mu)
    
    # if peak_number*3>=len(ydata):
    #     return print("Peak number will lead to overfitting")
    
    if peak_number!=len(sigma):
        return print("Missing input for initial guess")
    
    # Gather all the initial guesses for parameters to fit in a list
    # parameters=[mu_x1,mu_y1,sigma_x1,sigma_y1,factor1,rho1,mu_x2,mu_y2, sigma_x2...]
    init_guess = []
    for i in range(0,peak_number):
        init_guess.append(mu[i][0])
        init_guess.append(mu[i][1])
        init_guess.append(sigma[i][0])
        init_guess.append(sigma[i][1])
        init_guess.append(factor[i])
        init_guess.append(rho[i])
        
    # Generate bounds to reduce the risk of producing impossible or irrelevant modes
    low_bounds = [0.25*min(xdata),0.25*min(ydata),1.05,1.05,np.nanmin(zdata),0]*peak_number
    up_bounds  = [1.5*max(xdata),1.5*max(ydata),5, 5,  max(np.nansum(zdata), max(factor))*2.5,1]*peak_number
    
    # if tolerance>0:
    #   tolerance=tolerance/100
        
    # # binding=number_to_bool_list(binding,peak_number*3)
    # for i in range(0,len(binding)):
    #     if binding[i]==True:
    #         low_bounds[i]=init_guess[i]*(1-tolerance)
    #         up_bounds[i]=init_guess[i]*(1+tolerance)
    
    ### residual finction, i.e. the important stuff
    if log_fit==True:
        
        def res( params, X, Y, Z ):
            Z[Z==0]=0.000001
            th = np.log10(Normal_multivariate( X, Y, *params ))
            diff = np.log10(Z) - th
            diff[Z==0.000001]=0
            return np.concatenate( diff )
    else:
        def res( params, X, Y, Z ):
            th = Normal_multivariate( X, Y, *params )
            diff = Z - th
            diff[Z==0]=0
            return np.concatenate( diff )

    sol = least_squares( res, x0=init_guess, args=( XX, YY, zdata ),bounds=(low_bounds,up_bounds))
    
    # Returns the sorted fitted modes and their uncertainty.
    return sol #Popt, Perr
###############################################################################
def Normal_multivariate(X, Y, *params):    
    """
    The mathmatical expression of a lognormal distribution along two variables.
    The function can be used to genereate a theoretical double lognormal
    distribution and is also used by the Fit_lognormal function.
    It assumes a minimum of 1 peak, but can fit additional peaks if given prompt.

    Parameters
    ----------
    bin_mid : numpy.array
        An array of size bin midpoints used as the x values of the lognormal fit.
    *params: list
        params is a list contaning the triplet of information making out a:
        peak center: mu, peak spread: sigma, and population: factor
        The list should be structured:
            parameters=[mu_x1,mu_y1,sigma_x1,sigma_y1,factor1,rho1,mu_x2,mu_y2, sigma_x2...]
            
    mu : float
        The mode of the lognormal distribution.
    sigma : float
        The geometric standard devaition factor of the lognormal distribution.
    factor : float
        A scaling parameter needed to control and fit the particle numbers.
    rho : float
        Correlation factor between the two variables. Goes from 0 correpsonding
        to zero correlation to 1 corresponding to perfect correlation.
    Returns
    -------
    lognormal_function : numpy.array
        Returns an array of the same size as bin_mid populated by the sum of
        the desired peaks at diameter size in bin_mid.

    """
    
    x = np.log10(X)
    y = np.log10(Y)
    x_mu    = np.log10(np.array(params[0::6]))
    y_mu    = np.log10(np.array(params[1::6]))
    sigma_x = np.log10(np.array(params[2::6]))
    sigma_y = np.log10(np.array(params[3::6]))
    factor  = np.array(params[4::6])
    rho     = np.array(params[5::6])
    
    population = 0
    for i in range(0,len(x_mu)):
        #print(f"Iteration {i}: x_mu[i]: {x_mu[i]}, sigma_x[i]: {sigma_x[i]},y_mu[i]: {y_mu[i]},")
        #print(f" sigma_y[i]: {sigma_y[i]} factor[i]: {factor[i]}, rho[i]: {rho[i]}")
        x_diff = (x-x_mu[i])/sigma_x[i]
        y_diff = (y-y_mu[i])/sigma_y[i]
        Multiplier = factor[i]/(2*np.pi*sigma_x[i]*sigma_y[i]*np.sqrt(1-rho[i]**2))
        
        population += Multiplier*np.exp(-1/(2*(1-rho[i]**2))*
                    (x_diff**2+y_diff**2-2*rho[i]*x_diff*y_diff))
    
    return population
###############################################################################
###############################################################################
def ICRP_fraction(bin_mids, data_in=0, respvol=25, exposure_time=0):
    """
    Function to calculate the lung deposited fraction for an array of bin sizes
    based on the ICRP model.
    
    If no data is provided and only the bins are specified the function will
    return a vector, corresponding to the deposition fraction at each given
    size in the bin_mids parameter.
    
    If particle concentration data is specified, the function calculates the 
    lung deposited particle number, mass, or surface area in the specified 
    airway region, based on the average ICRP lung deposition model, the 
    measured particle distributions, the specified exposure time, and estimated 
    respirable volume. The unit of the input data determines the output, so if 
    mass data is used, the unit of the data should be ug/m3, in which case the 
    function will return ug particle deposited throughout the exposure. If 
    surface area data is used, the unit should be um2/m3 to get LDSA values of 
    um2 deposited throughout the exposure time. 

    Parameters
    ----------
    bin_mids : numpy.array
        Array of midpoints for all particle size bins. Sizes should be in nm.
    data_in : numpy.array, optional
        Particle data as returned by the load function, which is used to calculate
        the deposited particle fraction in the specified airway region, during
        the specified exposure time and with the specified respiration level. 
        The default is 0.
    respvol : float
        Respirable volume rate in L/min. A setting of 25 L/min corresponds to
        light exercise, while a setting of 9 L/min correspond to a person at 
        rest. Heavy exercise is 50 L/min. Default is 25 L/min.
    exposure_time : int, optional
        Time of exposure in seconds. If no value is specified, the deposited 
        fraction is calculated using the length of the dataset, meaning exposure 
        for the entire duration of the measurements. The default is 0.

    Returns
    -------
    ICRP : dict
        If no particle data is provided the function returns a dictionary,
        with vectors containing the deposition fraction for each airway
        region for each particle size in the bin_mids input parameter. 
        
        If particle data is also provided, the returned dictionary will also
        contain exposure data, meaning the number, mass, or surface area 
        (depending on the unit of the input particle data) of the
        deposited particles in each airway region at the given exposure time
        and respiratory rate.
        
        The keys for accessing the the different results:
            HA: Head airway deposition fraction
            TB: Tracheo bronchial deposition fraction
            AL: Alveolar deposition fraction
            TT: Total depostion fraction
            HA Deposition: Deposited particle number, mass or surface area in 
                           the head airways at the given exposure time and 
                           respiratory rate.
            TB Deposition: Deposited particle number, mass or surface area in 
                           the tracheo bronchial region at the given exposure 
                           time and respiratory rate.
            AL Deposition: Deposited particle number, mass or surface area in 
                           the Alveolar at the given exposure time and 
                           respiratory rate.
            TT Deposition: Total deposited particle number, mass or surface 
                           area in the airways at the given exposure time and 
                           respiratory rate.
    """
    
    #Dep_HA,Dep_TB,Dep_AL,Dep_TT=UL.Lung_Dep_ICRP(diff_cor_peak[meth][mate][rep][inst],data['bin_mids'][inst],14,15)
    #print("HA ", meth, mate, ": ",Dep_HA)
    ICRP = {}
    mids = bin_mids.copy()
    # Convert particle sizes from nm to um
    Dp = mids*1e-3 #um
    # Inhalable deposition fraction 
    IF = 1 - 0.5*(1 - 1 / (1 + 0.00076 * Dp**2.8 ))
    # Head airway deposition fraction
    DF_HA = IF * (1 / (1 + np.exp(6.84 + 1.183 * np.log(Dp))) + 1 / (1 + np.exp(0.924 - 1.885 * np.log(Dp))))
    # Tracheo bronchial deposition fraction
    DF_TB = (0.00352 / Dp) * (np.exp(-0.234 * (np.log(Dp) + 3.40)**2) + 63.9 * np.exp(-0.819 * (np.log(Dp) - 1.61)**2))
    # Alveolar deposition fraction
    DF_AL = (0.0155 / Dp) * (np.exp(-0.416 * (np.log(Dp) + 2.84)**2) + 19.11 * np.exp(-0.482 * (np.log(Dp) - 1.362)**2)) 
    # Total depostion fraction
    DF_TT = DF_HA + DF_TB + DF_AL
    # Fill out the ICRP dictionary of values
    ICRP['IF'] = IF
    ICRP['HA'] = DF_HA
    ICRP['TB'] = DF_TB
    ICRP['AL'] = DF_AL
    ICRP['TT'] = DF_TT
    
    # If particle data is given, calculate the deposited particle fraction in
    # the specified airway region, during the exposure time at the speicified
    # respiration rate.
    if data_in!=0:
        # Calculate the volume of inhaled air for the specified time
        respvol = respvol * 1e-3 / 60 # [L/min] -> [m3/s]
        if exposure_time == 0:
            duration = data_in[-1,0] - data_in[0,0]
            exposure_time = duration.seconds
        for i in ICRP.keys():
            # Calculate the deposited particles in eahc airway region
            Deposited_conc = data_in[:,2:] * ICRP[i]
            
            # Sum depostion of all sizebins and take the average to get the average
            # deposition during the measurement or the specified exposure time.
            Avg_Deposition = Deposited_conc.sum(axis=1).mean() 
            
            # Calculate the deposited number, mass or surface area, based on 
            # the volume of inhaled air
            ICRP[i+" Deposition"] = Avg_Deposition * respvol*exposure_time
    
    return ICRP

###############################################################################
###############################################################################
###############################################################################
def fit_linear(X, Y, intercept=True):
    """
    Function to detmine the correlation between two sets of values, which have been
    aligned so as to have sensible comparison points. 
    X and Y must have the same length. This can be accomplished by using the
    averaging function to generate time associated data of same dimensions. 
       
    Parameters
    ----------
    X: Numpy.array
        First set of values. 
    Y: Numpy.array
        Second set of values.
    intercept: boolean 
        Determines whether the fit should be forced through 0.
        
    Returns
    -------
    Fit: dictionary
        {A: Fitted slope
         B: Fitted intercept
         SE_A: Uncertainty of slope
         SE_B: Uncertainty of intercept}
    R2: float,
        Quality of fit
        """
    #Defining relevant sub-functions for the function to work
    def linear_func(x,A,B=0):
        #Calculates a first order equation.
        return A*x + B
    
    def R2(data,fit):
        #data: 
        # residual sum of squares
        ss_res = np.sum((data - fit) ** 2)
        # total sum of squares
        ss_tot = np.sum((data - np.mean(data)) ** 2)
        # r-squared
        return round((1 - (ss_res / ss_tot)),3)
    
    #Cleaning up the data and removing rows where either value is nan
    x = X.copy()
    y = Y.copy()
    
    mask=(x!=0)&(y!=0)
    z = np.column_stack((x[mask],y[mask])).astype('float64')
    z = z[~np.isnan(z).any(axis=1)]

    if type(x[0])==datetime.datetime:
        x = np.array(date2num(x[:]))

    #Apply the fit using curve_fit for a function with or without an intercept.
    if intercept==True:
        parameters, covariance = curve_fit(linear_func,z[:,0],z[:,1],p0=[1, 1])
        A, B = parameters
    
        SE = np.sqrt(np.diag(covariance))
        SE_A , SE_B = SE
    
        fit = linear_func(x,A,B)
        r2 = R2(y,fit)

    else:
        parameters, covarience = curve_fit(linear_func,z[:,0],z[:,1],p0=[1])
        A = parameters[0]
        SE_A = covarience[0][0]
        B = 0
        SE_B = 0
        
        fit = linear_func(x,A)
        r2 = R2(y,fit)
        
    Fit = {'A':A,'B':B,'SE_A':SE_A,'SE_B':SE_B}
    return Fit,r2

###############################################################################
###############################################################################
###############################################################################
def Lognormal(bin_mid, *params):
    """
    The mathmatical expression of a log10 (lognormal distribution). The function can be
    used to genereate a theoretical lognormal distribution and is also used by
    the Fit_lognormal function. 
    
    Parameters
    ----------
    bin_mid : numpy.array
        An array of size bin midpoints used as the x values of the lognormal fit.
    *params: list
        params is a list contaning the triplet of information making out a:
        peak center: mu, peak spread: sigma, and population: factor
        The list should be structured up to N peaks:
            paratmers=[mu1,sigma1,factor1,mu2,sigma2,factor2...factor N]
            
    mu : float
        The mode of the lognormal distribution, the center of peak.
    sigma : float
        The geometric standard devaition factor of the lognormal distribution.
    factor : float
        A scaling parameter needed to control and fit the particle numbers.

    Returns
    -------
    lognormal_function : numpy.array
        Returns an array of the same size as bin_mid populated by the sum of
        the desired peaks at diameter size in bin_mid.

    """
    bin_mid = np.log10(bin_mid)
    mu      = np.log10(np.array(params[0::3]))
    sigma   = np.log10(np.array(params[1::3]))
    factor  = np.array(params[2::3])
    
    population = 0
    for i in range(0,len(mu)):
            
        population+=((1/(np.sqrt(2*np.pi) * sigma[i])) *
                np.exp(-((bin_mid - mu[i])**2) /
                (2*sigma[i]**2)))*factor[i] 
   
    return np.log10(population)

###############################################################################
###############################################################################
###############################################################################

def MPS_Ceff(Pdiam,flowrate=0.6):
    """
    Function to estimate the collection efficiency of the mini particle sampler (MPS)
    as a function of particle size and flowrate. It should be noted, that the 
    expression is only valid up untill a particle diamter of 1 um, from where 
    a collection efficiency of 1 can be used. 
    
    The function assumes that 1.2/1.3 Quantfoil grids were used when sampling.
    
    Parameters
    ----------
    Pdiam : np.array()
        Array of particle diameters in um for which to determine the collection
        efficiency.
    flowrate : float, optional
        Flowrate used during sampling in lpm. The default is 0.6 lpm.

    Returns
    -------
    E_total : np.array
        Overall collection efficiency of the MPS at the given size and flow.
    E_impaction : np.array
        Collection efficiency from impaction alone at the given size and flow.
    E_diffusion : np.array
        Collection efficiency from diffusion to a flat surface alone at the given size and flow.
    E_interception : np.array
        Collection efficiency from interception alone at the given size and flow.
    E_edge : np.array
        Collection efficiency from diffusion to pore edges alone at the given size and flow.

    """
    ###########################################################################
    "Constants and conversions , units        , Descriptions" 
    rP = (Pdiam/2) * 1e-6      # m            ; Particle radius
    rhop = 1000                # kg/m3        ; Particle density
    lambda1 = 0.069 * 1e-6     # m            ; Molecule mean free path
    Lf = 0.02 * 1e-6           # m            ; Filter thickness
    N0 = 16003913722018.367    # #            ; Number of pores per surface unit of the carbon film
    r0 = 0.6 * 1e-6            # m            ; Pore radius
    A0 = np.pi*(r0**2)         # m2           ; Area of a single pore
    P  = A0*N0/100             # unitless     ; Porosity
    rc = r0/np.sqrt(P)         # m            ; Cylindrical aerosol stream passed through a unitary pore
    df = 2 * 1e-3              # m            ; TEM grid diameter
    Q  = flowrate/60 * 1e-3    # m3/s         ; Flowrate through the MPS
    U0 = Q / ((np.pi/4)*df**2) # m/s          ; Flow velocity
    nu = 1.85e-5               # kg/(m*s)     ; Fluid dynamic viscosity
    Kb = 1.381e-23             # kg*m2/(s2*K) ; Boltzmann constant
    T  = 300                   # K            ; Temperature
    Kn = lambda1 / rP          # unitless     ; Knudsen number
    Cc = 1 + Kn *(1.165 + 0.483*
         np.exp(-0.997/Kn))    # unitless     ; Cunningham slipfactor correction
    D = (Kb*T*Cc)/(6*np.pi*
                   nu*rP)      # m2/s         ; Diffusion coefficient    
    
    ############################## Interception ###############################
    # Calculate collection efficiency resulting from interception of particles
    lg = 1.126 * lambda1       # m            ; slip length
    Ng = lg/r0                 # unitless     ; Slip parameter
    NG = Ng*(1+Ng/2)           # unitless     ; Parameter
    Nr = rP/r0                 # unitless     ; Particle radius normalized for pore radius
    NR = Nr * (1-Nr/2)         # unitless     ; Parameter
    
    # Collection efficiency via interception
    E_interception = ((4*NR**2)/(1+4*NG))*(1+2*(NG/NR)) 
    
    ####################### Diffusion to pore edge ############################
    # Calculate collection efficiency resulting from particles diffusing to the pore edges
    Pe = U0 / (np.pi*Lf * D * 
               N0)             # unitless     ; Peclets number
    
    # Collection efficiency from pore edges calculated for different flow regimes
    E_edge = np.ones_like(Pe)
    E_edge[Pe > 25] = 2.56*Pe[Pe > 25]**(-2/3) -1.2*Pe[Pe > 25]**(-1)-0.177*Pe[Pe > 25]**(-4/3)
    E_edge[Pe <= 25] =  (1 - 0.81904*np.exp(-3.6568*Pe[Pe <= 25]**(-1)) - 0.09752*
                np.exp(-22.3045*Pe[Pe <= 25]**(-1))-0.03248*np.exp(-56.95*Pe[Pe <= 25]**(-1)) -
                0.0157*np.exp(-107.6*Pe[Pe <= 25]**(-1)))
    
    ######################### Diffusion to surface ############################
    # Calculate collection efficiency resulting from particles diffusing to flat filter surfaces
    alpha2 = 4.5                # unitless    ; Parameter
    Ds = D/ (rc*U0)             # unitless    ; Normalized diffusion coefficient
    alpha1 = (4.57-6.46*P
              +4.58*P**2)       # unitless    ; Parameter
    
    # Collection efficiency of filter via diffusion
    E_diffusion = 1-np.exp(-(alpha1*(Ds**(2/3)))/(1+(alpha1/alpha2)*Ds**(7/15))) 

    ############################## Impaction ##################################
    # Calculate collection efficiency resulting from impaction
    stk = (2*Cc*U0*rP**2*rhop
          )/(9*nu*r0)           # unitless     ; Stokes number
    eta = np.sqrt(P)/(1-
          np.sqrt(P))           # unitless     ; Parameter
    ei = (2*stk *np.sqrt(eta)+ 
          2 * (stk**2)*eta * 
          np.exp(-1/(stk*
          np.sqrt(eta)))-2*
          (stk**2)*eta)         # unitless     ; Parameter
    
    # Collection efficiency via impaction
    E_impaction = (2 *ei) / (1+eta) - (ei / (1 + eta))**2
    
    ############################## Overall Ceff ###############################
    # Calculate overall collection efficiency of the MPS
    E_total = 1 - (1-E_impaction)*(1-E_diffusion)*(1-E_interception)*(1-E_edge)

    E_total[Pdiam>1.] = 1
    
    return E_total, E_impaction, E_diffusion, E_interception, E_edge

###############################################################################
###############################################################################
###############################################################################
def Normal(bin_mid, *params):
    """
    The mathmatical expression of a lognormal distribution. The function can be
    used to genereate a theoretical lognormal distribution and is also used by
    the Fit_lognormal function. It assumes a minimum of 2 peaks, but can fit
    additional peaks if given prompt.

    Parameters
    ----------
    bin_mid : numpy.array
        An array of size bin midpoints used as the x values of the lognormal fit.
    *params: list
        params is a list contaning the triplet of information making out a:
        peak center: mu, peak spread: sigma, and population: factor
        The list should be structured:
            parameters=[mu1,sigma1,factor1,mu2,sigma2,factor2...]
            
    mu : float
        The mode of the lognormal distribution.
    sigma : float
        The geometric standard devaition factor of the lognormal distribution.
    factor : float
        A scaling parameter needed to control and fit the particle numbers.

    Returns
    -------
    lognormal_function : numpy.array
        Returns an array of the same size as bin_mid populated by the sum of
        the desired peaks at diameter size in bin_mid.

    """
    bin_mid = np.log10(bin_mid)
    mu      = np.log10(np.array(params[0::3]))
    sigma   = np.log10(np.array(params[1::3]))
    factor  = np.array(params[2::3])
    
    population = 0
    for i in range(0,len(mu)):
        # print(f"Iteration {i}: sigma[i] = {sigma[i]}, mu[i] = {mu[i]}, factor[i] = {factor[i]}")
        population+=((1/(np.sqrt(2*np.pi) * sigma[i])) * 
                     np.exp(-((bin_mid - mu[i])**2) /
                     (2*sigma[i]**2)))*factor[i] 
   
    return population

###############################################################################
###############################################################################
###############################################################################
def Normalize_dndlogdp(data_in,bin_edges):
    """
    Function to normalize an array of particle xxx concentrations (xxx can be
    mass, volume, surface area, and number), going from dxxx values to 
    dxxx/dlogDp.

    Parameters
    ----------
    data_in : numpy.array
        An array of data as returned by the Load_xxx functions with columns
        of datetime, total conc, and size bin data
    bin_edges : numpy.array
        Array containing the limits of all sizebins. The array should have one 
        more value than the length of the "data_in" parameter

    Returns
    -------
    Data_return : numpy.array
        An array of normalized data equivalent to the data_in shape with columns
        of datetime, total conc, and normalized size bin concentrations.

    """
    # Calculate the normalization vector
    dlogDp = np.log10(bin_edges[1:])-np.log10(bin_edges[:-1])
    
    # Copy the input data to not mess with the parent
    Data_return = data_in.copy()
    
    # Select relevant data
    data = Data_return[:,2:]
    
    # Apply normalization
    Normed = data/dlogDp
    # Normed_total = np.sum(Normed,axis=1)
    
    # Store the normalized data instead of the non-normalized and return it
    # Data_return[:,1] = Normed_total
    Data_return[:,1] = data_in[:,1]  # this is what we usually mean by Total
    Data_return[:,2:] = Normed
    
    return Data_return

###############################################################################
###############################################################################
###############################################################################
def num2mass(data_in, bin_mids, density=1.0, unit="mg", ICRP='none'):
    """
    Function to convert from number concentration to mass
    concentration, assuming spherical particles with a diameter equal to the
    size bin mid points, and a density as specified.

    Parameters
    ----------
    data_in : numpy.array
        An array of data as returned by the Load_xxx functions with columns
        of datetime, total conc, and size bin data. The unit should be #/cm3
    bin_mids : numpy.array
        Array containing the mid-points in nm of all sizebins. The array should have 
        the same number of bins as the "data_in" parameter.
    density : float/list, optinonal
        A density can be specified, either as a uniform single value, or as a list
        with length equal to the number of bins. The unit should be g/cm3.
        Default is a density of 1.0 g/cm3. 
    unit : str, optional
        Select the unit of the reported mass per m3: ng, mug, mg or g. Default is mg
    ICRP : str, optional
        Allows for adding ICRP lung deposition fraction:
        'none': Returns the total mass concentration (mg / m3), default
        'TT': Returns the total lung deposited mass concentration fraction (mg / m3)
        'HA': Returns the head airway deposited mass concentration fraction (mg / m3)
        'TB': Returns the tracheo bronchial deposited mass concentration fraction (mg / m3)
        'AL': Returns the alveolar deposted mass concentration fraction (mg / m3)
        
    Returns
    -------
    Data_return : numpy.array
        An array of mass concentration data equivalent to the data_in shape with 
        columns of datetime, total mass conc in gram, and size bin mass concentrations
        in ug/m3.

    """
    # Copy the original data
    Data_return = data_in.copy()
    bins = np.array(bin_mids, dtype="float64")
    
    # Determine conversion vector from number to volume, cm3
    
    #Calculates the volume vector for the bins in cm3
    if ICRP=='none': Num2Vol = (np.pi/6.)*(bins*1e-7)**3
    else: Num2Vol = (np.pi/6.)*(bins*1e-7)**3*ICRP_fraction(bins)[ICRP]
    
    #Determines the 
    if unit=='ng':
        mass_factor = 1e15
    elif unit=='mug':
        mass_factor = 1e12
    elif unit=='mg':
        mass_factor = 1e9
    elif unit=='g':
        mass_factor = 1e6
        
    # Convert from volume to mass via the specified density, g
    Num2mass = Num2Vol * density * mass_factor
    
    if len(bin_mids)==len(data_in[0,:])-2:
        # Apply the conversion vector to the particle number concentrations, g
        Data_return[:,2:] = Data_return[:,2:].astype("float64")* Num2mass 

        # Determine the total mass at each timestep, ug/m3
        Data_return[:,1] = Data_return[:,2:].sum(axis=1)
        
    elif len(bin_mids)==len(data_in[0,:]):
        
        # Apply the conversion vector to the particle number concentrations, g
        Data_return = Data_return.astype("float64")* Num2mass 
    else:
        return print('Error: Discrepency between number of bins and data')
       
    return Data_return

###############################################################################
###############################################################################
###############################################################################
def num2surface(data_in, bin_mids, unit="nm2", ICRP='none'):
    """
    Function to convert from number concentration to surface area
    concentration, assuming spherical particles with a diameter equal to the
    size bin mid points.

    Parameters
    ----------
    data_in : numpy.array
        An array of data as returned by the Load_xxx functions with columns
        of datetime, total conc, and size bin data
    bin_mids : numpy.array
        Array containing the mid-points of all sizebins. The array should have 
        the same number of bins as the "data_in" parameter
    unit : str, optional
        Select the unit of the reported surface area per cm3: nm2, mum2, mm2 or cm2.
        Default is nm2
    ICRP : str, optional
        Allows for choosing if the surface area should be augmented by the 
        ICRP fraction calculation:
        'none': Returns the total SA (nm2 / cm3)
        'TT': Returns the total lung deposited SA (nm2 / cm3)
        'HA': Returns the head airway deposited fraction (nm2 / cm3)
        'TB': Returns the tracheo bronchial deposited fraction (nm2 / cm3)
        'AL': Returns the alveolar deposted fraction (nm2 / cm3)
    Returns
    -------
    Data_return : numpy.array
        An array of surface area concentration data equivalent to the data_in 
        shape with columns of datetime, total surface area concentration in 
        nm2/cm3, and size bin surface area concentrations in nm2/cm3 of air.

    """
    Data_return = data_in.copy()
    bins = np.array(bin_mids, dtype="float64")
    
    # nm2 surface per particle in each size bin
    if ICRP=='none': Num2surface = 4*np.pi*(bins/2.)**2
    else: Num2surface = 4*np.pi*(bins/2.)**2*ICRP_fraction(bins)[ICRP]
    
    if unit=='nm2':
        unit_factor=1
    elif unit=='mum2':
        unit_factor=1e-6
    elif unit=='mm2':
        unit_factor=1e-12
    elif unit=='cm2':
        unit_factor=1e-14
    
    if len(bin_mids)==len(data_in[0,:])-2:
        
        # apply conversion (nm**2 / cm**3)
        Data_return[:,2:] = Data_return[:,2:].astype("float64")* Num2surface * unit_factor     
        
        # Calculate and store the total surface area
        Data_return[:,1] = Data_return[:,2:].sum(axis=1)
        
    elif len(bin_mids)==len(data_in[0,:]):
        data = Data_return.astype("float64")
        
        # apply conversion (nm**2 / cm**3)
        Data_return = data * Num2surface  * unit_factor
    else:
        return print('Error: Discrepency between number of bins and data')

    return Data_return

###############################################################################
###############################################################################
###############################################################################
def num2vol(data_in, bins_in, ICRP='none'):
    """
    Function to convert from number concentration to volume
    concentration, assuming spherical particles with a diameter equal to the
    size bin mid points.

    Parameters
    ----------
    data_in : numpy.array
        An array of data as returned by the Load_xxx functions with columns
        of datetime, total conc, and size bin data, or as size bin population only
    bins : numpy.array
        Array containing the mid-points of all sizebins. The array should have 
        the same number of bins as the "data_in" parameter
    ICRP: str
        Allows for adding ICRP lung deposition fraction:
        'none': Returns the total mass (nm3 / cm3)
        'TT': Returns the total lung deposited volume ratio (nm3 / cm3)
        'HA': Returns the head airway deposited volume ratio (nm3 / cm3)
        'TB': Returns the tracheo bronchial deposited volume ratio (nm3 / cm3)
        'AL': Returns the alveolar deposted volume ratio (nm3 / cm3)  
    Returns
    -------
    Data_return : numpy.array
        An array of volume concentration data equivalent to the data_in shape with 
        columns of datetime, total volume concentration in ul/m3, and size bin 
        volume concentrations in nm3/cm3.
    """
    Data_return = data_in.copy()
    bins = np.array(bins_in, dtype="float64")
    
    # Calculate volume per particle for each bin
    #Num2Vol = (4./3.)*np.pi*((bins*1e-7)/2.)**3
    if ICRP=='none': Num2Vol = (4./3.)*np.pi*((bins)/2.)**3
    else: Num2Vol = (4./3.)*np.pi*((bins)/2.)**3*ICRP_fraction(bins)[ICRP]
    
    if len(bins)==len(data_in[0,:])-2:
        # apply conversion to get nm3/cm3 
        Data_return[:,2:] = Data_return[:,2:].astype("float64")* Num2Vol
        
        # sum to get the total volume
        Data_return[:,1] = Data_return[:,2:].sum(axis=1)

    elif len(bins)==len(data_in[0,:]):
        
        # Select the relevant data and bins
        Data_return = Data_return.astype("float64")* Num2Vol 

    else:
        return print('Error: Discrepency between number of bins and data')
    return Data_return
    
###############################################################################
###############################################################################
###############################################################################
def Partector_Ceff(Psize):
    """
    Function to estimate the collection efficiency of the partectorTEM at the
    specified particle size in nm. The collection efficiency as a fraction is 
    returned and can be applied to the measured concentration to get a 
    corrected concentration.
    
    It should be noted, that the expression for the collection efficiency was fitted
    to data from experiments with NaCl and Ag particles in the size range from 
    3 to 320 nm, and may therefore not be accurate at um sizes! Especially, at
    sizes larger than 4-5 um, the estimate will fail, as impaction will start to
    play a role. There are currently no data on the matter, but theoretical 
    caculations suggest that D50 is a roughly 11 um, but an effect can be seen
    already at 4-5 um.
    
    Reference: Fierz, M., Kaegi, R., and Burtscher, H.;"Theoretical and 
    Experimental Evaluation of a Portable Electrostatic TEM Sampler", Aerosol
    Science and Technology, 41, issue 5, 2007.

    Parameters
    ----------
    Psize : float or np.array
        Either a single particle size given as a float, or an array of particle
        sizes to be used for calculating the collection efficiency. The sizes should
        be given in nm.

    Returns
    -------
    Collection_efficiency : float or np.array
        The calculated collection efficiency of the PartectorTEM at the specified
        particle size/sizes specified as a fraction (0-1).

    """
    Collection_efficiency = (0.43837287*Psize**(-0.48585362))
    
    return Collection_efficiency 
###############################################################################
def Partector_Ceff_upd(Psize):
    """
    Function to estimate the collection efficiency of the partectorTEM at the
    specified particle size in nm. The collection efficiency as a fraction is 
    returned and can be applied to the measured concentration to get a 
    corrected concentration.
    
    It should be noted, that the expression for the collection efficiency was fitted
    to data from experiments with NaCl and Ag particles in the size range from 
    3 to 320 nm, and may therefore not be accurate at um sizes! Especially, at
    sizes larger than 4-5 um, the estimate will fail, as impaction will start to
    play a role. There are currently no data on the matter, but theoretical 
    caculations suggest that D50 is a roughly 11 um, but an effect can be seen
    already at 4-5 um.
    
    Reference: Fierz, M., Kaegi, R., and Burtscher, H.;"Theoretical and 
    Experimental Evaluation of a Portable Electrostatic TEM Sampler", Aerosol
    Science and Technology, 41, issue 5, 2007.

    Parameters
    ----------
    Psize : float or np.array
        Either a single particle size given as a float, or an array of particle
        sizes to be used for calculating the collection efficiency. The sizes should
        be given in nm.

    Returns
    -------
    Collection_efficiency : float or np.array
        The calculated collection efficiency of the PartectorTEM at the specified
        particle size/sizes specified as a fraction (0-1).

    """
    Collection_efficiency = (0.43837287*Psize**(-0.48585362)+0.0010345*Psize**(0.48585362))
    
    return Collection_efficiency 

###############################################################################
###############################################################################
###############################################################################
def Partector_sample_vol(data):
    """
    Function for calculating the volume of air sampled through the TEM filter

    Parameters
    ----------
    data : np.array
        Partector data structured as [Datetime,LSDA,TEM,Flow]

    Returns
    -------
    Vol : float
        Volume of sampled air in cm3.

    """
    data=data.copy()
    
    #Make a check whether several filter samplings are within the time frame
    counter=0
    for i in range(1,len(data)):
        if data[i-1,2]==0 and data[i,2]==1:
            counter+=1
        if counter>1:
            return "Warning more than one sampling event detected"
    data=data[data[:,2]==1]
    start, end =data[0,0], data[-1,0]
    #cm3/min
    flow=np.nanmean(data[:,3])*1000
    #Multiply the average flow rate with the sampling time in min to get vol in cm3
    Vol = flow*(date2num(end)-date2num(start))*24*60
    
    return     Vol
        
###############################################################################
###############################################################################
###############################################################################
def Peak_finder(data,window: int=15,ratio: float=2.5,bin_size=0, method: str='median'):
    """
    Function for locating peaks based on diviance from standard deviation
    over a chosen period.

    Parameters
    ----------
    data : np.array
        Partector data structured as [Datetime,LSDA,TEM,Flow]
    window : int
        Width of the window around which the peak should be found, with each
        integer being a time step away from the investigated point.
    ratio : float
        Value over which the the difference from median should be larger than
        the standard deviation to be counted as a peak
    bin_size : int, str - default=0
        Designates what bin peak finding should be done for.
        Default is 0, which uses the total concentration column.
        Otherwise individual bins can be chosen, or 
        

    Returns
    -------
    mask : np.array
        array of boolean values indicating found peaks.

    """
    if method not in ["mean", "median", "sum", "min", "max"]:
        raise ValueError(
            "Invalid method. Choose from 'mean', 'median', 'sum', 'min', 'max'."
        )
    window          = int(window)
    Data_return     = pd.DataFrame(data[:,1:])
    # Med             = Data_return.copy()
    # Std             = Data_return.copy()
    
    Test             = np.array(getattr( Data_return.rolling(
        window=window, center=True, min_periods=2), method)())
    Std             = np.array(getattr( Data_return.rolling(
        window=window, center=True, min_periods=2), 'std')())
        
    # Med[:,1:]   = np.array(pd.DataFrame(Data_return[:,1:]).rolling(window=window,
    #                                         center=True, min_periods=1).median())
    #Change Med array from median value to the difference between median and actual value.
    # Med[:,1:]   = Data_return[:,1:]-Med[:,1:]
    # Std[:,1:]   = pd.DataFrame(Med[:,1:]).rolling(window=window, center=True, min_periods=1).std()
    #Med=Rolling_window(data,window_width=window)
        
    Test   = np.array(Data_return) - Test
    #Multiply the average flow rate with the sampling time in min to get vol in cm3
    if bin_size == 0:        
        mask = ((Test[:,0])**2)**0.5 > Std[:,0]*ratio
    elif bin_size == 'all':
        Test[:,0]=[0]*len(Test[:,0])
        for i in range (0,len(Test[0,1:])):
            Test[:,0] += ((Test[:,1+i])**2)**0.5 > Std[:,1+i]*ratio
        mask = (Test[:,0]/len(Test[0,1:]) > 1/4)
    else:
        mask = ((Test[:,bin_size+1])**2)**0.5 > Std[:,bin_size+1]*ratio
        
    return     mask

###############################################################################
    # def Peak_finder_old(data,window: int=15,ratio: float=2.5,bin_size=0):
    #     """
    #     Function for locating peaks based on diviance from standard deviation
    #     over a chosen period.

    #     Parameters
    #     ----------
    #     data : np.array
    #         Partector data structured as [Datetime,LSDA,TEM,Flow]
    #     window : int
    #         Width of the window around which the peak should be found, with each
    #         integer being a time step away from the investigated point.
    #     ratio : float
    #         Value over which the the difference from median should be larger than
    #         the standard deviation to be counted as a peak
    #     bin_size : int, str - default=0
    #         Designates what bin peak finding should be done for.
    #         Default is 0, which uses the total concentration column.
    #         Otherwise individual bins can be chosen, or 
            

    #     Returns
    #     -------
    #     mask : np.array
    #         array of boolean values indicating found peaks.

    #     """

    #     window          = int(window)
    #     Data_return     = data.copy()
    #     Med             = Data_return.copy()
    #     Std             = Data_return.copy()
        
    #     Med[:,1:]   = np.array(pd.DataFrame(Data_return[:,1:]).rolling(window=window,
    #                                             center=True, min_periods=1).median())
    #     #Change Med array from median value to the difference between median and actual value.
    #     Med[:,1:]   = Data_return[:,1:]-Med[:,1:]
    #     Std[:,1:]   = pd.DataFrame(Med[:,1:]).rolling(window=window, center=True, min_periods=1).std()
    #     #Med=Rolling_window(data,window_width=window)
        
    #     #Multiply the average flow rate with the sampling time in min to get vol in cm3
    #     if bin_size == 0:        
    #         mask = ((Med[:,1])**2)**0.5 > Std[:,1]*ratio
    #     elif bin_size == 'all':
    #         Med[:,0]=[0]*len(Med[:,0])
    #         for i in range (0,len(Med[0,2:])):
    #             Med[:,0] += ((Med[:,2+i])**2)**0.5 > Std[:,2+i]*ratio
    #         mask = (Med[:,0]/len(Med[0,2:]) > 1/4)
    #     else:
    #         mask = ((Med[:,bin_size+1])**2)**0.5 > Std[:,bin_size+1]*ratio
            
    #     return     mask
                
###############################################################################
###############################################################################
def PM_calc(data_in,bin_edges,*PM):
    """
    Function to calculate PM from size bins from an array similar to those
    returned from Load_xxx functions. 
    
    The calculated value is found according to 481 EN. 
    Note that the data should already be converted to mass e.g. ug/m3 as the
    output of this function will maintain the same units as the input.

    Parameters
    ----------
    data_in : numpy.array
        An array of data as returned by the Load_xxx functions with columns
        of datetime, total conc, and size bin data. The size bin data should
        be converted to mass based units e.g. ug/m3.
    bin_edges : numpy.array
        Array containing the limits of all sizebins. The array should have one
        additional value when compared to the sizebin data in the "data_in" 
        parameter
    *PM : integer or float
        PM limit or limits to use, given in um. Typical values are 0.1, 1, or
        10 to yield PM0.1, PM1, and PM10 respectively. Note that multiple PM
        limits can be specified at the same time e.g. 
        PM_calc(data,bins,0.1,1,10)
        In which case the returned data array will have a column for each PM mass
    Returns
    -------
    Data_return : numpy.array
        An array of mass concentration data. The first column is datetime. 
        The other column or columns are PM values corresponding to each time for
        the specified PM limit.

    """
    # Remove the datetime and total conc columns as these are not needed for PM
    # calculations
    data = data_in[:,2:]
    
    #calculate bin_mids
    bin_mids    =  np.round(np.array((bin_edges[:-1] + bin_edges[1:])/2,dtype=float),1)
    # Make a new array with the necessary shape
    Data_return = np.zeros((data_in.shape[0],len(PM)+1),dtype="object")
    
    # Set the first column of the new array equal to the datetime values of the input
    Data_return[:,0] = data_in[:,0]
    
    header = ["Datetime"]
    counter = 1
    delete_index = []
    
    # Run through the or all of the PM limits specified
    for i in PM:
        # convert the current PM limit frim um to nm

        PM_lim = i * 1000
        #EN 481 calculation
        Y=np.log(bin_mids/PM_lim)/(2**0.5*np.log(1.5))
    
        Frac=0.5*(1+np.vectorize(erf)(-Y))

            
        # Apply the fraction to the mass concentration of the bin. 
        mass_frac = np.nansum(data*Frac,axis=1)
        mass_frac[mass_frac==0] = np.nan#where(mass_frac==0,mass_frac,np.nan)
        # Add the two valuyes for a final mass concentration
        final_PM = mass_frac
        
        # Add a header
        header += ["PM{0}".format(i)]

        # Store all the PM values in a new array
        Data_return[:,counter] = final_PM
        counter += 1    
    
    # Delete columns if they are empty
    Data_return = np.delete(Data_return,delete_index,axis=1)
    
    return Data_return, header
###############################################################################
def PM_calc_old(data_in,bin_edges,*PM):
    """
    Function to calculate PM from size bins from an array similar to those
    returned from Load_xxx functions. 
    
    Note that the data should already be converted to mass e.g. ug/m3 as the
    output of this function will maintain the same units as the input.

    Parameters
    ----------
    data_in : numpy.array
        An array of data as returned by the Load_xxx functions with columns
        of datetime, total conc, and size bin data. The size bin data should
        be converted to mass based units e.g. ug/m3.
    bin_edges : numpy.array
        Array containing the limits of all sizebins. The array should have one
        additional value when compared to the sizebin data in the "data_in" 
        parameter
    *PM : integer or float
        PM limit or limits to use, given in um. Typical values are 0.1, 1, or
        10 to yield PM0.1, PM1, and PM10 respectively. Note that multiple PM
        limits can be specified at the same time e.g. 
        PM_calc(data,bins,0.1,1,10)
        In which case the returned data array will have a column for each PM mass
    Returns
    -------
    Data_return : numpy.array
        An array of mass concentration data. The first column is datetime. 
        The other column or columns are PM values corresponding to each time for
        the specified PM limit.

    """
    # Remove the datetime and total conc columns as these are not needed for PM
    # calculations
    data = data_in[:,2:]
    
    # Make a new array with the necessary shape
    Data_return = np.zeros((data_in.shape[0],len(PM)+1),dtype="object")
    
    # Set the first column of the new array equal to the datetime values of the input
    Data_return[:,0] = data_in[:,0]
    
    header = ["Datetime"]
    counter = 1
    delete_index = []
    
    # Run through the or all of the PM limits specified
    for i in PM:
        # convert the current PM limit frim um to nm
        PM_lim = i * 1000
        
        # Determine the number of bins smaller than the specified PM limit
        bin_within_range = np.array(bin_edges<PM_lim).sum()
        
        
        if bin_within_range == 0:
            # If none of the size bins are above the PM limit, continue to the
            # next specified PM limit 
            print("PM{0} is smaller than all size bins, so it will not be included in the output array".format(i))
            delete_index += [counter]
            counter += 1
            continue
        
        elif bin_within_range == 1:
            # If only the first bin is partially within the PM limit, calculate
            # the fraction within the limit and multiply with the mass concentration
            bin_frac = (PM_lim - bin_edges[0]) / (bin_edges[1]-bin_edges[0])
            final_PM = data[:,0]*bin_frac
            
            # Add a header
            header += ["PM{0}".format(i)]
            
        elif bin_within_range < bin_edges.shape[0]:
            # If several bins are within the PM limit, sum the relevant ones
            Initial_PM = data[:,:bin_within_range-1].sum(axis=1)
            
            # Determine the fraction of the highest relevant bin within the specified
            # PM limit
            bin_frac = (PM_lim - bin_edges[bin_within_range-1]) / (bin_edges[bin_within_range] - bin_edges[bin_within_range-1])
            
            # Apply the fraction to the mass concentration of the bin. Here it
            # is assumed that the mass concentration is evenly distributed within
            # the size bin
            mass_frac = data[:,bin_within_range-1]*bin_frac
    
            # Add the two valuyes for a final mass concentration
            final_PM = Initial_PM + mass_frac
            
            # Add a header
            header += ["PM{0}".format(i)]
            
        else:
            # If all bins are smaller than the specified PM limit, simply sum them
            final_PM = data.sum(axis=1)
            
            # Add a header
            header += ["PM{0}".format(i)]
       
        # Store all the PM values in a new array
        Data_return[:,counter] = final_PM
        counter += 1    
    
    # Delete columns if they are empty
    Data_return = np.delete(Data_return,delete_index,axis=1)
    
    return Data_return, header

###############################################################################
###############################################################################
###############################################################################
def Rolling_window(data,start=0,end=0,window_width=60,window_type="median",crop=0):
    """
    A function used to take the median or mean value within a rolling window, 
    in order to adjust the value of the central data point. A start and end 
    point of the dataseries can be set, and the width of the rolling window
    adjusted. Furthermore it can be specified if datapoints should be kept,
    maintaining the shape of the original dataset, or whether datapoints should
    be dropped, so the dataseries is rebinned in terms of time and reduced in
    length by a factor equal to the window size. The function is very good at 
    removing spikes from datasets e.g. from DiscMini or sensor datasets when 
    using a median window. 

    Parameters
    ----------
    data : numpy.array
        Data array similar to those returned from load functions. The rolling
        window will be applied to all columns except the first, which is expected
        to be datetime values.
    start : datetime, optional
        Starttime of the dataset. All datapoints prior to the starting point
        will be dropped. The default is 0, meaning that all datapoints are used.
    end : datetime, optional
        Endtime of the dataset. All datapoints after the endpoint will be 
        dropped. The default is 0, meaning that all datapoints are used.
    window_width : int, optional
        Width of the rolling window. The default is 60 to e.g. convert from
        seconds to minutes.
    window_type : str, optional
        Set which window type to use. Currently only "mean" and "median" rolling
        windows are available. The default is "median".
    crop : boolean, optional
        Whether to keep all datapoints or drop the datapoints, so to resize the
        dataset by a factor equal to the window size. The default is 0.

    Returns
    -------
    Data_return : numpy.array
        Data array wwhere datapoints have been adjusted by the rolling median 
        or mean operation. The data array will have the same shape as the input 
        data if crop = 0, or will have been rebinned if crop=1.

    """
    
    Data_return = data.copy()
    if window_type == "median":
        Data_return[:,1:] = pd.DataFrame(data[:,1:]).rolling(window=window_width, center=True, min_periods=1).median()
    elif window_type == "mean":
        Data_return[:,1:] = pd.DataFrame(data[:,1:]).rolling(window=window_width, center=True, min_periods=1).mean()
    else:
        print('No such method implemented, currently implemented window_types are: "median" and "mean"')
        return
        
    # Find the index of times fulfilling the start and/or end time conditions
    if (start != 0) or (end != 0):
        if (start != 0) & (end == 0):
            index = Data_return[:,0]>=start
        elif (start == 0) & (end != 0):
            index = Data_return[:,0]<=end
        else:
            index = (Data_return[:,0]>=start) & (Data_return[:,0]<=end)
        
        # Select the data rows fulfilling the time criterias
        Data_return = Data_return[index,:]
    if crop:
        Data_return = Data_return[::window_width,:]
    return Data_return

###############################################################################
###############################################################################
###############################################################################
def Save_2_Excel(data_in,header,filename,path):
    """
    Function to store data to an excel file.

    Parameters
    ----------
    data_in : numpy.array
        An array of data as returned by the Load_xxx functions with columns
        of datetime, total conc, and size bin data
    header : list
        A list of column names as returned by the Load_xxx functions
    filename : str
        Desired file name without the file extension. E.g. "ELPI_test" will be
        stored as "ELPI_test.xlsx" at the specified path
    path : str
        Path to the desired folder in which to store the file. E.g. 
        'L:\\PG-Nanoteknologi\\PERSONER\\xxx'

    Returns
    -------
    Nothing is returned by this function

    """
    # Combine the data with the header in a pandas.DataFrame for easy saving 
    DataFile = pd.DataFrame(data_in,columns=header)
    
    # Write the DataFrame to an excel file with the specified path and filename
    DataFile.to_excel("{0}\\{1}.xlsx".format(path,filename),index=False)
    
    # Print in the consol that the function finished
    print("File is now saved")

###############################################################################
###############################################################################
###############################################################################
def Save_plot(figure,filename,path,extension=".png"):
    """
    Function to save a figure at the specified path and with the given filename 
    and extension.

    Parameters
    ----------
    figure : matplotlib.figure.Figure
        Handle for the figure to save. Such handles are always returned by the
        plotting functions of the NFA Python Library.
    filename : str
        Desired file name without the file extension. E.g. "ELPI_test" will be
        stored as "ELPI_test.png" at the specified path
    path : str
        Path to the desired folder in which to store the plot. E.g. 
        'L:\\PG-Nanoteknologi\\PERSONER\\xxx'
    extension : str, optional
        Decired file extension e.g. ".tif" or ".bmp". The default is ".png".
    Returns
    -------
    Nothing is returned by this function

    """
    # Save the figure with a dot per inch of 500 to have a good resoulution.
    figure.savefig("{0}\\{1}{2}".format(path,filename,extension),dpi=500)
    print("Plot was saved")

###############################################################################
###############################################################################
###############################################################################
def segment_dataset(data_in,segments,labels=None):
    """
    Function to segment a dataset into different activities e.g. background and
    different workplace processes. Each activity can have multiple time 
    segments as long as a start time always has a corresponding end time. The
    times for each activity should be kept in serperate lists and joined in an
    overall list with all the activity start and end times.
    example:
        # The background has 2 segments from 12 to 14.30 and from 15.30 to 16.20
        background = [datetime.datetime(2018,11,28,12,0,0), datetime.datetime(2018,11,28,14,30,0),
                      datetime.datetime(2018,11,28,15,30,0), datetime.datetime(2018,11,28,16,20,0)
                      ]
        # pouring occured from 14.33 to 15.24
        pouring = [datetime.datetime(2018,11,28,14,33,0), datetime.datetime(2018,11,28,15,24,0)]
        
        # sanding occured from 16.23 to 17.00
        sanding = [datetime.datetime(2018,11,28,16,23,0), datetime.datetime(2018,11,28,17,0,0)]
        
        # Store all time lists in an overall list
        Activities = [background, pouring, sanding]
        
        # Feed to function for indexing
        segment_dataset(data_in,Activities)
        
    Parameters
    ----------
    data_in : numpy.array
        An array of data as returned by the Load_xxx functions with columns
        of datetime, total conc, and size bin data
    segments : list
        list of lists containing start and end times for activities. See example
        in the function description.

    Returns
    -------
    index : numpy.array
        Array of indexes, indicating which datapoints belongs to each activity.
        Non-categorized datapoints will have an index of 0, while others will
        have 1, 2, 3 and so on, in the order given in the segments variable.

    """
    
    # Generate an array of zeros for indexing
    index = np.zeros(data_in.shape[0],dtype="O")
    
    if not labels:
        labels = range(1,len(segments)+1)
    
    # Loop through the different time segments
    for j,i in enumerate(segments):
        
        # If only one time segment was given
        if len(i) == 2:
            start = i[0]
            end = i[1]
            
            # Find the indexes, where the experiment time is between the specified
            # start and end time and give these indexes a value of j+1
            index[(data_in[:,0]>start) & (data_in[:,0]<end)] = labels[j]
            
        # If there is more than one time segment for the given activity
        else:
            # store all starting points in one and end points in another variable
            starts = i[::2]
            ends = i[1::2]
            
            # Loop through all time segments and find the indexes, where the 
            # experiment time is between the specified start and end times and 
            # set these indexes to j+1
            for k in range(len(starts)):
                index[(data_in[:,0]>=starts[k]) & (data_in[:,0]<=ends[k])] = labels[j]
            
    return index

###############################################################################
###############################################################################
###############################################################################
def time_crop(data_in,start=0,end=0):
    """
    Function to crop a dataset, by setting a start and/or an end time. This is
    needed e.g. when making direct comparison between datasets in order to
    ensure the same number of datapoints.

    Parameters
    ----------
    data_in : numpy.array
        An array of data as returned by the Load_xxx functions with columns
        of datetime, total conc, and size bin data
    start : datetime, optional
        Specify starting time to crop by setting a datetime value e.g.
        datetime.datetime(2023,01,28,15,00,00). The default is 0.
    end : datetime, optional
        Specify end time to stop cropping by setting a datetime value e.g.
        datetime.datetime(2023,01,28,15,00,00). The default is 0.

    Returns
    -------
    Data_return : numpy.array
        New array following the same structure as the input array but with new
        times.
    """
    # Store the time
    times = data_in[:,0]
    
    # Find the index of times fulfilling the start and/or end time conditions
    if (start != 0) or (end != 0):
        if (start != 0) & (end == 0):
            index = times>=start
        elif (start == 0) & (end != 0):
            index = times<=end
        else:
            index = (times>=start) & (times<=end)
        
        # Select the data rows fulfilling the time criterias
        Data_return = data_in[index,:]
        
    return Data_return    

###############################################################################
###############################################################################
###############################################################################
def time_rebin(data_in, start=0, end=0, avg_time=5, APS=False):
    """
    Function to calculate the average values across time with a set time step.
    The function returns nan values where data is not present in the current
    dataset, which is usefull when having to combine several datasets with
    different lengths.
    
    NOTE! The function currently only works if the specified start time is
    viable with the avg_time, meaning that if avg_time is set to 1, so every
    minute, the starting time may not include seconds. Similarly, if the 
    avg_time is set to 0.5, the starting time must be at either 0 or 30 seconds
    on the clock.

    Parameters
    ----------
    data_in : numpy.array
        data in the format from a load_XXXX_function. First column must be datetime
    start : datetime, optional
        Define a starting time from which the data set should be started.
        This value can be before or after first entry.
        The default 0, so it uses the first entry as start
    end : datetime, optional
        Define an ending time from which the data set should be concluded.
        This value can be before or after last entry.
        The default 0, so it uses the last entry as end
    avg_time : float, optional
        Averaging time steps. The unit is in minutes, with a default of 5 
        minutes.

    Returns
    -------
    avg : numpy.array
        Returns the data 
    std : numpy.array
        Returns the standard deviation associated with each generated average.

    """

    data = data_in.copy()
    if APS==True:
        #Split up the data into the two columns with time and total concentration
        Time_Total=data[:,:2].copy()
        cor_data=[]
        for t in range(0,len(data)):
            row=[]
            #It then expands each 16*52 array into a single file line added to Time_total
            for O in range(0,16):
                for A in range(0,51): #(0,52)
                    row.append(data[t,2][O,A])
            cor_data.append(np.array(row))
        data=np.column_stack((Time_Total,cor_data))
        
    #Adds half a timebin so that the reported time relfects the average around that time
    data[:,0] = data[:,0]+datetime.timedelta(0,avg_time*60/2)
    df = pd.DataFrame(data[:, 1:], index=pd.to_datetime(data[:, 0]).tz_localize(None))
    
    freq = str(str(avg_time)+'min')
    # Calculate the frequence averages and standard deviation
    avg = df.resample(freq).mean()
    std = df.resample(freq).std()
        
    if start!=0:
        #Removes the offset from a frequency divisable step, as the data otherwise spits out nan
        offset = date2num(start)*24*60/avg_time
        start_time = pd.Timestamp(num2date(int(offset)/24/60*avg_time)).tz_localize(None)
    else:
        #Removes the offset from a frequency divisable step, as the data otherwise spits out nan
        offset = date2num(data[0,0])*24*60/avg_time
        start_time = pd.Timestamp(num2date(int(offset)/24/60*avg_time)).tz_localize(None)
    
    if end!=0:
        end_time = end
    else:
        end_time = data[-1,0]
        
    time_index = pd.date_range(start=start_time, end=end_time, freq=freq)
    avg                             = avg.reindex(time_index)
    std = std.reindex(time_index)
    # Reset the index to make the 'datetime' index a column
    avg = avg.reset_index()
    std = std.reset_index()
    # Convert the DataFrame to a NumPy array
    avg = avg.to_numpy()
    std = std.to_numpy()
    
    if APS==True:
        #Reforms the structure of the inserted APS data
        avg_cor=avg[:,:3].copy()
        std_cor=std[:,:3].copy()
        for t in range(0,len(avg_cor)):
            cor_array=[]
            std_array=[]
            for O in range(0,16):
                cor_array.append(avg[t,2+O*51:53+O*51])
                std_array.append(std[t,2+O*51:53+O*51])
                # cor_array.append(avg[t,2+O*52:54+O*52])
                # std_array.append(std[t,2+O*52:54+O*52])
            #cor_data.append(cor_array)
            avg_cor[t,2]=np.array(cor_array).astype(float)
            std_cor[t,2]=np.array(std_array).astype(float)
        avg=avg_cor
        std=std_cor
    return avg, std

###############################################################################
###############################################################################
###############################################################################
def time_shift(data_in,delta):
    """
    Function to adjust the times of a dataset. All times can be shifted in either
    positive or negative direction by the specified number of seconds in the
    delta parameter. 

    Parameters
    ----------
    data_in : numpy.array
        An array of data as returned by the Load_xxx functions with columns
        of datetime, total conc, and size bin data
    delta : int
        Number of seconds to shift the dataset. The shift can be both positive
        and negative

    Returns
    -------
    Data_return : numpy.array
        New array following the same structure as the input array but with new
        times.

    """
    # Store the datetime values
    time    = data_in[:,0]
    
    # Store the needed time shift
    delta   = datetime.timedelta(0,delta)
    
    # Shift the times
    new_time = time + delta
    
    # Copy the original dataset and insert the shifted times
    Data_return = data_in.copy()
    Data_return[:,0] = new_time
    
    return Data_return    

###############################################################################
###############################################################################
###############################################################################
def Unnormalize_dndlogdp(data_in,bin_edges):
    """
    Function to unnormalize an array of particle xxx concentrations (xxx can be
    mass, volume, surface area, and number), going from dxxx/dlogDp to dx values.

    Parameters
    ----------
    data_in : numpy.array
        An array of normalized data as returned by the Load_xxx functions with 
        columns of datetime, total conc, and size bin data
    bin_edges : numpy.array
        Array containing the limits of all sizebins. The array should have one 
        more value than the length of the "data_in" parameter

    Returns
    -------
    Data_return : numpy.array
        An array of normalized data equivalent to the data_in shape with columns
        of datetime, total conc, and normalized size bin concentrations.

    """
    # Calculate the normalization vector
    dlogDp          = np.log10(bin_edges[1:])-np.log10(bin_edges[:-1])
    
    # Copy the input data to not mess with the parent
    Data_return     = data_in.copy()
    
    # Select relevant data
    data            = Data_return[:,2:]
    
    # Apply normalization
    UnNormed        = data*dlogDp
    UnNormed_total  = np.sum(UnNormed,axis = 1)
    
    # Store the normalized data instead of the non-normalized and return it
    Data_return[:,2:]= UnNormed
    Data_return[:,1] = UnNormed_total
    
    return Data_return

###############################################################################
###############################################################################
###############################################################################
#%% The following is a list of simple functions fed into the peak_fit function
def EXP_FUNC(t,k, P0, E,t_E,t_P):
    
    """
    Calculates the concentration P(t) for a population experincing
    a first order decay: dP/dt = - k*P
    or if E>0: dP/dt = E- k*P
    
    Parameters
    ----------
    t: float
        Time at which the concentration is investigated (s)
    k: float
        Rate constant for the second order order decay (cm3/s)
    P0: float
        Starting concentration of particles (1/cm3)
    E: float
        Emission strength (1/cm3s)
    t_E: float
        Begining time of the emission in seconds relative to 00:00 (s)
    t_P: float
        Duration of emission and thus the time of peak concentration. (s)
        
    Returns
    -------
    P(t): array 
        An array of the resulting concentrations P at times t (1/cm3)
    """
    
    k       = abs(k)
    P0      = abs(P0)
    E       = abs(E)
    # Adjusts time so that t=0 is the initiation time for emission
    t       = t-abs(t_E)
    t_P     = abs(t_P)
    # Calculate the peak concentration based on t_P
    Pmax    = np.exp(-k*t_P)*(E*(np.exp(k*t_P)-1)+k*P0)/k
    # Generate an array for the resulting particle concentration 
    P       = len(t)*[0]
    for i in range(0,len(t)):
        
        if t[i]<0:
            P[i]    = P0
        elif t[i]>=t_P:
            P[i]    = Pmax * np.exp(-k * (t[i]-t_P))
        else:
            P[i]    = np.exp(-k*t[i] )*(E*(np.exp(k*t[i])-1)+k*P0)/k
    return P

###############################################################################
def LIN_FUNC(t, C, P0,E,t_E,t_P):

    """
    Calculates the concentration P(t) for a population experincing
    a second order decay: dP/dt = - c*P^2
    or if E>0: dP/dt = E- c*P^2
    
    Parameters
    ----------
    t: float
        Time at which the concentration is investigated (s)
    C: float
        Rate constant for the second order order decay (cm3/s)
    P0: float
        Starting concentration of particles (1/cm3)
    E: float
        Emission strength (1/cm3s)
    t_E: float
        Begining time of the emission in seconds relative to 00:00 (s)
    t_P: float
        Duration of emission and thus the time of peak concentration. (s)
        
    Returns
    -------
    P(t): array 
        An array of the resulting concentrations P at times t (1/cm3)
    """
    
    C       = abs(C)
    P0      = abs(P0)
    E       = abs(E)
    # Adjusts time so that t=0 is the initiation time for emission
    t       = t-abs(t_E)
    t_P     = abs(t_P)
    # Adding easier units
    c       = C**0.5
    e       = E**0.5

    Pmax    = e*np.tanh(np.arctanh(c*P0/e)+c*e*t_P)/c
    P       = len(t)*[0]
    for i in range(0,len(t)):
        
        if t[i]<0:
            P[i]    = P0
        elif t[i]>=t_P:
            P[i]    = 1/(C*(t[i]-t_P) + 1/Pmax)
        else:
            P[i]    = e*np.tanh(np.arctanh(c*P0/e)+c*e*t[i])/c
    return P
###############################################################################
def THREE_FUNC(t, K, C, P0, E,t_E,t_P):
    
    """
    Calculates the concentration P(t) for a population experincing
    a third order decay of the form: dP/dt = - (K*P + C*P^2)
    or if E>0: dP/dt = E - (K*P + C*P^2)
    
    Parameters
    ----------
    t: float
        Time at which the concentration is investigated (s)
    K: float
        First order rate constant (1/s)
    C: float
        Second order rate constant (cm3/s)
    P0: float
        Starting concentration of particles (1/cm3)
    E: float
        Emission strength (1/cm3s)
    t_E: float
        Begining time of the emission in seconds relative to 00:00 (s)
    t_P: float
        Duration of emission and thus the time of peak concentration. (s))
        
    Returns
    -------
    P(t): array 
        An array of the resulting concentrations P at times t (1/cm3)
    """
    
    K       = abs(K)
    C       = abs(C)
    E       = abs(E)
    P0      = abs(P0)
    # Adjusts time so that t=0 is the initiation time for emission
    t       = t-abs(t_E)
    t_P     = abs(t_P)
    
    Det     = (K**2+4*C*E)**0.5

    Pmax    =  (Det*np.tanh(np.arctanh((2*C*P0+K)/Det)+1/2*Det*t_P)-K)/(2*C)

    P       = len(t)*[0]
    for i in range(0,len(t)):
        
        if t[i]<0:
            P[i]    = P0
        elif t[i]>=t_P:
            P[i]    = 1/((Pmax*C+K)/(Pmax*K)*np.exp(K*(t[i]-t_P))-C/K)
        else:
            P[i]    = (Det*np.tanh(np.arctanh((2*C*P0+K)/Det)+1/2*Det*t[i])-K)/(2*C)

    return P

###############################################################################
def Determine_fit(Exp_fit,Lin_fit,Three_fit,conc):
    
    """
    Function to determine which of the three fit functions fit the best,
    with weight given to the simpler models.
    
    Parameters
    ----------
    Exp_fit : dict
        The fit dictionary as returned from the EXP_FUNC.
    Lin_fit : dict
        The fit dictionary as returned from the LIN_FUNC.
    Three_fit : dict
        The fit dictionary as returned from the THREE_FUNC.
    conc : numpy.array
        Array of the investigated .

    Returns
    -------
    fit : dict
        The best fitting fit of the three options.
    Pmax : TYPE
        DESCRIPTION.

    """
    Step_size1  = 1.1
    Step_size2  = 1.25
    if len(Three_fit)>0 and len(Lin_fit)>0 and len(Exp_fit)>0:
        if ((1-Three_fit['r2'])*Step_size1 < 1-Lin_fit['r2']) and (
            (1-Three_fit['r2'])*Step_size2 < 1-Exp_fit['r2']):                
            Pmax    = Three_fit['Pmax']
            fit     = Three_fit

        elif ((1-Three_fit['r2'])*Step_size1 > 1-Lin_fit['r2']) and (
            (1-Lin_fit['r2'])*Step_size1 < 1-Exp_fit['r2']): 
            Pmax    = Lin_fit['Pmax']
            fit     = Lin_fit

        else:
            Pmax    = Exp_fit['Pmax']
            fit     = Exp_fit

    elif len(Three_fit)>0 and len(Lin_fit)>0 and len(Exp_fit)==0:
        if (1-Three_fit['r2'])*Step_size1 < 1-Lin_fit['r2']:
            Pmax    = Three_fit['Pmax'] 
            fit     = Three_fit

        else:
            Pmax    = Lin_fit['Pmax']
            fit     = Lin_fit

    elif len(Three_fit)>0 and len(Lin_fit)==0 and len(Exp_fit)>0:            
        if (1-Three_fit['r2'])*Step_size1 < 1-Exp_fit['r2']:
            Pmax    = Three_fit['Pmax'] 
            fit     = Three_fit

        else:
            Pmax    = Exp_fit['Pmax']
            fit     = Exp_fit

    elif len(Three_fit)>0 and len(Lin_fit)==0 and len(Exp_fit)==0:  #!
        Pmax    = Three_fit['Pmax'] 
        fit     = Three_fit

    elif len(Lin_fit)>0 and len(Exp_fit)>0:
        if (1-Lin_fit['r2'])*Step_size1 < 1-Exp_fit['r2']:
            Pmax    = Lin_fit['Pmax']
            fit     = Lin_fit

        else:
            Pmax    = Exp_fit['Pmax']
            fit     = Exp_fit

    elif len(Lin_fit)>0 and len(Exp_fit)==0:
        Pmax    = Lin_fit['Pmax']
        fit     = Lin_fit

    elif len(Exp_fit)>0:
        Pmax    = Exp_fit['Pmax']
        fit     = Exp_fit

    else:
        Pmax=max(conc)
        fit={'fit_type':None,
                 'E':       0,
                 't_0':     0,
                 't_max':   0,
                 'Errors':{
                     'E':  0}}

    return fit, Pmax
###############################################################################
def R2(data,fit):
    """Calculates the R^2 value for a fitted set of values
    
    Parameters
    ----------
    data: array
        single column of data to which a fit is made.
    fit: array 
        single column of fitted values of equal dimension of the data.

    Returns
    -------
    R^2: float 
        The resulting R^2 value for the fit"""
    #data: 
    # mask=
    # residual sum of squares
    ss_res = np.sum((data - fit) ** 2)
    # total sum of squares
    ss_tot = np.sum((data - np.mean(data)) ** 2)
    # r-squared
    return round((1 - (ss_res / ss_tot)),3)

###############################################################################
def Peak_fitter(Data,t_0=0,t_Max=0,bin_mids=[],Peak_filter=0,Decay_length=600,Search_range=30,Forced_function=0):
    """Find the location of the peak by identifying the maximum value,
    and fits one of three functions to the peak, as well as provide emission calculations
    
    Parameters
    ----------
    Data : numpy.array
        An array of data as returned by the Load_xxx functions with 
        columns of datetime, total conc
    t_0 : datetime
        Precise or estimateted time of the begining of the peak
    t_max : datetime
        Precise or estimateted time of the top of the peak
    bin_mids: list
        If the instrument is sized, apply here the list of bin_mids
    Peak_filter : float
        Setpoint of a minimum the peak has to be larger than to be recongnized. 
    Decay_length : float
        Length in seconds of the decaying peak that should be fitted.
        A longer stretch of time to fit normally yields better results,
        but as the values return towards the background level the decay should 
        be limited. 
    Search_range : float
        Range in seconds around the estimated t_max where the peak is expected to be found.
    Forced_function: str (optional)
        Gives the option to force which decay to fit the peak with.
        The default is 0, which lets the function find the best fitting decay rate.
        Other options are: 'Exp', 'Lin' and 'Three'
        
    Returns
    -------
    Output : dictonary
        A dictonary of values:
            PT : Found peak time (datetime)
            BG : Background concentration before the begining of the peak (float)
            Fit : Fit type and  associated values and errors
            P0 : Fitted conctration at the peak in 1/cm3 (float)
            Emission_rate : Fitted value for the emission rate in 1/cm3s (float)
    """   
    output      = {}
    ###################################################################
    """Find the location of the peak by identifying the maximum value"""
    ###################################################################                 
    #Calculates the date as number
    ex_date     = int(date2num(Data[0,0]))
    
    #Initial data cleaning to remove nan values
    Data_clean  = np.isnan(Data[:,1].astype(float))
    Data_clean  = [not elem for elem in Data_clean]
            
    Data        = Data[Data_clean]
    
    #Establish the search range in seconds
    if t_0 == 0:
        Peak_Start  = Data[0,0]
    else:
        Peak_Start  = t_0   
    T0 = Peak_Start.hour*60*60 + Peak_Start.minute*60 + Peak_Start.second
    
    if t_Max    == 0:
        Peak_End    = Data[-1,0]
        if t_0  == 0:
            Search_range    = (Data[-1,0]-Data[0,0]).total_seconds()
    else:
        Peak_End    = t_Max   
        
    t_max   = Peak_End.hour*60*60 + Peak_End.minute*60 + Peak_End.second
    
    Time    = (date2num(Data[:,0])-ex_date)*24*60*60

    #Determines if the peaks should be found for differnet size bins if applicable.
    if len(bin_mids) == 0: #[]:
        Sized           = False
        Concentration   = Data[:, 1]  
    else:
        Sized           = True
        Concentration   = Data[:,1:]        
        
    if max(Data[:,1])<Peak_filter:
        Peak        = False
        search_mask = (Time > (T0 - Search_range)) & (Time < t_max + Search_range)
    elif max(Data[:,1] )>=Peak_filter:
        Peak        = True
        search_mask = (Time > T0-Search_range) & (Time <= (t_max + Decay_length + Search_range)) 
    
    #Round to the nearest second
    Time    = Time[search_mask]
    for i in range(0,len(Time)):
        Time[i]     = round(Time[i])

    concentration = Concentration[search_mask]            
    ###################################################################
    """Set up the for loop for the size bins"""
    ###################################################################
    if Sized == True:
        peak_time_array = []
        ###################################################################
        """Find the location of the peak or valley by identifying the maximum value or minimum value"""
        ###################################################################
        BG_tot      = 0
        Pmax_tot    = 0
        E_tot       = 0
        #for s in range(0,len(Concentration[0,1:])):
        for s in range(0,len(bin_mids)):
            peak_index  = np.argmax(concentration[:,s+1])
            peak_time   = Time[peak_index]
            
            conc        = concentration[:,s+1]
            #Removes zero values from the data set
            mask        = conc>0
            conc        = conc[mask]
            Timed       = (date2num(Data[:,0])-ex_date)*24*60*60
            Timed       = Timed[search_mask][mask]
            
            if t_Max == 0:
                t_P = peak_time - T0
            else: t_P = t_max-T0
        
            #Generate the name for the reporting folder with the bin size
            name= str(bin_mids[s])
            output[name]={}
            ###################################################################
            """Find the location of the pre-peak minimum and determine the background value"""
            ###################################################################
            if len(conc) < 10:
                output[name]    = {
                    'PT':           Peak_End,
                    'BG':           "NaN",
                    'Pmax':         "NaN",
                    'Emission_rate': 0
                    }
            else:    

                #Determines the background averages and spread before the peak
                BG_min_index    = np.argmin(conc)
                BG_avg          = np.average(conc[:BG_min_index+1])
    
                if Peak == True: #If a peak is found, a more thorough search starts for determining the emission rate
                    
                    #Guess the emission based on the difference between the max value and the    
                    E_guess     = (np.max(concentration[:,s+1])-BG_avg)/t_P
                    
                    #Defines the decay period between the peak and the return to the background level
                    Exp_fit     = {}
                    Lin_fit     = {}
                    Three_fit   = {}
                    ###############################################################
                    """Fit the found peak to an exponential decay assuming first order equation"""
                    ###############################################################
                    if Forced_function == 0 or Forced_function == 'Exp':
                        try:
                            min_bounds  = [1E-10,0,E_guess*0.15,T0-Search_range,t_P-Search_range]
                            max_bounds  = [np.inf,BG_avg*5,E_guess*5,T0+Search_range,t_P+Search_range]
                            parameters, covariance =curve_fit(EXP_FUNC,Timed,conc,p0=[1/(60*60),BG_avg,E_guess,T0,t_P],bounds=(min_bounds,max_bounds))#(0,np.inf))
                            fit_k, fit_P0, fit_E, fit_tE, fit_tP= parameters
                
                            SE          = np.sqrt(np.diag(covariance))
                            SE_k , SE_P0, SE_E, SE_tE, SE_tP = SE
                            
                            Pmax        = EXP_FUNC([fit_tP+fit_tE],*parameters)[0]#fit_k, fit_P0, fit_E, fit_tE, fit_tP)[0]
                            # residual sum of squares
                            exp_fit     = EXP_FUNC(Timed,fit_k, fit_P0, fit_E, fit_tE, fit_tP)
                            r2_ex       = R2(conc,exp_fit)
                            # Fill out the dictionary with values
                            Exp_fit     = {
                                    'fit_type':'Exp',
                                    'P0':      float(fit_P0),
                                    'Pmax':    float(Pmax),
                                    'k':       float(fit_k),
                                    'E':       float(fit_E),
                                    't_0':     float(fit_tE),
                                    't_max':   float(fit_tP), 
                                    'r2':      round(r2_ex,3),
                                    'Errors':{
                                        'P0':   float(SE_P0),
                                        'k':    float(SE_k),
                                        'E' :   float(SE_E),
                                        't_0':  float(SE_tE),
                                        't_max':float(SE_tP)  }
                                     }
                            
                            if r2_ex < 0.5:
                                Exp_fit     = {}
                        except: Exp_fit     = {}
                    else: Exp_fit   = {}
                    ###############################################################
                    """Fit the found peak to a decay with a second order equation"""
                    ###############################################################
                    if Forced_function == 0 or Forced_function == 'Lin':
                        try:
                            min_bounds  = [1E-14,0,E_guess*0.15,T0-Search_range,t_P-Search_range]
                            max_bounds  = [np.inf,BG_avg*5,E_guess*5,T0+Search_range,t_P+Search_range]
                            parameters, covariance =curve_fit(LIN_FUNC,Timed,conc, p0=[1E-6,BG_avg,E_guess,T0,t_P],bounds=(min_bounds,max_bounds))#(0,np.inf))
                            fit_c, fit_P0, fit_E, fit_tE, fit_tP = parameters
                
                            SE          = np.sqrt(np.diag(covariance))
                            SE_c , SE_P0, SE_E, SE_tE, SE_tP  = SE
                
                            Pmax        = LIN_FUNC([fit_tP+fit_tE],*parameters)[0]#fit_c, fit_P0, fit_E, fit_tE, fit_tP)[0]
                            # residual sum of squares
                            lin_fit     = LIN_FUNC(Timed,*parameters)#fit_c, fit_P0, fit_E, fit_tE, fit_tP)
                            r2_lin      = R2(conc,lin_fit)
                            # Fill out the dictionary with values
                            Lin_fit     = {
                                'fit_type': 'Lin',
                                'P0':       float(fit_P0),
                                'Pmax':     float(Pmax),
                                'c':        float(fit_c),
                                'E':        float(fit_E),
                                't_0':      float(fit_tE),
                                't_max':    float(fit_tP), 
                                'r2':       round(r2_lin,3),
                                'Errors':{
                                    'P0':    float(SE_P0),
                                    'c':     float(SE_c),
                                    'E' :    float(SE_E),
                                    't_0':   float(SE_tE),
                                    't_max': float(SE_tP)  }
                                     }

                            if r2_lin < 0.5:
                                Lin_fit     = {}
                        except: Lin_fit     = {}
                    else: Lin_fit   = {}
                    ###############################################################
                    """Fit the found peak to an combined first and second order equation"""
                    ###############################################################
                    if Forced_function == 0 or Forced_function == 'Three':
                        try:
                            min_bounds  = [1E-10,1E-14,0,E_guess*0.15,T0-Search_range,t_P-Search_range]
                            max_bounds  = [np.inf,np.inf,BG_avg*5,E_guess*5,T0+Search_range,t_P+Search_range]
                            parameters, covariance =curve_fit(THREE_FUNC,Timed,conc, p0=[1E-4,1E-6,BG_avg,E_guess,T0,t_P],bounds=(min_bounds,max_bounds))#(0,np.inf))
                            fit_K, fit_C, fit_P0, fit_E, fit_tE, fit_tP = parameters.copy()
                    
                            SE          = np.sqrt(np.diag(covariance))
                            SE_K , SE_C,  SE_P0, SE_E, SE_tE, SE_tP = SE.copy()
                            
                            Pmax        = THREE_FUNC([fit_tP+fit_tE],*parameters)#fit_K,fit_C, fit_P0, fit_E, fit_tE, fit_tP)[0]
                            # residual sum of squares
                            three_fit   = THREE_FUNC(Timed,*parameters)#fit_K, fit_C, fit_P0, fit_E, fit_tE, fit_tP)
                            r2_three    = R2(conc,three_fit)
                            # Fill out the dictionary with values
                            Three_fit   = {
                                'fit_type': 'Three',
                                'P0':       float(fit_P0),
                                'Pmax':     float(Pmax),
                                'K':        float(fit_K),
                                'C':        float(fit_C),
                                'E':        float(fit_E),
                                't_0':      float(fit_tE),
                                't_max':    float(fit_tP), 
                                'r2':       round(r2_three,3),
                                'Errors':{
                                    'P0':    float(SE_P0),
                                    'K':     float(SE_K),
                                    'C':     float(SE_C),
                                    'E' :    float(SE_E),
                                    't_0':   float(SE_tE),
                                    't_max': float(SE_tP)  }
                                     }
                            
                            if r2_three < 0.5:
                                Three_fit   = {}
                        except: Three_fit   = {}
                    else: Three_fit     = {}
                    ###############################################################   
                    """Determine the most appropriate fit"""
                    ###############################################################

                    fit, Pmax=Determine_fit(Exp_fit,Lin_fit,Three_fit,conc)
                        
                    ###############################################################   
                    """Fill out the values from chosen fit into the output"""
                    ###############################################################    

                    output[name] = {'PT':num2date(ex_date+(fit['t_0']+fit['t_max'])/24/3600),  #Time of peak
                        'BG':float(round(BG_avg,1)), #Background concentration in 1/cm3
                        'Fit':fit,          #Adds the best fitting fit function
                        'Pmax': float(Pmax),          #Peak concentration in 1/cm3
                        'Emission_rate':fit['E'],       #Average emission across the period in 1/cm3s
                        'SE_E':fit['Errors']['E']
                         }
                    
                    if fit['t_0'] == 0:
                        output[name]['PT']  = num2date(ex_date+peak_time)
                    else:
                        peak_time_array.append(date2num(output[name]['PT']))
                else:
                    output[name]    = {'PT':num2date(ex_date+t_max/24/3600),
                        'BG':float(min(conc)),
                        'Pmax':float(max(concentration[:,s])),
                        'Emission_rate':0
                        }
                    
                BG_tot      = BG_tot+output[name]['BG']
                Pmax_tot    = Pmax_tot+output[name]['Pmax']
                # E_tot=E_tot+fit['E']
                
                E_tot       = E_tot+output[name]['Emission_rate']
        #Adds together the P0, BG and emission rates for all the bin-sizes to give a total value
        if type(np.nanmean(peak_time_array))==float:
            PT  = num2date(np.nanmean(peak_time_array))
            
        else:
            PT  = num2date(ex_date+t_max/24/3600)
            
        output['Total'] = {'PT':PT,#ex_date+peak_time/24/3600),
                'BG':float(BG_tot),
                'Pmax':float(Pmax_tot),
                'Emission_rate':float(E_tot)
                }
    else:
        
        ###################################################################
        """Find the location of the peak or valley by identifying the maximum value or minimum value"""
        ###################################################################                  
        if Peak == True:
            peak_index  = np.argmax(concentration)
            peak_time   = Time[peak_index]
            
            output['Total']= {}
            conc        = concentration
            
            mask        = conc>0
            conc        = conc[mask]
            Timed       = (date2num(Data[:,0])-ex_date)*24*60*60
            Timed       = Timed[search_mask][mask]
            
            if t_max == 0:
                t_P = peak_time - T0
            else: t_P = t_max-T0
            
            ###################################################################
            """Find the location of the pre-peak minimum and determine the background value"""
            ###################################################################
            if len(conc) < 10:
                output['Total']={'PT':"NaN",
                    'BG':"NaN",
                    'P0':"NaN",
                    'Emission_rate':0
                    }
                pass

            #print("Pre-peak min occured at ",int(valley_time/3600),":",int(valley_time%3600/60),":",int(valley_time%60))
            BG_min_index= np.argmin(conc)
                                           
            #Determines the background averages and spread before the peak
            BG_avg      = np.average(conc[:BG_min_index]+1)
            #Guess the emission based on the difference between the max value and the    
            E_guess     = (np.max(concentration)-BG_avg)/t_P
                    
            #elif Peak == True: #If a peak is found, a more thorough search starts for determining the emission rate
            Exp_fit     = {}
            Lin_fit     = {}
            Three_fit   = {}
            ###############################################################
            """Fit the found peak to an exponential decay assuming first order equation"""
            ###############################################################
            if Forced_function == 0 or Forced_function == 'Exp':
                try:
                    min_bounds  = [1E-10,0,E_guess*0.15,T0-Search_range,t_P-Search_range]
                    max_bounds  = [np.inf,BG_avg*5,E_guess*5,T0+Search_range,t_P+Search_range]
                    parameters, covariance =curve_fit(EXP_FUNC,Timed,conc,p0=[1/(60*60),BG_avg,E_guess,T0,t_P],bounds=(min_bounds,max_bounds))#(0,np.inf))
                    fit_k, fit_P0, fit_E, fit_tE, fit_tP= parameters
        
                    SE          = np.sqrt(np.diag(covariance))
                    SE_k , SE_P0, SE_E, SE_tE, SE_tP = SE
                    
                    Pmax        = EXP_FUNC([fit_tP+fit_tE],*parameters)[0]#fit_k, fit_P0, fit_E, fit_tE, fit_tP)[0]
                    # residual sum of squares
                    exp_fit     = EXP_FUNC(Timed,fit_k, fit_P0, fit_E, fit_tE, fit_tP)
                    exp_fit     = EXP_FUNC(Timed,fit_k, fit_P0, fit_E, fit_tE, fit_tP)
        
                    # residual sum of squares
                    r2_ex       = R2(conc,exp_fit)
                    #Fill out the dictionary with values        
                    Exp_fit     = {
                        'fit_type':'Exp',
                        'P0':      float(fit_P0),
                        'Pmax':    float(Pmax),
                        'k':       float(fit_k),
                        'E':       float(fit_E),
                        't_0':     float(fit_tE),
                        't_max':   float(fit_tP), 
                        'r2':      round(r2_ex,3),
                        'Errors':{
                            'P0':       float(SE_P0),
                            'k':        float(SE_k),
                            'E' :       float(SE_E),
                            't_0':      float(SE_tE),
                            't_max':    float(SE_tP)  }
                             }
                    
                    if r2_ex < 0.5:
                        Exp_fit    = {}
                
                except: Exp_fit     = {}
            else:   Exp_fit   = {}
            ###############################################################
            """Fit the found peak to a decay with a second order equation"""
            ###############################################################
            if Forced_function == 0 or Forced_function == 'Lin':
                try:
                    min_bounds  = [1E-14,0,E_guess*0.15,T0-Search_range,t_P-Search_range]
                    max_bounds  = [np.inf,BG_avg*5,E_guess*5,T0+Search_range,t_P+Search_range]
                    parameters, covariance =curve_fit(LIN_FUNC,Timed,conc, p0=[1E-6,BG_avg,E_guess,T0,t_P],bounds=(min_bounds,max_bounds))#(0,np.inf))
                    fit_c, fit_P0, fit_E, fit_tE, fit_tP = parameters
                
        
                    SE          = np.sqrt(np.diag(covariance))
                    SE_c , SE_P0, SE_E, SE_tE, SE_tP  = SE
                
                    Pmax        = LIN_FUNC([fit_tP+fit_tE],*parameters)[0]#fit_c, fit_P0, fit_E, fit_tE, fit_tP)[0]
                    # residual sum of squares
                    lin_fit     = LIN_FUNC(Timed,*parameters)#fit_c, fit_P0, fit_E, fit_tE, fit_tP)
                    r2_lin      = R2(conc,lin_fit)
                    #Fill out the dictionary with values  
                    Lin_fit     = {'fit_type':'Lin',
                             'P0':      float(fit_P0),
                             'Pmax':    float(Pmax),
                             'c':       float(fit_c),
                             'E':       float(fit_E),
                             't_0':     float(fit_tE),
                             't_max':   float(fit_tP), 
                             'r2': round(r2_lin,3),
                             'Errors':{
                                 'P0':    float(SE_P0),
                                 'c':     float(SE_c),
                                 'E' :    float(SE_E),
                                 't_0':   float(SE_tE),
                                 't_max': float(SE_tP)  }
                             }
                    
                    if r2_lin < 0.5:
                        Lin_fit     = {}
                    
                except: Lin_fit     = {}
            else: Lin_fit   = {}
            ###############################################################
            """Fit the found peak to an combined first and second order equation"""
            ###############################################################
            if Forced_function == 0 or Forced_function == 'Three':
                try:

                    min_bounds  = [1E-10,1E-14,0,E_guess*0.15,T0-Search_range,t_P-Search_range]
                    max_bounds  = [np.inf,np.inf,BG_avg*5,E_guess*5,T0+Search_range,t_P+Search_range]
                    parameters, covariance =curve_fit(THREE_FUNC,Timed,conc, p0=[1E-4,1E-6,BG_avg,E_guess,T0,t_P],bounds=(min_bounds,max_bounds))#(0,np.inf))

                    fit_K, fit_C, fit_P0, fit_E, fit_tE, fit_tP = parameters.copy()
            
                    SE          = np.sqrt(np.diag(covariance))
                    SE_K , SE_C,  SE_P0, SE_E, SE_tE, SE_tP = SE.copy()
            
                    Pmax        = THREE_FUNC([fit_tP+fit_tE],*parameters)#fit_K,fit_C, fit_P0, fit_E, fit_tE, fit_tP)[0]
                    # residual sum of squares
                    three_fit   = THREE_FUNC(Timed,*parameters)#fit_K, fit_C, fit_P0, fit_E, fit_tE, fit_tP)
                    r2_three    = R2(conc,three_fit)

                    #Fill out the dictionary with values  
                    Three_fit   = {'fit_type':'Three',
                             'P0':      float(fit_P0),
                             'Pmax':    float(Pmax),
                             'K':       float(fit_K),
                             'C':       float(fit_C),
                             'E':       float(fit_E),
                             't_0':     float(fit_tE),
                             't_max':   float(fit_tP), 
                             'r2': round(r2_three,3),
                             'Errors':{
                                 'P0':    float(SE_P0),
                                 'K':     float(SE_K),
                                 'C':     float(SE_C),
                                 'E' :    float(SE_E),
                                 't_0':   float(SE_tE),
                                 't_max': float(SE_tP)  }
                             }
                    
                    if r2_three < 0.5:
                        Three_fit   = {}
                except: Three_fit   = {}
            else: Three_fit     = {}
            ###############################################################   
            """Determine the releasd amount of particles"""
            ###############################################################
            fit, Pmax=Determine_fit(Exp_fit,Lin_fit,Three_fit,conc)
            ###############################################################   
            """Determine the emission rate"""
            ###############################################################    
           
            output['Total'] = {'PT':num2date(ex_date+(fit['t_0']+fit['t_max'])/24/3600),  #Time of peak
                'BG':   float(round(BG_avg,1)), #Background concentration in 1/cm3
                'Fit':  fit,          #Adds the best fitting fit function
                'Pmax': float(Pmax),          #Peak concentration in 1/cm3
                'Emission_rate':fit['E'],       #Average emission across the period in 1/cm3s
                'SE_E': fit['Errors']['E']}

        else:
            output['Total'] =    {'PT':t_P,
                'BG':   float(min(concentration)),
                'Pmax': float(max(concentration)),
                'Emission_rate':0
                 }

    return output
