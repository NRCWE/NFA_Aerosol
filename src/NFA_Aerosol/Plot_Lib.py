# -*- coding: utf-8 -*-
"""
This is the NFA library for plotting data from our aerosol instruments and
some of our sensors as well. 

The current version of this document is:
    
Plot_Lib v 1.1
The most recent addition is the Load_Grimm_new function

@author: B279683
"""
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.dates as mdates
import datetime as datetime
from scipy.stats import sem
import matplotlib.ticker as ticker

params = {'legend.fontsize': 20,
         'axes.labelsize': 25,
         'axes.titlesize': 25,
         'xtick.labelsize': 20,
         'ytick.labelsize': 20,
         'figure.figsize' : (19, 10)}
plt.rcParams.update(params)

###############################################################################
###############################################################################
###############################################################################
  
def Boxplot_PM(data_in,labels,y_lim=(0,0),tick_dist = 5):
    """
    Function to genereate a boxplot of PM levels from an array returned by the
    PM_calc function.

    Parameters
    ----------
    data_in : np.array
        Array as returned by the UL.PM_calc function
    labels : list
        List of x axis labels. One can use the header returned from the 
        UL.PM_calc function.
    y_lim : tuple, optional
        Lower and upper limit on the y-axis of the boxplot given as (lower,upper). 
        The default is (0,0) in which case it will select a value automatically.
        If only a lower or upper limit is specified the other will default to
        the max of the dataset e.g. (0,xx) will plot from lowest datapoint to xx.
    tick_dist : int
        Distance between the ticks on the y-axis. Default is 5.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Handle for the returned figure for saving.
    ax : matplotlib.axes._subplots.AxesSubplot
        Handle for the axis object of the plot.

    """
    data = data_in[:,1:]
    fig, ax = plt.subplots()
    
    dat_min = data.min()
    dat_max = data.max()
    if (y_lim[0] != 0) or (y_lim[1] != 0):
        if (y_lim[0] == 0) & (y_lim[1] != 0):
            ax.set_ylim(dat_min*0.98,y_lim[1])
        elif (y_lim[0] != 0) & (y_lim[1] == 0):
            ax.set_ylim(y_lim[0],dat_max*1.02)
        else:
            ax.set_ylim(y_lim[0],y_lim[1])
    
    ax.boxplot(data)
    ax.set_xticklabels(labels[1:])
    ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_dist))
    ax.grid(which="both",axis="both")
    ax.set_ylabel("Mass concentration, $\mu$g/cm$^{3}$")
    
    return fig, ax

###############################################################################
###############################################################################
###############################################################################


def Direct_compare(data_in1,data_in2, bin_edges):
    """
    Function to compare the total number concentration and the mean particle 
    size distribution (PSD) from two identical instruments. The total number 
    concentration for both instruments are plotted in a time series along with
    the mean PSD during the measurement. In addition the ratio of the total
    number concentration are plotted along with the mean and std of the ratios,
    which can be used as a correction factor. Furthermore, the ratio of the two
    mean PSDs is plotted, which can be used to identify size bins where the two
    instruments do not agree. However, the PSD ratio plots should be 
    interpretered with care as low concentrations in the sizebins can quickly 
    lead to very high ratio values, which are not necessarily an issue.

    Parameters
    ----------
    data_in1 : numpy.array
        An array of data as returned by the Load_xxx functions with columns
        of datetime, total conc, and size bin data
    data_in2 : numpy.array
        An array of data as returned by the Load_xxx functions with columns
        of datetime, total conc, and size bin data
    bin_edges : numpy.array
        Array containing the limits of all sizebins in nm. The array should 
        have one more value than the length of the "data_in" .
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        Handle for the returned figure for saving.
    axs : numpy.array
        Array with handles, one for each of the axes objects of the plot.

    """
    # Extract time data
    tim1 = data_in1[:,0]
    tim2 = data_in2[:,0]
    
    # Extract total number concentrations
    tot1 = data_in1[:,1].astype("float64")
    tot2 = data_in2[:,1].astype("float64")
    
    # Extract size bin data
    d1 = data_in1[:,2:].astype("float64")
    d2 = data_in2[:,2:].astype("float64")
    
    # Calculate the midpoint of all sizebins for plotting the PSD
    mids = bin_edges[1:]-(bin_edges[1:]-bin_edges[:-1])/2
    
    # Caculate the ratio between total number concentrations, and deal with
    # cases of concentration = 0, which can give np.nan or np.inf values.
    ratio_tot = tot1/tot2
    ratio_tot[np.isnan(ratio_tot)] = 0
    inf_index = np.where(np.isinf(ratio_tot))[0]
    ratio_tot[inf_index] = np.nan
    
    # Calculate the mean PSD for each dataset
    bar1 = d1.mean(axis=0)
    bar2 = d2.mean(axis=0)
    
    # Caculate the ratio between the two mean PSDs, and deal with
    # cases of concentration = 0, which can give np.nan or np.inf values.
    ratio_PSD = bar1/bar2
    ratio_PSD[np.isnan(ratio_PSD)] = 0
    inf_index1 = np.where(np.isinf(ratio_PSD))[0]
    ratio_PSD[inf_index1] = np.nan
    
    # Generate canvas and axis
    fig, axs = plt.subplots(nrows=2,ncols=2)
    fig.subplots_adjust(hspace=0.4,wspace=0.2)
    
    """############################ Plot 1 #################################"""
    # Plot the total number concentration of both datasets
    axs[0,0].plot(tim1,tot1,lw=2,color="r",label="Instrument 1")
    axs[0,0].plot(tim2,tot2,lw=2,color="b",label="Instrument 2")
    
    # Set y- and x-labels, title, legend, format the time axis, and rotate ticklabels 
    axs[0,0].set_ylabel("N$_{Total}$, cm$^{-3}$",fontsize=15)
    axs[0,0].set_xlabel("Time, HH.MM.SS",fontsize=15)
    axs[0,0].set_title("Total number concentration",fontsize=15)
    axs[0,0].legend(fontsize=15)
    axs[0,0].xaxis.set_major_formatter(mdates.DateFormatter("%H.%M.%S"))
    for tick in axs[0,0].get_xticklabels():
        tick.set_rotation(-45)
    
    """############################ Plot 2 #################################"""
    # Plot the ratio of total number concnetration from instruments 1 and 2
    axs[1,0].scatter(tim1,ratio_tot,lw=2,color="g")
    
    # Plot the mean ratio
    axs[1,0].axhline(np.nanmean(ratio_tot),c="k",label="Mean Ratio: {:.2f}".format(np.nanmean(ratio_tot)))
    
    # Plot the plus and minus std lines
    axs[1,0].axhline(np.nanmean(ratio_tot)+np.nanstd(ratio_tot),c="k",ls="--", label = "Std of Ratio: $\pm$ {:.2f}".format(np.nanstd(ratio_tot)))
    axs[1,0].axhline(np.nanmean(ratio_tot)-np.nanstd(ratio_tot),c="k",ls="--")
    
    # Plot a 1:1 line, which corresponds to perfect agreement between the two instruments
    axs[1,0].axhline(1,c="red",ls="--",label="1:1 line")
    
    # Set y- and x-labels, title, legend, format the time axis, and rotate ticklabels 
    axs[1,0].set_ylabel("Ratio",fontsize=15)
    axs[1,0].set_xlabel("Time, HH.MM",fontsize=15)
    axs[1,0].set_title("Ratio of Total Number Concentration, instrument1/instrument2",fontsize=15)
    axs[1,0].legend(fontsize=15)
    axs[1,0].xaxis.set_major_formatter(mdates.DateFormatter("%H.%M"))
    if inf_index.size > 0:
        axs[1,0].scatter(tim1[inf_index],np.zeros(inf_index.size),color="blue",label="Inf err")    
    
    """############################ Plot 3 #################################"""
    # Plot the mean PSD for both data series
    axs[0,1].bar(mids,bar1, width = bin_edges[1:]-bin_edges[:-1],ec="k",color="r",alpha=0.5,label="Instrument 1")
    axs[0,1].bar(mids,bar2, width = bin_edges[1:]-bin_edges[:-1],ec="k",color="b",alpha=0.5,label="Instrument 2")
    
    # Set y and x scale to log, ands et y- and x-labels, title, and legend
    axs[0,1].set_yscale("log")
    axs[0,1].set_xscale("log")
    axs[0,1].set_ylabel("dN, cm$^{-3}$",fontsize=15)
    axs[0,1].set_xlabel("Dp, nm",fontsize=15)
    axs[0,1].set_title("Mean PSD",fontsize=15)
    axs[0,1].legend(fontsize=15)
    
    """############################ Plot 4 #################################"""
    # Plot the %-error between instruments 1 and 2
    axs[1,1].bar(mids,ratio_PSD, width = bin_edges[1:]-bin_edges[:-1],ec="k",color="g")
    axs[1,1].axhline(1,c="red",ls="--",label="1:1 line")
    
    # Set y and x scale to log, ands et y- and x-labels, and title
    axs[1,1].set_yscale("log")
    axs[1,1].set_xscale("log")
    axs[1,1].set_ylabel("Ratio",fontsize=15)
    axs[1,1].set_xlabel("Dp, nm",fontsize=15)
    axs[1,1].set_title("Ratio of PSDs, instrument1/instrument2",fontsize=15)
    
    # Set ticks on the plots to be longer and enable grid lines
    for i in axs.flatten():
        i.tick_params(axis="y",which="both",direction='out', length=6, width=2)
        i.grid(axis="both",which="both",alpha=0.3)
        i.tick_params(axis='both', which='major', labelsize=10)
        
    return fig, axs

###############################################################################
###############################################################################
###############################################################################

def Plot_PSD(*data_in, labels=None, ylog=True, xlog=True, y_lim=(0, 0), datatype="number"):
    """
    Function to plot PSDs from one or multiple instruments. Instruments can have
    different bin mids. 

    Parameters
    ----------
    *data_in : list of tuples
        Each tuple should contain (bin_mids, size distribution data), where size distribution data
        has the same structure as returned from Load_Instrument functions with e.g. 1st column as datetime.
    labels : list, optional
        List of labels for the plots. The default is None.
    ylog : bool, optional
        Flag to turn on/off log scale for y-axis. Defaults are True.
    xlog : bool, optional
        Flag to turn on/off log scale for x-axis. Defaults are True.
    y_lim : tuple, optional
        Limits for the y-axis. Defaults to (0, 0), which means auto-scaling.
    datatype : str, optional
        The type of data, which is specified on y-axis label either as (dN, cm-3), (dN/dlogDp, cm-3),
        or (dm, µm/cm-3) for keywords "number", "normed", and "mass" respectively. Defaults to 'number'.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Handle for the returned figure for saving.
    ax : numpy.array
        Handle for the axis object of the plot.
    
    Example
    
    fig, ax = Plot_PSD_1((data['bin_mids_NS'], data['smps_Lab']), 
                       (data['bin_mids_FMPS'], data['fmps_Lab']), 
                       labels=["Data 1", "Data 2"], ylog=True, xlog=True)
    
    Created by PLF
    
    """
    colors = ["red", "blue", "green", "orange", "magenta", "cyan", "k", "purple", "yellow", 'pink']

    fig, ax = plt.subplots()
    for idx, (bin_mids, dataset) in enumerate(data_in):
        particle_data = dataset[:, 2:].astype("float")
        mean_psd = particle_data.mean(axis=0)
        sem_psd = sem(particle_data, axis=0)
        
        color = colors[idx % len(colors)]
        label = labels[idx] if labels and idx < len(labels) else None

        ax.plot(bin_mids, mean_psd, label=label, lw=3, color=color)
        ax.fill_between(bin_mids, mean_psd - sem_psd, mean_psd + sem_psd, alpha=0.5, color=color)

    # Set axis scales and labels
    if xlog:
        ax.set_xscale("log")
    if ylog:
        ax.set_yscale("log")
    ax.grid(True, which="both")

    y_label = "dN/dlogDp, cm$^{-3}$" if datatype == "normed" else "dN, cm$^{-3}$" if datatype == "number" else "dm, $\mu$g$^{-3}$"
    ax.set_ylabel(y_label)
    ax.set_xlabel("Dp, nm")

    # Adjust y-axis limits if specified
    if y_lim != (0, 0):
        ax.set_ylim(y_lim)

    if labels:
        ax.legend()

    return fig, ax

###############################################################################
###############################################################################
###############################################################################

def Plot_time_segments(data_in,ix,labels,elapsed=0,ylog=1):
    """
    Function to plot total number concentration in different colors depending
    on their indexes e.g. based on activities.

    Parameters
    ----------
    data_in : numpy.array
        An array of data as returned by the Load_xxx functions with columns
        of datetime, total conc, and size bin data
    ix : numpy.array
        Array of indexes as returned by the UL.time_segment function, indicating 
        which datapoints belongs to different activities.
    labels : list
        List of strings used for labeling the different activities. The number
        of labels should match the number of activities and index values.
    elapsed : Boolean, optional
        Boolean flag (1 or 0) setting whether to used elapsed time or use local
        time on the x-axis. The default is 1.
    ylog : boolean, optional
        Boolean flag (1 or 0) to turn on/off log y-axis. Default is 1.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Handle for the returned figure for saving.
    ax : numpy.array
        Handle for the axis object of the plot.

    """
    
    # List of colors used to color segments of each the activties
    colors = ["grey","blue","red","green","orange","magenta","cyan","k","purple","yellow"]
    
    # Store the time data in a new variable
    time = data_in[:,0]
    
    # Make the canvas/figure
    fig, ax = plt.subplots()
    
    # If elapsed time is active, adjust the datetime values to start at the new
    # year, so that the first displayed time and date is "1 00:00"
    if elapsed:
        fake_start = datetime.datetime(2023,1,1,0,0,0)
        delta_time = time-time[0]
        time = delta_time + fake_start
        
        # Get the number of days in the dataset.
        total_time = time[-1]-time[0]
        days = total_time.days
    
    # Make an extra label, which is never used
    labels = ["Original"] + labels
    
    # loop through labels
    for j in range(len(labels)):
        # Generate a masked array of the total number concentration
        total = np.ma.array(data_in[:,1],dtype="float")
        
        # Mask the points that does not match the current index
        total[ix!=j] = np.ma.masked
        
        # The whole time series is plotted in a faded grey color to avoid empty stretches
        if j ==0:
            ax.plot(time,data_in[:,1],color=colors[j],alpha=0.5,lw=3)
        
        # Plot the time segments corresponding to each index in ix 
        else:
            ax.plot(time,total,color=colors[j],label=labels[j],lw=4)
    
    # If elapsed time flag is on and there are more than 1 day, rotate the 
    # x-ticklabels, make more space for them and specify the day as well as the 
    # Hour and minute
    if (elapsed!=0) and (days>0):
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d - %H:%M"))
        ax.set_xlabel("Time, DD - HH:MM")
        ax.xaxis.labelpad = 20
        ax.set_xticklabels(ax.get_xticklabels(), rotation=-45, ha="left")
        plt.subplots_adjust(hspace=0.05,bottom = 0.25)
    # Otherwise just make the x-label and format the axis ticks
    else:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.set_xlabel("Time, HH:MM")
            
    # Make the legend, enable grid lines, set ylabel and set y-scale to log if specified
    ax.legend()
    ax.grid(which="both",axis="both")
    ax.set_ylabel("Concentration, #/cm$^{-3}$")
    if ylog == 1:
        ax.set_yscale("log")
    
    return fig, ax

###############################################################################
###############################################################################
###############################################################################

def Plot_timeseries(data_in, bin_edges, y_tot=(0,0), y_3d=(0,0), elapsed = 0, log = 1, datatype = "number"):
    """
    Function to plot both the total number number concentration and the number 
    concentrations measured in individual sizebins as a colored mesh plot.

    Parameters
    ----------
   data_in : numpy.array
       An array of data as returned by the Load_xxx functions with columns
       of datetime, total conc, and size bin data
    bin_edges : numpy.array
        Array containing the limits of all sizebins. The array should have one 
        more value than the length of the "data_in" parameter
    y_tot : tuple, optional
        Lower and upper limit on the y-axis of the total concentration plot
        given as (lower,upper). 
        The default is (0,0) in which case it will select a value automatically.
    y_3d : tuple, optional
        Lower and upper limit on the colorbar of the 3d timeseris plot
        given as (lower,upper). 
        The default is (0,0) in which case it will select a value automatically.
    elapsed : boolean, optional
        Flag to set time as elapsed time rather than local time. Set to 1 to
        switch to elapsed time. The default is 0.
    log : boolean, optional
        Flag to set whether to use log-scale on the 3D mesh colorbar scale. 
        Default is on
    datatype : string, optional
        Keyword to specify the datatype. The available options are "number",
        "normed" and "mass". Here "normed" refers to dN/dlogDp. Default is 
        "number"

    Returns
    -------
    fig : matplotlib.figure.Figure
        Handle for the returned figure for saving.
    axs : numpy.array
        Array with two handles, one for each of the two axis objects of the plot.

    """
    
    time = data_in[:,0]
    total = data_in[:,1].astype('float64')
    data = data_in[:,2:].astype('float64')
    
    # Generate canvas and axis
    fig, axs = plt.subplots(nrows=2,ncols=1, sharex=True)
    ax1,ax2 = axs
    
    # If elapsed time is active, adjust the datetime values to start at the new
    # year, so that the first displayed time and date is "1 00:00"
    if elapsed:
        fake_start = datetime.datetime(2023,1,1,0,0,0)
        delta_time = time-time[0]
        time = delta_time + fake_start
        
        total_time = time[-1]-time[0]
        days = total_time.days
    
    # Plot the total number concentration
    ax1.plot(time,total,lw=2,color="r")
    
    # Change y-scale, show grid, set y-label, and set y limits 
    ax1.grid(axis="both",which="both")
    tot_min = total.min()
    tot_max = total.max()
    if (y_tot[0] != 0) or (y_tot[1] != 0):
        if (y_tot[0] == 0) & (y_tot[1] != 0):
            ax1.set_ylim(tot_min*0.98,y_tot[1])
        elif (y_tot[0] != 0) & (y_tot[1] == 0):
            ax1.set_ylim(y_tot[0],tot_max*1.02)
        else:
            ax1.set_ylim(y_tot[0],y_tot[1])
    ax1.set_yscale("log")
    if datatype == "number":
        ax1.set_ylabel("N$_{Total}$, cm$^{-3}$")
    elif datatype == "mass":
        ax1.set_ylabel("m$_{Total}$, $\mu$g/m$^{3}$")
    elif datatype == "normed":
        ax1.set_ylabel("dN/dlogDp, cm$^{-3}$")
    ax1.set_xlabel("")
    
    # Generate an extra time bin, which is needed for the meshgrid
    dt = time[1]-time[0]
    time = time - dt
    time = np.append(time,time[-1]+dt)
    
    # generate 2d meshgrid for the x, y, and z data of the 3D color plot
    y, x = np.meshgrid(bin_edges, time)
    
    # Set the upper and/or lower limit of the color scale based on input
    if (y_3d[0] == 0) & (y_3d[1] != 0):
        y_3d_min = np.nanmin(data)
        y_3d_max = y_3d[1]
        
    elif (y_3d[0] != 0) & (y_3d[1] == 0):
        y_3d_min = y_3d[0]
        y_3d_max = np.nanmax(data)
        data[data<y_3d[0]]= y_3d[0]   
    elif (y_3d[0] != 0) or (y_3d[1] != 0):
        y_3d_min = y_3d[0]
        y_3d_max = y_3d[1]
        data[data<y_3d[0]]= y_3d[0]
    else:
        y_3d_min = np.nanmin(data)
        y_3d_max = np.nanmax(data)
        
    # Fill the generated mesh with particle concentration data
    if log:
        # Set datapoints smaller than 1 equal to 1 in order to avoid issues when 
        # plotting log transformed values
        if np.all(data):
            print("No zeros, continuing")
        else:
            print("""There is a 0 value in the dataset, meaning that log plotting is not possible.\nEither specify a lower limit or turn off log y-scale""")
            return [], []
        
        # Make the colormesh plot
        c = ax2.pcolormesh(x, y, data, cmap='jet',norm=LogNorm(vmin=y_3d_min, vmax=y_3d_max),shading='flat')
        
        # Set y-axis to log scale
        ax2.set_yscale("log")
    else:
        c = ax2.pcolormesh(x, y, data, cmap='jet',vmin=y_3d_min, vmax=y_3d_max,shading='flat')
    
    # Adjust axis labels, formats and spacing between plots
    if (elapsed!=0) and (days>0):
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d - %H:%M"))
        ax2.set_xlabel("Time, DD - HH:MM")
        ax2.xaxis.labelpad = 20
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=-45, ha="left")
        plt.subplots_adjust(hspace=0.05,bottom = 0.25)
    else:
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax2.set_xlabel("Time, HH:MM")
        plt.subplots_adjust(hspace=0.05)
        
    # Make the y-scal logarithmic and set a label
    ax2.set_yscale("log")
    ax2.set_ylabel("Dp, nm")
    
    # Insert coloarbar and label it
    col = fig.colorbar(c, ax=axs)
    if datatype == "number":
        col.set_label('dN, cm$^{-3}$')
    elif datatype == "mass":
        col.set_label('dm, $\mu$g/m$^{3}$')
    elif datatype == "normed":
        col.set_label('dN/dlogDp, cm$^{-3}$')
    
    # Set ticks on the plot to be longer
    ax1.tick_params(axis="y",which="both",direction='out', length=6, width=2)
    ax2.tick_params(axis="y",which="both",direction='out', length=6, width=2)
    
    return fig,axs

###############################################################################
###############################################################################
###############################################################################

def Plot_timeseries_multiple(data_in, bin_edges, y_tot=(0,0), y_3d=(0,0), log = 1,sharex=0):
    """
    Function to compare several datafiles generated by instruments with size
    distribution capabilities. Comparison is made between the total number number 
    concentration and the number concentrations measured in individual sizebins. 
    
    Parameters
    ----------
    data_in : list
        List of numpy.arrays as returned by the Load_xxx functions with columns
        of datetime, total conc, and size bin data
    bin_edges : numpy.array
        Array containing the limits of all sizebins. The array should have one 
        more value than the length of the arrays in the "data_in" parameter. 
        The sizebins have to be common for the compared datafiles.
    y_tot : tuple, optional
        Lower and upper limit on the y-axis of the total concentration plot
        given as (lower,upper). 
        The default is (0,0) in which case it will select a value automatically.
    y_3d : tuple, optional
        Lower and upper limit on the colorbar of the 3d timeseris plot
        given as (lower,upper). 
        The default is (0,0) in which case it will select a value automatically.
    log : boolen, optinal
        Flag to set whether the y-scale should be plotted as logarithmic. Set to
        1 to turn it on and 0 for off. Default is 1.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Handle for the returned figure for saving.
    axs : numpy.array
        Array with handles, one for each of the axes objects of the plot.

    """
    # Determine number of datasets given to the function
    Number_of_datasets = len(data_in)
    
    time = [i[:,0] for i in data_in]
    total = [i[:,1].astype('float64') for i in data_in]
    data = [i[:,2:].astype('float64') for i in data_in]
    
    time_plus = []
    for i in range(Number_of_datasets):
        # Generate an extra time bin, which is needed for the meshgrid
        dt = time[i][2]-time[i][1]
        time_adjust = time[i] - dt/2 
        extra = time_adjust[-1]+dt
        time_plus += [np.hstack((time_adjust,extra))]
        
    # Determine the lowest and highest particle concentration out of all datasets
    # for the colorbar and total number concentration plot limits
    minimum = np.array([i.min() for i in data]).min()
    maximum = np.array([i.max() for i in data]).max()
    
    tot_min = np.array([i.min() for i in total]).min()
    tot_max = np.array([i.max() for i in total]).max()
    
    # Determine the y-limits of the total concentration plot
    if (y_tot[0] == 0) & (y_tot[1] != 0):
        y_totlimlow = tot_min*0.98
        y_totlimhigh = y_tot[1]
        
    elif (y_tot[0] != 0) & (y_tot[1] == 0):
        y_totlimlow = y_tot[0]
        y_totlimhigh = tot_max*1.02
    
    elif (y_tot[0] != 0) or (y_tot[1] != 0):
        y_totlimlow = y_tot[0]
        y_totlimhigh = y_tot[1]
    
    else:
        y_totlimlow = tot_min*0.98
        y_totlimhigh = tot_max*1.02
        
    
    # Generate canvas and axis depending on number of datasets
    if sharex:
        fig, axs = plt.subplots(nrows=Number_of_datasets,ncols=2, sharey='col',sharex="col")
    else:
        fig, axs = plt.subplots(nrows=Number_of_datasets,ncols=2, sharey='col')
    if Number_of_datasets == 2:
        fig.subplots_adjust(hspace=0.5,wspace=0.3)
    elif Number_of_datasets == 3:
        fig.subplots_adjust(hspace=0.5,wspace=0.3)
    elif Number_of_datasets == 4:
        fig.subplots_adjust(hspace=0.7,wspace=0.3)
        
    # Run through all datasets
    for i in range(Number_of_datasets):
        
        # Store current dataset in a temporary variable
        data_temp = data[i]
        
        # Plot the total number concentration
        axs[i,0].plot(time[i],total[i],lw=2,color="r")
                
        # Enable grid lines, specify x and y-label, format the axis ticklabels
        # and set titles
        axs[i,0].grid(axis="both",which="both")
        axs[i,0].set_ylabel("N$_{Total}$, cm$^{-3}$",fontsize=15)
        axs[i,0].set_xlabel("")
        axs[i,0].xaxis.set_major_formatter(mdates.DateFormatter("%H.%M"))
        axs[i,0].set_title("Instrument {0}".format(i+1),fontsize=15)
        
        # generate a 2d mesh from the particle size bins and the instrument times
        y, x = np.meshgrid(bin_edges, time_plus[i])
        
        # Set the upper and/or lower limit of the color scale based on input
        if (y_3d[0] == 0) & (y_3d[1] != 0):
            y_3d_min = minimum
            y_3d_max = y_3d[1]
            
        elif (y_3d[0] != 0) & (y_3d[1] == 0):
            y_3d_min = y_3d[0]
            y_3d_max = maximum
            data_temp[data_temp<y_3d[0]]= y_3d[0]   
            
        elif (y_3d[0] != 0) or (y_3d[1] != 0):
            y_3d_min = y_3d[0]
            y_3d_max = y_3d[1]
            data_temp[data_temp<y_3d[0]]= y_3d[0]
            
        else:
            y_3d_min = minimum
            y_3d_max = maximum
            
        # Fill the generated mesh with particle concentration data
        if log:
            # Set datapoints smaller than 1 equal to 1 in order to avoid issues when 
            # plotting log transformed values
            if np.all(data_temp):
                print("No zeros, continuing")
            else:
                print("""There is a 0 value in the dataset, meaning that log plotting is not possible.\nEither specify a lower limit or turn off log y-scale""")
                return [], []
            
            # Make the colormesh plot
            c = axs[i,1].pcolormesh(x, y, data_temp, cmap='jet',norm=LogNorm(vmin=y_3d_min, vmax=y_3d_max),shading='flat')
            
            # Set y-axis to log scale
            axs[i,1].set_yscale("log")
            axs[i,0].set_yscale("log")
        else:
            c = axs[i,1].pcolormesh(x, y, data_temp,vmin=y_3d_min, vmax=y_3d_max, cmap='jet',shading='flat')
            
        # Format axis ticklabels, set log scale, set y and x labels
        axs[i,1].axis([x.min(), x.max(), y.min(), y.max()])
        axs[i,1].xaxis.set_major_formatter(mdates.DateFormatter("%H.%M"))
        axs[i,1].set_xlabel("")
        axs[i,1].set_ylabel("Dp, nm",fontsize=15)
        axs[i,1].set_title("Instrument {0}".format(i+1),fontsize=15)
    
    # Produce colorbar and set labels
    col = fig.colorbar(c, ax=axs)
    col.set_label('dN/dlogDp, cm$^{-3}$',fontsize=15)
    axs[-1,0].set_xlabel("Time, HH:MM",fontsize=15)
    axs[-1,1].set_xlabel("Time, HH:MM",fontsize=15)  
    axs[0,0].set_ylim(y_totlimlow,y_totlimhigh)
    
    # Set ticks on the plot to be longer and reduce fontsize if many plots are generated
    for i in axs.flatten():
        i.tick_params(axis="y",which="both",direction='out', length=6, width=2)
        
        
        if Number_of_datasets == 2:
            i.tick_params(axis='both', which='major', labelsize=15)
        elif Number_of_datasets == 3:
            i.tick_params(axis='both', which='major', labelsize=15)
        else:
            i.tick_params(axis='both', which='major', labelsize=10)
    
    return fig,axs

###############################################################################
###############################################################################
###############################################################################

def Plot_totalconc(data_in,log=0,elapsed=0,y_lim=(0,0)):
    """
    Function to plot a single data series exported from e.g. a TSI CPC, nanoscan
    OPS, or similar

    Parameters
    ----------
    data_in : numpy.array
        Array as returned by Load_xxx with a column of datetimes and one with
        particle number concentrations. The data_in can also contain size bin 
        data but these will not be used with this plot.
    log : boolean
        Flag to set the y-axis to log-scale. Set to 1 to turn it on. Default is 0 
    elapsed : boolean, optional
        Flag to set time as elapsed time rather than local time. Set to 1 to
        switch to elapsed time. The default is 0.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Handle for the returned figure for saving.
    ax : matplotlib.axes._subplots.AxesSubplot
        Handles for the axis of the plot.

    """
    
    time = data_in[:,0]
    data = data_in[:,1].astype('float64')
    
    # Generate canvas and figure axis
    fig,ax = plt.subplots()
    
    # If elapsed time is active, adjust the datetime values to start at the new
    # year, so that the first displayed time and date is "1 00:00"
    if elapsed:
        fake_start = datetime.datetime(2023,1,1,0,0,0)
        delta_time = time-time[0]
        time = delta_time + fake_start
        
        total_time = time[-1]-time[0]
        days = total_time.days
        
    if log:
        # Set datapoint below 1 to be 1 in order to avoid issues when plotting log transformed values
        data1 = data.copy()
        data1[data1<1]=1
        
        # Plot the data
        ax.plot(time,data1)
        
        # Set the y-axis to logarithmic and adjust limits
        ax.set_ylim(np.nanmin(data1)*0.98,np.nanmax(data1)*1.02)
        ax.set_yscale("log")
    else:
        # Adjust limits
        ax.set_ylim(0,np.nanmax(data)*1.02)
        
        # Plot the data
        ax.plot(time,data)
    
    # Set the limits of the x axis
    ax.set_xlim(time.min(),time.max())
    
    dat_min = data.min()
    dat_max = data.max()
    if (y_lim[0] != 0) or (y_lim[1] != 0):
        if (y_lim[0] == 0) & (y_lim[1] != 0):
            ax.set_ylim(dat_min*0.98,y_lim[1])
        elif (y_lim[0] != 0) & (y_lim[1] == 0):
            ax.set_ylim(y_lim[0],dat_max*1.02)
        else:
            ax.set_ylim(y_lim[0],y_lim[1])
            
    # Adjust format, labels, spacings, and xticks
    if (elapsed!=0) and (days>0):
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d - %H:%M"))
        ax.set_xlabel("Time, DD - HH:MM")
        ax.xaxis.labelpad = 20
        ax.set_xticklabels(ax.get_xticklabels(), rotation=-45, ha="left")
        plt.subplots_adjust(hspace=0.05,bottom = 0.25)
    else:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.set_xlabel("Time, HH:MM")
    
    # Define y-axis labels and enable grid lines
    ax.set_ylabel("Concentration, #/cm$^{-3}$")
    ax.grid(axis="both",which="both")
    
    return fig, ax

###############################################################################
###############################################################################
###############################################################################

def Plot_totalconc_multiple(data_in,labels,log=0,elapsed=0):
    """
    Function to plot the total concentration of several data series. The datasets
    will be plotted on the same figure for direct comparison.

    Parameters
    ----------
    data_in : list
        List of numpy.arrays as returned by the Load_xxx function with columns
        of datetime and particle conc
    labels : list
        List of labels to use in the plot legend. The number of labels has to
        match the number of datasets.
    log : boolean
        Flag to set the y-axis to log-scale. Set to 1 to turn it on. Default is 0 
    elapsed : boolean, optional
        Flag to set time as elapsed time rather than local time. Set to 1 to
        switch to elapsed time. The default is 0.
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        Handle for the returned figure for saving.
    ax : matplotlib.axes._subplots.AxesSubplot
        Handles for the axis of the plot.

    """
    # Determine number of datasets given to the function
    Number_of_datasets = len(data_in)
    
    time = [i[:,0] for i in data_in]
    data = [i[:,1].astype('float64') for i in data_in]
    
    
    # Generate canvas and figure axis
    fig,ax = plt.subplots()
    
    # If elapsed time is active, adjust the datetime values to start at the new
    # year, so that the first displayed time and date is "1 00:00"
    if elapsed:
        fake_start = datetime.datetime(2023,1,1,0,0,0)
        days = []
        for i in range(Number_of_datasets):
        
            delta_time = time[i]-time[i].min()
            time[i] = delta_time + fake_start
            
            total_time = time[i].max()-time[i].min()
            days += [total_time.days]
    
    # empty lists to store the minimum of each dataset
    minimum = []
    maximum = []
    
    # Set the y-axis to logarithmic if specified and set y-limits
    if log:
        for i in range(Number_of_datasets):
            data[i][data[i]<1]=1
            
            minimum += [np.nanmin(data[i])]
            maximum += [np.nanmax(data[i])]
       
            # Run through all datasets and plot them
            ax.plot(time[i],data[i], label = labels[i])
                
        # Set y-limits and set the y-scale to logarithmic
        ax.set_ylim(min(minimum)*0.98,max(maximum)*1.02)
        ax.set_yscale("log")
    else:
        # Run through all datasets and plot them
        for i in range(Number_of_datasets):
            minimum += [np.nanmin(data[i])]
            maximum += [np.nanmax(data[i])]
            
            ax.plot(time[i],data[i], label = labels[i])
      
        ax.set_ylim(min(minimum)*0.98,max(maximum)*1.02)
    
    # Adjust format, labels, spacings, and xticks
    if (elapsed!=0) and (max(days)>0):
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d - %H:%M"))
        ax.set_xlabel("Time, DD - HH:MM")
        ax.xaxis.labelpad = 20
        ax.set_xticklabels(ax.get_xticklabels(), rotation=-45, ha="left")
        plt.subplots_adjust(hspace=0.05,bottom = 0.25)
    else:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        ax.set_xlabel("Time, HH:MM:SS")
        
    # Define y-axis label, enable grid lines, and generate legend
    ax.set_ylabel("Concentration, #/cm$^{-3}$")
    ax.grid(axis="both",which="both")
    ax.legend()
    
    return fig, ax

###############################################################################
###############################################################################
###############################################################################
  
def Plot_PSD_different_instruments(*data_in, labels=None, ylog=True, xlog=True, y_lim=(0, 0), datatype="number"):
    """
    Similar to Plot_PSD only this function can plot the PSDs for multiple instruments
    that have different bin mids. 

    Parameters
    ----------
    *data_in : list of tuples
        Each tuple contains (bin_mids, size distribution data).
    labels : list, optional
        List of labels for the plots. The default is None.
    ylog, xlog : bool, optional
        Flags to turn on/off log scales for y-axis and x-axis. Defaults are True.
    y_lim : tuple, optional
        Limits for the y-axis. Defaults to (0, 0), which means auto-scaling.
    datatype : str, optional
        The type of data: 'number', 'normed', or 'mass'. Defaults to 'number'.

    Returns
    -------
    fig, ax : matplotlib figure and axes objects.
    
    Example
    
    fig, ax = Plot_PSD_1((data['bin_mids_NS'], data['smps_Lab']), 
                       (data['bin_mids_FMPS'], data['fmps_Lab']), 
                       labels=["Data 1", "Data 2"], ylog=True, xlog=True)
    
    Created by PLF
    
    """
    colors = ["red", "blue", "green", "orange", "magenta", "cyan", "k", "purple", "yellow", 'pink']

    fig, ax = plt.subplots()
    for idx, (bin_mids, dataset) in enumerate(data_in):
        particle_data = dataset[:, 2:].astype("float")
        mean_psd = particle_data.mean(axis=0)
        sem_psd = sem(particle_data, axis=0)
        
        color = colors[idx % len(colors)]
        label = labels[idx] if labels and idx < len(labels) else None

        ax.plot(bin_mids, mean_psd, label=label, lw=3, color=color)
        ax.fill_between(bin_mids, mean_psd - sem_psd, mean_psd + sem_psd, alpha=0.5, color=color)

    # Set axis scales and labels
    if xlog:
        ax.set_xscale("log")
    if ylog:
        ax.set_yscale("log")
    ax.grid(True, which="both")

    y_label = "dN/dlogDp, cm$^{-3}$" if datatype == "normed" else "dN, cm$^{-3}$" if datatype == "number" else "dm, $\mu$g$^{-3}$"
    ax.set_ylabel(y_label)
    ax.set_xlabel("Dp, nm")

    # Adjust y-axis limits if specified
    if y_lim != (0, 0):
        ax.set_ylim(y_lim)

    if labels:
        ax.legend()

    return fig, ax
