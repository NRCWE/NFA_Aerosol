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
from matplotlib.collections import LineCollection, PathCollection
import matplotlib.ticker as ticker
from scipy.stats import sem, theilslopes
from scipy.optimize import curve_fit
import datetime as datetime
import Utility_Lib as UL

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
  
def Boxplot_PM(data_in,labels,y_lim=(0,0),tick_dist = 5,ax_in=None):
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
    ax_in : matplotlib.axes._axes.Axes
        If an axis is given, then the plot will be made using that handle, rather
        than generating a new figure. This is relevant in cases where subplots
        are being used.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Handle for the returned figure for saving.
    ax : matplotlib.axes._subplots.AxesSubplot
        Handle for the axis object of the plot.

    """
    if ax_in == None:
        fig, ax = plt.subplots()
    else:
        ax = ax_in
    data = data_in[:,1:]
    
    
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
    
    return ax.figure,ax

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
def Plot_APS_correlation(data,y_3d=(1E-3,0),log_3d=1,ax_in=None):
    """
    Plot for returning the correlated plot between aerodynamic and optical 
    size distribution for the APS.
    The data can be loaded with either a single set of data or a timeseries,
    in which case the average will be plotted.

    Parameters
    ----------
    data : np.array
        Array as returned by the IL.Load_APS_full function set to correlation.
        The array can either be a single time instance as [time,total,cor_array]
        or as the extracted cor_array, or it can span time by returning the average
        cor_array over the inserted time period.
    y_3d : tuple, optional
        Lower and upper limit on the colorbar of the 3d timeseris plot
        given as (lower,upper). 
        The default is (1E-3,0) in which case it will select a max value automatically, while
        ensuring a lower limit of 1E-3.
    log_3d : boolean, optional
        Flag to set whether to use log-scale on the 3D mesh colorbar scale. 
        Default is on
    ax_in : matplotlib.axes._axes.Axes
        If an axis is given, then the plot will be made using that handle, rather
        than generating a new figure. This is relevant in cases where subplots
        are being used.
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        Handle for the returned figure for saving.
    ax : matplotlib.axes._subplots.AxesSubplot
        Handle for the axis object of the plot.

    """
    
    #Bin_edges for Optic and Aerodynamic bins
    Optic_bins=[370,475.58,611.28,785.7,1009.89,1298.06,1668.44,2144.52,2756.44,3542.96,
                4553.91,5853.33,7523.53,9670.3,12429.6,15976.3,20535]
    Aero_bins=[523,568,604.5,649.5,698,750,806,866.5,931.5,1001,1075.5,1155.5,1241.5,
               1334,1434,1541,1655.5,1779,1912,2055,2208.5,2373,2550,2740.5,2945,
               3164.5,3400.5,3654.5,3927,4219.5,4534.5,4873,5236.5,5627,6046.5,
               6498,6983,7504,8064,8665.5,9312,10008.5,10755,11555,12415,
               13340,14340,15410,16555,17790,19120,20535]

    #Determines the shape of the data returned
    if len(data.shape)==1:
        Data=data[2].copy()
    #If the data spans multiple rows of times, the average will be plotted.
    elif data.shape[1]==3:
        #If the data is 
        Data=data[0,2].copy()
        for A in range(0,len(data[0,2][0,:])):
            for O in range(0,len(data[0,2][:,0])):
                avg=[]
                for t in range(0,len(data)):
                    avg=np.append(avg,data[t,2][O,A])
                Data[O,A]=np.mean(avg)
    elif data.shape[1]==52:
        Data=data.copy()
        
    # Set the upper and/or lower limit of the color scale based on input
    if (y_3d[0] == 0) & (y_3d[1] != 0):
        y_3d_min = np.nanmin(Data)
        y_3d_max = y_3d[1]
        
    elif (y_3d[0] != 0) & (y_3d[1] == 0):
        y_3d_min = y_3d[0]
        y_3d_max = np.nanmax(Data)
        Data[Data<y_3d[0]]= y_3d[0]   
    elif (y_3d[0] != 0) or (y_3d[1] != 0):
        y_3d_min = y_3d[0]
        y_3d_max = y_3d[1]
        Data[Data<y_3d[0]]= y_3d[0]
    else:
        y_3d_min = np.nanmin(Data)
        y_3d_max = np.nanmax(Data)
        
    #Plot colormesh of peak
    if ax_in == None:
        fig, ax = plt.subplots()
    else:
        ax = ax_in
    
    if log_3d:
        # Set datapoints smaller than 1 equal to 1 in order to avoid issues when 
        # plotting log transformed values
        if np.all(Data):
            print("No zeros, continuing")
        else:
            print("""There is a 0 value in the dataset, meaning that log plotting is not possible.\nEither specify a lower limit or turn off log y-scale""")
            return [], []
        
        # Make the colormesh plot
        c = ax.pcolormesh(Aero_bins,Optic_bins, Data, cmap='jet',norm=LogNorm(vmin=y_3d_min, vmax=y_3d_max),shading='flat')
        
    else:
        c = ax.pcolormesh(Aero_bins,Optic_bins, Data, cmap='jet',vmin=y_3d_min, vmax=y_3d_max,shading='flat')
    
    #Plot the location of the points of interest 
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylabel('Optic Dp (nm)')
    ax.set_xlabel('Aerodynamic Dp (nm)')
    col = fig.colorbar(c)
    col.set_label('dN, cm$^{-3}$')
    fig.set_size_inches(15, 15)
    plt.show()
    if ax_in == None:
        return fig, ax
    else:
        return ax.fig, ax
###############################################################################

def Plot_correlation(X, Y, ax_in=False, intercept=True, uniform_scaling=True, outlier_influence=True):
    """
    Function to plot the correlation between two sets of values, which have been
    aligned so as to have sensible comparison points. 
    X and Y must have the same length. This can be accomplished by using the
    averaging function to generate time associated data of same dimensions. 
       
    Parameters
    ----------
    X: Numpy.array
        First set of values. 
    Y: Numpy.array
        Second set of values.  
    ax_in : matplotlib.axes._subplots.AxesSubplot
        Handles for the axis of the plot.
        Usefull for plotting multiple correlations in the same figure.
        
        Example:
        '''
        df: dataframe of values from different instruments with associated label
        instruments: list of instruments used for comparison
        
        fig, axes = plt.subplots(len(instruments)-1, len(instruments)-1)

        # Fill the grid with custom scatter plots, or leave empty for unwanted pairs
        for i in range(len(instruments)-1):  # Loop over instruments 0:-1 for the horizontal axis
            for j in range(i,len(instruments)-1):  # Loop over instruments 1: for the vertical axis
                if i == j+1:  # Skip diagonal (Instrument 2 vs Instrument 2, etc.)
                    axes[j, i].axis('off')
                else:  # Use custom scatter plot function
                    _, _ = UL.cor_plot(df[instruments[i]], df[instruments[j+1]], axes[j, i],intercept=False)
                    if i==0:
                        axes[j, i].set_ylabel(instruments[j+1])
                    if j==len(instruments)-2:
                        axes[j, i].set_xlabel(instruments[i])
         '''                 
    uniform_scaling: boolean, optional
        Boolean that can be turned off so as to not scale axis the max value.
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        Handle for the returned figure for saving.
    ax : matplotlib.axes._subplots.AxesSubplot
        Handles for the axis of the plot.
        """
        
    #Defining relevant sub-functions for the function to work
    def linear_func(x,A,B=0):
        #Calculates a first order equation.
        return A*x + B
    
    def R2(data,fit):
        # residual sum of squares
        ss_res = np.sum((data - fit) ** 2)
        # total sum of squares
        ss_tot = np.sum((data - np.mean(data)) ** 2)
        # r-squared
        return round((1 - (ss_res / ss_tot)),3)
    
    #Cleaning up the data and removing rows where either value is nan
    z=np.column_stack((X.copy(),Y.copy())).astype('float64')
    z=z[~np.isnan(z).any(axis=1)]
    z=z[~np.isinf(z).any(axis=1)]
    x=z[:,0]
    y=z[:,1]
    
    if type(x[0])==datetime.datetime:
        x=np.array(mdates.date2num(x[:]))
    
    # This method works well but it's sensitive towards outliers
    if outlier_influence:
        #Apply the fit using curve_fit for a function with or without an intercept.
        if intercept==True:
            parameters, covariance =curve_fit(linear_func,x,y,p0=[1, 1])
            A, B = parameters
            SE = np.sqrt(np.diag(covariance))
            SE_A , SE_B = SE
        
        else:
            parameters, covarience =curve_fit(linear_func,x,y,p0=[1])
            A=parameters[0]
            SE_A=covarience[0][0]
            B=0
            SE_B=0
    else:
        # This method is less sensitive towards 
        A, B, _, _ = theilslopes(y,x)
        
    #Generate the fit and calculate the R^2 value
    fit=linear_func(x,A,B)
    r2=R2(y,fit)

    if uniform_scaling==True:
        factor=max(max(abs(x)),max(abs(y)))
    else: 
        factor=1
    if min(x)<=0:
        x_min=min(x)/factor
    else:
        x_min=0
    if max(x)<=0:
        x_max=0
    else: 
        x_max= z.max()/factor
    
    fit_x=np.linspace(x_min,x_max,20)
    fit_y=fit_x*A+B/factor
    if B==0:
        label=f"y={round(A,2)}$\cdot$x"#"y={:.2f}$\cdot$x+{:.2f}, R$^2$={:.2f}"
    elif B>0:
        label=f"y={round(A,2)}$\cdot$x \n + {round(B,2)}"#"y={:.2f}$\cdot$x+{:.2f}, R$^2$={:.2f}"
        
    else:
        label=f"y={round(A,2)}$\cdot$x \n {round(B,2)}"#"y={:.2f}$\cdot$x+{:.2f}, R$^2$={
    #If no ax is provided, figure and ax is generated here
    if ax_in==False:
        figure, ax = plt.subplots()
        plt.xticks(fontsize=25)  
        plt.yticks(fontsize=25) 
        ax.legend(fontsize=25)
        ax.grid(True)
    else:
        ax = ax_in
    #Plot the 1:1 line
    ax.plot([x_min,x_max],[x_min,x_max],ls="--",c="k",lw=3)
    #Plot the data with scatter plot
    ax.plot(x/factor,y/factor,'bo')
    #Plot the fit with associated uncertainty
    ax.plot(fit_x, fit_y, 'r-',lw=3)#, label=label.format(A, B, r2))
    ax.text(0.05, 0.95, label, transform=ax.transAxes, fontsize=25,
        verticalalignment='top')
    if B==0:
        ax.text(0.05, 0.80, f"r$^2$: {round(r2,2)}", transform=ax.transAxes, fontsize=25,
        verticalalignment='top')
    else:
        ax.text(0.05, 0.65, f"r$^2$: {round(r2,2)}", transform=ax.transAxes, fontsize=25,
        verticalalignment='top')
        
    if outlier_influence:
        ax.fill_between(fit_x, fit_y - ((SE_A*fit_x)**2+(SE_B/factor)**2)**0.5, fit_y + ((SE_A*fit_x)**2+(SE_B/factor)**2)**0.5, alpha=0.33)
    return ax.figure,ax

###############################################################################
def Plot_correlation_df(df, fig_text="", *Plotsettings):
    """
    Function to plot multiple correlation plots together in a n*n grid, where n is the number of instruments minus 1.
    X and Y must have the same length. This can be accomplished by using the
    averaging function to generate time associated data of same dimensions. 
       
    Parameters
    ----------
    df: dataframe
        dataframe of the instruments structured with an instrument per column,
        with instrument handle equal to 
        
    fig_text: str, optional
        Second set of values.  
        
    *Plotsettings: list of input to 
        Input to the Plot_calibration function:
            intercept=True, uniform_scaling=True, outlier_influence=True
        Call on this
    unifomr_scaling: boolean, optional
        Boolean that can be turned off so as to not scale axis the max value.
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        Handle for the returned figure for saving.
    gs : matplotlib.gridspace._subplots.AxesSubplot
        Handles for the gridspace of the plot.
        """
        
    instruments = list(df.columns)
    n=0
    
    if len(instruments)==2:
        fig, ax= Plot_correlation(df[instruments[0]], df[instruments[1]], *Plotsettings)#
        fig.figsize=(18, 18)
        ax.set_ylabel(instruments[1])
        ax.set_xlabel(instruments[0])
    else:
        n = len(instruments) - 1
        
        fig = plt.figure(figsize=(18, 18), constrained_layout=True)
        gs = fig.add_gridspec(n, n)
        
        axes = {}
        # Create only the subplots you need (upper triangle incl. diagonal)
        for i in range(n):
            for j in range(i, n):
                ax = fig.add_subplot(gs[j, i])    # row=j, col=i
                axes[(j, i)] = ax
        
                # Your custom correlation plot
                try:
                    _, _ = Plot_correlation(df[instruments[i]], df[instruments[j+1]], ax, *Plotsettings)#intercept=intercept)
                except Exception:
                    pass
        
                if i == 0:
                    ax.set_ylabel(instruments[j+1])
                if j == n - 1:
                    ax.set_xlabel(instruments[i])
    if len(fig_text)>0:
        # One description box spanning top row, columns 2..n (i.e., gs[0, 1:])
        if n > 1:
            desc_ax = fig.add_subplot(gs[0, 1:])  # uses only empty cells
            desc_ax.axis('off')
            desc_ax.text(
                0.0, 0.5,
                fig_text,
                ha='left', va='center', fontsize=40, wrap=True,
                bbox=dict(boxstyle="round,pad=0.6", fc="whitesmoke", ec="0.5", lw=1),
                transform=desc_ax.transAxes
            )
        else:
            # Fallback if there's only one column in the grid
            ax.text(
                0.5, 0.90,
                fig_text,
                ha='center', va='top',transform=ax.transAxes, fontsize=40
            )
    plt.tight_layout()
    if n==0:
        return fig, ax
    else:
        return fig, gs

###############################################################################
def rounder(mean,sem):
    try:
        # sem=Sem
        # mean=Mean
        Multiplier=0
        if sem>10:
            while sem>10:
                Multiplier+=1
                sem/=10
                mean/=10
        elif sem<1:
            while sem<1:
                Multiplier-=1
                sem*=10
                mean*=10
    
        if str(sem)[0]=='1':
            floater=3
            rounder=1
        else:
            floater=1
            rounder=0
    
        sem=str(round(sem,rounder))[:1+floater]
        mean=str(round(mean,rounder))
        for i in range(0,len(mean)):
            if mean[i]=='.':
                break
        mean=mean[:i+floater]
        
        if (Multiplier<3) & (Multiplier>=-2):
            mean=str(float(mean)*10**(Multiplier))
            sem=str(float(sem)*10**(Multiplier))
            for m in range(0,len(mean)):
                if mean[m]=='.':
                    break
            for s in range(0,len(sem)):
                if sem[s]=='.':
                    break
            if Multiplier>0:
                Multiplier=0
            mean=mean[:m-Multiplier+floater]
            sem=sem[:s-Multiplier+floater]
            return  f"{mean}+/-{sem}"
        #f"{float(mean)*10**(Multiplier)}+/-{float(sem)*10**(Multiplier)}"
        
        else: return f"{mean}+/-{sem} E{Multiplier}"
    except: return f"{mean}+/-{sem}"
###############################################################################
###############################################################################

def Plot_PM_timeseries(data, bin_edges, PM_values=[0.5,2.5,10],Fraction=False,
                       datatype="number", ax_in=False, colors=False, cummulative=False):
    """
    Parameters
    ----------
    data : numpy array
        An array of data as returned by the Load_xxx functions with columns
        of datetime, total conc, and size bin data. the data can be either mass, 
    bin_edges : numpy.array
        Array containing the limits of all sizebins. 
    PM_values : list, optional
        A list of Dp values for which PM should be plotted.
        The default is [0.5,2.5,10].
    Fraction : boolean
        Decides whether the PM should be plotted as values, or as fraction of the biggest PM value.
        By default the fraction is off.
    datatype : string, optional
        Keyword to specify the datatype. The available options are "number",
        "surface" and "mass". Default is "number"
    ax_in : matplotlib.axes._subplots.AxesSubplot, Optional
        Handles for the axis of the plot.
        Usefull for plotting multiple correlations in the same figure.
    colors : list, optional
        List of colors for the plots. A default list is provided below.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Handle for the returned figure for saving.
    ax : matplotlib.axes._subplots.AxesSubplot
        Handles for the axis of the plot.

    """
    #Set up standardized values for later use
    if colors==False:
        colors = ["brown","chocolate","darkorange","gold","olive","darkgreen","teal","deepskyblue","darkblue"]
    
    y_label={"Number"   :   "N$_{Total}$, cm$^{-3}$",
             "Surface"  :   "S$_{Total}$, nm$^{2}$ cm$^{-3}$",
             "Mass"     :   "m$_{Total}$, $\mu$g/m$^{3}$"}
    Legend_label={'Number': 'PN',
                  'Surface':'PS',
                  'Mass':   'PM'}
    
    #Corrects for the first letter not being capital in the datatype
    if datatype[0]=="n":
        datatype="N"+datatype[1:]
    elif datatype[0]=="s":
        datatype="S"+datatype[1:]
    elif datatype[0]=="m":
        datatype="M"+datatype[1:]
 
    data=data.copy()
    
    #Generate a dictionary of the lists of PM values
    PM={}
    for i in range(0,len(PM_values)):
        pm=str(PM_values[i])
        PM[pm]=UL.PM_calc(data, bin_edges, PM_values[i])[0][:,1].astype('float')
                   
    #If no ax is provided, figure and ax is generated here
    if ax_in==False:
        figure, ax = plt.subplots()
        plt.xticks(fontsize=25)  
        plt.yticks(fontsize=25)         
    else:
        ax = ax_in
        
    """  
    Determines the type of plot. The default is a plot with total concentration
    and the total concentration each pn/pm values reases.
    If the Fraction has been turned on, the fractional values will be calculated instead
    """
    if Fraction==True:
        ax.plot(data[:,0],data[:,1],color='k',label='Total',lw=3)
        ax2 = ax.twinx()
        PM_fr=PM.copy()
        for i in range(0,len(PM_values)):
            pm=str(PM_values[i])
            


            PM_fr[pm]=PM[pm]/PM[str(PM_values[-1])]
            if i == 0:
                sem_PM = np.nanstd(PM[pm], axis=0)
                # sem_PM = sem(PM[pm], axis=0,nan_policy="omit")
                Label=f"{Legend_label[datatype]}{pm}: {rounder(np.nanmean(PM[pm]),sem_PM)}"
                ax2.fill_between(data[:,0], PM_fr[pm],alpha=0.75,color=colors[i],label=Label)
                
            else:
                if cummulative==True:
                    sem_PM = np.nanstd(PM[pm], axis=0)
                    # sem_PM = sem(PM[pm], axis=0,nan_policy="omit")
                    Label=f"{Legend_label[datatype]}{pm}: {rounder(np.nanmean(PM[pm]),sem_PM)}"
                else:
                    sem_PM = np.nanstd(PM[pm]-PM[str(PM_values[i-1])], axis=0)
                    # sem_PM = sem(PM[pm]-PM[str(PM_values[i-1])], axis=0,nan_policy="omit")
                    Label=f"{Legend_label[datatype]}{pm}: {rounder(np.nanmean(PM[pm]-PM[str(PM_values[i-1])]),sem_PM)}"
                ax2.fill_between(data[:,0], PM_fr[str(PM_values[i-1])], PM_fr[pm],alpha=0.75,color=colors[i],label=Label)
        
        ax2.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
        ax2.set_ylim(0,1)
        ax2.set_ylabel(f"{datatype} fraction")
        ax2.legend(loc='best',title="Average values",fontsize=25,title_fontsize=25)

    else:
        for i in range(0,len(PM_values)):
            pm=str(PM_values[i])
            # sem_PM = sem(PM[pm], axis=0,nan_policy="omit")
            # Label=f"{pm} $\mu$m: {rounder(np.nanmean(PM[pm]),sem_PM)}"
            if i == 0:
                sem_PM = np.nanstd(PM[pm], axis=0)
                # sem_PM = sem(PM[pm], axis=0,nan_policy="omit")
                Label=f"{Legend_label[datatype]}{pm}: {rounder(np.nanmean(PM[pm]),sem_PM)}"
                ax.fill_between(data[:,0], PM[pm],alpha=1,color=colors[i],label=Label)
            else:
                if cummulative==True:
                    sem_PM = np.nanstd(PM[pm], axis=0)
                    # sem_PM = sem(PM[pm], axis=0,nan_policy="omit")
                    Label=f"{Legend_label[datatype]}{pm}: {rounder(np.nanmean(PM[pm]),sem_PM)}"
                else:
                    sem_PM = np.nanstd(PM[pm]-PM[str(PM_values[i-1])], axis=0)
                    # sem_PM = sem(PM[pm]-PM[str(PM_values[i-1])], axis=0,nan_policy="omit")
                    Label=f"{Legend_label[datatype]}{pm}: {rounder(np.nanmean(PM[pm]-PM[str(PM_values[i-1])]),sem_PM)}"
                ax.fill_between(data[:,0], PM[str(PM_values[i-1])], PM[pm],color=colors[i],label=Label)
        ax.legend(loc='best',title="Average values",fontsize=25,title_fontsize=25)

    #Set the axis settings    
    
    ax.set_ylim(0)
    ax.set_ylabel(y_label[datatype])
    
            
    #Checks wether the data spans more than two days to use suitable x-label and ticks
    if data[-1,0]-data[0,0]>datetime.timedelta(2,0):
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m %H:%M"))# - %H:%M
        ax.set_xlabel("Time, DD/MM - HH:MM")
        ax.xaxis.labelpad = 20
        ax.set_xticklabels(ax.get_xticklabels(), rotation=-45, ha="left")
        plt.subplots_adjust(hspace=0.05,bottom = 0.25)
    # Otherwise just make the x-label and format the axis ticks
    else:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.set_xlabel("Time, HH:MM")
    

    
    return ax.figure,ax

###############################################################################
###############################################################################
###############################################################################

def Plot_PSD(*data_in, labels=None, colors=None, linestyles=None, ylog=True, xlog=True, y_lim=(0, 0), datatype="number", ax_in=None):
    """
    Similar to Plot_PSD only this function can plot the PSDs for multiple instruments
    that have different bin mids. 

    Parameters
    ----------
    *data_in : list of tuples
        Each tuple contains (bin_mids, size distribution data).
        Size distribution data can either be as returned from load function, or
        an array exlusively with bin populations. 
    labels : list, optional
        List of labels for the plots. The default is None.
    colors: list, optional
        List of colors for the plots. A default list is provided below.
    linestyles: list, optional
        list of linestyles for the plots. The default is None.
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
    
    """
    if colors==None:
        colors = ["red", "blue", "green", "orange", "magenta", "cyan", "k", "purple", "yellow", 'pink']
    #dlinestyles='-'
    if ax_in == None:
        fig, ax = plt.subplots()
    else:
        ax = ax_in
        
    for idx, (bin_mids, dataset) in enumerate(data_in):
        #Checking whether the data format is as the data returned from a load function
        if len(dataset[0,:])==len(bin_mids)+2:
            particle_data = dataset[:, 2:].astype("float")
        #Or if it is just a dataset of bin population
        elif len(dataset[0,:])==len(bin_mids):
            particle_data = dataset[:, :].astype("float")    
        #If the data fits neither array width it returns an error message.
        else: return print("Error: Issue with size of bin_mids and data")
        
        mean_psd = np.nanmean(particle_data,axis=0)
        
        #runs through the supplied or default colors, labels and linestyles to correctly add it to the plot
        color = colors[idx % len(colors)]
        label = labels[idx] if labels and idx < len(labels) else None
        ls = linestyles[idx] if linestyles and idx < len(linestyles) else None
        
        ax.plot(bin_mids, mean_psd, label=label, lw=3, color=color,ls=ls)
        
        #confirms that there is sufficent data to make standard error plot
        if len(particle_data[:,0])>1:
            sem_psd = sem(particle_data, axis=0,nan_policy="omit")
            ax.fill_between(bin_mids, mean_psd - sem_psd, mean_psd + sem_psd, alpha=0.5, color=color)

    # Set axis scales and labels
    if xlog:
        ax.set_xscale("log")
    if ylog:
        ax.set_yscale("log")
    ax.grid(True, which="both")

    y_label = ("dN/dlogDp, cm$^{-3}$" if datatype == "normed" else "dN, cm$^{-3}$" if datatype == "number"
               else "dA, nm$^{2}$/cm$^{3}$" if datatype == "surface" else "dm, $\mu$g$^{-3}$")
    ax.set_ylabel(y_label)
    ax.set_xlabel("Dp, nm")

    # Adjust y-axis limits if specified
    if y_lim != (0, 0):
        ax.set_ylim(y_lim)

    if labels:
        ax.legend()
    
    return ax.figure, ax
    
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

def Plot_timeseries(data_in, bin_edges, y_tot=(0,0), y_3d=(1,0), elapsed = 0, log = 1,log_3d=1, datatype = "number",normal=True):
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
        The default is (1,0) in which case it will select a max value automatically, while
        ensuring a lower limit of 1.
    elapsed : boolean, optional
        Flag to set time as elapsed time rather than local time. Set to 1 to
        switch to elapsed time. The default is 0.
    log : boolean, optional
        Flag to set whether to use log-scale on the reported total concentration. 
        Default is off
    log_3d : boolean, optional
        Flag to set whether to use log-scale on the 3D mesh colorbar scale. 
        Default is on
    datatype : string, optional
        Keyword to specify the datatype. The available options are "number",
        "surface" and "mass". Default is "number"
    normal:
        Determines whether the data has been normalized. If it hasn't ' a normalization is done.
        
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
    
    if normal==False:
        dlogDp = np.log10(bin_edges[1:])-np.log10(bin_edges[:-1])
        data=data/dlogDp
    # Generate canvas and axis
    
    
    fig, axs = plt.subplots(nrows=2,ncols=1, sharex=True)
    ax1,ax2 = axs
    
    # If elapsed time is active, adjust the datetime values to start at the new
    # year, so that the first displayed time and date is "1 00:00"
    if elapsed:
        fake_start = datetime.datetime(2023,1,1,0,0,0)
        delta_time = time-time[0]
        time = datetime.timedelta(0,delta_time) + fake_start
        
        total_time = datetime.timedelta(0,time[-1]-time[0])
        days = total_time.days
    
    # Plot the total number concentration
    ax1.plot(time,total,lw=2,color="r")
    
    # Change y-scale, show grid, set y-label, and set y limits 
    ax1.grid(axis="both",which="both")
    tot_min = np.nanmin(total)
    tot_max = np.nanmax(total)
    if (y_tot[0] != 0) or (y_tot[1] != 0):
        if (y_tot[0] == 0) & (y_tot[1] != 0):
            ax1.set_ylim(tot_min*0.98,y_tot[1])
        elif (y_tot[0] != 0) & (y_tot[1] == 0):
            ax1.set_ylim(y_tot[0],tot_max*1.02)
        else:
            ax1.set_ylim(y_tot[0],y_tot[1])
            
    if log==1:
        ax1.set_yscale("log")

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
    if log_3d:
        # Set datapoints smaller than 1 equal to 1 in order to avoid issues when 
        # plotting log transformed values
        if np.all(data):
            print("No zeros, continuing")
        else:
            print("""There is a 0 value in the dataset, meaning that log plotting is not possible.\nEither specify a lower limit or turn off log y-scale""")
            return [], []
        
        # Make the colormesh plot
        c = ax2.pcolormesh(x, y, data, cmap='jet',norm=LogNorm(vmin=y_3d_min, vmax=y_3d_max),shading='flat')
        
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
        ax1.set_ylabel("N$_{Total}$, cm$^{-3}$")
        col.set_label('dN/dlogDp, cm$^{-3}$')
    elif datatype == 'surface':
        ax1.set_ylabel("S$_{Total}$, nm$^{2}$ cm$^{-3}$")
        col.set_label('dS/dlogDp, nm$^{2}$ cm$^{-3}$')
    elif datatype == "mass":
        ax1.set_ylabel("m$_{Total}$, $\mu$g/m$^{3}$")
        col.set_label('dm/dlogDp, $\mu$g/m$^{3}$')

    # Set ticks on the plot to be longer
    ax1.tick_params(axis="y",which="both",direction='out', length=6, width=2)
    ax2.tick_params(axis="y",which="both",direction='out', length=6, width=2)

    # Add the colorbar to the axis handles, enabling adjustments after the function is run
    axs = np.append(axs,col)
    
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
#%%
def Plot_timeseries_APS(data_in, y_tot=(0,0), y_3d=(-1,1), elapsed = 0, log = 1, datatype = "number",normal=False):
    """
    Function to plot both the total number number concentration and the difference
    between the optical and aerodynamically assigned size for .

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
        The default is (1,0) in which case it will select a max value automatically, while
        ensuring a lower limit of 1.
    elapsed : boolean, optional
        Flag to set time as elapsed time rather than local time. Set to 1 to
        switch to elapsed time. The default is 0.
    log : boolean, optional
        Flag to set whether to use log-scale on the reported total concentration. 
        Default is off
    log_3d : boolean, optional
        Flag to set whether to use log-scale on the 3D mesh colorbar scale. 
        Default is on
    datatype : string, optional
        Keyword to specify the datatype. The available options are "number",
        "surface" and "mass". Default is "number"
    normal:
        Determines whether the data has been normalized. If it hasn't ' a normalization is done.
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        Handle for the returned figure for saving.
    axs : numpy.array
        Array with two handles, one for each of the two axis objects of the plot.

    """
    Optic_bins=[487.0,615.3,777.41,982.22,1241.0,1567.94,1981.03,2502.95,
                3162.36,3995.51,5048.15,6378.12,8058.48,10181.54,12863.94,
                16253.03,20535.0]
    Aero_bins=[487.0,528.0,562.5,604.5,649.5,698.0,750.0,806.0,866.5,
                931.5,1001.0,1075.5,1155.5,1241.5,1334.0,1434.0,1541.0,
                1655.5,1779.0,1912.0,2055.0,2208.5,2373.0,2550.0,2740.5,
                2945.0,3164.5,3400.5,3654.5,3927.0,4219.5,4534.5,4873.0,
                5236.5,5627.0,6046.5,6498.0,6983.0,7504.0,8064.0,8665.5,
                9312.0,10008.5,10755.0,11555.0,12415.0,13340.0,14340.0,
                15410.0,16555.0,17790.0,19120.0,20535.0]

    Aero, Optic= UL.APS_cor_summation(data_in)

    Aero_rebin=Optic.copy()
    Aero_rebin[:,2:]=Aero_rebin[:,2:]*0
    # fraction=1
    Aero_count=1
    for O in range(1,len(Optic_bins)):
        for A in range(Aero_count,len(Aero_bins)):
            if Aero_bins[A]<=Optic_bins[O]:
                Aero_rebin[:,O+1]+=Aero[:,A+1]

            elif Aero_bins[A]>Optic_bins[O]:
                fraction=(Optic_bins[O]-Aero_bins[A-1])/( Aero_bins[A]-Aero_bins[A-1])
                Aero_rebin[:,O+1]+=Aero[:,A+1]*fraction
                Aero_rebin[:,O+2]+=Aero[:,A+1]*(1-fraction)
                Aero_count=A+1
                break
    
    time = data_in[:,0]
    total = data_in[:,1].astype('float64')
    data_3d = Optic[:,2:].astype('float64')-Aero_rebin[:,2:].astype('float64')
    
    for i in range(0,len(data_3d[0,:])):
        data_3d[:,i]=data_3d[:,i]/total[:]
    
    if normal==False:
        dlogDp = np.log10(Optic_bins[1:])-np.log10(Optic_bins[:-1])
        data_3d=data_3d/dlogDp
        

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
            
    if log==1:
        ax1.set_yscale("log")

    ax1.set_xlabel("")

    # Generate an extra time bin, which is needed for the meshgrid
    dt = time[1]-time[0]
    time = time - dt
    time = np.append(time,time[-1]+dt)
    
    # generate 2d meshgrid for the x, y, and z data of the 3D color plot
    y, x = np.meshgrid(Optic_bins, time)
    
    # Set the upper and/or lower limit of the color scale based on input
    if (y_3d[0] == 0) & (y_3d[1] != 0):
        y_3d_min = np.nanmin(data_3d)
        y_3d_max = y_3d[1]
        
    elif (y_3d[0] != 0) & (y_3d[1] == 0):
        y_3d_min = y_3d[0]
        y_3d_max = np.nanmax(data_3d)
        data_3d[data_3d<y_3d[0]]= y_3d[0]   
    elif (y_3d[0] != 0) or (y_3d[1] != 0):
        y_3d_min = y_3d[0]
        y_3d_max = y_3d[1]
        data_3d[data_3d<y_3d[0]]= y_3d[0]
    else:
        y_3d_min = np.nanmin(data_3d)
        y_3d_max = np.nanmax(data_3d)
    
    
    # Fill the generated mesh with particle concentration data

    # Make the colormesh plot
    c = ax2.pcolormesh(x, y, data_3d, cmap='jet',vmin=y_3d_min, vmax=y_3d_max,shading='flat')
    
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
        ax1.set_ylabel("N$_{Total}$, cm$^{-3}$")
        col.set_label('dn(O-A)/dlogDp, % cm$^{-3}$')
    elif datatype == 'surface':
        ax1.set_ylabel("S$_{Total}$, nm$^{2}$ cm$^{-3}$")
        col.set_label('dS(O-A)/dlogDp, % nm$^{2}$ cm$^{-3}$')
    elif datatype == "mass":
        ax1.set_ylabel("m$_{Total}$, $\mu$g/m$^{3}$")
        col.set_label('dm(O-A)/dlogDp, % $\mu$g/m$^{3}$')

    # Set ticks on the plot to be longer
    ax1.tick_params(axis="y",which="both",direction='out', length=6, width=2)
    ax2.tick_params(axis="y",which="both",direction='out', length=6, width=2)

    # Add the colorbar to the axis handles, enabling adjustments after the function is run
    axs = np.append(axs,col)
    
    return fig,axs
#%%
def Plot_timeseries_APS_full(data_in,bins=[[],[]], y_tot=(0,0), y_3d=(0,0), y_3diff=(-1,1), log_3d=False, elapsed = 0, log = 1, datatype = "number",normal=False):
    """
    Function to plot both the total number number concentration and the difference
    between the optical and aerodynamically assigned size for the correlated APS.
    Ideally this would allow for determining the presence of fibres as a discrapency
    between optically and aerodynamically determined particle size.
    
    Parameters
    ----------
    data_in : numpy.array
       An array of data as returned by the Load_xxx functions with columns
       of datetime, total conc, and size bin data
    y_tot : tuple, optional
        Lower and upper limit on the y-axis of the total concentration plot
        given as (lower,upper). 
        The default is (0,0) in which case it will select a value automatically.
    y_3d : tuple, optional
        Lower and upper limit on the colorbar of the 3d timeseris plot
        given as (lower,upper). 
        The default is (1,0) in which case it will select a max value automatically, while
        ensuring a lower limit of 1.
    y_3diff: tuple, optional
        Lower and upper limit on the colorbar of the realtive difference between
        the optically assigned particel and the re-sized aerodynamically assigned
        particle size for a 3d timeseris plot.
        The default is (-1,1) in which case it will span -100% to 100% d
        
    elapsed : boolean, optional
        Flag to set time as elapsed time rather than local time. Set to 1 to
        switch to elapsed time. The default is 0.
    log : boolean, optional
        Flag to set whether to use log-scale on the reported total concentration. 
        Default is off
    log_3d : boolean, optional
        Flag to set whether to use log-scale on the 3D mesh colorbar scale. 
        Default is on
    datatype : string, optional
        Keyword to specify the datatype. The available options are "number",
        "surface" and "mass". Default is "number"
    normal:
        Determines whether the data has been normalized. If it hasn't ' a normalization is done.
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        Handle for the returned figure for saving.
    axs : numpy.array
        Array with two handles, one for each of the two axis objects of the plot.

    """
    params = {'legend.fontsize': 20,
             'axes.labelsize': 25,
             'axes.titlesize': 25,
             'xtick.labelsize': 20,
             'ytick.labelsize': 20,
             'figure.figsize' : (22, 14)}
    plt.rcParams.update(params)
    if len(bins)==0:#[[],[]]:
        Aero_bins=[523.0,562.5,604.5,649.5,698.0,750.0,806.0,866.5,
                    931.5,1001.0,1075.5,1155.5,1241.5,1334.0,1434.0,1541.0,
                    1655.5,1779.0,1912.0,2055.0,2208.5,2373.0,2550.0,2740.5,
                    2945.0,3164.5,3400.5,3654.5,3927.0,4219.5,4534.5,4873.0,
                    5236.5,5627.0,6046.5,6498.0,6983.0,7504.0,8064.0,8665.5,
                    9312.0,10008.5,10755.0,11555.0,12415.0,13340.0,14340.0,
                    15410.0,16555.0,17790.0,19120.0,20535.0]
        Optic_bins=[370,475.58,611.28,785.7,1009.89,1298.06,1668.44,2144.52,
                    2756.44,3542.96,4553.91,5853.33,7523.53,9670.3,12429.63,
                    15976.31,20535]
    else: 
        Aero_bins=bins[0]
        Optic_bins=bins[1]

    Aero, Optic= UL.APS_cor_summation(data_in)

    Aero_rebin=Optic.copy()
    Aero_rebin[:,2:]=Aero_rebin[:,2:]*0
    # fraction=1
    Aero_count=1
    for O in range(1,len(Optic_bins)):
        for A in range(Aero_count,len(Aero_bins)):
            if Aero_bins[A]<=Optic_bins[O]:
                Aero_rebin[:,O+1]+=Aero[:,A+1]

            elif Aero_bins[A]>Optic_bins[O]:
                fraction=(Optic_bins[O]-Aero_bins[A-1])/( Aero_bins[A]-Aero_bins[A-1])
                Aero_rebin[:,O+1]+=Aero[:,A+1]*fraction
                Aero_rebin[:,O+2]+=Aero[:,A+1]*(1-fraction)
                Aero_count=A+1
                break
    
    time = data_in[:,0]
    total = data_in[:,1].astype('float64')
    data_3d = Optic[:,2:].astype('float64')-Aero_rebin[:,2:].astype('float64')
    
    for i in range(0,len(data_3d[0,:])):
        data_3d[:,i]=data_3d[:,i]/total[:]
    
    if normal==False:
        dlogDp=np.log10(Aero_bins[1:])-np.log10(Aero_bins[:-1])
        Aero=Aero[:,2:].astype('float64')/dlogDp
        dlogDp = np.log10(Optic_bins[1:])-np.log10(Optic_bins[:-1])
        Optic=Optic[:,2:].astype('float64')/dlogDp
        #data_3d=data_3d.copy()/dlogDp

    # Generate canvas and axis
    fig, axs = plt.subplots(nrows=4,ncols=1, sharex=True)
    axT,axA,axO,axDiff = axs
    
    # If elapsed time is active, adjust the datetime values to start at the new
    # year, so that the first displayed time and date is "1 00:00"
    if elapsed:
        fake_start = datetime.datetime(2023,1,1,0,0,0)
        delta_time = time-time[0]
        time = delta_time + fake_start
        
        total_time = time[-1]-time[0]
        days = total_time.days
    
    # Plot the total number concentration
    axT.plot(time,total,lw=2,color="r")
    
    # Change y-scale, show grid, set y-label, and set y limits 
    axT.grid(axis="both",which="both")
    tot_min = total.min()
    tot_max = total.max()
    if (y_tot[0] != 0) or (y_tot[1] != 0):
        if (y_tot[0] == 0) & (y_tot[1] != 0):
            axT.set_ylim(tot_min*0.98,y_tot[1])
        elif (y_tot[0] != 0) & (y_tot[1] == 0):
            axT.set_ylim(y_tot[0],tot_max*1.02)
        else:
            axT.set_ylim(y_tot[0],y_tot[1])
            
    if log==1:
        axT.set_yscale("log")

    axT.set_xlabel("")

    # Generate an extra time bin, which is needed for the meshgrid
    dt = time[1]-time[0]
    time = time - dt
    time = np.append(time,time[-1]+dt)
    

    # Set the upper and/or lower limit of the color scale based on input
    if (y_3diff[0] == 0) & (y_3diff[1] != 0):
        y_3d_min = np.nanmin(data_3d)
        y_3d_max = y_3diff[1]
        
    elif (y_3diff[0] != 0) & (y_3diff[1] == 0):
        y_3d_min = y_3diff[0]
        y_3d_max = np.nanmax(data_3d)
          
    elif (y_3diff[0] != 0) or (y_3diff[1] != 0):
        y_3d_min = y_3diff[0]
        y_3d_max = y_3diff[1]
        data_3d[data_3d<y_3diff[0]]= y_3diff[0]
    else:
        y_3d_min = np.nanmin(data_3d)
        y_3d_max = np.nanmax(data_3d)
        
    y, x = np.meshgrid(Optic_bins, time)
    Diff = axDiff.pcolormesh(x, y, data_3d, cmap='PiYG',vmin=y_3d_min, vmax=y_3d_max,shading='flat')
    
    # Set the upper and/or lower limit of the color scale based on input
    if (y_3d[0] == 0) & (y_3d[1] != 0):
        y_3d_min = np.nanmin(Aero)
        if y_3d_min<0:
            y_3d_min=0
        y_3d_max = y_3d[1]
        
    elif (y_3d[0] != 0) & (y_3d[1] == 0):
        y_3d_min = y_3d[0]
        y_3d_max = np.nanmax(Aero)
        Aero[Aero<y_3d[0]]= y_3d[0]
        Optic[Optic<y_3d[0]]= y_3d[0] 
    elif (y_3d[0] != 0) or (y_3d[1] != 0):
        y_3d_min = y_3d[0]
        y_3d_max = y_3d[1]
        Aero[Aero<y_3d[0]]= y_3d[0]
        Optic[Optic<y_3d[0]]= y_3d[0] 
    else:
        y_3d_min = np.nanmin(Aero)
        if y_3d_min<0:
            y_3d_min=0
        y_3d_max = np.nanmax(Aero)

    # generate 2d meshgrid for the x, y, and z data of the 3D color plot
    y, x = np.meshgrid(Aero_bins, time)
    # Fill the generated mesh with particle concentration data
    if log_3d ==True:
        # Set datapoints smaller than 1 equal to 1 in order to avoid issues when 
        # plotting log transformed values
        if (np.all(Aero)) & (np.all(Optic)):
            print("No zeros, continuing")
        else:
            print("""There is a 0 value in the dataset, meaning that log plotting is not possible.\nEither specify a lower limit or turn off log y-scale""")
            return [], []
        
        # Make the colormesh plot
        Ae = axA.pcolormesh(x, y, Aero, cmap='jet',norm=LogNorm(vmin=y_3d_min, vmax=y_3d_max),shading='flat')
        y, x = np.meshgrid(Optic_bins, time)
        
        Op = axO.pcolormesh(x, y, Optic, cmap='jet',norm=LogNorm(vmin=y_3d_min, vmax=y_3d_max),shading='flat')

    else:
        
        
        # Make the colormesh plot
        Ae = axA.pcolormesh(x, y, Aero, cmap='jet',vmin=y_3d_min, vmax=y_3d_max,shading='flat')
        # generate 2d meshgrid for the x, y, and z data of the 3D color plot
        y, x = np.meshgrid(Optic_bins, time)
        
        Op = axO.pcolormesh(x, y, Optic, cmap='jet',vmin=y_3d_min, vmax=y_3d_max,shading='flat')

    # Adjust axis labels, formats and spacing between plots
    if (elapsed!=0) and (days>0):
        axDiff.xaxis.set_major_formatter(mdates.DateFormatter("%d - %H:%M"))
        axDiff.set_xlabel("Time, DD - HH:MM")
        axDiff.xaxis.labelpad = 20
        axDiff.set_xticklabels(axDiff.get_xticklabels(), rotation=-45, ha="left")
        plt.subplots_adjust(hspace=0.05,bottom = 0.25)
    else:
        axDiff.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        axDiff.set_xlabel("Time, HH:MM")
        plt.subplots_adjust(hspace=0.05)
        
    # Make the y-scal logarithmic and set a label
    axA.set_yscale("log")
    axA.set_ylabel("Aero, nm")
    axO.set_yscale("log")
    axO.set_ylabel("Optic, nm")
    axDiff.set_yscale("log")
    axDiff.set_ylabel("Diff, nm")
    
    
    # Insert colorbar and label it

    col_Diff = fig.colorbar(Diff, ax=axs)
    col_conc = fig.colorbar(Ae, ax=axs)
    col_Diff.formatter(ticker.PercentFormatter(1.0))
    
    if datatype == "number":
        axT.set_ylabel("N$_{Total}$, cm$^{-3}$")
        col_conc.set_label('dn/dlogDp, cm$^{-3}$')
        col_Diff.set_label('(Optic - Aero)/total, %')
    elif datatype == 'surface':
        axT.set_ylabel("S$_{Total}$, nm$^{2}$ cm$^{-3}$")
        col_Diff.set_label('dS(O-A)/dlogDp, % nm$^{2}$ cm$^{-3}$')
    elif datatype == "mass":
        axT.set_ylabel("m$_{Total}$, $\mu$g/m$^{3}$")
        col_Diff.set_label('dm(O-A)/dlogDp, % $\mu$g/m$^{3}$')

    # Set ticks on the plot to be longer
    axT.tick_params(axis="y",which="both",direction='out', length=6, width=2)
    axDiff.tick_params(axis="y",which="both",direction='out', length=6, width=2)
    
    # Add the col_Difforbar to the axis handles, enabling adjustments after the function is run
    axs = np.append(axs,col_Diff)
    
    return fig,axs
###############################################################################
