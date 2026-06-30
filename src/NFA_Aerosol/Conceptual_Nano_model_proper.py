# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 09:11:31 2025

@author: b351796
"""

import pickle
import sys
import numpy as np
from datetime import datetime, date, time, timedelta
import pandas as pd
from scipy.optimize import curve_fit, least_squares
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import os as os
from matplotlib.dates import date2num, num2date
import matplotlib.pyplot as plt
local_package_path = os.path.abspath(r"C:/Users/B351796/Documents/GitHub/NFA_Aerosol/src/NFA_Aerosol")
sys.path.insert(0, local_package_path)
import Instrument_Lib as IL 
import Plot_Lib as PL
import Utility_Lib as UL

# Author Alexander C Ø Jensen Last updated 07-01-2020
# 
# The model has been published in Jensen et al 2018 doi.org/10.3390/environments5050052
# 
# The script uses the file "ACOJ_model_input_parameters.xlsx" as initialization 
# and for input parameters. 
# The source data should be formatted as found in the file "PROC-ART release
# library.xlsx".
# As output two files are saved with the local computer time at execution as unique
# identifier. One of the two files is a figure here time series of
# the modelled concentrations and size distribution are plotted. The other file is an
# .xlsx file containing the initialization values and concentration time series in the 
# NF and FF.
# 
# Any questions can be directed to alj@nfa.dk

input_file=r"C:\Users\B351796\Desktop\Python Library\Input_document_example.xlsx"

def MAD_model(input_file,starting_time=datetime(2025,1,1,12,52,20)):
    #param['src_sizes'],src_time,src_totc,param['src_data'],src_dlogdp
    # ???warning('off')
    param = {'Constants':{'g'  : 9.82, # gravitational acceleration  m/s^2,
                          'kB' : 1.381e-23, # Boltzmann constant J/K = N*m/K = kg*m^2/s^2 / K 
                          'dyn_visc_air' : 1.846e-5, #Dynamic viscosity of air kg m-1 s-1
                          'kin_visc_air' : 1.568e-5}, #Kinematic viscosity of air m2/s   },
             'Settings':{}, # Here the start up values for settings as loaded from input_file + source_file
             'NF':{},       # Here the values and qualities associated with the NF goes
             'FF':{},       # Here the values and qualities associated with the FF goes
             'Particle':{},
             'Result':{}}
    param = import_xlsx_parameters(param,input_file)    
    
    # param = define_case(param) 
    # param = set_parameters(param)
    # param['Settings']['opt']   = odeset('reltol',1e-6) #???
    # param['Settings']['time_to_calculate'] = 0:param['Settings']['time']-1
    param['Settings']['time']  = np.arange(0,param['Settings']['time_to_calculate'])
    # Runs the ODE solver based on the function diff_eq_model 
    result = solve_ivp(diff_eq_model,
        t_span=[0,param['Settings']['time_to_calculate']],y0=param['Settings']['initial_conc'],
        t_eval=param['Settings']['time'],args=[param],rtol=1E-6) #,param['opt'])
    param['Result']['Time']=result['t']
    param['Result']['Concentrations']=np.transpose(result['y'])

    # for adding time dependant variables they need their own time vector and interpolate with interp1.

    # t_var and c_var are variables for time and concentration and not
    # initial particle number concentration [1/cm3] nor time span for the 
    # evaluation if there are several sources then source vector and 
    # interpolate expand with only selection of which source to use when.
    
    n=len(param['Settings']['bin_mids'])
    param['Result']['Concentrations'] = param['Result']['Concentrations']*1e-6 # Change concentration from unit #/m3 to #/cm3
    
    param['Result']['Concentrations'][param['Result']['Concentrations'] <= 1e-8] = 0 # Removes concentraions that are less than 1e-8 #/cm3 since Excel does not save them due to 8 decimal characters in the CSV style
    
    # Calculate the total number concentration and dN/dlogdp
    result_totc_nf    = np.sum(param['Result']['Concentrations'][:,0:n],1)
    result_totc_ff    = np.sum(param['Result']['Concentrations'][:,n:n*2],1)
    # param['result_psd_nf']     = bsxfun(@rdivide,param['Result']['Concentrations'][:,1:n],param['Settings']['src_dlogdp'])
    # param['result_psd_ff']     = bsxfun(@rdivide,param['Result']['Concentrations'][:,n+1:n*2],param['Settings']['src_dlogdp'])
    param['Result']['Time']=[starting_time+timedelta(0,int(param['Result']['Time'][i])) for i in range (0,len(param['Result']['Time']))]
    # Convert the results to regular data format of: time, tot, bins
    param['Result']['NF'] = np.column_stack((param['Result']['Time'],result_totc_nf,param['Result']['Concentrations'][:,0:n]))
    param['Result']['FF'] = np.column_stack((param['Result']['Time'],result_totc_ff,param['Result']['Concentrations'][:,n:n*2]))
    #Convert the bin sizes to nm
    param['Settings']['bin_mids']=param['Settings']['bin_mids']*1E9
    param['Settings']['bin_edges']=param['Settings']['bin_edges']*1E9
    # fig,ax=PL.Plot_timeseries(param['Result']['NF'], param['Settings']['bin_edges'],y_3d=(1,1.2E4),log=0,datatype='number',normal=False)
    # ax[0].plot(param['Result']['FF'][:,0],param['Result']['FF'][:,1])
    # fig,ax=PL.Plot_timeseries(param['Result']['FF'], param['Settings']['bin_edges'],y_3d=(1,1.2E4),log=0,datatype='number',normal=False)
    return param
############################################################################### 
#%% Starting up function
def import_xlsx_parameters(param,input_file):
 
    # import properties about the particle release from xlsx document
    
    Settings = np.array(pd.read_excel(input_file,sheet_name='Input_parameters',usecols='B'))#, sep=seperator, header=6, encoding='latin-1'))
    
    # [param[xlsx_inputs_num,param[xlsx_inputs_txt,param[xlsx_inputs_raw] = xlsread('ACOJ_model_input_parameters.xlsx','ACOJ_model_input_parameters','B1:B48')
    
    param['Settings']['#_ff_sections'] = int(Settings[0])             # File location
    param['Settings']['Emission_limit'] = float(Settings[1])          # Sheet name
    param['Settings']['include_background'] = int(Settings[2])        # Include background
    param['Settings']['include_deposition'] = int(Settings[3])        # Include deposition
    param['Settings']['include_coagulation'] = int(Settings[4])       # Include coagulation
    param['Settings']['Source_mode'] = str(Settings[5][0])            # Column of the source 
     
    # Model time parameterization
    param['Settings']['source_time']      = int(Settings[8])          # Source active time [sec]
    param['Settings']['source_pause']     = int(Settings[9])          # Pause between source activation [sec]
    param['Settings']['source_repetitions']       = int(Settings[10]) # Repetitions [-]
    param['Settings']['time_to_calculate']        = int(Settings[11]) # Modeling time [sec]
    # Chamber parameters
    param['Settings']['rho']   = float(Settings[14])                  # Density  kg/m^3    
    param['Settings']['T']     = float(Settings[15])                  # Temperature  K 
    param['Settings']['p']     = float(Settings[16])                  # pressure  atm 
    param['Settings']['Us']    = float(Settings[17])                  # Friction velocity m/s
    # NF parameterization    
    param['NF']['V']          = float(Settings[21])                   # [m3] Volume of Near-field
    param['NF']['beta']       = float(Settings[22])/60                # [m3/s] Inter-zonal exhange rate 
    param['NF']['Au']         = float(Settings[23])                   # [m2] Area facing upwards
    param['NF']['Ad']         = float(Settings[24])                   # [m2] Area facing downwards
    param['NF']['Av']         = float(Settings[25])                   # [m2] Area facing vertically
    # FF parameterization    
    param['FF']['V']          = float(Settings[28])                   # [m3] Volume of Far-field
    param['FF']['Q']          = float(Settings[29])/60                # [m3/s] Ventilation from FF to exhaust
    param['FF']['Au']         = float(Settings[30])                   # [m2] Area facing upwards
    param['FF']['Ad']         = float(Settings[31])                   # [m2] Area facing downwards
    param['FF']['Av']         = float(Settings[32])                   # [m2] Area facing vertically

    # import properties about the particle release and background from the input document
    if 'Inst' in param['Settings']['Source_mode']:
        print('helo')
        Source = np.array(pd.read_excel(input_file,sheet_name='Instrument',header=None))#, sep=seperator, header=6, encoding='latin-1'))
      
        param['Settings']['bin_edges'] = Source[0,1:].astype(float)*1E-9       #convert nm to m 
        param['Settings']['bin_mids'] = (param['Settings']['bin_edges'][1:]+param['Settings']['bin_edges'][:-1])/2
        param['Settings']['src_dlogdp'] = np.log10(param['Settings']['bin_edges'][1:])-np.log10(param['Settings']['bin_edges'][:-1])
        #np.log10(param['Settings']['bin_edges'])-np.mean(np.diff(np.log10(param['Settings']['bin_edges'])))
        
        param['Settings']['source_strength'] = Source[1,1:-1].astype(float)      #1/s per bin
        param['Settings']['Background'] = Source[2,1:-1].astype(float)           #1/m3 per bin
        
    elif 'mode' in param['Settings']['Source_mode']:
        Source = np.array(pd.read_excel(input_file,sheet_name='Particle mode',header=None))#, sep=seperator, header=6, encoding='latin-1'))
        
        Bins=Source[1,1:4].astype(int)                               #Has lower limit, upper limit and number of bins   
        param['Settings']['bin_edges'] =  [Bins[0]]
        step=(np.log10(Bins[1])-np.log10(Bins[0]))/Bins[2]
        for i in range(0,Bins[2]):
            
            param['Settings']['bin_edges'].append(param['Settings']['bin_edges'][-1]*10**(step))
            
        param['Settings']['bin_edges']=np.round(np.array(param['Settings']['bin_edges']),1)*1E-9      # Convert from nm to m
        param['Settings']['bin_mids']=(param['Settings']['bin_edges'][1:]+param['Settings']['bin_edges'][:-1])/2        #convert nm to m 
        param['Settings']['src_dlogdp'] = np.log10(param['Settings']['bin_edges'][1:])-np.log10(param['Settings']['bin_edges'][:-1])
        #np.log10(param['Settings']['bin_edges'])-np.mean(np.diff(np.log10(param['Settings']['bin_edges'])))
        
        #Collect the data regarding the designated source modes and strengths
        Factor = Source[4,1:].astype(float)             # Total source strength of each mode 1/s
        GMD = Source[5,1:].astype(float)*1E-9           # Array of means for each source converted to m
        GSD = Source[6,1:].astype(float)                # 
        
        mask=~np.isnan(Factor)
        Mode=[]
        for i in range(0,len(GMD[mask])): #This combines them into a set for easy calculation via UL.Normal
            Mode.append(GMD[i])
            Mode.append(GSD[i])
            Mode.append(Factor[i])
            
        param['Settings']['source_strength']=UL.Normal(param['Settings']['bin_mids'],*Mode)       #1/s per bin
        param['Settings']['source_strength']=param['Settings']['source_strength']*param['Settings']['src_dlogdp'] #Unnormalizes the data
        param['Settings']['source_strength'][param['Settings']['source_strength']<param['Settings']['Emission_limit']]=0 #Puts a limit on the lower levels of emissions
        
        #Collect the data regarding the designated source modes and strengths
        Factor = Source[9,1:].astype(float)             # Total source strength of each mode 1/s
        GMD = Source[10,1:].astype(float)*1E-9           # Array of means for each source converted to m
        GSD = Source[11,1:].astype(float)                # 
        
        mask=~np.isnan(Factor)
        Mode=[]
        for i in range(0,len(GMD[mask])): #This combines them into a set for easy calculation via UL.Normal
            Mode.append(GMD[i])
            Mode.append(GSD[i])
            Mode.append(Factor[i])
        
        param['Settings']['Background']=UL.Normal(param['Settings']['bin_mids'],*Mode)       #1/m3 per bin
        param['Settings']['Background']=param['Settings']['Background']*param['Settings']['src_dlogdp'] #Unnormalizes the data  
        
    # else: return print("Incorrect input in source mode. Check B7 in input_parameters")
    
    # Checks which aspects to add to the physical considerations. 
    if param['Settings']['include_background']==1:
        pass
    else:
        param['Settings']['Background'] = np.zeros(len(param['Settings']['bin_mids'])) 
    NF_init = param['Settings']['Background'] 
    FF_init = np.tile(param['Settings']['Background'] ,param['Settings']['#_ff_sections'])
    
    # else:
    #     NF_init = np.zeros((1,len(param['Settings']['bin_mids']))) 
    #     FF_init = np.zeros((1,len(param['Settings']['bin_mids'])*param['Settings']['#_ff_sections'])) #???
 
    param['Settings']['initial_conc'] = np.append(NF_init, FF_init) # #/m3   
     

    if param['Settings']['include_deposition'] == 1:
        param   = create_deposition_parameters(param)

    if param['Settings']['include_coagulation'] == 1:
        param   = create_coagulation_parameters(param)
    
    
    param['Settings']['source']    = np.tile(param['Settings']['source_strength'],[param['Settings']['source_time'],1])
    Pause_array     = np.zeros((param['Settings']['source_pause'],len(param['Settings']['bin_mids'])))
    param['Settings']['source']    = np.tile(np.row_stack((param['Settings']['source'], Pause_array)),[param['Settings']['source_repetitions'],1])
    #Fills out the remining simulation time for the source profile 
    param['Settings']['source']    = np.row_stack((param['Settings']['source'], np.zeros([param['Settings']['time_to_calculate']-np.shape(param['Settings']['source'])[0],np.shape(param['Settings']['source'])[1]])))

        
    return param

###############################################################################
#%% create deposition parameters
def create_deposition_parameters(param):
    # Deposition is based on Lai & Nazaroff, (2000) doi.org/10.1016/S0021-8502(99)00536-4
    # Assuming density 1 g/ccm. 
    constants= param['Constants']
    settings = param['Settings']
    # Cunningham correction, Hinds (1999) 3.22 
    Cn  = 1+1/(101.325*settings['bin_mids']*1e6)*(15.6+7*np.exp(-0.059*101.325*settings['bin_mids']*1e6))  
        
    #Diffusion coefficient
    D   = constants['kB']*settings['T']*Cn/(3*np.pi*constants['dyn_visc_air']*settings['bin_mids']) # m2/s 
    
    # Schmidt number
    Sc  = constants['kin_visc_air']/D          
    
    Rp  = settings['bin_mids']/2*settings['Us']/(2*constants['kin_visc_air']) # from Lai & Nazaroff 2000 table 2 Unitless
 
    a   = 0.5*np.log((10.92*Sc**(-1/3)+4.3)**3/(Sc**(-1)+0.0609))+np.sqrt(3)*np.arctan((8.6-10.92*Sc**(-1/3))/(np.sqrt(3)*10.92*Sc**(-1/3)))
    b   = 0.5*np.log((10.92*Sc**(-1/3)+Rp)**3/(Sc**-1+7.669e-4*Rp**3))+np.sqrt(3)*np.arctan((2*Rp-10.92*Sc**(-1/3))/(np.sqrt(3)*10.92*Sc**(-1/3)))        
    I   = (3.64*Sc**(2/3)*(a-b)+39)
    
    # Gravitational settling velocity [m/s]
    vg  = constants['g']*1/6*np.pi*settings['bin_mids']**3*settings['rho']*Cn/(6*np.pi*constants['dyn_visc_air']*(settings['bin_mids']/2)) 
 
    # from Lai & Nazaroff (2000) table 3
    # Values if I are interpolated to obtain right value for I of the
    # respective size below 10 nm particle size.
    small_sizes = settings['bin_mids']<=10e-9 #m
    if len(settings['bin_mids'][small_sizes])>0:
        I_small = np.array([[0.001, 29.1],[ 0.0015, 49.1], [0.002, 71.0],[ 0.003, 120.3],
                            [0.004,174.9],[0.005, 234.2],[0.006, 297.4],[0.007, 364.0],
                            [0.008,432.7],[0.009, 504.5],[0.01, 579.3]])
        Interpolation=interp1d(I_small[:,0]*1e-6,I_small[:,1])
        I[small_sizes] = Interpolation(settings['bin_mids'][small_sizes])#,settings['bin_mids'][small_sizes])
    
    # Deposition velocity vertical surfaces
    vdv     = settings['Us']/I 
    
    # Depostition velocity downwards facing surfaces
    vdd     = vg/(1-np.exp(-vg*I/settings['Us']))
    
    # Depostitopn velocity upwards facing surfaces
    vdu     = vg/(np.exp(vg*I/settings['Us'])-1) 
    
    param['Particle']['b_dep']         = np.zeros((settings['#_ff_sections']+1,len(settings['bin_mids'])))
    # near field surface deposition
    param['Particle']['b_dep'][0,:]    = (param['NF']['Au']*vdu+param['NF']['Ad']*vdd+param['NF']['Av']*vdv)
 
    # far field surface deposition
    if settings['#_ff_sections'] >= 1:
        for i in range(0,settings['#_ff_sections']):
            param['Particle']['b_dep'][i+1,:] = (param['FF']['Au']*vdu+param['FF']['Ad']*vdd+param['FF']['Av']*vdv)
    
    return param
############################################################################### 
#%% create coagulation parameters
def create_coagulation_parameters(param):
    
    # This section deterimines the division of particle volume fractions
    # into the size bins after coagulation.
 
    # Particle volume for spherical particles
    param['Particle']['Vol']  = 1/6*np.pi*param['Settings']['bin_mids']**3 #m3
    
    param['Particle']['b_coag']        = coag_kernel(param) # m3/s	
 
    n = len(param['Settings']['bin_mids'])
 
    param['Particle']['res_s_vol'] = np.zeros((n,n))    #m3
    param['Particle']['res_L_vol'] = np.zeros((n,n))    #m3
    param['Particle']['frac_s_vol'] = np.zeros((n,n))   # unitless
    param['Particle']['frac_L_vol'] = np.zeros((n,n))   # unitless
    
    for qq in range(0,n):
        for ii in range(qq,n):

            res_vol_less = param['Particle']['Vol'][ param['Particle']['Vol']<=param['Particle']['Vol'][ii]+param['Particle']['Vol'][qq] ]
            res_vol_larger = param['Particle']['Vol'][ param['Particle']['Vol']>=param['Particle']['Vol'][ii]+param['Particle']['Vol'][qq] ]

            if len(res_vol_larger)==0:
                param['Particle']['res_s_vol'][ii,qq] = len(res_vol_less)-1
                param['Particle']['res_L_vol'][ii,qq] = float('nan')

                # res_vol = vol[-1]
                param['Particle']['frac_s_vol'][ii,qq] = (param['Particle']['Vol'][ii]+param['Particle']['Vol'][qq])/res_vol_less[-1]
                param['Particle']['frac_L_vol'][ii,qq] = 0  

            elif res_vol_less[-1]==res_vol_larger[0]:
                #??? is this case even possible?
                param['Particle']['res_s_vol'][ii,qq] = len(res_vol_less)-1
                param['Particle']['res_L_vol'][ii,qq] = n-len(res_vol_larger)
                param['Particle']['frac_s_vol'][ii,qq] = 0.5
                param['Particle']['frac_L_vol'][ii,qq] = 0.5

            else:

                param['Particle']['res_s_vol'][ii,qq] = len(res_vol_less)-1
                param['Particle']['res_L_vol'][ii,qq] =  n-len(res_vol_larger)
               
                # calculate fraction into each volume
                # param['Particle']['frac_s_vol'][ii,qq] = (param['Particle']['Vol'][ii]+param['Particle']['Vol'][qq]-res_vol_less[-1])/(-res_vol_less[-1]+res_vol_larger[0])
                param['Particle']['frac_s_vol'][ii,qq] = (param['Particle']['Vol'][ii]+param['Particle']['Vol'][qq]-res_vol_larger[0])/(res_vol_less[-1]-res_vol_larger[0]) 
                param['Particle']['frac_L_vol'][ii,qq] = 1-param['Particle']['frac_s_vol'][ii,qq]
                
    # for qq in range(0,n):
    #     for ii in range(qq,n):

    #         res_vol_less = param['Particle']['Vol'][ param['Particle']['Vol']<=param['Particle']['Vol'][ii]+param['Particle']['Vol'][qq] ]
    #         res_vol_larger = param['Particle']['Vol'][ param['Particle']['Vol']>=param['Particle']['Vol'][ii]+param['Particle']['Vol'][qq] ]

    #         if len(res_vol_larger)==0:
    #             param['Particle']['res_s_vol'][ii,qq] = res_vol_less[-1]
    #             param['Particle']['res_L_vol'][ii,qq] = float('nan')

    #             # res_vol = vol[-1]
    #             param['Particle']['frac_s_vol'][ii,qq] = (param['Particle']['Vol'][ii]+param['Particle']['Vol'][qq])/res_vol_less[-1]
    #             param['Particle']['frac_L_vol'][ii,qq] = 0  

    #         elif res_vol_less[-1]==res_vol_larger[0]:
    #             #??? is this case even possible?
    #             param['Particle']['res_s_vol'][ii,qq] = res_vol_less[-1]
    #             param['Particle']['res_L_vol'][ii,qq] = res_vol_larger[0]
    #             param['Particle']['frac_s_vol'][ii,qq] = 0.5
    #             param['Particle']['frac_L_vol'][ii,qq] = 0.5

    #         else:

    #             param['Particle']['res_s_vol'][ii,qq] = res_vol_less[-1]
    #             param['Particle']['res_L_vol'][ii,qq] = res_vol_larger[0]
               
    #             # calculate fraction into each volume
    #             param['Particle']['frac_s_vol'][ii,qq] = (param['Particle']['Vol'][ii]+param['Particle']['Vol'][qq]-res_vol_larger[0])/(res_vol_less[-1]-res_vol_larger[0]) 
    #             param['Particle']['frac_L_vol'][ii,qq] = 1-param['Particle']['frac_s_vol'][ii,qq]
                
    return param
###############################################################################
#%% coagulation kernel
def coag_kernel(param):
    # (c) Miikka Dal Maso 2013
    #
    # Version history:
    # 2013-05-24    0.1.0
    # Modified Alexander CØ Jensen 2016
    # Reference Seinfeld and Pandis, Atmospheric Chemistry and Physics, 2006, pp 600
    
    # Particle diameter
    Dp      = param['Settings']['bin_mids'] # [m]
    
    # Temperature
    T       = param['Settings']['T'] # [K]
    
    # Pressure atm
    p       = param['Settings']['p'] # [atm]
    
    # Boltzmann's constant
    kB      = param['Constants']['kB']  #J/K -> kg*m^2/s^2 / K 
    
    # Density
    rho    = param['Settings']['rho']
    n       = len(Dp)
    
    # Mean free path 
    lamda   = (6.73e-8*T*(1+(110.4/T)))/(296*p*1.373)    #[m]    
    
    #  viscosity
    myy     = (1.832e-5*(T**(1.5))*406.4)/(5093*(T+110.4)) # [kg m-1 s-1]
 
    K=np.zeros((n,n))
    for qq in range(0,n):
        for ii in range(qq,n):

            r1 = Dp[ii]/2 
            r2 = Dp[qq]/2 # big_R,         
    
            kn1=lamda/r1   #dimensionless            				  #1*1
            kn2=lamda/r2             									  #28*1
            
            CC1 = 1 + (kn1*(1.142 + 0.558*np.exp(-(0.999/kn1))))  # dimensionless
            CC2 = 1 + (kn2*(1.142 + 0.558*np.exp(-(0.999/kn2))))
    
            D1= (kB*T*CC1)/(6*np.pi*myy*r1)     # [m2/s]     	  #1*1 koaguloituvan
            D2= (kB*T*CC2)/(6*np.pi*myy*r2)     # [m2/s]          #28*1 jakauman
    
            M2= 4/3*np.pi*(r2**3)*rho # kg							
            M1= 4/3*np.pi*(r1**3)*rho									
    
            c2= np.sqrt((8*kB*T)/(np.pi*M2))	# 	m2/s2				
            c1= np.sqrt((8*kB*T)/(np.pi*M1))							
    
            c12= np.sqrt((c2**2)+(c1**2))		# m2/s2				
    
            r12= r2+r1	 #m												 	
    
            D12= D2+D1		#m2/s											
    
            CCONT= 4*np.pi*r12*D12	 # m3/s							
    
            CFR= np.pi*r12*r12*c12	# m^4/s^2							
    
            L2=(8*D2)/(np.pi*c2)	# m									
            L1=(8*D1)/(np.pi*c1)	# m										
    
            SIG2=(1/(3*r12*L2))*((r12+L2)**3-(r12*r12+L2*L2)**1.5) - r12  # 
            SIG1=(1/(3*r12*L1))*((r12+L1)**3-(r12*r12+L1*L1)**1.5) - r12 
    
            SIG12= np.sqrt((SIG2**2)+(SIG1**2))
    
            KO=CCONT/((r12/(r12+SIG12))+(CCONT/CFR))				#1*27  koagulaatiokerroin
            K[ii,qq] = KO
    return K
###############################################################################
#%% diff_eq_model
def diff_eq_model(t,c,param):
    
    # Initialising
    dc = np.zeros(np.shape(c)) #1/m3s
    
    # Source release in the NF
    dc = add_source(t,dc,param) 
    
    # Transport between NF and FF
    dc = add_transport(dc,c,0,1,param['NF']['beta'],param['NF']['V'],param['FF']['V'],param) 
    
    # Removal by general room ventilation located in the FF
    dc = add_exhaust(dc,c,1,param['FF']['V'],param['FF']['Q'],param) 
    
    # If deposition is included
    if param['Settings']['include_deposition'] == 1:
        dc = add_deposition(dc,c,0,param['NF']['V'],param) # NF deposition
        dc = add_deposition(dc,c,1,param['FF']['V'],param) # FF deposition
    
    # If coagulation is included
    if param['Settings']['include_coagulation'] == 1:
        dc = add_coagulation(dc,c,0,param) # NF coagulation
        dc = add_coagulation(dc,c,1,param) # FF coagulation
        
    return dc   

###############################################################################
#%% add_source
def add_source(t,dc,param):
    
    # Source release in the NF
    Interpolation    = interp1d(param['Settings']['time'],param['Settings']['source'],axis=0,bounds_error=False,fill_value=0,) # interpolate source to model time
    source_t = Interpolation(t)
    # If the source is NaN replace with zero release
    source_t[np.isnan(source_t)] = 0
    for ii in range(0,len(source_t)):           
        dc[ii]  = dc[ii]+source_t[ii]/param['NF']['V'] # Source term
        
    return dc

###############################################################################
#%% add_transport
def add_transport(dc,c_var,i,j,beta,v_i,v_j,param):
    n=len(param['Settings']['bin_mids'])
    # Transport between two connected boxes
    for ii in range (0,n):
         dc[n*i+ii] = dc[n*i+ii]-beta/v_i*c_var[n*i+ii] # Flow to farfield
         dc[n*i+ii] = dc[n*i+ii]+beta/v_i*c_var[n*j+ii] # Flow from farfield
         
         dc[n*j+ii] = dc[n*j+ii]+beta/v_j*c_var[n*i+ii] # Flow from nearfield
         dc[n*j+ii] = dc[n*j+ii]-beta/v_j*c_var[n*j+ii] # Flow to nearfield
    return dc
###############################################################################
#%% add_exhaust
def add_exhaust(dc,c_var,i,v_i,b_ex,param):
    n=len(param['Settings']['bin_mids'])
    # Removal of pollutants via ventilation
    for ii in range(0,n):
        dc[n*i+ii] = dc[n*i+ii]-b_ex/v_i*(c_var[n*i+ii]-param['Settings']['Background'][ii]) # Flow to exhaust
    return dc
###############################################################################
#%% add_deposition
def add_deposition(dc,c_var,i,v_i,param):
    n=len(param['Settings']['bin_mids'])
    # Deposition in box i
    for ii in range(0,n):
           dc[n*i+ii] = dc[n*i+ii]-param['Particle']['b_dep'][i,ii]/v_i*c_var[n*i+ii] # Depositon on surfaces
    return dc
###############################################################################
#%% add_coagulation
def add_coagulation(dc,c_var,i,param):
    
    # Coagulation function
    # Loss from smaller (LFS) is the smaller size bin of the two colliding
    # particles
    # Loss from larger (LFL) is the larger size bin of the two colliding
    # particles
    # Gain to smaller (GTS) is the smaller size bin of the resulting particle
    # Gain to larger (GTL) is the larger size bin of the resulting particle    
    
    n = len(param['Settings']['bin_mids'])
    
    dc_LFS = np.zeros((n,n))                    # 1/m3s
    dc_LFL = np.zeros((n,n))                    # 1/m3s
    dc_GTS = [0]*n                              # 1/m3s
    dc_GTL = [0]*n                              # 1/m3s
    b_coag = param['Particle']['b_coag']        # m3/s	
    
    # for ii in range(0,n):
    #     for qq in range(0,ii):
    #         if ii>=qq:
    #             if ii == qq:
    for qq in range(0,n):
        for ii in range(qq,n):     
                if ii == qq:
                    dc_LFS[ii,qq] = c_var[n*i+ii]*c_var[n*i+qq]*b_coag[ii,qq]
    
                else:    
                    dc_LFS[ii,qq] = c_var[n*i+ii]*c_var[n*i+qq]*b_coag[ii,qq]
                    dc_LFL[ii,qq] = c_var[n*i+ii]*c_var[n*i+qq]*b_coag[ii,qq]
    
                if ii == qq:
                    #The dc_GT series look at the table for the 
                    dc_GTS[int(int(param['Particle']['res_s_vol'][ii,qq]))] = dc_GTS[int(param['Particle']['res_s_vol'][ii,qq])] + 0.5*dc_LFS[ii,qq]*param['Particle']['frac_s_vol'][ii,qq]
                    dc_GTS[int(param['Particle']['res_s_vol'][ii,qq])] = dc_GTS[int(param['Particle']['res_s_vol'][ii,qq])] +0.5*dc_LFL[ii,qq]*param['Particle']['frac_s_vol'][ii,qq]
    
                    if ~np.isnan(param['Particle']['res_L_vol'][ii,qq]):
                        dc_GTL[int(param['Particle']['res_L_vol'][ii,qq])] = dc_GTL[int(param['Particle']['res_L_vol'][ii,qq])] + 0.5*dc_LFS[ii,qq]*param['Particle']['frac_L_vol'][ii,qq]
                        dc_GTL[int(param['Particle']['res_L_vol'][ii,qq])] = dc_GTL[int(param['Particle']['res_L_vol'][ii,qq])] + 0.5*dc_LFL[ii,qq]*param['Particle']['frac_L_vol'][ii,qq]
                        
                else:
                    dc_GTS[int(param['Particle']['res_s_vol'][ii,qq])] = dc_GTS[int(param['Particle']['res_s_vol'][ii,qq])] + 0.5*dc_LFS[ii,qq]*param['Particle']['frac_s_vol'][ii,qq]
                    dc_GTS[int(param['Particle']['res_s_vol'][ii,qq])] = dc_GTS[int(param['Particle']['res_s_vol'][ii,qq])] + 0.5*dc_LFL[ii,qq]*param['Particle']['frac_s_vol'][ii,qq]
    
                    if ~np.isnan(param['Particle']['res_L_vol'][ii,qq]):
                        dc_GTL[int(param['Particle']['res_L_vol'][ii,qq])] = dc_GTL[int(param['Particle']['res_L_vol'][ii,qq])] + 0.5*dc_LFS[ii,qq]*param['Particle']['frac_L_vol'][ii,qq]
                        dc_GTL[int(param['Particle']['res_L_vol'][ii,qq])] = dc_GTL[int(param['Particle']['res_L_vol'][ii,qq])] + 0.5*dc_LFL[ii,qq]*param['Particle']['frac_L_vol'][ii,qq]                    

    total_loss = np.sum(dc_LFS,0) + np.sum(dc_LFL,1)
    total_gain = np.array(dc_GTS) + np.array(dc_GTL)
        
    for ii in range(0,n):
        dc[n*i+ii] = dc[n*i+ii]+total_gain[ii]-total_loss[ii]

    return dc
###############################################################################

#%%
param=MAD_model(input_file)

NF_norm=param['Result']['FF'].copy()
NF_norm[:,2:]=NF_norm[:,2:]/param['Settings']['src_dlogdp']
# mask=(NF_norm[:,0]>param['Result']['FF'][param['Settings']['source_time']*param['Settings']['source_repetitions'],0]-timedelta(0,30)) & (
#     NF_norm[:,0]<param['Result']['FF'][param['Settings']['source_time']*param['Settings']['source_repetitions'],0]+timedelta(0,120))
# mask2=(NF_norm[:,0]>param['Result']['FF'][param['Settings']['source_time']*param['Settings']['source_repetitions']*22,0]-timedelta(0,1800)) & (
#     NF_norm[:,0]<param['Result']['FF'][param['Settings']['source_time']*param['Settings']['source_repetitions']*25,0])

mask= ( NF_norm[:,0]>datetime(2025,1,1,12,54,20)) & ( NF_norm[:,0]<datetime(2025,1,1,12,59,30)) 
mask2= ( NF_norm[:,0]>datetime(2025,1,1,13,15)) & ( NF_norm[:,0]<datetime(2025,1,1,13,49)) 
PL.Plot_PSD((param['Settings']['bin_mids'],NF_norm[mask]),(param['Settings']['bin_mids'],NF_norm[mask2]),ylog=True,y_lim=(1.1,1.2E4))
plt.xlim(10,1E4)
fig,ax=PL.Plot_timeseries(param['Result']['FF'], param['Settings']['bin_edges'],y_tot=(0,0),y_3d=(1,1.1E4),log=0,datatype='number',normal=False)
ax[0].axvspan(NF_norm[mask][0,0],NF_norm[mask][-1,0],color='black',alpha=0.3,label='Par')
ax[0].axvspan(NF_norm[mask2][0,0],NF_norm[mask2][-1,0],color='black',alpha=0.3,label='Par')
ax[0].plot(param['Result']['FF'][:,0][mask],param['Result']['FF'][:,1][mask],color='black',lw=4)
ax[0].plot(param['Result']['FF'][:,0][mask2],param['Result']['FF'][:,1][mask2],color='black',lw=4)
plt.xlim(datetime(2025,1,1,12,52,20),datetime(2025,1,1,13,45))

# NF_surf=UL.num2surface(NF_norm,param['Settings']['bin_mids'],unit='mum2')
# PL.Plot_PSD((param['Settings']['bin_mids'],NF_surf[mask]),(param['Settings']['bin_mids'],NF_surf[mask2]),ylog=True,y_lim=(0,0),datatype='surface')
# fig,ax=PL.Plot_timeseries(NF_surf,param['Settings']['bin_edges'],log=0,datatype='surface',y_3d=(0.01,1E5),y_tot=(0,8.2E5))
# ax[0].plot(NF_surf[:,0][mask],NF_surf[:,1][mask],color='black',lw=4)
# ax[0].plot(NF_surf[:,0][mask2],NF_surf[:,1][mask2],color='black',lw=4)

# NF_mass=UL.num2mass(NF_norm,param['Settings']['bin_mids'],unit='mug')
# PL.Plot_PSD((param['Settings']['bin_mids'],NF_mass[mask]),(param['Settings']['bin_mids'],NF_mass[mask2]),ylog=True,y_lim=(0.000,0),datatype='mass')
# PL.Plot_timeseries(NF_mass,param['Settings']['bin_edges'],log=0,datatype='mass',y_3d=(1E-5,1E5))

# PL.Plot_PM_timeseries(param['Result']['FF'],param['Settings']['bin_edges'],PM_values=[0.1,0.2,0.5,1,2.5,4,10])
# PL.Plot_PM_timeseries(UL.num2mass(param['Result']['FF'],param['Settings']['bin_mids'],param['Settings']['rho'],'mug'),
#                       param['Settings']['bin_edges'],datatype='mass',PM_values=[0.1,0.2,0.5,1,2.5,4,10])
# start   =   param['Result']['FF'][0,0]
# end     =   param['Result']['FF'][param['Settings']['source_time']*param['Settings']['source_repetitions'],0]
# Howsa=UL.Peak_fitter(param['Result']['FF'],start,end,param['Settings']['bin_mids'],0,30,60)

# Emission_list=[]
# error_list=[]
# for i in param['Settings']['bin_mids']:
#     Emission_list.append(Howsa[str(i)]['Emission_rate'])
#     error_list.append(Howsa[str(i)]['SE_E'])