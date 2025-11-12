
from typing import Dict

from ..utils import to_oneD_array

import numpy as np


def hbv(
        pcp: np.ndarray,
        temp: np.ndarray,
        evap: np.ndarray, 
        parameters:Dict[str, float], 
        initialize:bool = True,
        # init_days:int = None, todo
        routing:bool = False,
        )->np.ndarray: 

    """
    HBV hydrological model implementation in numpy

    Parameters
    ----------
    pcp : np.ndarray
        array of precipitation values (mm/day)
    temp : np.ndarray
        array of temperature values (°C)
    evap : np.ndarray
        array of potential evapotranspiration values (mm/day)
    parameters : Dict[str, float]
        dictionary of model parameters:

        - 'BETA': Shape coefficient for soil moisture accounting
        - 'FC': Field capacity of the soil (mm)
        - 'K0': Recession coefficient for quick runoff component (1/day)
        - 'K1': Recession coefficient for upper groundwater zone (1/day)
        - 'K2': Recession coefficient for lower groundwater zone (1/day)
        - 'LP': Evapotranspiration correction factor
        - 'MAXBAS': Routing parameter (days)
        - 'PERC': Percolation rate from upper to lower groundwater zone (mm/day)
        - 'UZL': Threshold for upper groundwater zone (mm)
        - 'PCORR': Precipitation correction factor
        - 'TT': Temperature threshold for snow/rain separation (°C)
        - 'CFMAX': Degree-day factor for snowmelt (mm/°C/day)
        - 'SFCF': Snowfall correction factor
        - 'CFR': Refreezing coefficient
        - 'CWH': Water holding capacity of snowpack
    routing : bool
        whether to apply routing to the simulated runoff or not    
    initialize : bool
        whether to initialize the model by running it once with the whole input data or not

    Returns
    -------
    np.ndarray
        array of simulated discharge values (mm/day)
    
    """

    assert pcp.shape == temp.shape == evap.shape, \
        "Input arrays must have the same shape"
    pcp = to_oneD_array(pcp, dtype=np.float32)
    temp = to_oneD_array(temp, dtype=np.float32)
    evap = to_oneD_array(evap, dtype=np.float32)

    for k,v in parameters.items():
        assert isinstance(v, (int, float)), f"Parameter {k} must be int or float"
    
    # Initialize time series of model variables
    SNOWPACK = 0.001
    MELTWATER = 0.001
    SM = 0.001
    SUZ = 0.001
    SLZ = 0.001
    ETact = 0.001
    Qsim = np.full(pcp.shape, dtype=np.float32, fill_value=np.nan)
    Qsim[:] = 0.001
    
    # Start loop

    if initialize:
        init_days = pcp.shape[0]-1
    else:
        init_days = 0
    
    time_step = 0
    init_day_counter = 0
    init_done = False

    total_steps = 0

    while time_step<pcp.shape[0]:
        
        # Separate precipitation into liquid and solid components
        PRECIP = pcp[time_step]*parameters['PCORR']
        RAIN = np.multiply(PRECIP, temp[time_step]>=parameters['TT'])
        SNOW = np.multiply(PRECIP, temp[time_step]<parameters['TT'])
        SNOW = SNOW*parameters['SFCF']
        
        # Snow
        SNOWPACK = SNOWPACK+SNOW
        melt = parameters['CFMAX']*(temp[time_step]-parameters['TT'])
        melt = melt.clip(0,SNOWPACK)
        MELTWATER = MELTWATER+melt
        SNOWPACK = SNOWPACK-melt
        refreezing = parameters['CFR']*parameters['CFMAX'] * (parameters['TT']- temp[time_step])
        refreezing = refreezing.clip(0,MELTWATER)
        SNOWPACK = SNOWPACK+refreezing
        MELTWATER = MELTWATER-refreezing
        tosoil = MELTWATER-(parameters['CWH']*SNOWPACK)
        tosoil = tosoil.clip(0,None)
        MELTWATER = MELTWATER-tosoil

        # Soil and evaporation
        soil_wetness = (SM/parameters['FC']) ** parameters['BETA']
        # soil_wetness = soil_wetness.clip(0,1.0)
        soil_wetness = max(0., min(soil_wetness, 1.0))
        recharge = (RAIN+tosoil) * soil_wetness
        SM = SM+RAIN+tosoil-recharge
        excess = SM-parameters['FC']
        excess = excess.clip(0,None)
        SM = SM-excess
        evapfactor = SM/(parameters['LP']*parameters['FC'])
        # evapfactor = evapfactor.clip(0,1.0)
        evapfactor = max(0., min(evapfactor, 1.0))
        ETact = evap[time_step]*evapfactor
        ETact = np.minimum(SM, ETact)
        SM = SM-ETact

        # Groundwater boxes
        SUZ = SUZ+recharge+excess
        PERC = np.minimum(SUZ, parameters['PERC'])
        SUZ = SUZ-PERC
        Q0 = parameters['K0']*np.maximum(SUZ-parameters['UZL'], 0.0)
        SUZ = SUZ-Q0
        Q1 = parameters['K1']*SUZ
        SUZ = SUZ-Q1
        SLZ = SLZ+PERC
        Q2 = parameters['K2']*SLZ
        SLZ = SLZ-Q2
        Qsim[time_step] = Q0+Q1+Q2
        
        time_step = time_step+1
        init_day_counter = init_day_counter+1

        total_steps += 1
                
        # Go back to date_start once we've reached the initialization period
        if (init_done==False) & (init_day_counter==init_days): 
            time_step = 0
            init_done = True

    ## Check water balance closure
    #storage_diff = SM[-1]-SM[0]+SUZ[-1]-SUZ[0]+SLZ[-1]-SLZ[0]+SNOWPACK[-1]-SNOWPACK[0]+MELTWATER[-1]-MELTWATER[0]
    #error = np.mean(P*365.25) - np.mean(Qsim*365.25) - np.mean(ETact*365.25) - (365.25*storage_diff/len(P))
    #if error > 1:
    #    print "Warning: Water balance error of "+str(round(error*1000) / 1000)+" mm/yr"

    if routing:
        # Add routing delay to simulated runoff, uses MAXBAS parameter
        parameters['MAXBAS'] = np.round(parameters['MAXBAS']*100) / 100
        window = parameters['MAXBAS']*100
        if int(window) < 0:
            raise ValueError(f"MAXBAS parameter must be greater than 0 {parameters['MAXBAS']}")
        
        w = np.empty(int(window))
        for x in range(0, int(window)):
            w[x] = window/2 - abs(window/2-x-0.5)
        w = np.concatenate([w, [0.0]*200])
        w_small = [0.0]*int(np.ceil(parameters['MAXBAS'])) 
    
        #w_small = np.arange(2, 10, dtype=np.float)
        for x in range(0, int(np.ceil(parameters['MAXBAS']))):
            w_small[x] = np.sum(w[x*100:x*100+100])
        w_small = w_small/np.sum(w_small)
        Qsim_smoothed = np.full(Qsim.shape, fill_value=0.0)

        for ii in range(len(w_small)):
            Qsim_smoothed[ii:len(Qsim_smoothed)] = Qsim_smoothed[ii:len(Qsim_smoothed)] \
                + Qsim[0:len(Qsim_smoothed)-ii]*w_small[ii]
        
        Qsim = Qsim_smoothed
    
    return Qsim
