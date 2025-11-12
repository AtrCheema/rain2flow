
__all__ = ['hbv_nb']

from typing import Dict

import numba as nb
import numpy as np

from ..utils import to_oneD_array


def hbv_nb(
        prec:np.ndarray,
        temp:np.ndarray,
        pet:np.ndarray,
        parameters:Dict[str, float], 
        initialize:bool = True,
        routing:bool = True,
        )->np.ndarray: 

    """
    Same as hbv function above but compatible with numba!
    parameters : dict
        a dictionary of 15 model parameters
    routing : bool
        whether to apply triangular routing to the output or not
    initialize : bool
        whether to initialize the model by running it once with the whole input data or not
    """

    prec = to_oneD_array(prec, dtype=np.float32)
    temp = to_oneD_array(temp, dtype=np.float32)
    pet = to_oneD_array(pet, dtype=np.float32)

    assert len(prec) == len(temp) == len(pet), "Input arrays must have the same length"

    assert isinstance(parameters, dict), "Parameters must be a dictionary"
    assert all([k in parameters for k in [
        'PCORR', 'TT', 'SFCF', 'CFMAX', 'CFR', 'CWH', 
        'FC', 'BETA', 'LP', 'K0', 'K1', 'K2', 
        'UZL', 'PERC', 'MAXBAS']]), "Missing required parameters"

    assert all([isinstance(v, (int, float)) for v in parameters.values()]), "All parameters must be floats"

    assert len(parameters) == 15, f"Expected 15 parameters for HBV, got {len(parameters)}"

    parameters = {k:float(v) for k,v in parameters.items()}
    
    return hbv_(
        prec,
        temp,
        pet,
        #**{k:np.array([v]) for k,v in parameters.items()},
        **parameters,
        initialize=initialize,
        routing=routing,
    )    


@nb.jit(nopython=True)
def hbv_(
        p,    #:np.ndarray,
        temp,
        pet,
        # ** parameters **
        PCORR,
        TT,
        SFCF,
        CFMAX,
        CFR,
        CWH,
        FC,
        BETA,
        LP,
        K0,
        K1,
        K2,
        UZL,
        PERC,
        MAXBAS,

        initialize,
        routing,

        # ** states **
        q0 = 0.001,
        snowpack0 = 0.001,
        MELTWATER0 = 0.001,
        SM0 = 0.001,
        SUZ0 = 0.001,
        SLZ0 = 0.001,
        ETact0 = 0.001,
        ):

    # Initialize time series of model variables
    SNOWPACK = snowpack0
    MELTWATER = MELTWATER0
    SM = SM0
    SUZ = SUZ0
    SLZ = SLZ0
    ETact = ETact0

    Qsim = np.full(len(p), np.nan, dtype=np.float32)
    Qsim[0] = q0

    # Start loop

    if initialize:
        init_days = p.shape[0]-1
    else:
        init_days = 0
    
    time_step = 0
    init_day_counter = 0
    init_done = False

    total_steps = 0

    while time_step < p.shape[0]:
        
        # Separate precipitation into liquid and solid components
        PRECIP = p[time_step] * PCORR
        RAIN = np.multiply(PRECIP, temp[time_step]>= TT).item()
        SNOW = np.multiply(PRECIP, temp[time_step]< TT).item()
        SNOW = SNOW * SFCF
        
        # Snow
        SNOWPACK = SNOWPACK+SNOW
        melt = CFMAX * (temp[time_step] - TT)
        # melt = np.clip(melt, 0., SNOWPACK)
        melt = max(0., min(melt, SNOWPACK))
        MELTWATER = MELTWATER + melt
        SNOWPACK = SNOWPACK - melt
        refreezing = CFR * CFMAX * (TT- temp[time_step])
        #refreezing = np.clip(refreezing, 0, MELTWATER)
        refreezing = max(0., min(refreezing, MELTWATER))
        SNOWPACK = SNOWPACK + refreezing
        MELTWATER = MELTWATER - refreezing
        tosoil = MELTWATER-(CWH * SNOWPACK)
        # tosoil = np.clip(tosoil, 0, None)
        tosoil = max(0., tosoil)
        MELTWATER = MELTWATER - tosoil

        # Soil and evaporation
        soil_wetness = (SM / FC) ** BETA
        #soil_wetness = soil_wetness.clip(0,1.0)
        # soil_wetness = np.clip(soil_wetness, 0,1.0)
        soil_wetness = max(0., min(soil_wetness, 1.0))
        recharge = (RAIN + tosoil) * soil_wetness
        SM = SM+RAIN + tosoil - recharge
        excess = SM - FC
        # excess = np.clip(excess, 0, None)
        excess = max(0., excess)
        SM = SM-excess
        evapfactor = SM/(LP * FC)
        # evapfactor = np.clip(evapfactor, 0, 1.0)
        evapfactor = max(0., min(evapfactor, 1.0))
        ETact = pet[time_step] * evapfactor
        ETact = np.minimum(SM, ETact)
        SM = SM-ETact

        # Groundwater boxes
        SUZ = SUZ + recharge + excess
        #perc = np.minimum(SUZ, PERC)
        perc = min(SUZ, PERC)
        SUZ = SUZ - perc
        # Q0 = K0 * np.maximum(SUZ- UZL, 0.0)
        Q0 = K0 * max(SUZ - UZL, 0.)
        SUZ = SUZ-Q0
        Q1 = K1 * SUZ
        SUZ = SUZ-Q1
        SLZ = SLZ + perc
        Q2 = K2 * SLZ
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
        MAXBAS = np.round(MAXBAS * 100) / 100
        window = MAXBAS * 100
        if int(window) < 0:
            raise ValueError(f"MAXBAS parameter must be greater than 0 {MAXBAS}")
        
        w = np.empty(int(window), dtype=np.float32)
        for x in range(0, int(window)):
            w[x] = window/2 - abs(window/2-x-0.5)
        # w = np.concatenate([w, [0.0]*200])
        # w = np.concatenate([w, np.full(200, 0.0, dtype=np.float32)])
        w = np.append(w, np.full(200, 0.0, dtype=np.float32))

        # w_small = [0.0] * int(np.ceil(MAXBAS)) 
        w_small = np.full(int(np.ceil(MAXBAS)) , fill_value=0.0, dtype=np.float32)
    
        #w_small = np.arange(2, 10, dtype=np.float)
        for x in range(0, int(np.ceil(MAXBAS))):
            w_small[x] = np.sum(w[x*100:x*100+100])

        w_small = w_small/np.sum(w_small)
        Qsim_smoothed = np.full(Qsim.shape, fill_value=0.0, dtype=np.float32)

        for ii in range(len(w_small)):
            Qsim_smoothed[ii:len(Qsim_smoothed)] = Qsim_smoothed[ii:len(Qsim_smoothed)] \
                + Qsim[0:len(Qsim_smoothed)-ii]*w_small[ii]
        
        Qsim = Qsim_smoothed
    
    return Qsim
