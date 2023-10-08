#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
# cython: profile=True
# saved???????????
# hm??????

# This program creates synthetic nest data and runs it through the
# MC nest survival algorithm. The purpose of this code is to make
# sure the MC estimates are being computed correctly.

# GENERAL NOTES:

# 4 Oct - Deleted a bunch of comments when I renamed the file. 
#         The script runs through without errors, but gives weird output.

# 6 oct - took out a bunch of typing info - can always add back in
# need to consider the type for numpy/decimal vars but also where they're stored?

# CYTHON NOTES:

# can add a .pxd (definition) file to make cdefs shareable with other modules using cimport
# *** should I be declaring all the types at the beginning, even if I haven't assigned anything to them?
# DTYPE_t is the C compile-time version of DTYPE. use if with all cdef statements instead of DTYPE.
# Still need to figure out how to integrate Cython and numpy in same pyx file - numpy is more efficient at some things
# can I use a regular python function (def) for vectorized computations (which would require a loop in C, I think)?
# need to figure out which types are precise, and use those where needed while using less precise elsewhere to save memory
# C-type things a little at a time to avoid breaking everything at once (oops...)
# use memoryviews instead of indexing arrays - type[:] = 1d, type[:,:] = 3d, etc for whatever type you're declaring
# balance btw making new functions to replace vectorized operations and somehow sequestering numpy code?
# for global vars to work, does it need to be cpdef?
# global pstorm, storm_start # NOTE: this does not get globals from python environment
    # can't mix cython memoryview (LHS) with numpy array (RHS)
    # may not be much speed benefit to making nestData a memoryview:
        #     Cannot convert nestsim._memoryviewslice to numpy.ndarray

    # -----------------------------------------------------------------------------------------------   
    # could probably also change whole columns (obs days) at once based on the init date values column 
    

# Questions:

# Difference between DTYPE and the actual type. How to type for things like matrix algebra, which seem to want only an integer?
# Why do globals from python become "NoneType" inside the Cython function?
# How to deal with survey_days when obs_int changes. maybe just need 3 diff arrays?
# why are there so many "weird" nests? (lots of NA values but make it to optimizer)
# there seem to be more of them than earlier tonight, even at the beginning of the loop

# older questions:
## need to introduce an observer that can mis-classify?
## no, just assign  certain percentage to an "unknown" category?
## then look at what happens when we are more likely to be uncertain about hatch than predation,
## or vice versa
## increase number of nests - more accurate. what number is needed?
            # does the timing of fail vs observer matter? i.e. if nest fails on day 28 and is observed on day 28,
            # does it fail before or after it's observed?



# NOTES FOR THIS SCRIPT:
# maybe best to avoid global declarations
# pass storm_days as an argument like before?
# is it OK to have the first survey day be the "start of the season"



########################################################################
## scenarios:

## 1. 3-day nest check interval
## 2. 7-day nest check interval
## 3. 5% of hatched nests mis-assigned to failed
## 4. 5% of hatched nests marked as uncertain fate and not included
## 5. 5% of all nests mis-assigned
## 6. 10% of all nests mis-assigned

# deleted a bunch of comments from sim_data.py
########################################################################

import cython
import decimal
# from decimal import Decimal
#from   decimal import * # why doesn't this import the exception class?
from datetime import datetime
# started = datetime.now() 
# from itertools import product
import itertools
# from line_profiler import LineProfiler 
import numpy as np
import os 
# from pandas import DataFrame
from   pathlib import Path 
from   scipy import optimize

cimport numpy as np
np.import_array()

DTYPE = np.float64
iDTYPE = np.int64
ctypedef np.float64_t DTYPE_t
ctypedef np.int64_t iDTYPE_t

from libc.stdint cimport uintptr_t
from libc.math cimport sin, cos, atan, sqrt, exp, isnan

# ctypedef fused allfloat:
#     np.longdouble_t
#     np.float_t
#     float

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# np.set_printoptions(threshold=12)

# rng = np.random.default_rng(seed=61389) # call random number generator
# rng = np.random.default_rng(seed=71358)  # call random number generator
rng = np.random.default_rng()  # call random number generator

# don't want a global rng. which functions create random numbres?
# these are all the places where "rng" is that aren't commented out:
#   1. stormGen - need to be consistent for each time it's called w/in a set
#                 but also vary between param sets
#   2. mk_obs: 
#        a. draw random initiation dates from distribution
#        b. random probability vector for each nest to compare to probSurv
#        c. storm mortality probability
#        d. probability of being discovered 
#
#   3. runOpt: random initial values for the optimizer

# SMALL FUNCTIONS #############################################################
# In order:
#   triangle: remaps vals from R^2 to lower left triangle of unit square
#   logistic: is the logistic function
#   in1d_sorted: compute intersection of arrays more quickly 
#   n_inter: 
#   to_dec: transform float to "decimal" type for added precision


cdef triangle(DTYPE_t x0, DTYPE_t y0):
    cdef DTYPE_t  r0, m, theta, r3, x3, y3
    cdef np.ndarray[DTYPE_t, ndim=1] ret = np.empty(shape=2) 
    if y0 > x0:
        ret = triangle(y0, x0)
        return ret # is this working??
    r0 = sqrt( x0**2 + y0**2)
    m  = 1.0

    if y0 != x0:
        m = y0/x0
    theta = atan(m)
    r3    = r0 * 1.0/(1.0 + m)
    x3    = r3 * cos(theta)
    y3    = r3 * sin(theta)
    cdef np.ndarray[DTYPE_t, ndim=1] retpair = np.array([x3, y3])
    return retpair


@cython.profile(False)
# cdef DTYPE_t logistic(DTYPE_t x):
def logistic(x):
    return 1.0/( 1.0 + exp(-x) )


def in1d_sorted(A, B): 
    # possible days observer could see nest: alive days in survey days
    # print("search for B:", B, "in A:", A) 
    # in ordered array combining A and B, what are indices of B values?
    idx = np.searchsorted(B, A) 
    # len(B) = out of bounds for B, so can't be part of intersection:
    idx[idx==len(B)] = 0 
    # return vals of A for which the vals in line 1 == vals in A
    return A[B[idx] == A]       
    # would need a for loop to compare statically typed vars


def n_inter(Y, Z, Bool):
    return Y[Z==Bool]


from decimal import Decimal

@cython.profile(False)
cpdef to_dec(DTYPE_t val):
    return(Decimal(val))
                                                           

# STORM GENERATION FUNCTION ##################################################
#
# process:
# - 1 start day for each storm in season; randomly chosen
# - subsequent storm days = range(start, duration, step=1)
# - add each to start; makes sequential dates (no order-costly to sort)
# NOTES:
#     rng.choice can pick w/o replacement (unlike sample from distr)
#     pstorm is real weekly storm probabilities from 1975-2021
#     storm start is the start day of each week 
#     more than 1 storm/week seems unrealistic anyway
# NB: if you call rng here, storms will always be the same 
#     need a way to have 162 different sets of storm days generate
#     from the same rng call

def stormGen(frequency, duration, sstart, pstorm):
    # print("args:", frequency, duration, sstart, pstorm)
    # print("possible storm start days:", sstart)
    # rng_ = np.random.default_rng(seed=71358) # do this part in python script
    # rng_  = np.random.default_rng() # don't seed the rng
    # just create nests in only one spot in the script and then pass as args 
    out       = rng.choice(a=sstart, size=frequency, replace=False, p=pstorm)
    print("random start days:", out)
    print("p storm:", list(pstorm))
    dr        = np.arange(0, duration, 1) 
    print(dr)
    stormDays = [out + x for x in dr]    
    print("storm days= ", stormDays)
    # # flatten to convert to a list instead of list of tuples:
    stormDays = np.array(stormDays).flatten() 
    return(stormDays)
   

# PAIRS FUNCTION #############################################################
#   list all possible sequential pairs from a list of dates
#   represents the first and last days of each observation interval
#   used later in the matrix calculations
#   index from 1 (so arr[x-1] exists) to x_max (excluded by range)

@cython.profile(False)
# cdef pairs(iDTYPE_t[:] arr):
def pairs(arr):
    # cdef Py_ssize_t x_max = arr.shape[0]
    x_max = arr.shape[0]
    # print("days:", str(arr))
    # print("PAIR FUNCTION: days", list(arr), "max index", x_max-1)
    # pair1 = np.zeros(shape=(2, x_max-1), dtype=np.int64)
    # pair1 = )
    # pair  = pair1.copy()
    # pair  = []
    

    # print("PAIR FUNCTION: days", arr, "max index", x_max-1)
    pair = np.zeros(shape=(2, x_max-1))
    # cdef Py_ssize_t x

    for x in range(1, x_max):
        pair[1,x-1] = arr[x]       
        # pair[x-1] = arr[x]       
        pair[0,x-1] = arr[x-1] 
    return pair


# SURVEY DAYS WITHOUT STORMS ##################################################
#   creates the survey days for a given param set
#   then makes sure none of them are storm days

cdef rmStorm(storm_days, obs_int):
    # print(storm_days)
    cdef np.ndarray survey_days = np.arange(
        start, breedingDays, obs_int, dtype=np.int64
        )
    # print(survey_days)
    # keep only values that aren't in storm_days:
    survey_days = survey_days[np.isin(survey_days, storm_days)==False] 
    # print(survey_days)
    return(survey_days)


# def set_par(params):
#     stormFreq = params[4]

# cpdef get_glob(x):
#     return global x

# NOTE: took out notes here

# ######################################################################
# #################### GLOBAL VARS + TYPES #############################
# region ###############################################################

# for now, make types of everything compatible
#     later, can try converting to c types for speed
# is it still best to avoid global vars where possible?


index        = 0
# index2       = 0
num_out      = 19
breedingDays = 150
discProb     = 0.8
start        = 1

# endregion ############################################################


########################################################################
################## FUNCTIONS ###########################################
# region ###############################################################

# NEST CREATION FUNCTION ###############################################
#   this function makes nests
#   unlike in older versions, now it doesn't return python objects
#   this means we have to calculate things like intervals later on
#   now this happens inside the "like" function

cpdef mk_obs(
        DTYPE_t[:] params, DTYPE_t[:] initprob, iDTYPE_t[:] storm_days,
        int[:] dates, int start
        ): 
    cdef iDTYPE_t num_nests = int(params[0])
    cdef DTYPE_t probSurv   = params[1] 
    cdef DTYPE_t SprobSurv  = params[2]   
    cdef iDTYPE_t obs_int   = int(params[6])
    cdef iDTYPE_t hatchTime = int(params[5])
    cdef DTYPE_t probMort_Flood     = params[8]
    cdef np.ndarray[DTYPE_t, ndim=1] probVec 
    cdef int nCol = 10 # num columns in output array
    survey_days = rmStorm(storm_days, obs_int)
    cdef np.ndarray[iDTYPE_t, ndim=2] nestData = np.zeros(
            shape=(num_nests, nCol), dtype=np.int64)
    
    probVec   = np.array([
        SprobSurv if i in storm_days else probSurv 
        for i in np.arange(start, start+breedingDays)
        ])
    stormTrue = (probVec == SprobSurv) # T/F storm on given day 
    TF_array  = np.zeros(shape=(num_nests,1))
    #  Column 1: nest ID -------------------------------------------------
    nestData[:,0]  = np.arange(num_nests)
    #  Column 2: initiation date, from real distribution -----------------
    initiation     = rng.choice(a=dates,size=num_nests, p=initprob)
    nestData[:,1]  = initiation 
    print("init dates: ",initiation)

    # Can plot the distribution of synthetic nest initiation dates ---------
    #  from matplotlib import pyplot as plt 
    #  plt.hist(initiation, bins=[
    #    0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,
    #    105,110,115,120,125,130,135])
    #  plt.title("nest initiation dates")
    #  #filepath2 = Path('{}/init_{}'.format(dir_name, repID))
    #  #filepath2 = Path('output/',dir_name/'init_{}'.format( repID))
    #  filepath2 = Path.cwd() / ('output') / (dir_name ) / ('init_' + repID)
    #  filepath2.parent.mkdir(parents=True, exist_ok=True)  
    #  plt.savefig(filepath2)
    # ----------------------------------------------------------------------

    # These columns will be filled in later:
    #     Column 3 = failure/hatch date                  
    #     Column 4 = cause of failure 
    #     Column 5 = I (first found)                         
    #     Column 6 = J (last active)
    #     Column 7 = K (last checked)                         
    #     Column 8 = fate assigned by observer
    #     Column 9 = length of time nest is active            
    #     Column 10 = storm while active (T/F)

    for n in range(num_nests):   # FOR EACH NEST CREATED
        #    STEPS:
        # 1. create index array for dates in incubation period
        # 2. period = whole incubation period, not just when active
        # 3. extract storm/regular survival probabilities for those days        
        # 4. vector of random probabilities for each day 
        # 5. if prob < p, nest stays alive for that day
        # 6. make it so nests can't fail the day they are initiated
        #          nests are always alive on init day
        # 7. stormTrue_int: when does storm=True during potential incubation
        print('----------------------------------------------------------------') 
        init          = nestData[n,1]                
        print("nest ID = ", nestData[n, 0], "| initiation =", init)
        period        = np.arange(init, init+hatchTime).astype(int)
        p             = probVec[period]                  
        prob          = rng.uniform(size=hatchTime) 
        alive         = np.less(prob, p)     
        alive[0]      = True                           
        stormTrue_inc = stormTrue[period]                
        if all(alive):
            nestData[n,3] = 1     # nest hatches if alive is all "True"
            hatch         = period[hatchTime-1]
            nestData[n,2] = hatch
            active        = sum(alive)   # number of days nest was active
            print("nest hatched on day", hatch, "! nest was alive: ", alive)
        else:   # if False in alive
            # fail date will be the smallest of value of period for which the corresponding alive = F
            fail               = np.amin(period[alive == False])  # fail date 
            alive[period>fail] = False              # set all days after that day to F in the 'alive' array
            nestData[n,2]      = fail                    # record to nest data
            active             = sum(alive) 
            print('nest failed on day', fail)
            print('nest was alive:', alive)
            if fail in storm_days:
                nestData[n,3] = 2   # during storms, all failed nests fail due to flooding 
                print("nest flooded during storm")
            # output of rng.binomial is successes over n trials
            # success = if random number is < probMortFlood
            elif rng.binomial(1, probMort_Flood, 1) == 1:
                nestData[n,3] = 2   # failed due to flood, normal day
                print("nest flooded, not during storm")
            else:
                nestData[n,3] = 3   # failed due to predation, normal day
                print("nest depredated")
        s = stormTrue_inc[0:(active)]  # 
        print("s=",s)
        storm_true    = any(s)               # T/F was this nest active during >1 storm?
        print("storm while active?", storm_true)
        # TF_array[n]   = storm_true  # make it its own array to tack on after we handle zeros
        TF_array[n]   = int(storm_true)  # make it its own array to tack on after we handle zeros
    
        ########## OBSERVER - full true nest history has already been recorded ##################################

        # print("time for observer")
        # print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print("> > > > > > > > > > > > > > > > > > > > > > > > ")
        alive_days = np.array(period[alive == True])
        print("time for observer - days nest is observable:", alive_days) # why isn't this being evaluated before error??
        
        # if alive_days is None:
        #     print("nest never alive")
        #     continue
        # if observable days>0, do the rest of this section; if not, skip                     
        if len(alive_days) > 0:
        # if len(alive_days) > 1: # make sure alive_days isn't a scalar so error is not raised
        # maybe need to check length before this line:
        # if len(alive_days) > 1 & len(survey_days) > 1: # make sure alive_days isn't a scalar so error is not raised
        # if alive_days.shape[0] > 1 & survey_days.shape[0] > 1: # make sure alive_days isn't a scalar so error is not raised
            # if they are both np.ndarray and len > 1, how is one of them too small?
            # now it's object of type 'NoneType' has no len() - try weeding out 0-length before evaluating
            # tried to take care of alive_days = None; how would survey_days be None??
            # according to debugger, survey_days and storm_days have not been created yet when error occurs...
            # does cython let you know if you try to assign an object of wrong type to a var name? thought it did...
            # tried adding dtype to survey_days, doesn't help
            # is the error even in the correct place?
            # they are both clearly not None, so maybe it's an error? 
            # back to trying to find a c/cython solution I guess...
            # maybe it's bc I didn't specify ndim for either of them??
            # this lets the code run but it must evaluate to False bc it then exits immediately
            # also don't know yet whether elif statements are working

            obs_choice = in1d_sorted(survey_days, alive_days)     # all possible observation days
            print('survey days when nest is observable:', obs_choice)
            prob_disc     = np.full(len(obs_choice), discProb)  # vector of same discovery probability for all possible days
            print('prob of discovery:', prob_disc)
            obs_prob      = rng.uniform(size=len(obs_choice))   # random probability for each of those days 
            discover      = np.less(obs_prob, prob_disc)        # if obs_prob is less than discProb, discover nest
            print("discovered?", discover)
            nestData[n,8] = len(alive_days)  # how long was nest active for?
        
            # maybe can make this and other cool python stuff into a python function? and then not have to worry 
            # about 
            if True in discover: # if at least one value is true, nest is discovered
                discover  = ~np.cumprod(~discover).astype(bool) # now all days after discovery date are discover == True
                # this is very clever - once there is a False, the product is zero and remains zero.
                obsDays = obs_choice[discover == True] # possible obs days where discover == True 
                print("days nest was observed as active: ", obsDays) # need to tack on extra obs for failed nests
                i = obsDays[0]       # which one is faster?
                nestData[n,4] = i

                # Now observer must assign a fate based on field cues -------------------------------------------------------------
                # - Assume more time since end date of nest = harder to assign fate
                # - Also assume that storms make it impossible to accurately assign fate 
                # - (True fate may have happened before storm, but observer will say nest was flooded)

                if False in alive:  # if there is at least one "False" in "alive" then the nest did not hatch
                    j = obsDays[-1] # for failed nests, j = last day nest was observed as active
                    nestData[n,5] = j
                    # get the index for the day 'j' in survey_days, then add 1 to get index of next possible obs
                    # i.e. the following value of survey_days (the next day a survey will happen)
                    next_ind      = np.argwhere(survey_days == j) + 1  # need an integer for addition (argwhere, not where)
                    # add on one more observation on the survey day *following* when the nest failed 
                    # this way observer can observe the failure
                    obsDays = np.append(obsDays, survey_days[next_ind])  # add on one more observation, taking storms into account
                    k = obsDays[-1]
                    nestData[n,6] = k
                    # This is the case where nests after storms are assigned unknown fate:
                    y = 2 # number of days between obs above which accuracy decreases
                    if (k - fail > y) | (True in stormTrue[j:k]):
                        fate = 9   # unknown fate if too much time has passed or storm occurred in last interval
                        nestData[n,7] = fate
                        print("assigned fate of unknown,", fate, ", bc of days since fate:", k-fail,", or storm in last interval:", stormTrue[j:k].astype(bool))
                    else: 
                        fate = nestData[n,3]   # correct fate assigned
                        nestData[n,7] = fate
                        print("assigned correct fate,", fate,". view days since fate:", k-fail,", and storm in last interval:", stormTrue[j:k].astype(bool))
                    # print("failed - fate, i,j,k:", fate, i, j, k)
                    print("marked as failed - fate, i,j,k:", fate, i, j, k)
                # NOTE: is this realistic? vv
                # hatch is more difficult to detect than failure:   
                else: 
                    if np.max(obsDays) == hatch: # if hatch occurs on last obs day, don't add an extra obs day
                        j = np.max(obsDays)      # j and k are both hatch day because the nest is still "alive"
                        nestData[n,5] = j
                        k = np.max(obsDays)
                        nestData[n,6] = k                  
                    else:
                        next_ind = np.argwhere(survey_days == np.max(obsDays)) + 1 # need integer for addition (argwhere, not where)
                        # add one more obs if nest hatched 1 day ago (assume chicks are still visible)
                        # need to ask whether j and k should actually be hatch date in this case
                        obsDays = np.append(obsDays, survey_days[next_ind])  # add on one more observation, taking storms into account
                        j = obsDays[-1]
                        nestData[n,5] = j
                        k = obsDays[-1]
                        nestData[n,6] = k
                    # if nest hatched more than x day(s) ago or storm in last interval, give it unknown fate
                    x = 1
                    if k-hatch > x | (True in stormTrue[j:k]): # need to have calculated k and j already
                        fate = 9   # unknown fate
                        nestData[n,7] = fate
                        print("assigned fate of unknown,", fate,", bc of days since hatch:", k-hatch,", or storm in last interval:", stormTrue[j:k].astype(bool))
                    else:
                        fate = 1   # fate assigned as "hatch"
                        nestData[n,7] = fate
                        print("assigned correct fate,", fate,". view days since hatch:", k-hatch,", and storm in last interval:", stormTrue[j:k].astype(bool))
                    #print("hatched - fate, i, j, k:", fate, i, j, k)
            else:
                print("not discovered")

        print(nestData[n,:])

        # elif len(alive_days) > 0 and len(alive_days) < 2: # don't think it ever got here, or it would have thrown error
            # obs_choice = survey_days[np.which(survey_days==alive_days)]
        # elif len(survey_days) > 0 and len(survey_days) < 2:
            # obs_choice = survey_days[np.which(survey_days==alive_days)]
        # else:
            # continue
        
    
    # Format the nest data:
    # nestData[nestData == 0]  = np.nan          # convert zeros to NaNs
    # nestData[0,0] = 0                          # make this index NaN back into a zero
    # TF_array = int(TF_array) # can't use int() on array
    
    nestData[:,8] = TF_array[:,0]              # tack on after converting to NaN but before removing NaN rows
    nestData[:,9] = obs_int
    # print(nestData[n,:])
    #nestData[nestData['ac`tive_len']>= obs_int] # remove nests with only one observation
    print("nests that were discovered:", np.where(nestData[:,4]))
    #            0          1         2         3              4           5               6             7         8             9          
    # columns = ['ID','initiation','end_date','true_fate','first_found','last_active','last_observed','ass_fate','active_len','storm_true']
    columns = ['ID','initiation','end_date','true_fate','first_found','last_active','last_observed','ass_fate','storm_true','obs_int']
    
    # cnames  = ', '.join([str(x) for x in columns])

    # columns = ['ID','initiation','end_date','true_fate','first_found',
    # #            0          1         2         3              4         
    
    # 'last_active','last_observed','ass_fate','storm_true','obs_int']
    #   5               6             7         8             9          

    # columns = ['ID','initiation','end_date','true_fate','first_found','last_active','last_observed','ass_fate','active_len','storm_true','long_obs1', 'long_obs2', 'long_obs3']
    # could add columns for long observation intervals (where storms occurred) but prob easier to do obs hist later
    # try using numpy.save instead of converting to pandas df and saving to csv (pickle is another option)
    # numpy.save is already optimized for use with numpy arrays and is faster than pickle for purely numeric arrays (like this one)
    # nestData = np.array(nestData)
    # # now = datetime.today().strftime('%d%b_%H%M%S')
    # now = datetime.today().strftime('%S')
    # # nests = DataFrame(nestData, columns=columns)
    # # filepath3 = Path.cwd() / ('output') / (dir_name ) / ('nest_' + repID)
    # # filepath3 = Path.cwd() / ('_output') / ('test / nestdata_' + now + '.csv')
    # # filepath3 = Path.cwd() / ('_output/test/n_' + now + '.csv')
    # filepath3 = Path.cwd() / ('_output/test/n_' + ind + '.csv')

    # filepath3.parent.mkdir(parents=True, exist_ok=True)  
    # # nests.to_csv(filepath3)
    # # np.savetxt(filepath3, nestData, delimiter=',', header=columns)
    # np.savetxt(filepath3, nestData, delimiter=',', header=cnames)

    #np.save(filepath3, nestData)

    ###print(nestData)   
    return(nestData) # return data as 2d numpy array instead of df
    
# LIKELIHOOD FUNCTION ##################################################

# This function computes the likelyhood of the data given the model parameter estimates.
#
# The model parameters are expected to be received in the following order:
# - a_s   = probability of survival during non-storm days
# - a_mp  = conditional probability of predation given failure during non-storm days
# - a_mf  = conditional probability of flooding given failure during non-storm days
# - a_ss  = probability of survival during storm days
# - a_mfs = conditional probability of predation given failure during storm days
# - a_mps = conditional probability of predation given failure during storm days

#@profile
# cdef like(
#         DTYPE_t a_s, DTYPE_t a_mp, DTYPE_t a_mf, DTYPE_t a_ss, DTYPE_t a_mfs, 
#         DTYPE_t a_mps, np.ndarray nestData, iDTYPE_t obs_int, ):
  
# cdef like(
#         DTYPE_t a_s, DTYPE_t a_mp, DTYPE_t a_mf, DTYPE_t a_ss, DTYPE_t a_mfs, 
#         DTYPE_t a_mps, np.ndarray n_Data, iDTYPE_t[:] storm_days  
#         # DTYPE_t a_mps, np.ndarray nestData, iDTYPE_t[:] storm_days  
#         ):
def like(a_s, a_mp, a_mf, a_ss, a_mfs, a_mps, n_Data, storm_days):
    #"nestsim.pyx:567:6: closures inside cpdef functions not yet supported" - keep as cdef?
    # print("nest data passed to like:", nestData)
    # print("nest data passed to like:", n_Data)
    # the new like function is much longer than the old one, because we need to construct
    # the nest info for each nest from just the values returned from mk_obs()
    # it also uses the observer data by default because the 1-day obs int is not interesting
    # did I just take the time-consuming part out of mk_obs and move it here instead of speeding things up overall?
    
    # This is the next place I need to call storm days
    obs_int = n_Data[0,9]
    survey_days = rmStorm(storm_days, obs_int)

    stillAlive  = np.array([1, 0, 0])
    mortFlood   = np.array([0, 1, 0])
    # mortPred    = np.array([0, 0, 1], dtype=np.intc) 
    mortPred    = np.array([0, 0, 1]) 
    # cdef int[:] still_Alive = stillAlive # these are 1x3 matrices, not 3x3
    # cdef int[:] mort_Flood  = mortFlood
    # cdef int[:] mort_Pred   = mortPred 
    # print("row 1 of nest data:", n_Data[1,:])
    # print('obs_int check: ', obs_int)
    # print('storm days check:', storm_days)
    # startMatrix  = np.array([[a_s,0,0], [a_mf,1,0], [a_mp,0,1]], dtype=np.intc)
    startMatrix  = np.array([[a_s,0,0], [a_mf,1,0], [a_mp,0,1]])
    # stormMatrix  = np.array([[a_ss,0,0], [a_mfs,1,0], [a_mps,0,1]], dtype=np.intc)
    stormMatrix  = np.array([[a_ss,0,0], [a_mfs,1,0], [a_mps,0,1]])
    # stormM  = np.array([[a_ss,0,0], [a_mfs,1,0], [a_mps,0,1]], dtype=np.int32)
    # startM  = np.array([[a_s,0,0], [a_mf,1,0], [a_mp,0,1]], dtype=np.int64)
    # stormM  = np.array([[a_ss,0,0], [a_mfs,1,0], [a_mps,0,1]], dtype=np.int64)
    # cdef int[:,:] start_Matrix = startMatrix
    # cdef int[:,:] storm_Matrix = stormMatrix
        # how is this matrix actually being incorporated during the analysis?
    # cdef np.ndarray[iDTYPE_t, ndim=1] nest, obsDays, stateF, stateI, TstateI
    # cdef np.ndarray[int, ndim=1] intList
    # cdef np.ndarray[iDTYPE_t, ndim=2] obs
    # cdef float lDay, lPer
    #cdef DTYPE_t lDay, lPer
    # cdef iDTYPE_t disc, endObs, fate, num, storm_nests
    # cdef int intElt
    # cdef Py_ssize_t i, row
    # COUNTERS    
    storm_nests      = 0 # counter for nests using storm matrix
    storm_nests_real = 0
    weird_nests      = 0
    # logLike = (0.0)          # initialize the overall likelihood counter
    # logLike = to_dec(logLike) # why did I initialize them both in the same place??
    logLike = Decimal(0.0)
    print("number of rows:", len(n_Data))

    print('#############################################################')
    for row in range(len(n_Data)):
        # NOTE: maybe the names should NOT be initialized outside the loop? is that why we are getting weird output?
    #            0          1         2         3              4           5               6             7         8             9          
    #columns = ['ID','initiation','end_date','true_fate','first_found','last_active','last_observed','ass_fate','storm_true','obs_int]
        
    # FOR EACH NEST -------------------------------------------------------------------------------------

        # nest    = row         # choose one nest (multiple output components from mk_obs())
        # print(row)
        nest = n_Data[row,] # Cannot convert nestsim._memoryviewslice to numpy.ndarray
        # print('#############################################################')
        # print('nest =',nest[0],"| fate=", nest[7],"| i=", nest[4],"| j=", nest[5], "| k=", nest[6], "| active days:", nest[8])
        disc    = nest[4].astype(int)   # first found
        # endObs  = nest[6]   # last observed
        endObs  = nest[6].astype(int)   # last observed
        fate    = nest[7].astype(int)   # assigned fate
        strue   = nest[8].astype(int)
        # obs_int = nest[9]
        # print("disc=", disc, "| endObs=", endObs)
        if strue:
            storm_nests_real = storm_nests_real + 1
        # print("NAs?", np.isnan(nest))
        # if np.any(np.isnan(nest) | np.less(nest, 1)): # if there are any np.isnan == True
        if (np.isnan(disc)):
            print("this nest was not discovered but made it through")
            # weird_nests = weird_nests + 1
            continue
        num     = len(np.arange(disc, endObs, obs_int)) + 1
        # could probably get obs_int from survey_days
        # or just tack it on the nest data
        # then pass in storm nests bc I don't have the params here
        # print("disc=", disc, "| endObs=", endObs, "| num=", num)
        # NOTE would it be easier to search storm days than survey days? 
        # also NOTE since I'm recreating the whole nest history and everything here, would
        # it be easier to just generate the rows of the nest df and then do this?
        # is there some faux-authenticity in looping through the nests that doesn't actually matter mathematically?
        # have to make this fast in some way for it to be worth externalizing it like this
        # observDays = in1d_sorted((np.linspace(disc, endObs,num=num)), survey_days)
        obsDays = in1d_sorted((np.linspace(disc, endObs,num=num)), survey_days)
        # print("days observed:", observDays)
        # print("days observed:", obsDays)
        # obsDays    = np.array(observDays, dtype=np.int64)
        obsPairs = pairs(obsDays)
        # obsPairs = np.fromiter(itertools.pairwise(obsDays), dtype=np.dtype((int,2))) # do elements of numpy arrays have to be floats?
        #          error: setting an array element with a sequence.
        
        # print("day 1", str(obsPairs[0,:])) # can't even convert one dimension to str
        # print("day 2", str(obsPairs[1,:]))
        # print("date pairs in observation period:", str(obsPairs) )
        # make a list of intervals between each pair of observations (necessary for likelihood function)
        int_List = obsPairs[1,:] - obsPairs[0,:]
        intList  = np.array(int_List, dtype=np.intc)
        # print("int list=", intList)
        # this makes each row an observation (1d matrix AKA array):
        # obs = np.full(( len(obsDays), 3), stillAlive, dtype=np.int64) # doesn't have "shape" arg
        obs     = [stillAlive for _ in range(len(obsPairs)+1)] # start off with all intervals = alive
        # change the last obs if nest failed:
        if fate == 2:
            obs[-1] = mortFlood
        elif fate == 3:
            obs[-1] = mortPred
        # print("fate, obs = ", fate, ",\n", obs) # check that last entry in obs corresponds to fate
        # print("obs (each row = 1 observation): \n", obs) # check that last entry in obs corresponds to fate
        # logLikelihood = (0.0)
        # logLikelihood = to_dec(logLikelihood)    # can't change type once you've assigned it?
        logLikelihood = Decimal(0.0)
        # for i in range(1, len(obs)): # remember "stop" (len(obs)) is omitted! 
        for i in range(len(obs)-1): # remember "stop" (len(obs)) is omitted! 
            # no, using this second one leads to ll=infinity
            # FOR EACH OBSERVATION OF THIS NEST ---------------------------------------------------------------------
            intElt  = (intList[i-1]).astype(int) # how was this working if we started at index 0?
                                    # which is the interval from the (i-1)th
                                  # to the ith observation
            stateF  = obs[i+1] 
            stateI  = obs[i]
            # stateF  = obs[i]       # how did this work in the pure python program?
            # stateI  = obs[i-1]         # nvm, it's clearly grabbing the right thing
            TstateI = np.transpose(stateI) # was this never returning the correct thing?? 
            # print("intElt:", intElt, "stateF:", stateF, "TstateI:", TstateI)
            if any(d in storm_days for d in range(i-1, i)):
                # if any of the days in the current observation interval (range) is also in storm days, use storm matrix
                # print("using storm matrix")
                lDay = np.dot(stateF, np.linalg.matrix_power(stormMatrix, intElt))
                # print("lDay", lDay)
                storm_nests = storm_nests + 1
                # this is the dot product of the current state of the nest and the storm matrix ^ interval length
           # look into using @ instead of nest dot calls 
            else:
                # print("using normal matrix")
                lDay = np.dot(stateF, np.linalg.matrix_power(startMatrix, intElt)) # sometimes getting "singular matrix" error
                # print("lDay", lDay)
            lPer = np.dot(lDay, TstateI)
            # print("lPer", lPer)
            # why is this zero? none of the vars that need arbitrary precision
            # (e.g. logLike, lPer, etc) has been c-typed 
            # logL = to_dec(np.log(lPer)) * -1
            logL = Decimal(- np.log(lPer))
            # print("ll from 1 observation:",logL)
            logLikelihood = logLikelihood + logL # add in the likelihood for this one observation
            # print("ll from this nest:",logLikelihood)
        logLike = logLike + logLikelihood        # add in the likelihood for the observation history of this nest
    print('#############################################################')
    print("nests that used storm matrix:", storm_nests)
    print("nests that experienced storms:", storm_nests_real)
    print("weird nests:", weird_nests)
    return(logLike)
    
# FUNCTION TO OPTIMIZE ################################################################################
#
# this function generates random initial values and runs "like" to see the likelihood of those values
# given the nest data
# optimizing it should optimize the values given to "like"

#@profile
# cpdef DTYPE_t like_smd(DTYPE_t[:] x, np.ndarray nData, iDTYPE_t obs_int):

# cpdef like_smd(DTYPE_t[:] x, np.ndarray nData, iDTYPE_t obs_int): 
cpdef like_smd(DTYPE_t[:] x, np.ndarray nData, iDTYPE_t[:] storm_days): 
    #        remember like is called from like_smd
    # going up from like args, need like_smd args to match
    # print("nest data passed to like_smd:",nData)
    # value to be minimized needs to be a float:
    # cdef DTYPE_t ret = 0.0
    ret = Decimal(0.0)

    # Step 1: Unpack the arguments:
    #
    # These are unbounded parameters that can take values from -inf to +inf. See below for details.
    #
    # s0   = survivorship
    # mp0  = conditional mortality from predation?
    # ss0  = survivorship during a storm
    # mps0 = conditional mortality from predation during a storm, seems awkward.
    #
    cdef DTYPE_t s0   = x[0]
    cdef DTYPE_t mp0  = x[1]
    cdef DTYPE_t ss0  = x[2]
    cdef DTYPE_t mps0 = x[3]
    # mp0  = x[1]
    # ss0  = x[2]
    # mps0 = x[3
    # mlInit = [s0, mp0, ss0, mps0]
    # print("random starting values for optimizer:", mlInit)

    # Step 2: Unpack the data
    #
    # data = arg[0]

    # Step 3: Transform the values so that they remain between 0 and 1
    #
    cdef DTYPE_t s1   = logistic(s0)
    cdef DTYPE_t mp1  = logistic(mp0)
    cdef DTYPE_t ss1  = logistic(ss0)
    cdef DTYPE_t mps1 = logistic(mps0)

    # Step 4: Further transformation to keep the values within the lower left triangle.
    #
    cdef np.ndarray[DTYPE_t, ndim=1 ] tri1 = triangle(s1, mp1) # remember triangle output is array
    cdef np.ndarray[DTYPE_t, ndim=1 ] tri2 = triangle(ss1, mps1)
    # cdef np.ndarray[iDTYPE_t, ndim=1 ] tri1 = triangle(s1, mp1) # remember triangle output is array
    # cdef np.ndarray[iDTYPE_t, ndim=1 ] tri2 = triangle(ss1, mps1)
    # cdef np.ndarray[np.int64, ndim=1 ] tri1 = triangle(s1, mp1) # remember triangle output is array
    # cdef np.ndarray[np.int64_t, ndim=1 ] tri2 = triangle(ss1, mps1)
    # cdef DTYPE_t tri2 = triangle(ss1, mps1)
    cdef DTYPE_t s2   = tri1[0]
    cdef DTYPE_t mp2  = tri1[1]
    cdef DTYPE_t ss2  = tri2[0]
    cdef DTYPE_t mps2 = tri2[1]

    # Step 5: Compute the depended conditional probability of mortality due to flooding.
    #
    cdef DTYPE_t mf2  = 1.0 - s2 - mp2
    cdef DTYPE_t mfs2 = 1.0 - ss2 - mps2

    # Step 6: Call the likelihood function
    #
    # The probabilities are passed in the following order
    # s2   = probability of survival during non-storm days
    # mp2  = conditional probability of predation given failure during non-storm days
    # mf2  = conditional probability of flooding given failure during non-storm days
    # ss2  = probability of survival during storm days (I guess this is also a
    #        conditional probability: Conditional probability of survival given there
    #        was a storm on that day.
    # mps2 = conditional probability predation given failure during storm days
    # mpf2 = conditional probability flooding given failure during storm days
    # data = the nest survivorship data
    #
    #warning1 = False
    mlInit = [s2, mp2, mf2, ss2, mps2, mfs2]
    print("random starting values for optimizer:", mlInit)
    ret = like(s2, mp2, mf2, ss2, mps2, mfs2, nData, storm_days)
    # 03 Aug: why does ret = infinity?
    print('like_smd(): Msg : ret = ', ret)
    return ret

# cdef printmem(int[:,:]):
#     print nrows()
# cdef runParam(float[:] paramsList):
#     cdef Py_ssize_t i
#     for i in range(0, len(paramsList)): # for each set of params

# region ######################################################################################################  
# THIS RUNS CODE FOR EACH INDIVIDUAL PARAM SET. iT CREATES NEST DATA REPLICATES WITHIN EACH PARAM SET.


cpdef runOpt(
        int nreps, int nruns, DTYPE_t[:] pstorm, int[:] dates, DTYPE_t[:] params,
        int start, DTYPE_t[:] initprob, iDTYPE_t[:] storm_start, dir_name
        ):
    # these args ARE mostly sourced from outside data/input, so maybe should stay so I can easily change them
    # In this function, need to:
    #   1. Load the params for this param set
    #   2. Create or import storm and survey days
    #   3. Try making the nest data
    #   4. Choose which nests to exclude 
    #   5. Call the rng for the starting vals
    #   6. Give it all to the optimizer

    # NOTE: 162 param combos - YIKES
    # NOTE: PARAM INDICES
    #    0         1          2         3           4           5       6       7           8               9      
    # numNests, ProbSurv, SprobSurv, stormDur, stormFreq, hatchTime, obsInt, discProb, probMortFlood, SprobMortFlood
 
    # NOTE: seem to be able to copy over values from memoryviews?
    # probably looping through each nest the optimizer works it out
    cdef iDTYPE_t numNests       = int(params[0])
    cdef DTYPE_t probSurv        = params[1] 
    cdef DTYPE_t SprobSurv       = params[2]   
    cdef iDTYPE_t stormDur       = int(params[3])
    cdef iDTYPE_t stormFreq      = int(params[4])
    cdef iDTYPE_t hatchTime      = int(params[5])
    cdef iDTYPE_t obs_int        = int(params[6])
    cdef DTYPE_t  probMortFlood  = params[8]
    cdef DTYPE_t  SprobMortFlood = params[9]

    print("~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*")                                                   
    print("~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*")                                                   
    par = np.array(
            [numNests, probSurv, SprobSurv, stormDur, stormFreq, hatchTime,
            obs_int, discProb, probMortFlood, SprobMortFlood ]
            )
    print("params for this set:", par)
    # storm days is not making any output, but it doesn't appear to be a type thing
    # random start days are being generated correctly.
    # the function itself matches the old one. What gives??
    # cdef np.ndarray[iDTYPE_t, ndim=1] storm_days = np.array(
        # stormGen(stormFreq, stormDur, storm_start, pstorm), dtype=np.int64)
    # print("storm days for this set of params:", storm_days)
    # storm_days = np.array(
    #     stormGen(stormFreq, stormDur, storm_start, pstorm), dtype=np.int64)
    # print("storm days for this set of params:", storm_days)
    cdef np.ndarray storm_days, survey_days
    s_days      = stormGen(stormFreq, stormDur, storm_start, pstorm) 
    print("p storm:", pstorm)
    storm_days  = np.array(s_days, dtype=np.int64)
    print("storm days for this set of params:", storm_days)
    survey_days = rmStorm(storm_days, obs_int)
    print("survey days for this set of params: ", survey_days)
    likeVal     = np.zeros(shape=(nreps*nruns, num_out))  
    nestDataBig = np.zeros(shape=(numNests*nreps, 10)) # the number of columns = nCol from inside mk_obs 
    print("start, stop, step: ", start, breedingDays, obs_int) # try print statements from inside the cython code
    # cdef int repID, discovered, excluded
    # cdef int discovered, excluded # for now, repID doesn't have to be int
    # cdef np.ndarray exclude
    # cdef np.ndarray[DTYPE_t, ndim=1] exclude 
    # cdef np.ndarray survey_days = np.arange(start, breedingDays, obs_int, dtype=np.int64) # error division by zero
    # survey_days = survey_days[np.isin(survey_days, storm_days)==False] # keep only values that aren't in storm_days
    cdef int nCol           = 10 # number of columns in table returned by this function
    # do I need to define these here instead of in mk_nests so they're in scope for both?
    # cdef np.ndarray[int, ndim=2] nestData = np.empty(shape=(numNests, nCol), dtype=np.int64) # empty array to fill w/ nest data for each replicate
    # cdef np.ndarray[iDTYPE_t, ndim=2] nest_Data = np.empty(
    #         shape=(numNests, nCol), dtype=np.int64
    #         ) # empty array to fill w/ nest data for each replicate
                                                       # will be returned by the function
    #            0          1         2         3              4           5               6             7         8             9          
    # columns = ['ID','initiation','end_date','true_fate','first_found','last_active','last_observed','ass_fate','storm_true', 'oba_int', 'obs_int']
    cdef np.ndarray nest_Data = np.zeros(shape=(numNests, nCol))
    # cdef iDTYPE_t[:,:] nd = nest_Data
    # print("nd col 4", nd[:,4])
    cdef int index2 = 0
    # cdef int ind    = 1
    # ind = 1
    # ind = 1
    for r in range(nreps):  # for each data replicate that uses this set of params
        # FOR EACH REPLICATE USING THIS PARAM SET -----------------------------------------------------------
        rID         = datetime.today().strftime('%H%M%S') # unique rep identifier
        print("this is data replicate ", rID)
        rep_ID      = int(f'{rID}{index2}')
        # print(type(rep_ID))
        repID       = int(f'{rID}{index2}')
        try:
            print("making nests")
            nest_Data    = mk_obs(
                params, initprob, storm_days, dates, start 
                )                                           # make nest data
        except IndexError:
            print("IndexError in nest data, go to next replicate")
            continue

        # nest_Data = np.array(nest_Data)
        # indstr = str(ind)
        # filepath3 = Path.cwd() / ('_output/test/n_' + indstr + '.csv')
        # stringy = 'output/test/n_'.join(str(ind))
        # filepath3 = Path.cwd() / ('output/test/n_' + str(ind) + '.csv') 
        # filepath3 = Path.cwd() / ( stringy + '.csv') 
        # filepath3   = f"/output/test/n_{ind}.csv"
        # now = datetime.today().strftime('%S')
        # # nests = DataFrame(nestData, columns=columns)
        # # filepath3 = Path.cwd() / ('output') / (dir_name ) / ('nest_' + repID)
        # # filepath3 = Path.cwd() / ('_output') / ('test / nestdata_' + now + '.csv')
        # filepath3 = Path.cwd() / ('_output/test/o_' + now + '.csv')
        # columns = ['ID','initiation','end_date','true_fate','first_found','last_active','last_observed','ass_fate','storm_true','obs_int']
        # cnames  = ', '.join([str(x) for x in columns])
        # # filepath3.parent.mkdir(parents=True, exist_ok=True)  
        # np.savetxt(filepath3, nest_Data, delimiter=',', header=cnames)
        # ind = ind + 1
        
        # NOTE: nestData [0]ID [1]init [2]end [3]fate [4]i [5]j [6]k [7]assigned fate [8]active duration [9]storm (T/F)
        # NOTE why is this still blue???
        print("number of nests before removal:", nest_Data.shape[0])
        print("remove undiscovered nests: ", np.argwhere(np.isnan(nest_Data).any(axis=1)).T )# r##print undiscovered nests
        nest_Data    = nest_Data[nest_Data[:,4] != 0] # trying to select only rows, not columns. but dimensions don't match
        # cdef iDTYPE_t[:,:] nd = nest_Data
        # discovered  =  nd.shape[0]                              # then count discovered nests
        discovered  =  nest_Data.shape[0]                              # then count discovered nests
        # get shape from a memoryview so we don't un-type nestData???
        print("number of nests after removing undiscovered:", nest_Data.shape[0], discovered)
        # select nests to be excluded: unknown fate or only one observation
        # exclude     = ((nest_Data[:,7] == 9) | (nest_Data[:,8]<obs_int))
        # print("observation length total:", nest_Data[:,6]-nest_Data[:,4])                         
        # exclude     = ((nest_Data[:,7] == 9) | ((nest_Data[:,6]-nest_Data[:,4])<obs_int))                         
        # the conditions for exclude may not make sense
        exclude = ((nest_Data[:,7]==9) | (nest_Data[:,6]==nest_Data[:,4]))
        excluded    = sum(exclude)      # exclude is a boolean array, sum gives you num True
        print("exclude these unknown-fate or 1-observation nests: ", nest_Data[exclude,:] )
        nest_Data    = nest_Data[~(exclude), :]    # remove excluded nests from data
        # cdef iDTYPE_t[:,:] nd = nest_Data
        print("number of nests after removing more nests:", nest_Data.shape[0])
        # storm_nests = sum(nd[:,9])                           # count num of nests active during >1 storm
        storm_nests = sum(nest_Data[:,9])                           # count num of nests active during >1 storm
        print("storm nests from df:", storm_nests)
        s   = rng.uniform(-10.0, 10.0)       # random initial values for optimizer
        mp  = rng.uniform(-10.0, 10.0)
        ss  = rng.uniform(-10.0, 10.0)
        mps = rng.uniform(-10.0, 10.0)
        z   = np.array([s, mp, ss, mps])
        print("main.py: Msg: Running optimizer")
        try:
            print("trying optimizer")
            ans  = optimize.minimize(
                like_smd, z, args=(nest_Data, storm_days), method='Nelder-Mead')
            #ans  = optimize.minimize(like_smd, z, args=(nestData), method='L-BFGS-B')
            ex = 0
        except decimal.InvalidOperation:
            print("Decimal error in optimizer - go to next replicate")
            ex = 100
            likeVal[r] = np.full(num_out, ex)
            continue            # skip the rest of this iteration, go to next
        except OverflowError:
            print("Overflow error in optimizer - go to next replicate")
            ex = 200
            likeVal[r] = np.full(num_out, ex)
            continue

        s0   = ans.x[0]         # Series of transformations of optimizer output.
        mp0  = ans.x[1]         # These make sure the output is between 0 and 1, 
        ss0  = ans.x[2]         # and that the three fate probabilities sum to 1.
        mps0 = ans.x[3]

        s1   = logistic(s0)
        mp1  = logistic(mp0)
        ss1  = logistic(ss0)
        mps1 = logistic(mps0)

        ret2 = triangle(s1, mp1)
        s2   = ret2[0]
        mp2  = ret2[1]
        mf2  = 1.0 - s2 - mp2

        ret3 = triangle(ss1, mps1)
        ss2  = ret3[0]
        mps2 = ret3[1]
        mfs2 = 1.0 - ss2 - mps2

        # These are the likelihood values for this data replicate:
        #                           0      1   2    3     4   5     6       7         8         9        10          11              12            13        14            15       16        17       18
        like_val    = np.array([repID, s2, mp2, mf2, ss2, mps2, mfs2, stormDur, stormFreq, probSurv, SprobSurv, probMortFlood, SprobMortFlood, hatchTime, numNests, obs_int, discovered, excluded, ex])
        likeVal[r]  = like_val # likelihood values for this data replicate recorded to larger array
        # 'NoneType' object does not support item assignment 
        # also, ret is infinity
        # sometimes it runs until the very end and sometimes it get stuck?
        
        print("likelihood vals for this replicate:", like_val)

        nestrows                            = nest_Data.shape[0]
        nestDataBig[index2:index2+nestrows] = nest_Data
        index2                              = index2 + 1
        
        print("index 2:", index2)
    
    # return()
    
    # nrows    = likeVal.shape[0]          # the number of rows per param set 
    # values[index:index+nrows,] = likeVal # fill the rows of "values" corresponding to the last param set
    # np.save(f, likeVal)                  # save likelihood values for this replicate to disk in case of error
    #                                      # save by leaving file open and writing to it
    #                                      # why does the nest file get a file extension but not this one?
                                         
    # index    = index + nrows             # increment index 

    now       = datetime.now().strftime("%H%M%S")
    # #nestfile  = Path.cwd() / ('output') / dir_name / ('nests' + now ) # don't add file extension?
    # # nestfile  = Path.cwd() / ('py_output') / dir_name / ('nests' + now ) # don't add file extension?
    nestfile  = Path.home() / '/mnt/c/Users/sarah/Documents/nest_models/py_output' / dir_name / ('nests' + now )
    nestfile.parent.mkdir(parents=True, exist_ok=True)
    np.save(nestfile, nestDataBig)                                    # save nest data for this replicate to file
    
    return(likeVal)


# cpdef run_params(int nreps, int[:] storm_days):
# def run_params(int nreps, int[:] storm_days, int[:] dates, DTYPE_t[:] params, int start, DTYPE_t[:] initprob):
cpdef run_params(
        int nreps, int nruns, DTYPE_t[:] pstorm, int[:] dates, 
        DTYPE_t[:] params, int start, DTYPE_t[:] initprob, 
        iDTYPE_t[:] storm_start, dir_name
        ):
    # as long as the data types for both functions match, why would this not run??
    # return runOpt(nreps, nruns, storm_days, dates, params, start, initprob)
    return runOpt(nreps, nruns, pstorm, dates, params, start, initprob, storm_start, dir_name)

# cpdef runOpt(
#         int nreps, int nruns, DTYPE_t[:] pstorm, int[:] dates, DTYPE_t[:] params,
#         int start, DTYPE_t[:] initprob, iDTYPE_t[:] storm_start, dir_name
#     ):
# endregion ###########################################################################################

########################################################################################################
############################### PARAMETERS #############################################################
# region ###############################################################################################

# These are the parameters used to create the synthetic nest survival data
# They are in lists because we will cycle through all possible param combinations

# numNests       = [150, 400 ]   # Number of nests created - not number found/monitored
# num_Nests      = [150, 400 ]   # Number of nests created - not number found/monitored
# num_Nests       = [200]
# prob_Surv       = [0.95 ]   # daily prob of survival
#probSurv       = [0.95]   # daily prob of survival

# cdef DTYPE_t probMort_Flood  = 0.1    # 10% of failed nests are due to flooding - not .1 of all nests
# # Sprob_Surv      = [0.2] # daily prob of survival during storms - kind of like intensity
# #SprobSurv       = [0.2] # daily prob of survival during storms - kind of like intensity
# cdef DTYPE_t SprobMort_Flood = 1.0    # all failed nests during storms fail due to flooding

# # storm_Dur       = [1, 3, 5 ]
# # storm_Dur        = [ 3, 4]
# # storm_Freq      = [1, 3, 5 ]
# # storm_Freq       = [3, 4]

# # hatch_Time      = [16, 20, 28 ]    # length of incubation - based on real species
# # hatch_Time       = [19]
# # cdef int breedingDays    = 150   # length of breeding season - add some extra days at the end to prevent indexing error
# cdef iDTYPE_t breedingDays    = 150   # length of breeding season - add some extra days at the end to prevent indexing error
# cdef int stormNests      = 0      # tally of nests active during a storm
# #storm_days = switch2(stormDur, stormFreq)   # make storms for the season

# #obs_int        = [3, 4, 5 ]
# # obs_Int        = [3 , 6]
# cdef DTYPE_t discProb       = 0.7

# rng            = np.random.default_rng(seed=61389)  # call the random number generator
# this only affects the very first set of nest data generated
# rng            = np.random.default_rng(seed=82985)


# this makes a list of every possible combination of the given parameter values
# paramsList      = list(product(num_Nests, prob_Surv, Sprob_Surv, storm_Dur, storm_Freq, hatch_Time, obs_Int))
# paramsArray     = np.array(paramsList)   # don't want prob surv to be an integer!


# endregion #########################################################################################
       
########################################################################################################
############################### MLE OPTIMIZATION #######################################################
########################################################################################################

# in old script:
# 1. cycled through param value combinations (sets)
# 2. cycled through number of nest data replicates for given combination
# 3. (cycled through nests) - already done here in mk_obs()
# 4. cycled through number of runs of the optimizer for given replicate

# print("this is script: ", Path(__file__).name)

# #nreps     = 100
# cdef int nreps     = 50
# #repIDs    = range(1, nreps, step=1) 
# # cdef int nruns     = 1
# cdef int num_out   = 19 # number of output params
# # dir_name  = datetime.today().strftime('%m%d%Y_%H%M%S') # name for unique directory to hold all output
# # cdef int nrows     = len(paramsList)*nreps*nruns  # number param combos * number reps * number runs
# # values    = np.zeros((nrows, num_out)  ) # MLE values for the entire thing
# # # cdef np.ndarray[DTYPE_t, ndim=2] values    = np.zeros((nrows, num_out)  ) # MLE values for the entire thing
# cdef int index     = 0                            # increments for each param set to allow saving MLE vals in chunks

# # todaysdate = datetime.today().strftime('%Y%m%d')
# # #likefile   = Path.cwd() / ('output') / dir_name / ("ml_values_" + todaysdate)
# # # likefile   = Path.cwd() / ('py-output') / dir_name / ("ml_values_" + todaysdate)
# # likefile   = Path.cwd() / ('py_output') / dir_name / ("ml_values_" + todaysdate)
# # likefile.parent.mkdir(parents=True, exist_ok=True)
# # #column_names     = np.array(['rep_ID', 'psurv_est', 'ppred_est', 'pflood_est', 'stormsurv_est', 'stormpred_est', 'stormflood_est', 'storm_dur', 'storm_freq', 'psurv_real', 'stormsurv_real','pflood_real', 'stormflood_real', 'hatch_time','num_nests', 'num_discovered','num_excluded','obs_int', 'exception'])
# # #colnames = ', '.join([str(x) for x in column_names])

# # # cdef np.ndarray[DTYPE_t, ndim=1] params
# # # cdef np.ndarray[DTYPE_t, ndim=2] likeVal
# # # cdef np.ndarray[np.int_t, ndim=2] nestDataBig

# cdef DTYPE_t probSurv, SprobSurv
# cdef int numNests, stormDur, stormFreq, hatchTime, obs_int, index2
# cdef np.ndarray survey_days, likeVal, nestDataBig
# # cdef np.ndarray[np.int_t, ndim=1] storm_nests 

# with open(likefile, "wb") as f: 
#     #f.write(colnames) # if it's a npy file, no point in adding colnames since it's not human-readable anyway

# for i in range(0, len(paramsList)): # for each set of params
        
# # FOR THIS SET OF PARAMETERS ------------------------------------------------------------------------
        
#         params     = paramsArray[i]   # load in the params 
#         # params     = list(params) # why is this here? why make it alist?

#         #print("##############################################################################################")
#         #print("params: ",params)
#         #print("index: ", index)
#         ##print("values dataframe: ", values)

#         # extract the parameter values for this set - since for loop doesn't have local scope, changing these here changes the global vals
#         numNests   = params[0].astype(int)
#         probSurv   = params[1]               # don't make prob surv an integer!
#         SprobSurv  = params[2]
#         stormDur   = params[4].astype(int)
#         stormFreq  = params[3].astype(int)
#         hatchTime  = params[5].astype(int)
#         obs_int    = params[6].astype(int)
#         ##print("number of reps, number of nests:", nreps, numNests)
        
#         # Generate pre-allocated empty arrays to store model output (MLE values and nest data) for this set
    # likeVal     = np.zeros(shape=(nreps*nruns, num_out))  
    # nestDataBig = np.zeros(shape=(numNests*nreps, 10)) # the number of columns = nCol from inside mk_obs 
#         index2      = 0                                    # index for nest data needs to increment with each replicate

#         # Generate some random values for the data
#         rng         = np.random.default_rng()           # call random number generator
        #   start       = rng.integers(1, high=5)           # random day of first survey from first 5 days of breeding season
#         # global storm_days  
#         storm_days  = stormGen(stormDur, stormFreq)     # python is treating as local var called before assignment
#         # tried using global storm_days within the mk_obs function; maybe just move this line into mk_obs
#         # or explicitly add it as a function arg?
 
#         survey_days = np.arange(start, breedingDays, obs_int)
#         survey_days = survey_days[np.isin(survey_days, storm_days)==False] # keep only values that aren't in storm_days
         
#         # # might be best to make storm days for all replicates and param sets the same?
#         #print("start = ", start)
#         #print("storm days = ", storm_days)
#         #print("survey days = ", survey_days)


#         # idx = 1
#         runOpt(nreps, storm_days)
#         # for r in range(nreps):  # for each data replicate that uses this set of params
#         #     # nest data are not repeating between replicates - good.
#         #     # should look at data output, see if I can write a script to see if vals are as expected
#         #     # ALSO do not appear to repeat between runs of the script - maybe remove seed since I don't know
#         #     # exactly what effect it is having?
#         #     # nope, looks like data are repeatable between runs of the script

#         # # FOR EACH REPLICATE USING THIS PARAM SET -----------------------------------------------------------

#         #     #repID       = datetime.today().strftime('%m%d%Y_%H%M%S')   # unique identifier for data replicate
#         #     # repID       = datetime.today().strftime('%m%d%H%M%S')   # unique identifier for data replicate - simplified
#         #     repID       = datetime.today().strftime('%H%M%S')   # unique identifier for data replicate - simplified
#         #                                                         # what if these aren't being created quickly enough?
#         #     ##print("this is data replicate ", repID)
#         #     #rng = np.random.default_rng()        # call random number generator
#         #     # repID       = f'{repID}{idx}'
#         #     repID       = f'{repID}{index2}'
#         #     try:
#         #         nestData    = mk_obs(params,repID)              # make nest data
#         #     except IndexError:
#         #         print("IndexError in nest data, go to next replicate")
#         #         continue

#         #     #print("remove undiscovered nests: ", np.argwhere(np.isnan(nestData).any(axis=1)).T )# r##print undiscovered nests
#         #     # nestData    = nestData[~np.isnan(nestData).any(axis=1),:]  # remove undiscovered nests
#         #     nestData    = nestData[nestData[4] != 0,:]
#         #     discovered  =  len(nestData)                               # then count discovered nests

#         #     # select nests to be excluded: unknown fate or only one observation
#         #     exclude     = ((nestData[:,7] == 9) | (nestData[:,8]<obs_int))                         
#         #     excluded    = sum(exclude)      # exclude is a boolean array, sum gives you num True
#         #     nestData    = nestData[~(exclude), :]    # remove excluded nests from data
#         #     #print("exclude these unknown-fate or 1-observation nests: ", exclude)
                          
#         #     storm_nests = sum(nestData[:,9])                           # count num of nests active during >1 storm

#         #     #rng = np.random.default_rng()        # call random number generator
#         #    # probably don't need to call it again
#         #    # unless that is why the storms are the same? idk need to mess with this.
#         #    # overrides the seed, so only the first nest data is reproducible

#         #     s   = rng.uniform(-10.0, 10.0)       # random initial values for optimizer
#         #     mp  = rng.uniform(-10.0, 10.0)
#         #     ss  = rng.uniform(-10.0, 10.0)
#         #     mps = rng.uniform(-10.0, 10.0)
#         #     z   = np.array([s, mp, ss, mps])

#         #     #print("main.py: Msg: Running optimizer")

#         #     try:
#         #         ans  = optimize.minimize(like_smd, z, args=(nestData), method='Nelder-Mead')
#         #         #ans  = optimize.minimize(like_smd, z, args=(nestData), method='L-BFGS-B')
#         #         ex = 0

#         #     except decimal.InvalidOperation:
#         #         print("Decimal error in optimizer - go to next replicate")
#         #         ex = 100
#         #         likeVal[r] = np.full(num_out, ex)
#         #         continue            # skip the rest of this iteration, go to next

#         #     except OverflowError:
#         #         print("Overflow error in optimizer - go to next replicate")
#         #         ex = 200
#         #         likeVal[r] = np.full(num_out, ex)
#         #         continue

#         #     s0   = ans.x[0]         # Series of transformations of optimizer output.
#         #     mp0  = ans.x[1]         # These make sure the output is between 0 and 1, 
#         #     ss0  = ans.x[2]         # and that the three fate probabilities sum to 1.
#         #     mps0 = ans.x[3]

#         #     s1   = logistic(s0)
#         #     mp1  = logistic(mp0)
#         #     ss1  = logistic(ss0)
#         #     mps1 = logistic(mps0)

#         #     ret2 = triangle(s1, mp1)
#         #     s2   = ret2[0]
#         #     mp2  = ret2[1]
#         #     mf2  = 1.0 - s2 - mp2

#         #     ret3 = triangle(ss1, mps1)
#         #     ss2  = ret3[0]
#         #     mps2 = ret3[1]
#         #     mfs2 = 1.0 - ss2 - mps2

#         #     # These are the likelihood values for this data replicate:
#         #     #                           0      1   2    3     4   5     6       7         8         9        10          11              12            13        14            15       16        17       18
#         #     like_val    = np.array([repID, s2, mp2, mf2, ss2, mps2, mfs2, stormDur, stormFreq, probSurv, SprobSurv, probMortFlood, SprobMortFlood, hatchTime, numNests, obs_int, discovered, excluded, ex])
#         #     likeVal[r]  = like_val # likelihood values for this data replicate recorded to larger array
#         #     #print("likelihood vals for this replicate:", like_val)

#         #     nestrows                            = nestData.shape[0]
#         #     nestDataBig[index2:index2+nestrows] = nestData
#         #     index2                              = index2 + 1
            
#         #     ##print("index 2:", index2)

#         nrows    = likeVal.shape[0]          # the number of rows per param set 
#         values[index:index+nrows,] = likeVal # fill the rows of "values" corresponding to the last param set
#         np.save(f, likeVal)                  # save likelihood values for this replicate to disk in case of error
#                                              # save by leaving file open and writing to it
#                                              # why does the nest file get a file extension but not this one?
                                             
#         index    = index + nrows             # increment index 

#         now       = datetime.now().strftime("%H%M%S")
#         #nestfile  = Path.cwd() / ('output') / dir_name / ('nests' + now ) # don't add file extension?
#         nestfile  = Path.cwd() / ('py_output') / dir_name / ('nests' + now ) # don't add file extension?
#         nestfile.parent.mkdir(parents=True, exist_ok=True)
#         np.save(nestfile, nestDataBig)                                    # save nest data for this replicate to file

#         ##print("index= ", index)
#         ##print("replicate finish:", datetime.now())

#     column_names     = np.array(['rep_ID', 'psurv_est', 'ppred_est', 'pflood_est', 'stormsurv_est', 'stormpred_est', 'stormflood_est', 'storm_dur', 'storm_freq', 'psurv_real', 'stormsurv_real','pflood_real', 'stormflood_real', 'hatch_time','num_nests', 'obs_int', 'num_discovered','num_excluded', 'exception'])
#     # header values need to be stored to np.ndarray for np.savetxt; actually, needs to be a string
#     colnames = ', '.join([str(x) for x in column_names])
#     filepath     = Path.cwd() / ('py_output') / dir_name / ('ml_val_' + dir_name + '.csv')
#     filepath.parent.mkdir(parents=True, exist_ok=True)
#     #headerTF     = False if exists(filepath) else True 
# #values.to_csv(filepath, mode='a', header=headerTF) # want to write the csv after all the runs, replicates, and param sets
# # .to_csv doesn't work for numpy objects, but since it's all numeric, should be able to do:
# # .savetext('filename', df, delimiter=',') - if strings included, add fmt argument

#     #np.savetxt(filepath, values, delimiter=',',header=colnames) # saves with gaps in each array
#     np.savetxt(filepath, values, delimiter=',',header=colnames) # save once to csv - not many times
# ##print("finish time:", datetime.now())
# # if 
# # (needs to be outside the loop)
#     #end = datetime.now()
# ##print("running time:", started - end,"; number of reps:", nreps)


        #filepath         = Path('{}/likelihood_values_{}.csv'.format(dir_name, dir_name))
        # filepath         = Path.cwd() / ('output') / dir_name / ('ml_val_' + dir_name)
        # headerTF         = False if exists(filepath) else True 
        
        # # append values from this param set to the csv:
        # #likelihoodValues.to_csv(filepath, mode='a', header=headerTF)
        # values.to_csv(filepath, mode='a', header=headerTF)
        # this output is very strange - extra rows, etc

        # when you do 500 reps, if it breaks you still get nothing. maybe put this csv command into the other loop?

