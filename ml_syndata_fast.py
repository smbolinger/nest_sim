#!/usr/bin/env python3

# This program creates synthetic nest data and runs it through the
# MC nest survival algorithm. The purpose of this code is to make
# sure the MC estimates are being computed correctly.

# to run a line-by-line profile of a function, decorate it with @profile and then run:
#    python kernprof.py -l script.py 
# you can convert the output to text file:
#    python -m line_profiler script.py.lprof > filename.txt
# but it doesn't give me any output for like() and like_smd() (see profile_ml_syndata_fast.txt)

# can add stderr to stdout in the same file using script.py > output.txt 2>&1
# script should #print any time there's an exception, but this will also show me warnings
# AND approx the time they occurred? i.e. which replicate or optimizer run
# (when print statemets are enabled)

# can try writing files with JSON to speed things up

# Lingering questions:
# - Should I have all replicates use the same survey and storm days, and start of season?

# ml_syndata_fast.py (09 Aug 2023):
# removed a bunch of comments, but for now they are all still in ml_syndata_print.py


# v8 (07 Aug 2023):
# Going to delete a bunch of comments and clean this up a bit.
# thought: mostly seems easier to just figure out code that works, even if slow
# and THEN google to find a more efficient route (rather than looking for that upfront)
# also: easier to look at output if you open it in VSCode and then split screen vertically
# (can see starting vals while looking at each individual nest)

# new script after v8 called ml_syndatav2.py for ease of typing
# Notes:
# inconsistencies in output: 
# - observation intervals are wrong (-1, 0, others)
# - days nest is observed - also wrong. should not just be when discovered == True unless
#   I do what I did with the failure date and set the rest of the discovered array to True
# follow-up: did that and it works now

# seems as though i,j,k are being calculated correctly, but interval still isn't
# found a clever way to set discover == True after the first True (via stackexchange)
# "interval" in that context was actually number of days between final fate and final obs
# going to set pSurv a bit higher to see if we can get some hatches
# also, the ML values still aren't being recorded to the np.array "values"

# when are we removing undiscovered nests? I don't see it in mk_obs
# found it.
# why are storm days always different with the rng seed, but nest data always the same?
# I guess maybe it's not identical. last time (1807) there were 0 hatches out of 30 nests!
# could easily output a T/F from mk_obs saying whether there was a storm in the last interval
# date pairs also aren't working - just giving first and last day of observation
# and I'm not even sure what the s is about
# - based on the output, using active+1 is not giving the correct results 
# still no nests in like() are using the storm matrix...
# but actually, might be because none of their observation periods overlaps with storms. hmm...

# look into using @ instead of nested dot multiplication

# V6:

# well, I meant to edit v5 in vim, but somehow stayed in v4, so v5 is actually the version
# before v4

# need to figure out why ret is infinity
# looks like even in mc_v1.py or mc_v2.py, ret was returning an integer, but only one instead of 
# optimizing. look further back
# in markov_likelihood_new_storm18a.py, appears to work (based on the output)
# looks like (appropriately) mc-no-loops.py is the first script to convert mk_obs to use
# fewer loops
# by the time we get to mc-no-loopsv7, looks like the mk_obs function works similarly to
# the current version. 
# in the output from 1-26, ret yields a single number; by 1-28 (markov_no-loops_new.py) it gives multiple zeros
# how did I never try to figure out why ret was messed up? just wanted to get the
# functions working. need to know which output is from which script, keep better
# track of changes
# looks like by 1/30/2023 (markov_no-loops_new3.py) the ret output is normal again...
# between those two scripts, not much difference
# - no difference in likd_smd(), minimal in like()
# - some differences in how they are called at the end of the script
# I guess I never tried to figure out why my likelihood values were so low?
# strange behavior - like() is just taking the observed nests and running them through again and again...
# only shows a (single) value for ret each time it reaches nest 15, but that value is Infinity
# just repeats over and over until we get to the second set of params, which has the same storm days
# but different nest data? then it goes straight to the optimizer, no #printing from like(), and shows
# multiple values for ret but they are all zero
# then after 1 run it immediately shows the error about values being None (even though it wasn't at the beginning)
# nestData is only 20 rows in debug, so why does it loop through them over and over?
# UPDATE: there are just a bunch of #print statements in like() that weren't there before, so
# every time the optimizer runs, they run
# still don't know why ret = 0 the whole time
# also, why did it not #print them the second time the loop ran? in this case, it's because none of the nests was found


# this could be useful for toggling #print statements on/off?

# from __future__ import #print_function
# enable_#print = 0

# def #print(*args, **kwargs):
#     if enable_#print:
#         return __builtins__.#print(*args, **kwargs)

# #print('foo') # doesn't get #printed
# enable_#print = 1
# #print('bar') # gets #printed


# v3:

# v2 runs but produces strange ML values and was #printing all the nest data a bajillion times



# going to remove some excess comments and try to fix the ML calculation and whatever else

# custom folding that probably doesn't actually work in settings.json:
        # {
        #     "folding": {
        #       "markers": {
        #         "start": "# -----------",
        #         "end": "# --------"
        #       }
        #     }
        #   } , 
#import csv 



import decimal
#from   decimal import * # why doesn't this import the exception class?
from datetime import datetime

started = datetime.now() 
#import h5py 
from itertools import product
import itertools
from line_profiler import LineProfiler 
#import logging
import numpy as np
#from   numpy import linalg 
from os.path import exists
import os 
from   pandas import DataFrame 
from   pathlib import Path 
#import scipy 
#import scipy 
from   scipy import optimize
import scipy.stats as stats

# region ######################################################################################################
################################ SET UP LOGGING AND PROFILER    ###############################################



os.chdir(os.path.dirname(os.path.abspath(__file__)))



# profiler = LineProfiler()

# def profile(func):
#     def inner(*args, **kwargs):
#         profiler.add_function(func)
#         profiler.enable_by_count()
#         return func(*args, **kwargs)
#     return inner

# endregion ###################################################################################################

# def #print_stats():
#     profiler.#print_stats()
#print("this is script: ", Path(__file__).name)
#print("start time:", datetime.now())
##################################################################################################
# This function remaps values from R^2 into the lower left triangle located within the unit square.
def triangle(x0, y0):
    if y0 > x0:
        ret = triangle(y0, x0)
        return ret[1], ret[0]

    r0 = np.sqrt( x0**2 + y0**2)
    m  = 1.0
    if y0 != x0:
        m = y0/x0

    theta = np.arctan(m)
    r3    = r0 * 1.0/(1.0 + m)
    x3    = r3 * np.cos(theta)
    y3    = r3 * np.sin(theta)
    return x3, y3

# This is just the logistic function
def logistic(x):
    #return 1.0/( 1.0 + math.exp(-x) )
    return 1.0/( 1.0 + np.exp(-x) )

# Must be called AFTER the param set is specified
# define p from actual storm data!
# import the storm frequency distribution:
# don't need the week number since it's just being used in the random number generator
#pstorm = np.zeros()
# with open("summer_storms_weekly_noSept.csv") as f:
#     #readCSV = csv.reader(f, delimiter=",")
#     #pstorm = np.zeros(shape=len(readCSV))
#     #pstorm = readCSV[2] # it's the second column because there's an index column
#     # think I can do this in 1 step with numpy
#     pstorm = np.genfromtxt(f, delimiter=",", skip_header=1, usecols=2) # 2nd column bc R added an index col
#                                                         # no need for names
# f.close()
# pstorm is still in frequencies - convert to probability
#scipy.special.softmax()
# maybe make a softmax function to keep memory usage lower? from stackexchange:
# np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)
#def softmax(z):
#np.exp(z) / np.sum(np.exp(z), axis=0, keepdims=True)
# def softmax(x):
#     e_x = np.exp(x - np.max(x))
#     return e_x / e_x.sum()
# from scipy.special import softmax 
# probs = softmax(pstorm)
#
#with open("stormprob.csv") as f:
#    
#     pstorm = np.genfromtxt(f, delimiter=",", skip_header=1, usecols=1)# 2nd column bc R added an index col
#f.close()

#with open("stormprob4.csv") as f:
#    
#     pstorm = np.genfromtxt(f, delimiter=",", skip_header=1, usecols=1)# 2nd column bc R added an index col
#f.close()
with open("stormprob_real_.csv") as f:
     pstorm = np.genfromtxt(f, delimiter=",", skip_header=1, usecols=3)# 4th column 
f.close()
#
with open("stormprob_real_.csv") as f:
    storm_weeks  = np.genfromtxt(f, delimiter=",", skip_header=1, usecols=1)# 3rd column 
f.close()

# get the first Julian day of each week, then subtract 110 to start at 01 Apr
storm_start = (storm_weeks * 7) - 90
##print(pstorm)
#pstorm = pstorm.pop()
##print(pstorm)
#start=1
#breedingDays = 150
#storm_season = np.arange(start+7, start+breedingDays, step=7)
##print(len(np.arange(start+7, start+breedingDays, step=7)))
##print(len(pstorm))


# Make storms - must be called AFTER the param set is specified
def stormGen(frequency, duration):

    # process: 
    # - pick random days  as start days- number is equal to number of storms in the season
    # - create a range based on the duration of the storm (if storms are 2 days long, range = (0, 1,2)
    # - add each number in range to the start day to get sequential days of the storm including the start date (+0)
    # - not in order, but doesn't matter. apparently sort() is very costly, so not bothering with it.

    out = rng.choice(a=storm_start, size=frequency, replace=False, p=pstorm)
        # use rng.choice (preferred over np.random.choice) because can pick w/o replacement (unlike sampling from distr)
        # pstorm is real weekly storm probabilities from 1975-2021
        # storm start is the start day of each week (so you don't get overlapping storms)
        # more than 1 storm/week seems unrealistic anyway

    dr = np.arange(0, duration, 1)             # create range based on storm duration
    stormDays = [out + x for x in dr]          # add each number in range to the dates generated above
    stormDays = np.array(stormDays).flatten()  # flatten to convert to a list instead of list of tuples
    return(stormDays)



###################################################################################################
# region ###################### PARAMETERS #######################################################

# These are the parameters used to create the synthetic nest survival data
# They are in lists because we will cycle through all possible param combinations

#numNests       = [100,300 ]   # Number of nests created
numNests       = [300]
#probSurv       = [0.9, 0.95 ]   # daily prob of survival
probSurv       = [0.95]   # daily prob of survival

probMortFlood  = 0.1    # 10% of failed nests are due to flooding - not .1 of all nests
#SprobSurv      = [0.2, 0.4] # daily prob of survival during storms - kind of like intensity
SprobSurv      = [0.2] # daily prob of survival during storms - kind of like intensity
SprobMortFlood = 1.0    # all failed nests during storms fail due to flooding

#stormDur       = [1, 2, 3 ]
stormDur       = [ 3, 4]
#stormFreq      = [1, 2, 3, 4 ]
stormFreq      = [3, 4]

#hatchTime      = [16, 19, 28 ]    # length of incubation
hatchTime      = [19]
breedingDays   = 150   # length of breeding season - add some extra days at the end to prevent indexing error
stormNests     = 0      # tally of nests active during a storm
#storm_days = switch2(stormDur, stormFreq)   # make storms for the season

#obs_int        = [3, 4, 5 ]
obs_int        = [3 , 5]
discProb       = 0.7

rng            = np.random.default_rng(seed=61389)  # call the random number generator
# rng            = np.random.default_rng(seed=82985)


# this makes a list of every possible combination of the given parameter values
paramsList      = list(product(numNests, probSurv, SprobSurv, stormDur, stormFreq, hatchTime, obs_int))
paramsArray     = np.array(paramsList)   # don't want prob surv to be an integer!

with open("init_dates_reall.csv") as f:
    initprob = np.genfromtxt(f, delimiter=",", skip_header=1, usecols=2)
    #dates = np.genfromtxt(f, delimiter=",", skip_header=1, usecols=1) #  why doesn't this second one work?
f.close()
# loadtext may be faster, but maybe not worth switching

initprob = initprob / sum(initprob) # convert to probability

with open("init_dates_reall.csv") as f:
    #dates = np.genfromtxt(f, delimiter=",", skip_header=1, usecols=1, dtype=None) 
    dates = np.genfromtxt(f, delimiter=",", skip_header=1, usecols=1, dtype=int) # makes everything "-1"
    #dates = np.genfromtxt(f, delimiter=",", skip_header=1, usecols=1, dtype=None, deletechars="b") # doesn't work to get rid of leading b's
f.close()
# changed dates from factor to integer in R before exporting
# add another 10 to recalibrate to 01 apr

dates = dates +10

# endregion #########################################################################################
       

###################################################################################################
# region ################# NEST CREATION FUNCTION ################################################

def in1d_sorted(A,B): # possible days observer could see nest is intersection of observable and survey days
    idx = np.searchsorted(B, A)
    idx[idx==len(B)] = 0
    return A[B[idx] == A]
 

# This function makes nests
# could make the observer a different function in order to get more useful returns from the functions?
#@profile 
def mk_obs(params, repID): 
    #print('mk_obs params:', params)
    num_nests      = params[0].astype(int)    # make these local variables
    probSurv       = params[1] 
    SprobSurv      = params[2]   
    obs_int        = params[6].astype(int)
    #hatchTime      = params[5].astype(int)     # why does this one not work? no actual issue
    hatch_Time     = params[5].astype(int)     # why does this one not work?

    probVec        = [SprobSurv if i in storm_days else probSurv for i in np.arange(start, start+breedingDays)]
        # add extra probabilities to the end of the vector so it won't throw the indexing error

    probVec        = np.array(probVec) # works
    stormTrue      = (probVec == SprobSurv)      # T/F is there a storm on given day 
                                                 # i.e. days where we use SprobSurv (storm days)
    #print("storm?", np.column_stack((np.arange(start,start+breedingDays), stormTrue))) # works now  
    nCol           = 10 # number of columns in table returned by this function
    nestData       = np.zeros(shape=(num_nests, nCol)) # empty array to fill w/ nest data
                                                       # will be returned by the function
       # one big difference between the new mk_obs function and the one used in older versions
       # is that we aren't returning a list of lists, so we don't have a list of observation 
       # dates and intervals for each nest. we are calculating those later on.
       # this means the new function can return all the nest data at once instead of 1 by 1
       #  
    TF_array       = np.zeros(shape=(num_nests,1))
    
    # Column 1 will be nest ID ---------------------------------------------------------------------------
    nestData[:,0]  = np.arange(num_nests)           # number of IDs = number of nests
    
    # ----------------------------------------------------------------------------------------------------
    # Column 2 = nest initiation date, drawn for now from normal dist bc I don't have reason to believe 
    # otherwise
    #mu, sigma      = 68, 20
    
    #initiation     = (rng.normal(mu, sigma, num_nests)).astype(int)  # number of samples = number of nests
    initiation     = rng.choice(a=dates,size=num_nests, p=initprob)
    # remember that real init dates start at 20 Apr, not 1 Apr - already accounted for this
    nestData[:,1]  = initiation 
    #print("init dates: ",initiation)

    # Can plot the distribution of synthetic nest initiation dates --------------------------------------
  #  from matplotlib import pyplot as plt 
  #  plt.hist(initiation, bins=[0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120,125,130,135])
  #  plt.title("nest initiation dates")
  #  #filepath2 = Path('{}/init_{}'.format(dir_name, repID))
  #  #filepath2 = Path('output/',dir_name/'init_{}'.format( repID))
  #  filepath2 = Path.cwd() / ('output') / (dir_name ) / ('init_' + repID)
  #  filepath2.parent.mkdir(parents=True, exist_ok=True)  
  #  plt.savefig(filepath2)

    # -----------------------------------------------------------------------------------------------
    # These columns will be filled in later:
    # Column 3 = nest failure/hatch date               |   Column 4 = cause of failure 
    # Column 5 = I (first found)                       |   Column 6 = J (last active)
    # Column 7 = K (last checked)                      |   Column 8 = fate assigned by observer
    # Column 9 = length of time nest is active         |   Column 10 = T/F was there a storm in the interval 
    # -----------------------------------------------------------------------------------------------   
    # could probably also change whole columns (obs days) at once based on the init date values column 
    for n in range(num_nests):
        #print('----------------------------------------------------------------') 
        #print("nest ID = ", nestData[n, 0]) 
        init          = nestData[n,1]                
        #print("initiation =", init)
        
        period        = np.arange(init, init+hatch_Time).astype(int) # index array for dates in incubation period
        # period is the whole incubation period, not just when nest is active
        p             = probVec[period]                  # extract storm/regular survival probabilities for those days        
        prob          = rng.uniform(size=hatch_Time)      # vector of random probabilities for each day 
        alive         = np.less(prob, p)                 # if prob < p, nest stays alive for that day
        alive[0]      = True                             # nest is always alive on day of initiation
        stormTrue_inc = stormTrue[period]                # When does storm = True during potential incubation
        # fail date will be the smallest of value of period for which the corresponding alive = F
        # make it so nests can't fail the day they are initiated

        if all(alive):
            nestData[n,3] = 1       # nest hatches if alive is all "True"
            hatch = period[hatch_Time-1]
            nestData[n,2] = hatch
            active = sum(alive)   # number of days nest was active
            #print("nest hatched on day ", hatch, "! nest was alive: ", alive)
        #if False in alive:
        else:
            # does the timing of fail vs observer matter? i.e. if nest fails on day 28 and is observed on day 28,
            # does it fail before or after it's observed?
            fail = np.amin(period[alive == False])  # fail date 
            alive[period>fail] = False              # set all days after that day to F in the 'alive' array
            
            nestData[n,2] = fail                    # record to nest data
            active = sum(alive) 
            #print('nest failed on day:', fail)
            #print('nest was alive:', alive)

            if fail in storm_days:
                nestData[n,3] = 2   # during storms, all failed nests fail due to flooding 
                #print("nest flooded during storm")

            elif rng.binomial(1, probMortFlood, 1) == 1:
                nestData[n,3] = 2   # failed due to flood, normal day
                #print("nest flooded, not during storm")
            
            else:
                nestData[n,3] = 3   # failed due to predation, normal day
                #print("nest depredated")

        s = stormTrue_inc[0:(active)]  # 
        #print("s=",s)
        
        storm_true    = any(s)               # T/F was this nest active during >1 storm?
        #print("storm while active?", storm_true)
        TF_array[n]   = storm_true  # make it its own array to tack on after we handle zeros
    
########## OBSERVER - full true nest history has already been recorded ##################################

        #print("time for observer")
        alive_days = period[alive == True]
        #print("days nest is observable:", alive_days)
       
        # if observable days>0, do the rest of this section; if not, skip                     
        if len(alive_days) > 0:
            obs_choice = in1d_sorted(survey_days, alive_days)     # all possible observation days
                # this first one is > 2x as fast (see profile_03aug_1555.txt)
            #print('survey days when nest is observable:', obs_choice)

            prob_disc  = np.full(len(obs_choice), discProb)  # vector of same discovery probability for all possible days
            obs_prob   = rng.uniform(size=len(obs_choice))   # random probability for each of those days 
            discover   = np.less(obs_prob, prob_disc)        # if obs_prob is less than discProb, discover nest
            #print("discovered?", discover)
            nestData[n,8] = len(alive_days)  # how long was nest active for?
        
            if True in discover: # if at least one value is true, nest is discovered
                discover  = ~np.cumprod(~discover).astype(bool) # now all days after discovery date are discover == True
                # this is very clever - once there is a False, the product is zero and remains zero.
                obsDays = obs_choice[discover == True] # possible obs days where discover == True 
                #print("days nest was observed as active: ", obsDays) # need to tack on extra obs for failed nests
                
                i = obsDays[0]       # which one is faster?
                nestData[n,4] = i
                
            # Now observer must assign a fate based on field cues -------------------------------------------------------------
                # Assume more time since end date of nest = harder to assign fate
                # Also assume that storms make it impossible to accurately assign fate 
                # (True fate may have happened before storm, but observer will say nest was flooded)

                if False in alive:
                    # if there is at least one "False" in "alive" then the nest did not hatch
                    # now need to use the fail date to calculate j and k
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
                        #print("assigned fate of unknown,", fate, ", bc of days since fate:", k-fail,", or storm in last interval:", stormTrue[j:k].astype(bool))

                    else: 
                        fate = nestData[n,3]   # correct fate assigned
                        nestData[n,7] = fate
                        #print("assigned correct fate,", fate, ". view days since fate,", k-fail,", and storm in last interval:", k-fail, stormTrue[j:k].astype(bool))

                    #print("failed - fate, i,j,k:", fate, i, j, k)

                # hatch is more difficult to detect than failure:   
                else:
                    
                    if np.max(obsDays) == hatch:

                        j = np.max(obsDays)
                        nestData[n,5] = j
                        k = np.max(obsDays)
                        nestData[n,6] = k                  
                    else:

                        next_ind      = np.argwhere(survey_days == np.max(obsDays)) + 1
                            # need an integer for addition (argwhere, not where)
                        # add one more observation for both j and k in case of hatch and correct fate 
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
                        #print("assigned fate of unknown,", fate, ", bc of days since hatch:", k-hatch,", or storm in last interval:", stormTrue[j:k].astype(bool))
                                        
                    else:
                        fate = 1   # fate assigned as "hatch"
                        nestData[n,7] = fate
                        #print("assigned correct fate,", fate, ". view days since hatch,", k-hatch,", and storm in last interval:", stormTrue[j:k].astype(bool))

                    #print("hatched - fate, i, j, k:", fate, i, j, k)
            #else:
                #print("not discovered")
        #else:
            #print("not discovered")
    
# Format the nest data:
    nestData[nestData == 0]  = np.nan          # convert zeros to NaNs
    nestData[0,0] = 0                          # make this index NaN back into a zero
    nestData[:,9] = TF_array[:,0]              # tack on after converting to NaN but before removing NaN rows
    #nestData[nestData['ac`tive_len']>= obs_int] # remove nests with only one observation
    #            0          1         2         3              4           5               6             7         8             9          
    #columns = ['ID','initiation','end_date','true_fate','first_found','last_active','last_observed','ass_fate','active_len','storm_true']
    # #            0          1         2         3              4           5               6             7         8             9          
    # columns = ['ID','initiation','end_date','true_fate','first_found','last_active','last_observed','ass_fate','active_len','storm_true','long_obs1', 'long_obs2', 'long_obs3']
# could add columns for long observation intervals (where storms occurred) but prob easier to do obs hist later
# try using numpy.save instead of converting to pandas df and saving to csv (pickle is another option)
# numpy.save is already optimized for use with numpy arrays and is faster than pickle for purely numeric arrays (like this one)
    #nests = DataFrame(nestData, columns=columns)
    #filepath3 = Path('{}/nestdata_{}.csv'.format(dir_name, repID))
    #filepath3 = Path.cwd() / (dir_name ) / ('nest_' + repID)
    filepath3 = Path.cwd() / ('output') / (dir_name ) / ('nest_' + repID)
    filepath3.parent.mkdir(parents=True, exist_ok=True)  
    
    np.save(filepath3, nestData)
        
    #nests.to_csv(filepath3)
#   File "C:\Users\sarah\AppData\Local\Programs\Python\Python311\Lib\site-packages\pandas\io\common.py", line 856, in get_handle
#     handle = open(
#     OSError: [Errno 22] Invalid argument: '08032023_122946\\nestdata_08032023_123100.csv'
#   why would it just suddenly stop working?

    ##print(nestData)   
    return(nestData) # return data as 2d numpy array instead of df
    
#####################################################################################################
# LIKELIHOOD FUNCTION

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
def like(a_s, a_mp, a_mf, a_ss, a_mfs, a_mps, nestData):
  
    # the new like function is much longer than the old one, because we need to construct
    # the nest info for each nest from just the values returned from mk_obs()
    # it also uses the observer data by default because the 1-day obs int is not interesting
    
    from decimal import Decimal
    stillAlive  = np.array([1, 0, 0])
    mortFlood   = np.array([0, 1, 0])
    mortPred    = np.array([0, 0, 1])


    #global stormDays
    startMatrix = np.array([[a_s,0,0], [a_mf,1,0], [a_mp,0,1]]) # starting matrix, from etterson 2007
    stormMatrix = np.array([[a_ss,0,0], [a_mfs,1,0], [a_mps,0,1]]) # use this matrix for storm weeks
        # how is this matrix actually being incorporated during the analysis?

    logLike = Decimal(0.0)          # initialize the overall likelihood counter

    for row in nestData:
        
    # FOR EACH NEST -------------------------------------------------------------------------------------

        nest    = row         # choose one nest (multiple output components from mk_obs())
        #print('#############################################################')
        #print('nest =',nest[0])
        #print('obs_int check: ', obs_int)

        disc    = nest[4].astype(int)    # first found
        endObs  = nest[6].astype(int)    # last observed
        fate    = nest[7].astype(int)    # assigned fate

        if np.isnan(disc):
            #print("this nest was not discovered but made it through")
            continue
        
        num     = len(np.arange(disc, endObs, obs_int)) + 1
        obsDays = in1d_sorted((np.linspace(disc, endObs, num=num)), survey_days)
        obsPairs = np.fromiter(itertools.pairwise(obsDays), dtype=np.dtype((int,2))) # do elements of numpy arrays have to be floats?
        #print("date pairs in observation period:", obsPairs) 

        # make a list of intervals between each pair of observations (necessary for likelihood function)
        intList = obsPairs[:,1] - obsPairs[:,0]

        obs     = [stillAlive for _ in range(len(obsPairs)+1)] # start off with all intervals = alive

        # change the last obs if nest failed:
        if fate == 2:
            obs[-1] = mortFlood
        elif fate == 3:
            obs[-1] = mortPred

        #print("fate, obs = ", fate, " , ", obs) # check that last entry in obs corresponds to fate

        # if hatch, leave as is?

        logLikelihood = Decimal(0.0)   # place this likelihood counter inside the for loop so it resets with each nest

        for i in range(len(obs)-1):
    # FOR EACH OBSERVATION OF THIS NEST ---------------------------------------------------------------------

            intElt  = (intList[i-1]).astype(int)  # access the (i-1)th element of intList,
                                    # which is the interval from the (i-1)th
                                  # to the ith observation

            #stateF  = obs[i]
            stateF  = obs[i+1] 
            stateI  = obs[i]
            #print("stateF:",stateF)
            TstateI = np.transpose(stateI)
            #print("TstateI:", TstateI)

            if any(d in storm_days for d in range(i-1, i)):
                # if any of the days in the current observation interval (range) is also in storm days, use storm matrix
                #print("using storm matrix")
                lDay = np.dot(stateF, np.linalg.matrix_power(stormMatrix, intElt))
                # this is the dot product of the current state of the nest and the storm matrix ^ interval length
           # look into using @ instead of nest dot calls 
            else:
                #print("using normal matrix")
                lDay = np.dot(stateF, np.linalg.matrix_power(startMatrix, intElt))

            lPer = np.dot(lDay, TstateI)

            logL = Decimal(- np.log(lPer))

            logLikelihood = logLikelihood + logL # add in the likelihood for this one observation

        logLike = logLike + logLikelihood        # add in the likelihood for the observation history of this nest

    return(logLike)
    
#####################################################################################################
# FUNCTION TO OPTIMIZE
# this function generates random initial values and runs "like" to see the likelihood of those values
# given the nest data
# optimizing it should optimize the values given to "like"
#@profile
def like_smd(x, *arg):

    # value to be minimized needs to be a float:
    ret = 0.0

    # Step 1: Unpack the arguments:
    #
    # These are unbounded parameters that can take values from -inf to +inf. See below for details.
    #
    # s0   = survivorship
    # mp0  = conditional mortality from predation?
    # ss0  = survivorship during a storm
    # mps0 = conditional mortality from predation during a storm, seems awkward.
    #
    s0   = x[0]
    mp0  = x[1]
    ss0  = x[2]
    mps0 = x[3]

    # Step 2: Unpack the data
    #
    data = arg[0]

    # Step 3: Transform the values so that they remain between 0 and 1
    #
    s1   = logistic(s0)
    mp1  = logistic(mp0)
    ss1  = logistic(ss0)
    mps1 = logistic(mps0)

    # Step 4: Further transformation to keep the values within the lower left triangle.
    #
    tri1 = triangle(s1, mp1)
    tri2 = triangle(ss1, mps1)
    s2   = tri1[0]
    mp2  = tri1[1]
    ss2  = tri2[0]
    mps2 = tri2[1]

    # Step 5: Compute the depended conditional probability of mortality due to flooding.
    #
    mf2  = 1.0 - s2 - mp2
    mfs2 = 1.0 - ss2 - mps2

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

    ret = like(s2, mp2, mf2, ss2, mps2, mfs2, data)
    # 03 Aug: why does ret = infinity?

    #print('like_smd(): Msg : ret = ', ret)
    return ret

# again, create new function for the observer (will fix later)

# why can't I just use like_smd with the observer values?

######################################################################################################
## scenarios:

## 1. 3-day nest check interval
## 2. 7-day nest check interval
## these two only work if you introduce an observer
## 3. 5% of hatched nests mis-assigned to failed
## 4. 5% of hatched nests marked as uncertain fate and not included
## do those make sense? or just a general mis-assignment of all nests, not just hatched?
## 5. 5% of all nests mis-assigned
## 6. 10% of all nests mis-assigned
## need to introduce an observer that can mis-classify?
## no, just assign  certain percentage to an "unknown" category?
## then look at what happens when we are more likely to be uncertain about hatch than predation,
## or vice versa
## increase number of nests - more accurate. what number is needed?

# endregion #######################################################################################

####################################################################################################
############################### MLE OPTIMIZATION ##################################################

# in old script:
# 1. cycled through param value combinations (sets)
# 2. cycled through number of nest data replicates for given combination
# 3. (cycled through nests) - already done here in mk_obs()
# 4. cycled through number of runs of the optimizer for given replicate

nreps     = 2 
#repIDs    = range(1, nreps, step=1) 
nruns     = 1
num_out   = 19 # number of output params
dir_name  = datetime.today().strftime('%m%d%Y_%H%M%S') # name for unique directory to hold all output

print("this is script: ", Path(__file__).name)
print("number of reps, number of nests:", nreps, numNests)
# let's make a list for now:
#values    = []
# but isn't tacking items onto a list verrrry slow?
nrows     = len(paramsList)*nreps*nruns
# number param combos * number reps * number runs

#values    = np.zeros(shape=(nrows, num_out)  ) # this is just giving 'None' - doesn't even say why
values    = np.zeros((nrows, num_out)  ) # this is just giving 'None' - doesn't even say why
# the first time we loop (first param set) values is the correct dimensions
# then becomes None before the second loop.
# number param combos * number reps * number runs

index     = 0

#todaysdate = datetime.today().strftime('%Y%m%d')
#nestfile   = 'nestdata{todaysdate}.hdf5'
#f = h5py.File(nestfile, 'w')
#filepath3 = Path.cwd() / ('output') / (dir_name ) / "nestdata{todaysdate}.npy"
#with open(filepath3, "wb+") as f:

for i in range(0, len(paramsList)): # for each set of params
    
# FOR THIS SET OF PARAMETERS ------------------------------------------------------------------------
    #subfile   = datetime.now().strftime("%H%M%S")
    
    params     = paramsArray[i]
    params     = list(params) # why is this here? why make it alist?

    #print("##############################################################################################")
    #print("params: ",params)
    #print("index: ", index)
    #print("values dataframe: ", values)

    # since for loop doesn't have a local scope, changing these here changes the global vals
    numNests   = params[0].astype(int)
    probSurv   = params[1]               # don't make prob surv an integer!
    SprobSurv  = params[2]
    stormDur   = params[4].astype(int)
    stormFreq  = params[3].astype(int)
    hatchTime  = params[5].astype(int)
    obs_int    = params[6].astype(int)

#    likeVal    = np.zeros(shape=(nreps*nruns, num_out))  # array to store model output
    likeVal    = np.zeros(shape=(nreps*nruns, num_out))  # empty array to store model output for this set

    #storm_days = switch2(stormDur, stormFreq)   # make storms for the season
    #storm_days = switch2(stormDur, stormFreq)   # make storms for this param set
        # need to sample w/o replacement
    start      = rng.integers(1, high=5)  # random day of first survey from first few (5?) days of breeding season
    #storm_season = np.arange(start+7, start+breedingDays, step=7)
    storm_days = stormGen(stormDur, stormFreq)   # make storms for this param set

    #survey_days    = [x for x in np.arange(start, breedingDays, obs_int) if x not in storm_days]
    #survey_days    = np.where(np.arange(start, breedingDays, obs_int) not in storm_days) 
                                   # needs to be set after the param set is specified
                                   # can probably make this a different, easier way
                                   # but in the interest of finishing, do it in two steps:

    survey_days    = np.arange(start, breedingDays, obs_int)
    survey_days    = survey_days[np.isin(survey_days, storm_days)==False] # keep only the values that aren't in storm_days
     
    # # might be best to make storm days for all replicates and param sets the same?
    #print("start = ", start)
    #print("storm days = ", storm_days)
    #print("survey days = ", survey_days)
    
    for r in range(nreps):  # for each data replicate that uses this set of params
        # nest data are not repeating between replicates - good.
        # should look at data output, see if I can write a script to see if vals are as expected
        # ALSO do not appear to repeat between runs of the script - maybe remove seed since I don't know
        # exactly what effect it is having?
        # nope, looks like data are repeatable between runs of the script

    # FOR EACH REPLICATE USING THIS PARAM SET -----------------------------------------------------------


        #repID       = datetime.today().strftime('%m%d%Y_%H%M%S')   # unique identifier for data replicate
        repID       = datetime.today().strftime('%m%d%H%M%S')   # unique identifier for data replicate - simplified
        #print("this is data replicate ", repID)

        try:
            nestData    = mk_obs(params,repID)              # make nest data
        except IndexError:
            #print("IndexError in nest data, go to next replicate")
            continue

        #print("remove undiscovered nests: ", np.argwhere(np.isnan(nestData).any(axis=1)).T )# r#print undiscovered nests
        nestData    = nestData[~np.isnan(nestData).any(axis=1),:]  # remove undiscovered nests
        discovered  =  len(nestData)                               # then count discovered nests

        # select nests to be excluded: unknown fate or only one observation

        #exclude     = (nestData[:,7] == 9)                       # select nests with unknown fate
        #exclude     = ((nestData[:,7] == 9) or (nestData[:,8]<obs_int))  # throws error            
        #exclude     = a.any((nestData[:,7] == 9) or (nestData[:,8]<obs_int))              
        #exclude     = nestData[(nestData[:,7] == 9) or (nestData[:,8]<obs_int)]  
        exclude     = ((nestData[:,7] == 9) | (nestData[:,8]<obs_int))                         
        excluded    = sum(exclude)      # exclude is a boolean array, sum gives you num True
        nestData    = nestData[~(exclude), :]    # remove excluded nests from data
        #print("exclude these unknown-fate or 1-observation nests: ", exclude)
                      
        storm_nests = sum(nestData[:,9])                           # count num of nests active during >1 storm

        rng = np.random.default_rng()        # call random number generator
       # probably don't need to call it again
       # unless that is why the storms are the same? idk need to mess with this.

        s   = rng.uniform(-10.0, 10.0)       # random initial values for optimizer
        mp  = rng.uniform(-10.0, 10.0)
        ss  = rng.uniform(-10.0, 10.0)
        mps = rng.uniform(-10.0, 10.0)
        z   = np.array([s, mp, ss, mps])

        #print("main.py: Msg: Running optimizer")

        try:
            ans  = optimize.minimize(like_smd, z, args=(nestData), method='Nelder-Mead')
            ex = 0

        except decimal.InvalidOperation:
            #print("Decimal error in optimizer - go to next replicate")
            ex = 100
            # ^ this won't ever show up if we are skipping the rest of the loop
            likeVal[r] = np.full(num_out, ex)
            continue            # skip the rest of this iteration, go to next

        except OverflowError:
            #print("Overflow error in optimizer - go to next replicate")
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
        like_val       = np.array([repID, s2, mp2, mf2, ss2, mps2, mfs2, stormDur, stormFreq, probSurv, SprobSurv, probMortFlood, SprobMortFlood, hatchTime, numNests, discovered, excluded, obs_int, ex])
        #like_val       = np.asarray([repID, s2, mp2, mf2, ss2, mps2, mfs2, stormDur, stormFreq, probSurv, SprobSurv, probMortFlood, SprobMortFlood, numNests, discovered, excluded, obs_int, ex])
        # I swear this worked before, trying something else

        likeVal[r]     = like_val # likelihood values for this data replicate recorded to larger array
        # repID is being converted into a float?
        #print("likelihood vals for this replicate:", like_val)

    #column_names     = ['rep_ID', 'psurv_est', 'ppred_est', 'pflood_est', 'stormsurv_est', 'stormpred_est', 'stormflood_est', 'storm_dur', 'storm_freq', 'psurv_real', 'stormsurv_real','pflood_real', 'stormflood_real', 'hatch_time','num_nests', 'num_discovered','num_excluded','obs_int', 'exception']
    #likelihoodValues = DataFrame(likeVal, columns=column_names) # this gives the likelihood vals from 1 set of params?
    #values           = fill_array(values, likeVal, index)
    nrows    = likeVal.shape[0] # the number of rows per param set 
    values[index:index+nrows,] = likeVal # fill the rows of "values" corresponding to the last param set
        # we are finally filling in values, and it == like_val and likeVal (at least for the first loop, with only one replicate)
    index    = index + nrows # increment index 
    #print("index= ", index)
    #print("replicate finish:", datetime.now())

column_names     = np.array(['rep_ID', 'psurv_est', 'ppred_est', 'pflood_est', 'stormsurv_est', 'stormpred_est', 'stormflood_est', 'storm_dur', 'storm_freq', 'psurv_real', 'stormsurv_real','pflood_real', 'stormflood_real', 'hatch_time','num_nests', 'num_discovered','num_excluded','obs_int', 'exception'])
# header values need to be stored to np.ndarray for np.savetxt; actually, needs to be a string
colnames = ', '.join([str(x) for x in column_names])
filepath     = Path.cwd() / ('output') / dir_name / ('ml_val_' + dir_name)
headerTF     = False if exists(filepath) else True 
#values.to_csv(filepath, mode='a', header=headerTF) # want to write the csv after all the runs, replicates, and param sets
# .to_csv doesn't work for numpy objects, but since it's all numeric, should be able to do:
# .savetext('filename', df, delimiter=',') - if strings included, add fmt argument

np.savetxt(filepath, values, delimiter=',',header=colnames)
#print("finish time:", datetime.now())
# if 
# (needs to be outside the loop)
end = datetime.now()
print("running time:", started - end,"; number of reps:", nreps)


    #filepath         = Path('{}/likelihood_values_{}.csv'.format(dir_name, dir_name))
    # filepath         = Path.cwd() / ('output') / dir_name / ('ml_val_' + dir_name)
    # headerTF         = False if exists(filepath) else True 
    
    # # append values from this param set to the csv:
    # #likelihoodValues.to_csv(filepath, mode='a', header=headerTF)
    # values.to_csv(filepath, mode='a', header=headerTF)
    # this output is very strange - extra rows, etc

    # when you do 500 reps, if it breaks you still get nothing. maybe put this csv command into the other loop?

