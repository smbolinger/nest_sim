# sudo vim -o file1 file2 [open 2 files] 

# NOTE 9-24: I fixed the calculation of true DSR and added a true DSR for discovered nests only.
# This second value shows there's some bias towards discovering nests that end up hatching.
# I.e. the DSR of the discovered nests is consistently higher than DSR for all nests (by about 0.02)
# (see out/24sep-dsr.txt)
# so the DSR estimates from MARK and the matrix model are high, but that's because DSR for discovered nests is high.
# What could be causing this?
# >> I don't think fateCuesProb is affecting failed nests more than hatched nests
# >> check whether nests excluded for unknown fate have a higher proportion of failed - no 
# >> check proportion hatched in discovered nests (high) and in discovered - excluded 
# NEED TO FIX HANDLING OF FINAL INTERVAL in prog_mark() function - don't have perfect knowledge

# NOTE 6-28 so have I basically introduced a second way for nests to die, and how does that affect the first one?
# i.e. with storms added in, DSR is not the same (because some die due to storms)
# need to either have a constant DSR or a separate storm DSR
# does having two separate DSR actually help answer the question???
# the question is about how storms affect our ability to classify, not about how they affect nest survival
# but I guess one informs the other...
# maybe just have nests fail as usual, and if it's during a storm, failure is flooding, otherwise predation

# ctrl-w switches between windows; add direction j,k,h,l 
# try a smaller version of the nest survival model to work on efficiency 
# optimizer may be the limiting step, but it is in turn affected by the functions 
# try to make those more efficient 
# NOTE need to convert T/F to 1/0 bc numpy arrays hold only numbers 
# if comparing survey_days and storm_days is too costly, can I just generate them once for all replicates? 
from datetime import datetime
import decimal
from decimal import Decimal
from itertools import product
import numpy as np 
from os.path import exists
import os
from pathlib import Path
from scipy import optimize
import scipy.stats as stats
rng = np.random.default_rng(seed=102891) 
# ----------------------------------------------------------------------------------------
# OPTIMIZER PARAMETERS: 
nruns   = 1 # number of times to run  the optimizer 
#nreps   = 1000 # number of replicates of simulated nest data
#nreps   = 500
#nreps   = 10
nreps   = 1
num_out = 21 # number zof output parameters
#dir_name = datetime.now().strftime("%m%d%Y")
# ---------------------------------------------------------------------------------------
# NEST MODEL PARAMETERS:
# NOTE the main problem is that my fate-masking variable (storm activity) also leads to certain nest fates
# NOTE 2: how many varying params is a reasonable number?
breedingDays   = [150] 
discProb       = [0.7] 
#numNests       = [500]
numNests       = [100]
#numNests       = [50]
#numNests        = [1000]
uncertaintyDays = [1]
print(
        "STATIC PARAMS:\nbreeding season length:",
        breedingDays,
        "; discovery probability:",
        discProb,
        "; number of nests:",
        numNests,
        "how long nest fate is discoverable (in days):",
        uncertaintyDays
        )
# only really need to be in lists if they are part of the combinations below
#probSurv       = [0.95 ]   # daily prob of survival
probSurv       = [0.93 ]   # daily prob of survival
probMortFlood  = [0.1]    # 10% of failed nests are due to flooding - not .1 of all nests
SprobSurv      = [0.2] # daily prob of survival during storms - kind of like intensity
SprobMortFlood = [1.0]   # all failed nests during storms fail due to flooding
stormDur       = [1,3]
stormFreq      = [3,5]
#obsFreq        = [3,5,7]
obsFreq        = [3]
#hatchTime      = [16,20,28]
hatchTime      = [20]
#assignUnknown  = [0,1] # whether or not nests that end during storms are marked "unknown" (vs. flooded)
assignUnknown  = [0] # whether or not nests that end during storms are marked "unknown" (vs. flooded)
# maybe better to have an uncertainty param of some sort? if random draw is less than x, etc
#observerError  = 
#probCorrect     = 0.8
fateCuesPresent = 0.8 # 80% chance that field cues about nest fate are present/observed
# based on around 80% success rate in identifying nest fate (from camera study)
# this is essentially uncertainty?
numMC           = 0 # number of nests misclassified
paramsList      = list(product(
        numNests, probSurv, SprobSurv, stormDur, stormFreq, hatchTime, 
        #obsFreq, probMortFlood, SprobMortFlood, breedingDays, discProb
        obsFreq, probMortFlood, breedingDays, discProb, assignUnknown
        ))
# NOTE the above probably only needs to be for the params that i'm testing?
#print("-----------------------------------------------------------------------------------------")
#print("USING NUMBER OF PARAMS DURING TESTING")
#print("-----------------------------------------------------------------------------------------")
#print(">> params list:", paramsList, "length:", len(paramsList))
paramsArray     = np.array(paramsList)   # don't want prob surv to be an integer!
#print(">> params array:", paramsArray, "length:", len(paramsArray))
#@#print(">> params list:", paramsList, "length:", len(paramsList), "& number of combinations to be tested:", len(paramsArray))
nrows   = len(paramsList)*nreps*nruns
# ---------------------------------------------------------------------------------------
# Import weekly storm probability and weekly nest initiation probability:
init= np.genfromtxt(
        fname="/mnt/c/Users/Sarah/Dropbox/nest_models/storm_init3.csv",
        dtype=float,
        delimiter=",",
        skip_header=1,
        usecols=2
        )
initProb = init / np.sum(init)
stormProb = np.genfromtxt(
        fname="/mnt/c/Users/Sarah/Dropbox/nest_models/storm_init3.csv",
        dtype=float,
        delimiter=",",
        skip_header=1,
        usecols=3 # 4th column 
        )
print(">> storm probability, by week:",stormProb, len(stormProb),"sum=", np.sum(stormProb))
print(">> initiation probability, by week:", initProb, len(initProb),"sum=",np.sum(initProb))
storm_weeks2 = np.arange(14,29,1)
print(">> storm weeks from list:", storm_weeks2, len(storm_weeks2))
weekStart = (storm_weeks2 * 7) - 90
weekStart = weekStart.astype(int)
print(">> start day for each week:", weekStart, len(weekStart))
# ---------------------------------------------------------------------------------------
# SAVE FILES 
dirName  = datetime.today().strftime('%m%d%Y_%H%M%S') # name for unique directory to hold all output
#print(">> directory name:", dirName)
todaysDate = datetime.today().strftime("%Y%m%d")
#likeFile   = Path.home() / '/mnt/c/Users/Sarah/Dropbox/nest_models/py_output/new' / dirName / ("ml_values.csv")
likeFile   = Path.home() / '/mnt/c/Users/Sarah/Dropbox/nest_models/py_output' / dirName / ("ml_values.csv")
#likeFile   = Path.home() / '/mnt/c/Users/Sarah/Dropbox/nest_models/py_output/ml_values.csv'
#likeFile   = Path.home() / '/mnt/c/Users/Sarah/Dropbox/nest_models/py_out/ml_values.csv'
likeFile.parent.mkdir(parents=True, exist_ok=True)
print(">> likelihood file path:", likeFile)
column_names     = np.array(['rep_ID', 'mark_s', 'psurv_est', 'ppred_est', 'pflood_est', 'stormsurv_est', 'stormpred_est', 'stormflood_est', 'storm_dur', 'storm_freq', 'psurv_real', 'stormsurv_real','pflood_real', 'stormflood_real', 'hatch_time','num_nests', 'obs_int', 'num_discovered','num_excluded', 'exception'])
# header values need to be stored to np.ndarray for np.savetxt; actually, needs to be a string
colnames = ', '.join([str(x) for x in column_names])
# --------------------------------------------------------------------------------------------
# FUNCTIONS
# Some are very small and specific (e.g. logistic function) while others are quite involved.
# --------------------------------------------------------------------------------------------
def decreaseProb(p, numNests, lastActive, lastChecked):
    # this one might be making the model too complicated
    # plus, the decreases are arbitrary values
    cueProb = np.array(numNests)
    cueProb[lastActive - lastChecked < 2] = p
    cueProb[lastActive - lastChecked == 2] = 0.9*p
    cueProb[lastActive - lastChecked > 2] = 0.81*p
    print("probability of nest fate cues:", cueProb)
    return(cueProb)

def searchSorted2(a, b):
    #out = np.zeros(a.shape)
    out = np.zeros((a.shape[0], len(b)))
    for i in range(len(a)):
        #out[i] = np.searchsorted(a[i], b[i])
        #print("sorted search of\n", b, "within\n", a[i])
        #print(">> sorted search of", b, "within", a[i])
        out[i] = np.searchsorted(a[i], b)
        #print("sorted search of\n", b, "within\n", a[i], ":\n", out, out.shape)
        #print("index positions:", out, out.shape)
        # shouldn't the output have the shape of b?
    #print(">> index positions:\n", out, out.shape)
    return(out)

def stormGen(frq, dur):
    out = rng.choice(a=weekStart, size=frq, replace=False, p=stormProb)
    #out = rng.choice(a=storm_start, size=frq, replace=False, p=stormProb)
    #out = rng.choice(a=weekStart, size=frq.astype(int), replace=False, p=stormProb)
    dr = np.arange(0, dur, 1)
    stormDays = [out + x for x in dr]
    stormDays = np.array(stormDays).flatten()
    print(">> storm days:", stormDays)
    return(stormDays)
# --------------------------------------------------------------------------------------------------------------------------
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
# --------------------------------------------------------------------------------------------------------------------------
# This is just the logistic function
# Trying out type hints (PEP 484) to keep output from overflowing
def logistic(x)->np.float128:
    #return 1.0/( 1.0 + math.exp(-x) )
    return 1.0/( 1.0 + np.exp(-x) )
# --------------------------------------------------------------------------------------------------------------------------
# This function computes the intersection of two arrays more quickly than intersect1d
#@profile
def in1d_sorted(A,B): # possible days observer could see nest is intersection of observable and survey days

    idx = np.searchsorted(B, A)
    idx[idx==len(B)] = 0
    return A[B[idx] == A]
# --------------------------------------------------------------------------------------------------------------------------
#def mk_surveys(stormDays, params):
def mk_surveys(stormDays, obsFreq, breedingDays):
    # first day of each week because the initiation probability is weekly 
    # the upper value should not be == to the total number of season days because then nests end 
    # after season is over 
    start     = rng.integers(1, high=5) # random day of first survey from first 5 breeding days          
    #dates     = np.arange(0, 70, step=7) 
    end       = start + breedingDays
    surveyDays = np.arange(start, end, step=obsFreq)
    print(">> all survey days, including those cancelled by storms:\n", surveyDays, len(surveyDays)) 
    stormSurvey = np.isin(surveyDays, stormDays) 
    print(">> was survey canceled by storm?:\n", stormSurvey, len(stormSurvey)) 
    surveyDays = surveyDays[np.isin(surveyDays, stormDays) == False] 
    # keep only values that aren't in storm_days 
    print(">> all survey days, minus storms:\n", surveyDays, len(surveyDays)) 
    return(surveyDays)

def survey_int(surveyDays):
    surveyInts = np.array( [surveyDays[n] - surveyDays[n-1] for n in range(1, len(surveyDays)-1) ] )
    #print(">> interval between current survey and previous survey:\n", surveyInts, len(surveyInts))
    return(surveyInts)
#
# --------------------------------------------------------------------------------------------------------------------------
# NEST DATA COLUMNS: 
# 0) ID number        | 4) flooded? (T/F)          | 8) j (last active)      | 12) final interval storm (T/F) 
# 1) initiation date  | 5) surveys til discovery   | 9) k (last checked)     | 13) num other storms 
# 2) end date         | 6) discovered? (T/F)       | 10) assigned fate
# 3) hatched? (T/F)   | 7) i (first found)         | 11) number of observation intervals 

#paramsList      = list(product(
#          0           1         2          3         4          5
#        numNests, probSurv, SprobSurv, stormDur, stormFreq, hatchTime, 
#        #obsFreq, probMortFlood, SprobMortFlood, breedingDays, discProb
#           6           7             8            9          10
#        obsFreq, probMortFlood, breedingDays, discProb, assignUnknown
#        ))
# --------------------------------------------------------------------------------------------------------------------------
def randArgs():
    # Choose random initial values for the optimizer
    # These will be log-transformed before going through the likelihood function
    s     = rng.uniform(-10.0, 10.0)       
    mp    = rng.uniform(-10.0, 10.0)
    ss    = rng.uniform(-10.0, 10.0)
    mps   = rng.uniform(-10.0, 10.0)
    srand = rng.uniform(-10.0, 10.0) # should the MARK and matrix MLE start @ same value?
    z = np.array([s, mp, ss, mps, srand])

    return(z)

def ansTransform(ans):
        
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
    
    ansTransformed = np.array([s2, mp2, mf2, ss2, mps2, mfs2], dtype=np.float128)
    #print(">> results as an array:\n", ansTransformed)
    #print(">> results (s, mp, mf, ss, mps, mfs, ex):\n", s2, mp2, mf2, ss2, mps2, mfs2, ex)
    return(ansTransformed)
    
def mk_nests(params, init, stormDays, surveyDays): 

    # 1. Unpack necessary parameters
    # NOTE about the params at the beginning of the script:
    # some have only 1 member, but they are still treated as arrays, not scalars
    hatchTime = int(params[5]) 
    stormDur  = int(params[3]) 
    stormFrq  = int(params[4]) 
    obsFreq   = int(params[6]) 
    numNests  = int(params[0]) 
    discProb  = params[9]
    pSurv     = params[1]           # daily survival probability
    pfMort    = params[7]           # (conditional) probability of failure due to flooding
    nCol      = 15                  # number of output columns
    assignUnknown = params[10]      # how uncertain nest fates are treated
    breedingDays  = params[8]       # number of days in breeding season

    #@#print(">> discovery probability:", discProb)
    #@#print(">> hatch time:", hatchTime, "| storm duration:", stormDur, "| storm frequency:", stormFrq)
    #@#print(">> number of nests:", numNests, "| frequency of observations:", obsFreq)
    #@#print(">> probability of survival:", pSurv, "and failure due to flooding:", pfMort)
    #@#print(">> total days in the season:", breedingDays)

    # 2. Create lists of storm days and survey days, plus the intervals between surveys
    surveyInts = survey_int(surveyDays)
    nestData = np.zeros(shape=(numNests, nCol), dtype=int) 
    nestData[:,0] = np.arange(1,numNests+1) # nest ID numbers 

    # 3. Create a list of nest initiation dates (one for each nest) and make sure you have the correct number
    initWeek = rng.choice(a=weekStart, size=numNests, p=init)  # random starting weeks; len(a) must equal len(p)
    initiation = initWeek + rng.integers(7)                    # add a random number from 1 to 6 (?) 
    nestData[:,1] = initiation                                 # record to a column of the data array
    #print(">> number of nests:", numNests) 
    #print(">> initiation weeks:\n", initWeek, "\n>> initiation dates:\n", initiation, len(initiation)) 

    # 4. Decide how long each nest is active
    # >> use a negative binomial distribution - distribution of number of failures until success 
    #    in this case, "success" is actually the nest failing, so use 1-pSurv (the failure probability) 
    #    makes sense because we want to know the number of days until the nest fails.  
    # >> if value is greater than the incubation time, then the nest hatches 
    # >> then use the value to calculate end dates for each nest (end = initiation + survival)
    survival = rng.negative_binomial(n=1, p=(1-pSurv), size=numNests) 
    # but since the last trial is the success, need to subtract 1
    survival = survival - 1
    #@#print(">> probability of survival=", pSurv, ">> probability of mortality=", 1-pSurv)
    #print(">> survival in days:\n", survival, len(survival)) 
    survival[survival > hatchTime] = hatchTime # set values > incubation time to = incubation time (nest hatched) 
    nestEnd = initiation + survival # add the number of days survived to initiation date to get end date   
    nestData[:,2] = nestEnd 
    #print(">> end dates:\n", nestEnd, len(nestEnd)) 
    hatched = survival >= hatchTime # the hatched nests survived for hatchTime days 
    nestData[:,3] = hatched.astype(int) # make it numeric for the numpy array 
    failed  = ~hatched
    # >>>>>> Remember that int() only works for single values 
    #print("-----------------------------------------------------------------------------")
    #print(">> did nest hatch? (hatch time =", hatchTime,"):\n", hatched, len(hatched)) 
    #print(
    #        ">> proportion of nests hatched:", 
    #        sum(hatched)/numNests,
    #        "and expected proportion based on DSR:",
    #        pSurv**hatchTime
    #        )     
    # if previous value == F (nest failed), did nest fail due to flooding?  

    # FAILED NESTS:
    # on a regular day, prob of mortality = 1-DSR and conditional prob of flooding = 0.05
    # on a storm day, prob of mortality = 0.9 and conditional prob of flooding = 1
    # OR make it even simpler - all nests during storms fail, and cause is always flooding
    # all failed nests not during storm fail due to predation

    # 5. Decide cause of failure for failed nests:
    # >> create a vector of probabilities, one for each nest, to decide whether nest flooded (if it failed) 
    # >> the probability value will be compared to the probability of failure due to flooding
    pflood = rng.uniform(low=0, high=1, size=numNests) 
    # need to check whether this is the correct distribution 
    # NOTE: still needs to be conditional on nest having failed already...  
    # NOTE np.concatenate is for joining existing axes, while np.stack creates new ones
    nestPeriod = np.stack((initiation, nestEnd))
    nestPeriod = np.transpose(nestPeriod)
    #print(">> start and end of nesting period:\n", nestPeriod, nestPeriod.shape)
    #stormNest1 = initiation < 64
    #stormNest1 = np.zeros((1,3))
    #stormNest1 = np.zeros(numNests, len(stormDays))
    #stormNest = np.any((stormDays>nestPeriod[:,0]) & (stormDays < nestPeriod[:,1]))
    #stormNest = np.isin(stormDays, np.arange(initiation, endNest))
    #stormNest = np.searchsorted(
    # difficult to apply searchsorted to a 2d array D:
    # WHY do I need this?
    # at least looping over the storm days shouldn't be as slow as looping over nest rows?
    #stormNest = np.zeros(shape=(len(stormDays), numNests))
    #for s in range(len(stormDays)):
    #    # with 2 conditions, doesn't work; workss with 1 condition
    #    stormNest = stormDays[s] > initiation & stormDays[s] < endNest
    #    print(stormNest)

    #print("storm while nest was active:", stormNest1)
    #stormNest1 = np.any(stormDays
    # the above was suggested on stack exchange but may not be appropriate for my issue
    # elementwise comparison of the arrays; then need to select for rows where there are any Trues
    #stormNest = np.where(stormNest1 == True).any(axis=1)
    #stormNest = np.where(stormNest1 == True)
    stormNestIndex = searchSorted2(nestPeriod, stormDays)
    stormNest = np.any(stormNestIndex == 1, axis=1) # if index == 1, then storm Day is within the period interval 
    numStorms = np.sum(stormNestIndex==1, axis=1) # axis=1 means summing over rows?
    # NOTE I *think* this is actually number of storm intervals, which is what we want for the likelihood function
    # so that would be good...

    #print(">> was there a storm during nesting period?", stormNest)
    #print("number of storm days during nesting period:", numStorms)
    #@#print(">> number of storm intervals during nesting period:\n", numStorms)
    stormNest = stormNest.astype(int) # of those nest, some tiny fraction do survive
    flooded = np.where(pflood>pfMort, 1, 0) # if pflood>pfMort, flooded=1, else flooded=0 
    #print(">> active during a storm:\n", stormNest, len(stormNest)) 
    # since it's 1 and 0, can use arithmetic: 
    #floodedAll = stormNest + flooded > 0 # both need to be true 
    floodedAll = stormNest + flooded > 1 
    # now I don't understand what floodedAll is supposed to be doing (9 jul 2024)
    # shouldn't it be >1?
   
    #print(">> flooded:\n", floodedAll, len(floodedAll)) 
    #print(">> number of nests flooded:\n", sum(floodedAll==True), "of", len(floodedAll)) 
    #@#print(">> hatched as integers:\n", hatched.astype(int)) 
    #print(">> inverse of hatched:\n", (~hatched).astype(int)) 
    #floodedAndFailed = (floodedAll + (~hatched).astype(int)) > 1 
    # flooded == true and hatched == False; see NOTE above 
    # where flooded == true (1) and ~hatched == true(1), result will be > 1; else marked false
    # where you put the parentheses in ~hatched.astype(int) is important!  
    #nestData[:,4] = floodedAndFailed.astype(int) 
    nestData[:,4] = floodedAll.astype(int) 
    #print(">> did nest fail due to flooding:\n", floodedAndFailed, len(floodedAndFailed)) 
    #print(">> number of nests failed due to flooding:\n", sum(floodedAndFailed==True), len(floodedAndFailed)) 
    trueFate = np.empty(numNests) 
    #print("length where flooded = True", len(trueFate[floodedAll==True]))
    trueFate.fill(1) # nests that didn't flood or hatch were depredated 
    #trueFate[flooded==True] = 2 
    trueFate[floodedAll==True] = 2 
    trueFate[hatched==True] = 0 # was nest discovered?  
    fates = [np.sum(trueFate==x) for x in range(3)]

    # TRUE DSR
    # Calculate proportion of nests hatched and use to calculate true DSR
    # Why use log(hatchProp)? I think my reasoning was:
    # where P = period survival and N = length of nesting period
    # >> P = DSR ^ N; log(P) = N*log(DSR) / N = log(DSR) so exp(log(DSR)) gives us DSR  
    # >> period survival = num hatched / num total 
    # but is this calculation correct? and does period survival == apparent success?
    # try calculating it a different way using a one-day interval
    hatchProp = sum(hatched)/numNests
    print("hatch proportion:\n", sum(hatched), "divided by", numNests, "equals", hatchProp)
    trueDSR =  np.exp(np.log(hatchProp)/hatchTime)
    #print(">> true fate (hatched, predated, flooded):\n", trueFate,"(",fates,")")
    #print(">>>> true DSR:", (sum(hatched)/numNests)/hatchTime, "and period survival:", sum(hatched)/numNests)
    #print(">>>> true DSR:", (hatchTime * np.log(sum(hatched)/numNests)), "and period survival:", sum(hatched)/numNests)
    print(">>>> true DSR:", np.exp((np.log(hatchProp)/hatchTime)), "and period survival:", sum(hatched)/numNests)
    # TRY THIS WAY: 
    # daily mortality = num failed (total-num hatched) divided by total exposure days (add together survival periods)
    # DSR = 1 - daily mortality
    trueDSR2 = 1 - ((numNests - hatched.sum()) / survival.sum()) 
    print(">>>> total exposure days:", survival.sum())
    print(">>>>> OR true DSR, calculated correctly:", trueDSR2)
    #@#print(">> true nest fates:\n", trueFate)
        # this isn't just a one-time check; prob of discovery on each survey day 
    # NOTE this name is a little misleading 
    #daysTilDiscovery = rng.negative_binomial(n=1, p=1-discProb, size=numNests) 
    daysTilDiscovery = rng.negative_binomial(n=1, p=discProb, size=numNests) 
    # see above for explanation of p 
    nestData[:,5] = daysTilDiscovery 
    print(">> survey days until discovery:\n", daysTilDiscovery, len(daysTilDiscovery)) 
    # NOTE position and position2 are not giving accurate indices
    # OR nest initiation is going on too long
    # surveys should go on for longer than nest initiation - until after last nest ends

    # what are the first and last *possible* survey dates for the nest?  
    # this finds the index of where initiation would be inserted in surveyDays 
    # use this index; we don't want the previous day, because the nest won't have been initiated 
    position = np.searchsorted(surveyDays, initiation) 
    print(">> position of initiation date in survey day list:\n", position, len(position)) 
    firstSurvey = surveyDays[position] 
    print(">> first possible survey date:\n", firstSurvey, len(firstSurvey)) 
    # NOTE last possible survey will be after nest ends (either hatch or fail) 
    #position2  = np.searchsorted(surveyDays, initiation+hatchTime) 
    # needs to be initiation + survival time, AKA nestEnd
    position2 = np.searchsorted(surveyDays, nestEnd)
    print(">> position of end date in survey day list:\n", position2, len(position2)) 
    lastSurvey = surveyDays[position2] 
    #print("last possible survey:", lastSurvey, lastSurvey.shape)
    possSurveyDates = np.stack((firstSurvey, lastSurvey)) # double parens so numpy knows it's not two separate arguments
    possSurveyDates = np.transpose(possSurveyDates)
    #lastSurvey = surveyDays[position2-1] 
    #@#print(">> last possible survey date (check that storms were excluded):\n", lastSurvey, len(lastSurvey)) 
    #@#print(">> start and end of nesting period:\n", nestPeriod, nestPeriod.shape)
    #@#print(">> first and last possible survey dates:\n", np.stack((firstSurvey, lastSurvey))) # need double parens
    #print(
    #        ">> start of nest, end of nest, first possible survey, last possible survey:\n",
    #        np.concatenate((nestPeriod,possSurveyDates), axis=1)
    #        )
    totalSurvey = position2-position 
    #@#print(">> total possible surveys to observe nest (position2 - position):\n", position2-position) 
    totalReal  = totalSurvey - daysTilDiscovery 
    #print(">> actual number of surveys for this nest:\n", totalReal, len(totalReal))
    #nestData[:,11] = totalReal-1 # number of observation intervals is number observations - 1 
    nestData[:,11] = totalReal  # number of observations
    # need to decide if it's number of observations or observation intervals

    discovered = daysTilDiscovery < (position2-position) 
        # this is not giving me the correct discribution (30% undiscovered) - it's less than 5% undiscovered 
    print(">> was nest discovered:\n", discovered, len(discovered)) 
    #print(">> was nest discovered:\n", sum(discovered==True), len(discovered)) 
    # this one (below) is not correct - would be proportion expected to be discovered on all survey days. 
    # but nests only need to be discovered once (unlike survival)
    # so overall probability of being discovered doesn't change, even over multiple surveys 
    #print(">> proportion of nests discovered:", sum(discovered)/numNests, "vs. expected proportion:", discProb**(hatchTime/obsFreq))
    #print(">> proportion of nests NOT discovered:", (numNests-sum(discovered))/numNests, "vs. expected proportion:", (1-discProb)**(hatchTime/obsFreq))
    #print(">> proportion of nests discovered:", sum(discovered)/numNests, "vs. expected proportion:", discProb)
    nestData[:,6] = discovered.astype(int) # convert to numeric for numpy array  

    #print(">> index of firstFound:\n", position+daysTilDiscovery) 
    firstFound = np.zeros(numNests) 
    firstFound[discovered==True] = surveyDays[position+daysTilDiscovery][discovered==True] 
    #firstFound = surveyDays[position+daysTilDiscovery]

    # how is the above line different from the one that's throwing the indexing error?
    # should be for indexing a nested list - is that what this is?
    # need the number of values on left side == number on right side
     # NOTE should everything for the observed nests be restricted to where discovered==True?  
     # firstFound needs to be defined beforehand 
     # you don't even need the False comparison if you start with a vector of zeros 
    print(">> nest first found:\n", firstFound, len(firstFound)) 
    lastActive = np.zeros(numNests)
    lastActive[discovered==True] = surveyDays[position2][discovered==True] 
    print( ">> nest last active:\n", lastActive, len(lastActive)) 
    # Last checked will be one survey after last active, unless hatch == True
    lastChecked = np.zeros(numNests) 
    lastChecked[discovered==True] = surveyDays[position2+1][discovered==True] 
    print(">> nest last checked, w/o hatch:\n", lastChecked, len(lastChecked)) 
    lastChecked[hatched==True] = lastActive[hatched==True]
    print(">> nest last checked:\n", lastChecked, len(lastChecked)) 
    #nestData[:,8]  = firstFound 
    nestData[:,7]  = firstFound 
    nestData[:,8]  = lastActive 
    nestData[:,9] = lastChecked 

    obsPeriod      = lastChecked - firstFound 
    print(">> length of observation period:\n", obsPeriod, len(obsPeriod)) # is the actual number of days in the period greater than expectedsPeriod) 
    stormDuring    = obsPeriod/obsFreq> totalReal #  BUT how to tell if it's final interval or not? 
    numStorms = (obsPeriod/obsFreq - totalReal) / obsFreq # number extra days in period divided by normal interval length 
    # I changed how surveyInts was defined; now it's current minus previous, not next minus current
    #stormInterval = surveyInts[position2] > obsFreq 
    # what is the above doing??
    stormInterval = surveyInts > obsFreq
    #@#print(">> was there a storm in this interval?\n", stormInterval)
    stormIntFinal = surveyInts[position2] > obsFreq 
    print(">> was there a storm in the final interval for a given nest?\n", stormIntFinal, len(stormIntFinal)) # basically ask is final interval > 3 
    # needs to be positionEnd -1 because of how intervals are calculated 

    # stormFinal will be true if the value of stormInterval at the index of nestEnd is true
    # index of nestEnd = position2
    # NOTE why was I making stormFinal in the first place? (6/27)
    #stormFinal = np.zeros(numNests)
    #stormFinal.fill(99)
    #stormFinal.fill(np.nan) # NaNs are not correctly interpreted
    #print(">> initialize array to hold stormFinal:\n", stormFinal)
    #stormFinal[discovered==True] = stormInterval[position2][discovered==True]
    #stormFinal[np.where(discovered==True)] = stormInterval[position2][np.where(discovered==True)]
    # stormFinal[discovered==True] is out of bounds, or the other side is
    #stormFinal = stormIntFinal[position2]
    # position2 is going to be relative to the list of survey days, so not appropriate for indexing nests
    # so trying to compare two unrelated arrays
    #print(">> stormFinal filled:\n", stormFinal)
    nestData[:,12] = stormIntFinal.astype(int) #nestData[:,13] = numStorms - as.integer(stormFinal) 
    #@#print(">> stormIntFinal in integer form:\n", nestData[:,12], stormIntFinal.shape)
    #nestData[:,13] = numStorms - stormIntFinal.astype(int) 
    nestData[:,13] = numStorms 

    assignedFate = np.zeros(numNests) # if there was no storm in the final interval, correct fate is assigned 
    #assignedFate.fill(99)
    assignedFate.fill(7)
    #@#print(">> assigned fate array & its shape:\n", assignedFate, assignedFate.shape)
    #print(">> proportion of nests assigned hatch fate:", np.sum((assignedFate==0)[discovered==True]),"vs expected:", pSurv**hatchTime)
    #print(">> proportion of nests assigned hatch fate:", np.sum(np.where(assignedFate==0 & discovered==True))/np.sum(discovered==True),"vs expected:", pSurv**hatchTime)

    #print(">> proportion of nests assigned hatch fate:", np.sum(assignedFate[assignedFate==0&discovered==True])/np.sum(assignedFate[discovered==True]),"vs expected:", pSurv**hatchTime)

    fateCuesProb = rng.uniform(low=0, high=1, size=numNests)
    #@#print(">> random probs to compare to fateCuesPresent:\n", fateCuesProb, fateCuesProb.shape)
    #if np.any(fateCuesProb, axis=0) < fateCuesPresent: # this is not evaluating, so all fates are 7
        # or somehow it's evaluating to False?
    #if fateCuesProb.any() < fateCuesPresent:
    #    assignedFate[stormIntFinal==False] = trueFate[stormIntFinal==False] 
    #else:
    #    assignedFate[stormIntFinal==False] = 7
    assignedFate[fateCuesProb < fateCuesPresent] = trueFate[fateCuesProb < fateCuesPresent]
    # fate cues prob should be afecting all nest fates equally, not just failures.

    #print(
    #        ">> proportion of nests assigned hatch fate:",
    #        #np.sum((assignedFate==0)[discovered==True])/np.sum(assignedFate[discovered==True]),
    #        np.sum((assignedFate==0)[discovered==True])/np.sum(discovered==True),
    #        "vs expected:", 
    #        #pSurv**hatchTime)
    #        np.sum((trueFate==0)[discovered==True])/np.sum(discovered==True)
    #        )
    #print(">> assigned fates:\n", assignedFate, len(assignedFate))

    # add another parameter: will storm nests be assigned a fate of unknown or flooded? 
    if assignUnknown == 1: 
        assignedFate[stormIntFinal==True] = 7 
        assignedFate[lastChecked-nestEnd > uncertaintyDays]
    else: 
        assignedFate[stormIntFinal==True] = 2 
        #csvHead = "ID, initiation, end, hatched, flooded, surveys_til_discovery, discovered, storm_during, i, j, k" 
        #           0      1         2    3          4      5                       6           7        8  9  10  11 
    #nestData[:,11] = assignedFate
    #nestData[:,7] = assignedFate
    aFates = [np.sum((assignedFate == x)[discovered==True]) for x in range(4)]
    print(
            ">> assigned fate (hatched, predated, flooded, unknown):", 
    #        #aFates[0:3][discovered==True], 
            aFates[0:3], 
    #        #numNests - np.sum(aFates)
           np.sum(discovered==True)- np.sum(aFates)
            )
    nestData[:,10] = assignedFate
    nestData[:,14] = trueDSR2

    hatchPropD = sum(hatched[assignedFate != 7])/len(discovered[discovered==True])
    print("proportion of discovered nests that hatched:\n", sum(hatched[assignedFate != 7]), "divided by",
            len(discovered[discovered==True]), "equals", hatchPropD)
    #trueDSR_disc =  np.exp(np.log(hatchPropD)/hatchTime)
    #trueDSR_disc = 1 - ((numNests - hatched.sum()) / survival.sum()) 
    numDisc = discovered.sum()
    numDiscH = discovered[hatched==True].sum()
    survDisc = survival[discovered==True].sum()
    print("number discovered:", numDisc, ", number discovered that hatched:", numDiscH, ", and exposure days:", survDisc)
    #trueDSR_disc = 1 - ((discovered.sum() - hatched.sum()) / survival.sum()) 
    trueDSR_disc = 1 - ((numDisc - numDiscH) / survDisc) 
    print("true DSR of discovered nests only:", trueDSR_disc)

    #@#print(">> nest data:\n----id--ini-end-hch-fld-std-dsc-i--j--k-fate-nobs-sfin-nstm\n", nestData)

    csvHead = "ID, initiation, end, hatched, flooded, surveys_til_discovery, discovered,\
               i, j, k, a_fate, num_obs, storm_during_final, num_storms" 

               #num_stm_survey_excl_final 
    np.savetxt("nestdata.csv", nestData, fmt="%d", delimiter=",", header=csvHead) 
    return(nestData) 

# NEST DATA COLUMNS: 
# 0) ID number        | 4) flooded? (T/F)          | 8) j (last active)      | 12) final interval storm (T/F) 
# 1) initiation date  | 5) surveys til discovery   | 9) k (last checked)     | 13) num other storms 
# 2) end date         | 6) discovered? (T/F)       | 10) assigned fate
# 3) hatched? (T/F)   | 7) i (first found)         | 11) number of observation intervals 

# --------------------------------------------------------------------------------------------------------------------------
# MAYFIELD
# Mayfield's original estimator was defined as: DSR = 1 - (# failed nests / # exposure days)
# so if DSR = 1 - daily mortality, daily mortality = # failed nests / # exposure days

# Johnson (1979) provided a mathematical derivation that allowed the calculation of variance for the estimate:
# for a single day, probability of survival is s and probability of failure is (1-s)
# > i.e. the probability of a nest surviving three days and failing on the fourth is s*s*s*(1-s) 
# for an interval of length k days, probability of survival is s**k and failure is s**(1/2k-1)(1-s)
# this assumes that a failed nest survived half (minus a day) of the interval and then failed
# Johnson's modified ML estimator: mortality = (f1 + sum(ft)) / (h1 + sum(t*ht) + f1 + 0.5 sum(t*ft)) 
# > created by differentiating the log-likelihood equation and setting to zero (maximizing)
# f1 and h1 represent an interval between visits of one day, which is not used in our studies
# so we end up with sum(ft) / (sum(t*ht) + 0.5*sum(t*ft)) where t is interval length and
# f and h represent number of failures and hatches, respectively

# The model used in Program MARK is based on Dinsmore (2002); allows for variance in DSR & use of covariates
# 

def mayfield(ndata):
#    I am assuming the nest data that is input has already been filtered to only discovered nests w/ known fate
#    dat = ndata[
    hatched = np.sum(ndata[:,3])
    failed = len(ndata) - hatched 
    print(">> Calculate Mayfield estimator.")
    print(">> number of nests hatched:", hatched, "and failed", failed)
    mayf = failed / (hatched + 0.5*failed)
    print(">> Mayfield estimator of daily mortality (1-DSR) =", mayf) 

    return(mayf)


def prog_mark(s, ndata):

    # 1. grab the data for the input for MARK i
    # (nest ID, date first found, date last active, date last checked, assigned fate)
    # inp[0] = ID | inp[1] = i | inp[2] = j | inp[3] = k | inp[4] = fate `
    inp = ndata[:,np.r_[0,7:11]] # doesn't include index 11
    #@#print("inp (ID, i, j, k, fate:)\n",inp)
    nocc = ndata[0,10] # load number of occasions from data
    l    = len(inp)
    #print("l=", l, "| s=", s, "| nocc=", nocc)
    #print("----------------------------")

    # 2. extract rows where j minus i does not equal zero (nest wasn't only observed as active for one day)
    #    - the model requires all nests to have at least two observations while active
    inp = inp[(inp[:,2]-inp[:,1]) != 0] # access all rows; create mask and index inp using it
    #@#print("----------------------------")
    #@#print(inp)
    #@#print("----------------------------")
    #@#print("length=", len(inp), "| s=", s)

    # 3. create vectors to store:
    # > a) the probability values for each nest
    # > b) the degrees of freedom for each value
    allp   = np.array(range(1,len(inp)), dtype=np.longdouble) # all nest probabilities 
    alldof = np.array(range(1,len(inp)), dtype=np.double) # all degrees of freedom
    #print("length of array:", len(allp))

    #4. fill the vectors
    for n in range(len(inp)-1): # want n to be the row NUMBER
        #print("nest ID=", n, "| i=", inp[n,1], "| j=", inp[n,2], "| k=", inp[n,3])

        # for the basic case where psurv is constant across all nests and times:
        #   - count the total number of alive days
        #   - count the number of days in the final interval (for failed nests)
        alive_days = inp[n,2] - inp[n,1] # interval from first found to last active = all nests are KNOWN to be alive
        alive_days = alive_days - 1 # since this is essentially 1-day intervals, need 1 fewer than total number

        final_int  = inp[n,3] - inp[n,2] 
        final_int  = final_int - 1
        # for failed nests, failure occurs in interval from last active to last checked 
        # for hatched nests, last active == last checked (assumption is that hatched nests are found on hatch day)
        #print("days nest was alive:",alive_days,"& final int:",final_int)
        # NOTE need nests to be alive for at least one interval

        # Probability equation: 
        # > daily probability of survival (DSR) raised to the power of intervals nest was known to be alive
        # > for hatched nests, that's it
        # > for failed nests, exact failure date can be unknown
        # > > but we know nest wasn't alive for the entire final interval
        # > > so add in probability of NOT surviving one interval (1-DSR)
        # > hatched nests also have one extra degree of freedom (dof)
        
        # FAILED NESTS will have final_int (last checked-last active) greater than zero
        if final_int > 0:
            p   = (s**alive_days)*(1-(s**final_int)) 
            dof = alive_days
            #print(">> nest", inp[n,0], "failed. likelihood=", p)
            # but if final_int==0 (survived), this will become zero
        # HATCHED NESTS will have final_int == 0
        else:
            p   = s**alive_days
            dof = alive_days + 1
            #print(">> nest", inp[n,0], "hatched. likelihood=", p)
            #print("probability=", p)

        allp[n]   = p # NOTE this line is throwing the Deprecation Warning
        #allp[n,0]   = p # NOTE this line is throwing the Deprecation Warning
        # apparently warning means that n is a 2d array, so need to index it
        # never mind, apparently it's one dimensional 
        alldof[n] = dof
        #    once we have all the probabilities then...?
        #    where do exposure days come into it?
        #    might just be alive_days + final_int
        #    technically, raise prob to power of frequency
        #    we are assuming each occurs only once
    #print(">> all nest cell probabilities:\n", allp)
    #print(">> all degrees of freedom:\n", alldof)
    #lnp  = np.log(allp) # vector of all log-transformed probabilities
    lnp  = -np.log(allp) # vector of all log-transformed probabilities
    #print("log of all nest cell probabilities:", lnp)
    #print(">> negative log likelihood of each nest cell probability:", lnp)
    #lnSum = lnp.sum()
    #NLL = -1*lnp.sum()
    NLL = lnp.sum()
    #print(">> sum to get negative log likelihood of the data:", NLL)
    return(NLL)

# --------------------------------------------------------------------------------------------------------------------------
def mark_wrapper(srn, ndata):
    # This function calls the program MARK function when given a random starting value (srn) and some nest data (ndata)
    # > values given to the optimizer are transformed before being given to the MARK function
    # > > this allows a larger range of values for the optimizer to work over without overflow
    # > > but the values given to the function are still between 0 and 1, as is required
    # > Create vector to store the log-transformed values, then fill:
    s = np.ones(numNests, dtype=np.longdouble)
    s = logistic(srn)
    # NOTE is this multiple random starting values (for each nest) or one random starting value?
    #@#print("logistic of random starting value for program MARK:", s, s.dtype)
    # the logistic function tends to overflow if it's a normal float; make it np.float128
    ret = prog_mark(s, ndata)
    #@#print("ret=", ret)
    return ret

 # ------------------------------------------------------------------------------------------------------------- 
 # this function does the matrix multiplication for a SINGLE interval of length intElt days 
 # during observaton, nest state is assessed on each visit to form an observation history 
 # the function calculates the negative log likelihood of one interval from the observation history 
 # these can then be multiplied together to get the overall likelihood of the observation history 
 # the function takes 4 arguments: 
 #    > intElt - length in days of the observation interval being assessed 
 #    > initial state (stateI) - state of the nest at the beginning of this interval 
 # for analysis, these nest states are coded as a 1-dimensional matrix (vector): 
 # 
 #               [ 1 0 0 ] (alive)       [ 0 1 0 ] (failed - predation)        [ 0 0 1 ] (failed - flooding) 
 #
 # the formula used is from Etterson et al. (2007) 
 #    > in this case, the nest started the interval alive and ended it alive as well 
 #    > daily nest probabilities: s - survival; mp - mortality from predation; mf - mortality from flooding 
 #    > since these are daily probabilities, we raise the base matrix to the power of the number of days in the interval # 
 #                                          _         _  intElt           _   _ 
 #                 [ 1 0 0 ]               |  s  0  0  |                 |  1  | 
 #                                 *       |  mp 1  0  |            *    |  0  | 
 #                                         |_ mf 0  1 _|                 |_ 0 _|  
 #                              
 #           {  transpose(stateI)  *  useM, raised to intElt power  *    stateF  } 
def llInt(stateI, stateF, intElt, useM): 
 #def llloop(stateI, stateF, useM, stormDuring, numObs): 
    TstateI = np.transpose(stateI)             # by default, vector is in column form; convert to row (horizontal) 
    pwr = np.linalg.matrix_power(useM, intElt) # raise the matrix to the power of the number of days in obs int 
    lik = np.linalg.multi_dot([stateF, pwr, TstateI])    # take the dot product of the 3 matrices 
 # then you can raise this individual interval to the number of (normal) intervals 
 # and then create a new multiplication for storm intervals, and raise it to the number of storm intervals...  
    logL = Decimal(np.log(lik))*(-1)           # Decimal gives more precision; *-1 to get negative log likelihood 
 # print("int element:", intElt, "| stateF:",stateF, "| TstateI:", TstateI) 
 # NOTE why doesn't this print as negative?  
     
     #print("-ll of this observation interval:", logL) 
     #logL = float(logL) 
     #logLikelihood = logLikelihood + logL 
     # add in the likelihood for this one observation #return(logLikelihood) return(logL) 
 # -------------------------------------------------------------------------------------------------------------- 
## COLUMNS:
# 0) ID number        | 4) flooded? (T/F)                  | 8) i (first found) 
# 1) initiation date  | 5) surveys til discovery           | 9) j (last active)
# 2) end date         | 6) discovered? (T/F)               | 10) k (last checked)
# 3) hatched? (T/F)   | 7) storm during observation? (T/F) | 11) total observation intervals

# --------------------------------------------------------------------------------------------------------------
# LIKELIHOOD FUNCTION ###############################################################################

# This function computes the overall likelyhood of the data given the model parameter estimates.
#
# The model parameters are expected to be received in the following order:
# - a_s   = probability of survival during non-storm days
# - a_mp  = conditional probability of predation given failure during non-storm days
# - a_mf  = conditional probability of flooding given failure during non-storm days
# - a_ss  = probability of survival during storm days
# - a_mfs = conditional probability of predation given failure during storm days
# - a_mps = conxditional probability of predation given failure during storm days
# - sM    = for program MARK?

#def like(argL, numNests, obsInt, nestData, storm_days, survey_days):
# --------------------------------------------------------------------------------------------------------------------------
            #logLikelihood = logLikelihood + logL # add in the likelihood for this one observation
            # print("-ll of this nest:", logLikelihood)

        #logLike = logLike + logLikelihood        # add in the likelihood for the observation history of this nest
        # print("total -ll so far:", logLike)
    #return(logLike)

# try to keep these in numpy:
def like(perfectInfo, hatchTime, argL, numNests, obsFreq, nestData, surveyDays):
    # perfectInfo == 0 or 1 to tell you whether you know all nest fates or not
# NEST DATA COLUMNS: 
# 0) ID number        | 4) flooded? (T/F)          | 8) i (first found)      | 12) final interval storm (T/F) 
# 1) initiation date  | 5) surveys til discovery   | 9) j (last active)      | 13) num other storms* 
# 2) end date         | 6) discovered? (T/F)       | 10) k (last checked) 
# 3) hatched? (T/F)   | 7) assigned fate           | 11) number of observation intervals 
# * should be number of storm intervals (excluding final)?
    # ---------------------------------------------------------------------------------------------------
    # 1. Unpack:
    #    a. Initial values for optimizer:
    a_s   = argL[0]                
    a_mp  = argL[1]
    a_mf  = argL[2]
    a_ss  = argL[3]
    a_mps = argL[4]
    a_mfs = argL[5]
    sM    = argL[6]
    #    b. Observation history values from nest data:
    #@#print(">> nest data going into likelihood function:\n", nestData, nestData.shape)
    flooded =  nestData[:,4]
    hatched = nestData[:,3]
    if perfectInfo == 0:
        numIntTotal = nestData[:,11]
        #print(">> number of observation intervals for each nest:\n", numIntTotal)
    else:
        numIntTotal = hatchTime
        #print(">> number of observation intervals for each nest, frequency = 1 day:\n", numIntTotal)
    # NOTE this SHOULD be number of obs (I think) but it's giving number of obs intervals, so I'll roll with it
    # ---------------------------------------------------------------------------------------------------
    # 2. Initialize the overall likelihood counter; Decimal gives more precision
    logLike = Decimal(0.0)         
    # ---------------------------------------------------------------------------------------------------
    # 3. Define state vectors (1x3 matrices) - these are all the possible nest states 
    #     [ 1 0 0 ] (alive)      [ 0 1 0 ] (failed - predation)     [ 0 0 1 ] (failed - flooding) 
    stillAlive = np.array([1,0,0]) 
    mortFlood  = np.array([0,1,0])
    mortPred   = np.array([0,0,1])
    # ---------------------------------------------------------------------------------------------------
    # 4. Create arrays to hold state vectors for all nests:
    #    a. state of nest on date nest was first found (stateFF)
    #    b. state of nest on date nest was last checked (stateLC) - this is the fate as observed
    # Could also calculate bassed on nest fate value
    #stateFF = np.empty((numNests, 3))     # make arrays with space for the state vectors for each nest
    stateEnd = np.empty((numNests, 3))     # make arrays with space for the state vectors for each nest
    # state at end of normal interval
    # FOR THE INITIAL STATE, just one vector (see notebook)
    stateLC = np.empty((numNests, 3)) 
    # instead of fill, use broadcasting
    # fill doesn't work with arrays as the fill value
    #stateFF.fill(stillAlive)            # fill arrays
    #stateLC.fill(mortPred)              # if nest was not flooded or hatched, then it was predated
    #stateFF[:] = stillAlive
    stateEnd[:] = stillAlive
    stateLC[:] = mortPred
    #@#print(">> state at the end of a normal interval:\n",stateEnd) 

    #print(stateFF, "\n", stateLC)
    stateLC[flooded==True] = mortFlood  # flooded status gets flooded state vector
    stateLC[hatched==True] = stillAlive # hatched nests stay alive the entire time
    #@#print(">> state on final nest observation:\n", stateLC)
    # ---------------------------------------------------------------------------------------------------
    # 5. Compose the matrix equation for one observation interval.
    #        The formula used is from Etterson et al. (2007) 
    #    For this, you need the nest state at the beginning and end of the interval, plus interval length
    #    > intElt - length in days of the observation interval being assessed 
    #    > initial state (stateI) - state of the nest at the beginning of this interval 
    #    > stateF - state of the nes at the end of this interval
    #    There is a transition matrix that is multiplied for each day in the interval 
    #    > in this case, the nest started the interval alive and ended it alive as well 
    #    > daily nest probabilities: s - survival; mp - mortality from predation; mf - mortality from flooding 
    #    > these are daily probabilities, so raise transition matrix to the power of number of days in interval  
    #
    #                                          _         _  intElt           _   _ 
    #                 [ 1 0 0 ]               |  s  0  0  |                 |  1  | 
    #                                 *       |  mp 1  0  |            *    |  0  | 
    #                                         |_ mf 0  1 _|                 |_ 0 _|  
    #                              
    #           {  transpose(stateI)  *  trMatrix, raised to intElt power  *    stateF  } 
    #
    #    Then, you can multiply this equation for the number of intervals (numIntTotal)
    #    Single in  wterval --> all intervals --> likelihood

    # in the following code, we calculate all matrix multiplications for all nests, and then turn them on/off
    # based on whether we need them. Would it be faster to only do the multiplications we need? That would require indexing
    # maybe calculate the one for normal and storm intervals (not final interval) once, and the final intervals for each nest
    # (based on what the actual fate is)

    trMatrix = np.array([[a_s,0,0], [a_mf,1,0], [a_mp,0,1]]) # transition matrix, from etterson 2007
    #TstateI = np.transpose(stateFF)             # by default, vector is in column form; convert to row (horizontal)
    TstateI = np.transpose(stillAlive)
    ##print(">> transpose of initial state vector:", TstateI, TstateI.shape)
    #TstateI = stateFF
    # numpy.transpose is transposing the entire matrix, not just the row
    #numIntTotal = numObs - 2                   # # number of intervals NOT including final interval
    #numIntTotal = numObs - 1      # total number of intervals 
    #numIntTotal = numObs      # total number of intervals 
    numIntStm   = nestData[:,13]
    numIntNorm  = numIntTotal - numIntStm
    ##print(">> total intervals:", numIntTotal, "\nstorm intervals:", numIntStm, "\nnormal (non-storm) intervals:", numIntNorm)
    ##print(">> transition matrix\n:", trMatrix, trMatrix.shape)
    # NOTE need the transpose of each row, not entire matrix. 
    #      > this is making a lot of the code wonky
    #print("transpose of initial state vectors:", TstateI, TstateI.shape)
    #                                             this will depend on how many storms/when they are
    # so the tranpose we want is now the columns?
    # maybe the matrix functions in numpy are smart enough to convert
    # nope. dimensions need to match
    pwr = np.linalg.matrix_power(trMatrix, obsFreq) # raise the matrix to the power of the number of days in obs int
    ##print(">> transition matrix raised to the obs int power:\n", pwr, pwr.shape)
    #    can also create a separate storm matrix with different values
    #    or just create a larger intElt to use for the longer observation intervals
    pwrStm = np.linalg.matrix_power(trMatrix, obsFreq*2) # power equation for storm intervals (longer obs int)
    # matrix equations for one single interval, based on start/end state and whether there was a storm---
    # Each interval (except final) is represented by one of the following 2 equations (ends in same state as it began):
    #normalInt   = np.linalg.multi_dot([stateFF, pwr, TstateI])    # take the dot product of the 3 matrices
    # line above is wrong equation - has initial state twice NEVER MIDN
    #normalInt   = np.linalg.multi_dot([stateFF, pwr, TstateI])    # take the dot product of the 3 matrices
    normalInt   = np.linalg.multi_dot([stateEnd, pwr, TstateI])    # take the dot product of the 3 matrices
    #@#print(">> likelihood of one interval:\n", normalInt, normalInt.shape)
    stormInt    = np.linalg.multi_dot([stateLC, pwrStm, TstateI]) # dot product for storm intervals
    # The final interval is one of these two (ends in final state):
    #normalFinal[stFin == False] = multi_dot([stateLC[stFin == False], pwr, TstateI])    # regular final interval
    #    not sure if necessary
    normalFinal = np.linalg.multi_dot([stateLC, pwr, TstateI])    # regular final interval
    #floodFinal = np.linalg.multi_dot([stateLC, pwr, TstateI])    # regular final interval
    #    why would a nest flood unless there was a storm?
    #predFinal = np.linalg.multi_dot([stateLC, pwr, TstateI])    # regular final interval
    #hatchFinal = normalInt
    stormFinal  = np.linalg.multi_dot([stateLC, pwrStm, TstateI]) # storm final interval
    #    are these two already too small bc we don't take the log until next step?
    #    then we will use a combination of these for each nest
    #    determined by how long the nest was represented by each of these 4 alternatives
    #    then you can raise this individual interval to the number of (normal) intervals
    #    and then create a new multiplication for storm intervals, and raise it to the number of storm intervals...
    # take the negative log of the likelihoods (matrix multiplications):
    #logLik       = Decimal(np.log(normalInt)) * -1
    # "conversion of numpy.ndarray to Decimal is not supported"
    # NOTE or use numpy.float128 instead of Decimal? That way all the code runs in numpy w/ no conversions
    #logLik      = np.zeros(numNests, dtype=np.float128) # this should give it enough precision & avoid errors
    #logLik      = np.ones(numNests, dtype=np.float128) # this should give it enough precision & avoid errors
    logLik      = np.ones(numNests, dtype=np.longdouble) # this should give it enough precision & avoid errors
    #logLik      = logLik * np.log(normalInt) * -1 # dtype changes to float64 unless you multiply it by itself
    #@#print("log likelihood:", np.log(normalInt))
    # NOTE why is the log likelihood already negative?
    logLik      = logLik * -1 * (np.log(normalInt))  # dtype changes to float64 unless you multiply it by itself
    #logLik      = logLik * (np.log(normalInt))  # dtype changes to float64 unless you multiply it by itself
    #@#print(">> check negative log likelihood:\n", np.log(normalInt), logLik)
    #@#print(">> negative log likelihood for one normal interval:\n", logLik, logLik.dtype, logLik.shape)
    #logLikStm    = Decimal(np.log(stormInt)) * -1
    logLikStm    = np.ones(numNests, dtype=np.longdouble)
    #logLikStm    = logLikStm * np.log(stormInt) * -1
    logLikStm    = logLikStm * (-np.log(stormInt))
    #logLikStm    = logLikStm * (np.log(stormInt))
    #@#print(">> negative log likelihood for one storm interval:\n", logLik, logLik.dtype, logLik.shape)
    #logLikFin    = np.log(normalFinal) * -1
    
    # NON-STORM FINAL INTERVALS:
   # if nestData[3] == 1:
   #     logLikFin = np.log(hatchFinal) * -1
   # elif nestData[4] == 0 & nestData[3] == 0:
   #     logLikFin = np.log(predFinal) * -1
    #elif nestData[4] == 1:
        #logLikFin = np.log(floodFinal) * -1
    #    logLikFin = np.log(stormFinal) * -1
    # calculate the log likelihood of the final interval (which can be different types):
    #logLikFin = np.zeros(numNests, dtype=np.float128)
    #print(logLikFin.shape)
    #logLikFin[np.where(nestData[3] == 1),] = np.log(hatchFinal[np.where(nestData[3] == 1),]) * -1
    #logLikFin[np.where(nestData[3] == 1),] = np.log(hatchFinal[np.where(nestData[3] == 1),]) * -1
    #print("final interval likelihood, hatched nests only:\n", logLikFin, logLikFin.dtype)
    # I'm not totally sure what I was doing with the above - trying to fill in likelihoods for each nest? 
    # does that even apply to the equation below?
    logLikFin    = np.ones(numNests, dtype=np.longdouble)
    logLikFin    = logLikFin * (-np.log(normalFinal))
    #logLikFin    = logLikFin * (np.log(normalFinal))
    #@#print(">> log likelihood final:\n", logLikFin)
    logLikFinStm = np.ones(numNests, dtype=np.longdouble)
    logLikFinStm = logLikFinStm * (-np.log(stormFinal))
    #logLikFinStm = logLikFinStm * (np.log(stormFinal))
    #@#print(">> log likelihood final, with storm:\n", logLikFinStm)
    # now, the joint negative log likelihood is:
    # (number of nonstorm intervals)*normalInt * (num storm intervals)*stormInt * finalInt
    # and final interval can be one of the two, so raise to power of zero or one, like turning it on/off
    # and obviously if there are no storm intervals before the final interval, also multiply by zero
    #logLikelihood = (normalInt ^ power) * (stormInt ) * normalFinal * stormFinal
    #logLikelihood = (normalInt ^ power) * (stormInt ) * normalFinal * stormFinal
    #logLikelihood = (logLik*numIntNorm) * (logLikStm^moreStorm*numStorms) * (logLikFin^(1-stormDuring)) * (logLikFinStm^stormDuring)
    # NOTE what IS stormDuring? is it a count (as defined below) or a T/F index?
    # looks like a probability? I think it's supposed to be T/F to turn the equation on/off
    # single interval to the power of how many of that interval there are
    stormDuringFin = nestData[:,12] # was there a storm during the final interval?
    #logLikelihood = (logLik*numIntNorm) * (logLikStm*numIntStm) * (logLikFin**(1-stormDuringFin)) * (logLikFinStm**stormDuringFin)
    #logLikelihood = (logLik*numIntNorm) + (logLikStm*numIntStm) + (logLikFin**(1-stormDuringFin)) + (logLikFinStm**stormDuringFin)
    #logLikelihood = (logLik*numIntNorm) + (logLikStm*numIntStm) + (logLikFin*(1-stormDuringFin)) + (logLikFinStm*stormDuringFin)
    logLikelihood = (logLik*numIntNorm) + (logLikStm*numIntStm) + (logLikFin*(1-stormDuringFin)) + (logLikFinStm*stormDuringFin)
    # if stormDuringFin == 1, then logLikFinStm will be used and the logLikFin expression will be zero (and vice versa)
    #print("likelihood equation: (",logLik,"*",numIntNorm,")(",logLikStm,"*",numIntStm,")(",logLikFin,"**(1 -",stormDuringFin,")(", logLikFinStm,"**",stormDuringFin)
    #print("likelihood equation: (",logLik,"*",numIntNorm,")+(",logLikStm,"*",numIntStm,")+(",logLikFin,"**(1 -",stormDuringFin,")+(", logLikFinStm,"**",stormDuringFin)
    #@#print(">> likelihood equation setup: \n (logLik*numIntNorm) + (logLikStm*numIntStm) +\
    #@#        (logLikFin*(1-stormDuringFin)) + (logLikFinStm*stormDuringFin)") 
    #@#for x in range(numNests):
        #print(">> likelihood equation: (",logLik[x],"*",numIntNorm[x],")+(",logLikStm[x],"*",numIntStm[x],")+(",logLikFin[x],"**(1 -",stormDuringFin[x],")+(", logLikFinStm[x],"**",stormDuringFin[x])
       #@#print(f">> likelihood equation for nest {x}: {logLik[x]:.2f} * {numIntNorm[x]:.2f}\
       #@#         + {logLikStm[x]:.2f} * {numIntStm[x]:.2f} + {logLikFin[x]:.2f}*(1-{stormDuringFin[x]:.2f})\
       #@#         + {logLikFinStm[x]:.2f}*{stormDuringFin[x]:.2f}")
# NOTE print only once (so do it outside of the function call, or it will be called repeatedly by optimizer)

    #@#print(">> negative log likelihood of each nest:", logLikelihood)
    #logLikelihood[moreStorm==True] = (logLik*numIntNorm) * (logLikStm*numStorms) * (stormFinal)
        # not quite right - still need to know if the storm was in the final interval or not
    # how can I calculate the joint likelihood of all nests without using a loop?
    #logLike       = np.prod(logLikelihood) 
    # since it's negative LOG likelihood, take the sum, not the product:
    # logLikelihood is an array of likelihoods for each nest - sum gives us joint likelihood
    logLike       = np.sum(logLikelihood)
    #@#print(">> joint negative log likelihood of all nests in replicate:",logLike)
    return(logLike)
# also need to deal with undiscovered nests, as neither MARK nor the MCMC model can handle them
    # need to know how many observation intervals, but also the length
    # i.e. was there a storm preventing survey on any day (would make that interval longer)
    # so storm activity has two effects: makes it harder to tell fate and (potentially) increases obs int
    # as in the original simdata.pyx script, probably need to reconstruct the observation history 
    # instead of storing it as output from the mk_nests function
    # is there a vectorized way to detect where obs int is extra long, and then record the length?
    # can maybe make a detector for when observation period as a whole is longer than expected
    # then for those nests, do something???
    # but do this inside the mk_nests function?
    # if you know a storm occurred on a survey day, does it matter which interval you extend?
    # could you just add it in arbitrarily, without knowing when storm was?
    # try it out, see if results are equal...
    # then could calculate how many storms and just add them in wherever
# can compare calculating the log likelihood using separate storm survival/failure probabilities vs not
# the other function takes the final state (after the entire observation period, not one interval)
# besides the 3 matrices, the other arguments it takes are: 
#   - stormDuring (how many storms replaced surveys during observation period)
#   - numObs (total number of surveys where nest was observed)

#def like_smd(x, ndata, obs, storm, survey):
# --------------------------------------------------------------------------------------------------------------------------
# The values are log-transformed before running them thru the likelihood function 
# So the values given to the optimizer are the untransformed values, meaning the 
# optimizer output will also be untransformed.
# therefore, need to transform the output as well.

def like_smd( x, perfectInfo, hatchTime, nestData, obsFreq, stormDays, surveyDays):

    # unpack the initial values:
    s0   = x[0]
    mp0  = x[1]
    ss0  = x[2]
    mps0 = x[3]
    sM   = x[4]
    #@#print("initial values:", s0, mp0, ss0, mps0, sM)

    # transform the initial values so all are between 0 and 1:
    s1   = logistic(s0)
    mp1  = logistic(mp0)
    ss1  = logistic(ss0)
    mps1 = logistic(mps0)
    #@#print("logistic-transformed initial values:", s1, mp1, ss1, mps1)

    # further transform so they remain in lower left triangle:
    tri1 = triangle(s1, mp1)
    tri2 = triangle(ss1, mps1)
    s2   = tri1[0]
    mp2  = tri1[1]
    ss2  = tri2[0]
    mps2 = tri2[1]
    #@#print("triangle-transformed initial values:", s2, mp2, ss2, mps2)

    # compute the conditional probability of mortality due to flooding:
    mf2  = 1.0 - s2 - mp2
    mfs2 = 1.0 - ss2 - mps2

    numNests = nestData.shape[0]
    #@#print(">> number of nests:", numNests)

    # call the likelihood function:
    argL = np.array([s2,mp2,mf2,ss2,mps2,mfs2, sM])
    #ret = like(argL, ndata, obs, storm, survey)
    #ret = like(argL, nestData, obsFreq, stormDays, surveyDays)
    ret = like(perfectInfo, hatchTime, argL, numNests, obsFreq,  nestData, surveyDays)
##def like(argL, numNests, obsFreq, nestData, surveyDays):

    return(ret)

# need to loop through the param combinations
# within the loop, need to unpack the params and run the optimizer
# where are storms and survey days created? during each loop or before all loops?

#def run_params(paramsList, dirName):
#
# NEST DATA COLUMNS: 
# 0) ID number        | 4) flooded? (T/F)          | 8) i (first found)      | 12) final interval storm (T/F) 
# 1) initiation date  | 5) surveys til discovery   | 9) j (last active)      | 13) num other storms 
# 2) end date         | 6) discovered? (T/F)       | 10) k (last checked) 
# 3) hatched? (T/F)   | 7) assigned fate           | 11) number of observation intervals 

    # NOTE need to create storms after params are chosen

    #paramsList      = list(product(
    #        numNests, probSurv, SprobSurv, stormDur, stormFreq, hatchTime, 
    #        obsFreq, probMortFlood, breedingDays, discProb
    #        ))
                    #repID,mark_s,s2,mp2,mf2,ss2,mps2,mfs2,dur,freq,pSurv,pSurvStorm, 
                    #pMFlood,hatchTime,numNests,obsFreq,discovered,excluded,
                    #ex
# NOTE is there even a reason to have this in a function?

with open(likeFile, "wb") as f:

    parID     = 0

    for i in range(0, len(paramsList)): # for each set of params
        #par = paramsList[i]
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print(">>>>>>>>> param set number:", parID)
        # paramsArray is a 2d array; loop through the rows 
        # each row is a set of parameters we are trying
        par        = paramsArray[i] 
        # in an array, all values have the same numpy dtype (float in this case) 
        # after selecting the row, unpack the params & change dtype as needed:
        numNests   = par[0].astype(int)
        pSurv      = par[1]
        pSurvStorm = par[2]
        freq       = par[4].astype(int)
        dur        = par[3].astype(int)
        hatchTime  = par[5].astype(int)
        obsFreq    = par[6].astype(int)
        pMFlood    = par[7]
        brDays     = par[8]
        pDisc      = par[9]
        # Generate a random list of storm days based on the real weekly probabilities
        # Then create a list of survey days 
        # NOTE need to choose if you want new storms/survey days for each replicate or each parameter set
        stormDays  = stormGen(freq, dur)
        #surveyDays = mk_surveys(stormDays, obsFreq, breedingDays)
        surveyDays = mk_surveys(stormDays, obsFreq, brDays)
        repID      = 0  # keep trackof replicates
        numOut     = 21 # number of output params
        print(">> nest params in this set:", pSurv, pSurvStorm, dur, freq, hatchTime, obsFreq, pMFlood)

        # Create a file to store the nest data as you go: nestfile
        now       = datetime.now().strftime("%H%M%S")
        nestfile  = Path.home()/'/mnt/c/Users/sarah/Dropbox/nest_models/py_output'/dirName/('nests'+now+'.npy')
        nestfile.parent.mkdir(parents=True, exist_ok=True)
        columns = [
                'rep_ID','ID','initiation','end_date','true_fate','first_found',
                'last_active','last_observed','ass_fate','n_obs','storm_true'
                ]
        colnames = ', '.join([str(x) for x in columns])

        with open(nestfile, "wb") as n:
            # append nest data from each replicate to nestfile as you loop through them
            for r in range(nreps):
                # if you write likelihoods to a file as you go, shouldn't need an array to store them
                # this should save memory, at least in this case?
                print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
                print(">>>>>>>>>>>> replicate ID:", repID)
                likeVal =  np.zeros(shape=(numOut), dtype=np.longdouble)
                try:
                    nestData = mk_nests(par, initProb, stormDays, surveyDays)
                    np.save(n, nestData) # make sure this is the correct kind of save
                except IndexError as error:
                    print(">> IndexError in nest data:", error,". Go to next replicate")
                    continue
                #print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
                #print(">> optimize over all nests with perfect information")
                #z = randArgs()
                #obsFreq2 = 1
                #perfectInfo2 = 1
                #print(">> z=", z)
                ## Run the optimizer with messages for exceptions
                #print(">> main.py: Msg: Running optimizer")
                #try:
                #    ans = optimize.minimize(
                #            like_smd, z, 
                #            #args=(nestData, obsInt, stormDays, surveyDays),
                #            args=(perfectInfo2, hatchTime, nestData, obsFreq2, stormDays, surveyDays),
                #            method='Nelder-Mead'
                #            )
                #    ex = 0.0
                #except decimal.InvalidOperation as error2:
                #    ex=100.0
                #    print(">> Error: invalid operation in decimal:", error2, "Go to next replicate.")
                #    continue
                #except OverflowError as error3:
                #    ex=200.0
                #    print(">> Error: overflow error:", error3, "Go to next replicate.")
                #    continue
#
                #print("Success?", ans.success, ans.message)
                ##print("answer:", ans.x)
                #ans1 = ansTransform(ans)
                ##ansTransformed = np.array([s2, mp2, mf2, ss2, mps2, mfs2])
                #
                #srand = z[4]
#
                #markAns1  = optimize.minimize(
                #        mark_wrapper, srand,
                #        args=(nestData), 
                #        method='Nelder-Mead'
                #        )
                ##print("MARK answer:", logistic(markAns.x))
                #print(">> success?", markAns1.success, markAns1.message) # did the optimizer converge?
                ## this MLE seems unlikely to need exceptions, but who knows...
                ##NOTE markAns is an "OptimizeResult" object - does not match the other objects in like_val
                #if markAns1.success == True:
                #    # Transform the MARK optimizer output so that it is between 0 and 1
                #    mark_s = logistic(markAns1.x[0])
                #    # even when accessing "x" part of markAns, still formatted like a list
                #    # according to scipy manual, x is the "solution array" so probably need to index it
                #    print(">> logistic of MARK answer:", mark_s)
                #else:
                #    mark_s = 10001
#
                # ------------------------------------------------------------------------------------------
                # ------------------------------------------------------------------------------------------
                print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
                #print("Calculate number of hatched nests and true DSR")
                hatched   = nestData[:,3]
                #hatchProp = sum(hatched)/numNests
                #trueDSR =  np.exp(np.log(hatchProp)/hatchTime)
                #trueDSR  = 1 - ((numNests - hatched.sum()) / survival.sum())
                trueDSR   = nestData[0,14]
                print(">> number of hatched nests:", sum(hatched), "and true DSR value:", trueDSR)
                print(">> proportion of hatched nests overall:", hatched.sum()/numNests)
    
                #print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
                #print(">> now run optimizer for discovered nests only")
                #print(">> nest data:\n----id--ini-end-hch-fld-std-dsc-stm--i--j--k-nobs\n", nestData)
                # NOTE rearrange columns to make more sense? 
                nestData    = nestData[np.where(nestData[:,6]==1)] # keep only discovered nests
                #print("shape of nest data:", nestData.shape)
                #print(">> nest data\n--id--ini-", nestData)
                print(
                        ">> nest data, discovered nests:\n----id--ini-end-hch-fld-std-dsc-i--j--k-fate-nobs-sfin-nstm\n",
                        nestData
                        )

                discovered  =  nestData.shape[0]                               # then count discovered nests
                hatched2    = nestData[:,3]

                # select nests to be excluded: unknown fate or only one observation (one or the other must be true)
                #print(
                #        ">> nests that were assigned unknown fate:",
                #        np.where(nestData[:,10] == 7),
                #        len(np.where(nestData[:,10]==7))
                #        ) 
                print(">> nests with only one observation while active:",np.where(nestData[:,7] == nestData[:,8])) 
                #exclude     = ((nestData[:,7] == 9) | (nestData[:,8]==nestData[:,10]))                         
                exclude     = ((nestData[:,10] == 7) | (nestData[:,7]==nestData[:,8]))                         
                # if there's only one observation, firstFound will == lastActive
                excluded    = sum(exclude)      # exclude is a boolean array, sum gives you num True
                print(">> exclude nest (unknown fate or only 1 observation)?", exclude)
                print(">> number nests discovered:", discovered, "and excluded:", excluded)
                #print(">> proportion of hatched nests in excluded:", exclude[hatched==True].sum()/discovered.sum())
                #print(">> out of excluded nests, which hatched?", exclude[hatched==True])
                print(">> proportion of hatched nests in excluded:", exclude[hatched2==True].sum()/excluded)
                #nestDataAll = nestData
                # the above doesn't work bc it's making a pointer to nestData, not saving a separate copy
                nestData    = nestData[~(exclude),:]    # remove excluded nests from data
                #hatched     = nestData[:,3]
                #nestDataExc    = nestData[~(exclude),:]    # remove excluded nests from data
                #print(">> exclude unknown-fate or 1-observation nests: \n", exclude)
                print(">> shape of nest data without unknown/1-observation nests:", nestData.shape)
                print(">> hatched in remaining nests:\n", nestData[:,3])
                print(">> proportion of hatched nests in remaining:", sum(nestData[:,3])/nestData.shape[0])

                z=randArgs()
                # optimizer for matrix function takes an array (z) but optimizer for MARK takes one 
                # param (srand) which just happens to be in z
                srand = z[4]
                perfectInfo = 0

                # Choose random initial values for the optimizer
                # >> moved this to a separate function
                # These will be log-transformed before going through the likelihood function
                #s     = rng.uniform(-10.0, 10.0)       
                # mp    = rng.uniform(-10.0, 10.0)
                #ss    = rng.uniform(-10.0, 10.0)
                #mps   = rng.uniform(-10.0, 10.0)
                #srand = rng.uniform(-10.0, 10.0) # should the MARK and matrix MLE start @ same value?
                #z = np.array([s, mp, ss, mps, srand])
                ex = np.float128("0.0")
                #ex = np.longdouble("0.0")

                #print("optimizing over all nests created:")
                #ansAll = minimizer(nestData, obsFreq, stormDays, surveyDays)
                #ansAll2 = ansTransform(ansAll)
                #print("optimizing over discovered nests only:")
                #ans = minimizer(nestDataExc, obsFreq, stormDays, surveyDays)
                #ans2 = ansTransform(ans)

                # Run the optimizer with messages for exceptions
                print(">> main.py: Msg: Running optimizer")
                try:
                    ans = optimize.minimize(
                            like_smd, z, 
                            #args=(nestData, obsInt, stormDays, surveyDays),
                            args=(perfectInfo, hatchTime, nestData, obsFreq, stormDays, surveyDays),
                            #method='Nelder-Mead'
                            method='SLSQP'
                            )
                    ex = 0.0
                except decimal.InvalidOperation as error2:
                    ex=100.0
                    print(">> Error: invalid operation in decimal:", error2, "Go to next replicate.")
                    continue
                except OverflowError as error3:
                    ex=200.0
                    print(">> Error: overflow error:", error3, "Go to next replicate.")
                    continue

                print("Success?", ans.success, ans.message)
                print("answer:", ans.x)
                res = ansTransform(ans)



                #s0   = ans.x[0]         # Series of transformations of optimizer output.
                #mp0  = ans.x[1]         # These make sure the output is between 0 and 1, 
                #ss0  = ans.x[2]         # and that the three fate probabilities sum to 1.
                #mps0 = ans.x[3]
               # 
                #s1   = logistic(s0)
                #mp1  = logistic(mp0)
                #ss1  = logistic(ss0)
                #mps1 = logistic(mps0)

                #ret2 = triangle(s1, mp1)
                #s2   = ret2[0]
                #mp2  = ret2[1]
                #mf2  = 1.0 - s2 - mp2

                #ret3 = triangle(ss1, mps1)
                #ss2  = ret3[0]
                #mps2 = ret3[1]
                #mfs2 = 1.0 - ss2 - mps2
                #print(">> results (s, mp, mf, ss, mps, mfs, ex):", s2, mp2, mf2, ss2, mps2, mfs2, ex)
                #s2, mp2, mf2, ss2, mps2, mfs2 = ans2[0], ans2[1], ans2[2], ans2[3], ans2[4], ans2[5]
                #print(">> results again (s, mp, mf, ss, mps, mfs):", s2, mp2, mf2, ss2, mps2, mfs2)

                #OPTIMIZER: MARK function
                markAns  = optimize.minimize(
                        mark_wrapper, srand,
                        args=(nestData), 
                        method='Nelder-Mead'
                        )
                #print("MARK answer:", logistic(markAns.x))
                print(">> success?", markAns.success, markAns.message) # did the optimizer converge?
                # this MLE seems unlikely to need exceptions, but who knows...
                #NOTE markAns is an "OptimizeResult" object - does not match the other objects in like_val
                if markAns.success == True:
                    # Transform the MARK optimizer output so that it is between 0 and 1
                    mark_s = logistic(markAns.x[0])
                    # even when accessing "x" part of markAns, still formatted like a list
                    # according to scipy manual, x is the "solution array" so probably need to index it
                    print(">> logistic of MARK answer:", mark_s)
                else:
                    mark_s = 10001

                # Compile the optimizer output for each replicate with other important info 
                # including params and the number of nests actually used in the analysis
                # can get a proportion because we know how many nests were generated initially
                #firstPart = np.array([repID, mark_s])
                #secondPart = np.array([dur, freq, pSurv, pSurvStorm, pMFlood, hatchTime, numNests, obsFreq, discovered, excluded, ex])
                #like_val   = np.concatenate((firstPart, res, secondPart))
                s2,mp2,mf2,ss2,mps2,mfs2 = res # unpack function output
                like_val = [
                        #repID,mark_s,s2,mp2,mf2,ss2,mps2,mfs2,dur,freq,pSurv,pSurvStorm,pMFlood,
                        repID,mark_s,s2,mp2,mf2,ss2,mps2,mfs2,dur,freq,trueDSR,pSurvStorm,pMFlood,
                        hatchTime,numNests,obsFreq,discovered,excluded,ex
                        ]
                # is there a reason this wasn't a list?
                print(">> like_val:\n", like_val)
                #, "lengths:", [len(x) for x in like_val])
                like_val = np.array(like_val, dtype=np.float128)
                #like_val   = np.array([
                #    repID,mark_s,s2,mp2,mf2,ss2,mps2,mfs2,dur,freq,pSurv,pSurvStorm, 
                #    pMFlood,hatchTime,numNests,obsFreq,discovered,excluded,ex
                #    ], dtype=np.float128) # NOTE problem could be multiple types in same array?

                    #repID,mark_s,s2,mp2,mf2,ss2,mps2,mfs2,stormDur,stormFreq,probSurv,SprobSurv, 
                    #probMortFlood,SprobMortFlood,hatchTime,numNests,obs_int,discovered,excluded,
                    #ex,stormMat
                #####################
                # this is being saved w/o the delimiter
                # stack exchange says to make it a list of only one item
                #np.savetxt(f, like_val, delimiter=",")
                column_names     = np.array(['rep_ID', 'mark_s', 'psurv_est', 'ppred_est', 'pflood_est', 'stormsurv_est', 'stormpred_est', 'stormflood_est', 'storm_dur', 'storm_freq', 'psurv_real', 'stormsurv_real','pflood_real', 'hatch_time','num_nests', 'obs_int', 'num_discovered','num_excluded', 'exception'])
                # header values need to be stored to np.ndarray for np.savetxt; actually, needs to be a string
                colnames = ', '.join([str(x) for x in column_names])
                #if repID == 0:
                #print(">> saving likelihood values")
                if parID == 0 | repID == 0: # only the first line gets the header
                    np.savetxt(f, [like_val], delimiter=",", header=colnames)
                    #np.savetxt(f, like_val, delimiter=",", header=colnames)
                    print(">> saving likelihood values")
                else:
                    np.savetxt(f, [like_val], delimiter=",")
                    #np.savetxt(f, like_val, delimiter=",")
                    print(">> saving likelihood values")
                # except this adds the header before every line
                # could just remove them later in R

                repID = repID + 1
                print(">> repID increased:", repID)

        parID = parID +1
        print(">> param set ID increased:", parID)
        


#with open(likeFile, "a") as f:
#with open(likeFile, "wb") as f:
#
#    likeVal = run_params(paramsArray, dir_name)
    # want to loop through param sets under "with" so you can save after each set?
    # still need an intermediate array to store values from each replicate w/in the param set
    # so open document within the run_params function?
#likeVal = run_params(paramsArray, dir_name)
#nestData   = mk_nests(par, inits)
#par = np.ndarray(20, 3, 3, 3, 25, 0.95, 0.5) # starting parameters
#par = np.array([20, 3, 3, 3, 25, 0.95, 0.05, 1]) # starting parameters
#sDays = np.array([20, 21, 22, 30, 31, 32, 45, 46, 47]) # storm days
#inits = np.array([0.05, 0.05, 0.1, 0.1, 0.2, 0.2, 0.1, 0.1, 0.05, 0.05]) # weekly prob of nest initiation


#stormDays  = stormGen(stormDur, stormFreq)
#stormDays  = stormGen(par[1], par[2])
#f = par[1]
#d = par[2]
#stormDays = stormGen(f, d)
#surveyDays = mk_surveys(stormDays)
# has to be created for each parameter set because observation interval/frequency changes
# same with storm days and frequency/duration 
#nestData   = mk_nests(par, stormDays, inits)
#nestData   = mk_nests(par, inits)
# why aren't both storm days and survey days arguments, or neithera = # initial values for the optimizer are translated to probabilities by like_smd()

#likelihood = like(`)
#def like(argL, numNests, obsInt, nestData, surveyDays):
