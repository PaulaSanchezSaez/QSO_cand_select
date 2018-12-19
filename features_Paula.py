import os
import astropy.io.fits as pf
from SF import fitSF_mcmc, var_parameters
import FATS
import numpy as np
import pandas as pd
import glob
from pgram_func2 import get_period_sigf
#from multiprocessing import Pool
from pathos.multiprocessing import ProcessingPool as Pool
import carmcmc as cm
import time
import sys, getopt
import ast


###############################################################################
#modify these and only these variables:

folder = '../' #Folder where FITS files are located.

ncores=28 #number of cores used to compute the features

output = '../output.csv' #Output text file with calculated features.

patern = 'bin3*fits' # Pattern with wildcard to match desired FITS files.

use_z = False # If True compute rest frame time for light curves

train_samp = False # the lc correspond to the training sample?
###############################################################################
# To modify the parameters from the terminal
myopts, args = getopt.getopt(sys.argv[1:],"f:n:p:o:z:t:")

###############################
# o == option
# a == argument passed to the o
###############################
for o, a in myopts:
    if o == '-f':
        folder = a
    elif o == '-n':
        ncores = a
    elif o == '-p':
        patern = a
    elif o == '-o':
        output = a
    elif o == '-z':
        use_z = a
    elif o == '-t':
        train_samp = a


folder = str(folder)

patern = str(patern)

ncores = int(ncores)
print ncores

output = str(output)

use_z = ast.literal_eval(use_z)

train_samp = ast.literal_eval(train_samp)

###############################################################################
#feature list for FATS

featurelist = [
'Mean',
'Std',
'Meanvariance',
'MedianBRP',
'Rcs',
#'PeriodLS',
#'Period_fit',
#'Color',
'Autocor_length',
#'SlottedA_length',
'StetsonK',
#'StetsonK_AC',
'Eta_e',
#'Amplitude',
'PercentAmplitude',
'Con',
'LinearTrend',
'Beyond1Std',
#'FluxPercentileRatioMid20',
#'FluxPercentileRatioMid35',
#'FluxPercentileRatioMid50',
#'FluxPercentileRatioMid65',
#'FluxPercentileRatioMid80',
#'PercentDifferenceFluxPercentile',
'Q31',
]

###############################################################################

#funcion to read every lc

def read_lc(lc_name):

    print "########## reading lc %s  ##########" % (lc_name)

    arch = pf.open(lc_name)

    if train_samp:

        head = arch[0].header
        datos = arch[1].data
        jd = datos['JD']
        flux = datos['fluxQ']
        errflux = datos['errfluxQ']
        mag = datos['Q']
        errmag = datos['errQ']
        zspec = head['ZSPEC']
        type = head['TYPE_SPEC']
        umag = head['UMAG']
        gmag= head['GMAG']
        rmag = head['RMAG']
        imag = head['IMAG']
        zmag = head['ZMAG']
        ra = head['ALPHA']
        dec = head['DELTA']


        return(ra,dec,jd,flux,errflux,mag,errmag,umag,gmag,rmag,imag,zmag,zspec,type)

    else:

        head = arch[0].header
        datos = arch[1].data
        jd = datos['JD']
        flux = datos['fluxQ']
        errflux = datos['errfluxQ']
        mag = datos['Q']
        errmag = datos['errQ']
        umag = head['UMAG']
        gmag= head['GMAG']
        rmag = head['RMAG']
        imag = head['IMAG']
        zmag = head['ZMAG']
        ra = head['ALPHA']
        dec = head['DELTA']

        return(ra,dec,jd,flux,errflux,mag,errmag,umag,gmag,rmag,imag,zmag)

################################################################################
#function to run carmcmc

def drw(jd, mag, errmag):

    model = cm.CarmaModel(jd, mag, errmag, p=1, q=0)
    sample = model.run_mcmc(10000)
    log_omega=sample.get_samples('log_omega')
    tau=np.exp(-1.0*log_omega)
    sigma=sample.get_samples('sigma')
    tau_mc=(np.percentile(tau, 50),np.percentile(tau, 50)-np.percentile(tau, 15.865),np.percentile(tau, 84.135)-np.percentile(tau, 50))
    sigma_mc=(np.percentile(sigma, 50),np.percentile(sigma, 50)-np.percentile(sigma, 15.865),np.percentile(sigma, 84.135)-np.percentile(sigma, 50))

    #tau_mc,sigma_mc=[-99,-99,-99], [-99,-99,-99]
    return(tau_mc[0],tau_mc[1],tau_mc[2],sigma_mc[0],sigma_mc[1],sigma_mc[2])

################################################################################
#function to run SF with mcmc

def var_amp(jd, mag, errmag):
    nwalkers,nit = 50, 100
    nthr=1

    P,exvar , exvar_err = var_parameters(jd,mag,errmag)
    A_mcmc , gamma_mcmc = fitSF_mcmc(jd,mag,errmag,2,nwalkers,nit,nthr)#[-99,-99,-99], [-99,-99,-99]

    return(P,exvar , exvar_err, A_mcmc[0],A_mcmc[1],A_mcmc[2],gamma_mcmc[0],gamma_mcmc[1],gamma_mcmc[2])

################################################################################
#function to run FATS

def fats_feats(jd, mag, errmag):

    lc = np.array([mag, jd, errmag])

    a = FATS.FeatureSpace(featureList=featurelist)

    a=a.calculateFeature(lc).result(method='dict')

    return(a)

################################################################################

def comp_feat(lc_name):

    if train_samp:


        ra,dec,jd,flux,errflux,mag,errmag,umag,gmag,rmag,imag,zmag,zspec,type = read_lc(lc_name)

        num_epochs = len(jd)
        time_range = jd[-1] - jd[0]
        time_rest =  time_range/(1.0+zspec)

        u_g = umag - gmag
        g_r = gmag - rmag
        r_i = rmag - imag
        i_z = imag - zmag

        if use_z: jd = jd/(1.0+zspec)


        tau, tau_lo, tau_up, sigma, sigma_lo, sigma_up  = drw(jd, mag, errmag)

        P, exvar, exvar_err, A_mcmc,A_low,A_up,gamma_mcmc,gamma_low,gamma_up = var_amp(jd, mag, errmag)

        best_period, peak, sig5, sig1 = get_period_sigf(jd, mag, errmag)

        other_feat = [ra,dec,zspec,type,num_epochs,time_range,time_rest,umag,gmag,rmag,imag,zmag,u_g,g_r,r_i,i_z,tau, tau_lo, tau_up, sigma, sigma_lo, sigma_up,P, exvar, exvar_err, A_mcmc,A_low,A_up,gamma_mcmc,gamma_low,gamma_up,best_period, peak, sig5, sig1]

        fats_results = fats_feats(jd, mag, errmag)

        fats_list = []

        for feat in featurelist:

            fats_list.append(fats_results[feat])

        feat_list = other_feat+ fats_list

        return(feat_list)

    else:

        ra,dec,jd,flux,errflux,mag,errmag,umag,gmag,rmag,imag,zmag = read_lc(lc_name)

        u_g = umag - gmag
        g_r = gmag - rmag
        r_i = rmag - imag
        i_z = imag - zmag

        num_epochs = len(jd)
        time_range = jd[-1] - jd[0]

        tau, tau_lo, tau_up, sigma, sigma_lo, sigma_up  = drw(jd, mag, errmag)

        P, exvar, exvar_err, A_mcmc,A_low,A_up,gamma_mcmc,gamma_low,gamma_up = var_amp(jd, mag, errmag)

        best_period, peak, sig5, sig1 = get_period_sigf(jd, mag, errmag)

        other_feat = [ra,dec,num_epochs,time_range,umag,gmag,rmag,imag,zmag,u_g,g_r,r_i,i_z,tau, tau_lo, tau_up, sigma, sigma_lo, sigma_up,P, exvar, exvar_err, A_mcmc,A_low,A_up,gamma_mcmc,gamma_low,gamma_up,best_period, peak, sig5, sig1]

        fats_results = fats_feats(jd, mag, errmag)

        fats_list = []

        for feat in featurelist:

            fats_list.append(fats_results[feat])

        feat_list = other_feat+ fats_list

        return(feat_list)

################################################################################

def multi_run_wrapper(args):
    #function necessary for the use of pool.map with different arguments
    return comp_feat(*args)



def run_parallele():

    agn=sorted(glob.glob(folder+patern))

    print "number of lc to process = %d" % (len(agn))
    '''
    pool = Pool(processes=ncores)
    results = pool.map(comp_feat,list(agn))
    pool.close()
    pool.join()
    '''

    proc_pool = Pool(ncpus=int(ncores),processes=ncores)
    results = proc_pool.map(comp_feat,list(agn))
    #while not proc_light_curves.ready():
    #    print "****Chunks left: %d****" % proc_light_curves._number_left
    #time.sleep(5)
    #results = proc_light_curves.get()
    proc_pool.close()
    proc_pool.join()

    #print results
    #print results

    if train_samp:
        head=['ra','dec','zspec','TYPE','num_epochs','time_range','time_rest','umag','gmag','rmag','imag','zmag','u_g','g_r','r_i','i_z','tau','tau_lo','tau_up','sigma','sigma_lo','sigma_up','P_var','exvar','exvar_err','A_mcmc','A_low','A_up','gamma_mcmc','gamma_low','gamma_up','best_period','peak','sig5','sig1']
        for feat in featurelist:
            head.append(feat)


    else:
        head=['ra','dec','num_epochs','time_range','umag','gmag','rmag','imag','zmag','u_g','g_r','r_i','i_z','tau','tau_lo','tau_up','sigma','sigma_lo','sigma_up','P_var','exvar','exvar_err','A_mcmc','A_low','A_up','gamma_mcmc','gamma_low','gamma_up','best_period','peak','sig5','sig1']
        for feat in featurelist:
            head.append(feat)


    #np.savetxt(output,results,delimiter=',',header=head)
    df = pd.DataFrame(results)
    df.to_csv(output, header=head, index=None)
    print "File %s writen" % (output)
    return (results)


run_parallele()
