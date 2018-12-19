#this code contain all the functions to calculate the variability features for a light curve
import numpy as np
#import pyfits as pf
import astropy.io.fits as pf
import os
import matplotlib.pyplot as plt
from decimal import Decimal
from scipy.integrate import quad
import emcee
import corner
import scipy.stats as st
from scipy.stats import chi2
import FATS


##########################################
#to get P, exvar y exvar_err

def var_parameters(jd,mag,err):
#function to calculate the probability of a light curve to be variable, and the excess variance

    #nepochs, maxmag, minmag, mean, variance, skew, kurt = st.describe(mag)

    mean=np.mean(mag)
    nepochs=float(len(jd))

    chi= np.sum( (mag - mean)**2. / err**2. )
    q_chi=chi2.cdf(chi,(nepochs-1))


    a=(mag-mean)**2
    ex_var=(np.sum(a-err**2)/((nepochs*(mean**2))))
    sd=(1./(nepochs-1))*np.sum(((a-err**2)-ex_var*(mean**2))**2)
    ex_verr=sd/((mean**2)*np.sqrt(nepochs))
    #ex_var=(np.sum(a)/((nepochs-1)*(mean**2)))-(np.sum(err**2)/(nepochs*(mean**2)))

    #ex_verr=np.sqrt(((np.sqrt(2./nepochs)*np.sum(err**2)/(mean*nepochs))**2)+((np.sqrt(np.sum(err**2)/nepochs**2)*2*np.sqrt(ex_var)/mean)**2))

    #ex_verr=np.sqrt((np.sqrt(2.0/nepochs)*np.mean(err*2)/mean**2)**2+(np.sqrt(np.mean(err*2)/nepochs)*(2.0*np.sqrt(ex_var)/mean))**2)

    #print q_chi,ex_var,ex_verr

    return [q_chi,ex_var,ex_verr]



#######################################
#determine single SF using emcee

def SFarray(jd,mag,err):#calculate an array with (m(t)-m(t+tau)), whit (err(t)^2+err(t+tau)^2) and another with tau=dt
    sfarray=[]
    tauarray=[]
    errarray=[]
    for i, item in enumerate(mag):
        for j in range(i+1,len(mag)):
            dm=mag[i]-mag[j]
            sigma=err[i]**2+err[j]**2
            dt=(jd[j]-jd[i])
            sfarray.append(np.abs(dm))
            tauarray.append(dt)
            errarray.append(sigma)
    sfarray=np.array(sfarray)
    tauarray=np.array(tauarray)
    errarray=np.array(errarray)
    return (tauarray,sfarray,errarray)


def Vmod(dt,A,gamma): #model
    return ( A*((dt/365.0)**gamma) )
    #return ( A*(dt**gamma) )


def Veff2(dt,sigma,A,gamma): #model plus the error
    return ( (Vmod(dt,A,gamma))**2 + sigma )

def like_one(theta,dt,dmag,sigma): #likelihood for one value of dmag

    gamma, A = theta
    aux=(1/np.sqrt(2*np.pi*Veff2(dt,sigma,A,gamma)))*np.exp(-1.0*(dmag**2)/(2.0*Veff2(dt,sigma,A,gamma)))

    return aux

def lnlike(theta, dtarray, dmagarray, sigmaarray): # we define the likelihood following the same function used by Schmidt et al. 2010
    gamma, A = theta

    '''
    aux=0.0

    for i in xrange(len(dtarray)):
    aux+=np.log(like_one(theta,dtarray[i],dmagarray[i],sigmaarray[i]))
    '''

    aux=np.sum(np.log(like_one(theta,dtarray,dmagarray,sigmaarray)))

    return aux



def lnprior(theta): # we define the prior following the same functions implemented by Schmidt et al. 2010

    gamma, A = theta


    if 0.0 < gamma < 10 and 0.0 < A < 2.0 :
        return ( np.log(1.0/A) + np.log(1.0/(1.0+(gamma**2.0))) )

    return -np.inf
    #return -(10**32)


def lnprob(theta, dtarray, dmagarray, sigmaarray): # the product of the prior and the likelihood in a logaritmic format

    lp = lnprior(theta)

    if not np.isfinite(lp):
    #if (lp==-(10**32)):
        return -np.inf
        #return -(10**32)
    return lp +lnlike(theta, dtarray, dmagarray, sigmaarray)


def fitSF_mcmc(jd,mag,errmag,ndim,nwalkers,nit,nthr): #function that fits the values of A and gamma using mcmc with the package emcee.
#It recives the array with dt in days, dmag and the errors, besides the number of dimensions of the parameters, the number of walkers and the number of iterations

    #we calculate the arrays of dm, dt and sigma

    dtarray, dmagarray, sigmaarray = SFarray(jd,mag,errmag)

    ndt=np.where((dtarray<=365) & (dtarray>=5))
    dtarray=dtarray[ndt]
    dmagarray=dmagarray[ndt]
    sigmaarray=sigmaarray[ndt]


    #definition of the optimal initial position of the walkers

    #p0 = np.random.rand(ndim*nwalkers).reshape((nwalkers,ndim)) #gess to start the burn in fase
    p0 = np.random.rand(ndim*nwalkers).reshape((nwalkers,ndim))*0.1+0.5

    #chi2 = lambda *args: -2 * lnlike(*args)
    #result = op.minimize(chi2, [0.1,0.1], args=(x, y, yerr))

    #run mcmc
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, threads=nthr, args=(dtarray, dmagarray, sigmaarray))
    #sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(dtarray, dmagarray, sigmaarray))

    pos, prob, state = sampler.run_mcmc(p0,50) #from pos we have a best gess of the initial walkers
    sampler.reset()
    print("Running MCMC...")
    sampler.run_mcmc(pos, nit,rstate0=state)
    print("Done.")

    # Compute the quantiles.
    samples=sampler.chain[:,50:,:].reshape((-1,ndim))

    #ac=sampler.acceptance_fraction

    #plt.hist(ac)
    #plt.show()

    #print samples

    A_fin=samples[:,1]
    gamma_fin=samples[:,0]

    #print gamma_fin
    #print A_fin

    #A_hist,A_bins=np.histogram(A_fin,200)
    #a_max=np.amax(A_hist)
    #na=np.where(A_hist==a_max)
    #a_mode=A_bins[na]

    #print "a_mode", a_mode

    #g_hist,g_bins=np.histogram(gamma_fin,200)
    #g_max=np.amax(g_hist)
    #ng=np.where(g_hist==g_max)
    #g_mode=g_bins[ng]

    #print "g_mode", g_mode

    #plt.plot(A_fin,gamma_fin,'.')
    #plt.show()

    #fig = corner.corner(samples, labels=["$m$", "$b$", "$\ln\,f$"])
    #fig.savefig("line-triangle.png")
    #plt.show()

    #plt.hist(A_fin,100)
    #plt.xlabel('A')
    #plt.show()


    #plt.hist(gamma_fin,100)
    #plt.xlabel('gamma')
    #plt.show()

    A_mcmc=(np.percentile(A_fin, 50),np.percentile(A_fin, 50)-np.percentile(A_fin, 15.865),np.percentile(A_fin, 84.135)-np.percentile(A_fin, 50))
    g_mcmc=(np.percentile(gamma_fin, 50),np.percentile(gamma_fin, 50)-np.percentile(gamma_fin, 15.865),np.percentile(gamma_fin, 84.135)-np.percentile(gamma_fin, 50))

    #print A_mcmc
    #print g_mcmc
    sampler.reset()
    return (A_mcmc, g_mcmc)



#######################################
#determine ensamble SF using emcee


def SFlist(jd,mag,err):#calculate an array with (m(t)-m(t+tau)), whit (err(t)^2+err(t+tau)^2) and another with tau=dt
    sfarray=[]
    tauarray=[]
    errarray=[]
    for i, item in enumerate(mag):
        for j in range(i+1,len(mag)):
            dm=mag[i]-mag[j]
            sigma=err[i]**2+err[j]**2
            dt=(jd[j]-jd[i])
            sfarray.append(np.abs(dm))
            tauarray.append(dt)
            errarray.append(sigma)

    return (tauarray,sfarray,errarray)

def SF_array_ensambled(list_objects,filt,magname,errname,nbin):

    dtlist=[]
    dmaglist=[]
    derrlist=[]



    #the dt, dmag and derr are determined for all the objects

    for i in xrange(len(list_objects)):

        arch=pf.open('../'+filt+'/'+list_objects[i])
        dat=arch[1].data
        head=arch[0].header
        z=head['ZSPEC']
        jd=dat['JD']
        mag=dat[magname]
        err=dat[errname]

        dta,dmaga,derra=SFlist(jd/(1+z),mag,err)

        dtlist+=dta
        dmaglist+=dmaga
        derrlist+=derra

        arch.close()


    dt=np.array(dtlist)
    derr=(np.array(derrlist))
    dmag=(np.array(dmaglist))

    return (dt,dmag,derr)




def fitSF_mcmc_ensambled(list_objects,filt,magname,errname,nbin,ndim,nwalkers,nit,nthr): #function that fits the values of A and gamma using mcmc with the package emcee.
#It recives the array with dt in days, dmag and the errors, besides the number of dimensions of the parameters, the number of walkers and the number of iterations

    #we calculate the arrays of dm, dt and sigma

    dtarray, dmagarray, sigmaarray = SF_array_ensambled(list_objects,filt,magname,errname,nbin)


    '''
    #test to explore the parameter space
    a_array=np.linspace(0,1,500)
    g_array=np.linspace(0,2,500)

    test_post=np.zeros((500,500))

    for i in xrange(len(g_array)):
        for j in xrange(len(a_array)):

            test_post[i,j]=np.exp(lnprob([g_array[i],a_array[j]], dtarray, dmagarray, sigmaarray))

    plt.contourf(a_array,g_array,test_post,100)
    plt.show()

    nn=np.where(test_post==np.amax(test_post))
    gg=g_array[nn[0]]
    aa=a_array[nn[1]]
    return (aa,gg)
    '''

    #definition of the optimal initial position of the walkers

    p0 = np.random.rand(ndim*nwalkers).reshape((nwalkers,ndim)) #gess to start the burn in fase

    #chi2 = lambda *args: -2 * lnlike(*args)
    #result = op.minimize(chi2, [0.1,0.1], args=(x, y, yerr))

    #run mcmc
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob,threads=nthr, args=(dtarray, dmagarray, sigmaarray))

    pos, prob, state = sampler.run_mcmc(p0,50) #from pos we have a best gess of the initial walkers
    sampler.reset()
    print("Running MCMC...")
    sampler.run_mcmc(pos, nit,rstate0=state)
    print("Done.")

    # Compute the quantiles.
    samples=sampler.chain[:,50:,:].reshape((-1,ndim))

    ac=sampler.acceptance_fraction

    #plt.hist(ac)
    #plt.show()

    #print samples

    A_fin=samples[:,1]
    gamma_fin=samples[:,0]

    print gamma_fin
    print A_fin

    #plt.plot(A_fin,gamma_fin,'.')
    #plt.show()

    #fig = corner.corner(samples, labels=["$m$", "$b$", "$\ln\,f$"])
    #fig.savefig("line-triangle.png")

    #plt.hist(A_fin,100)
    #plt.show()

    #plt.hist(gamma_fin,100)
    #plt.show()

    A_mcmc=(np.percentile(A_fin, 50),np.percentile(A_fin, 50)-np.percentile(A_fin, 15.865),np.percentile(A_fin, 84.135)-np.percentile(A_fin, 50))
    g_mcmc=(np.percentile(gamma_fin, 50),np.percentile(gamma_fin, 50)-np.percentile(gamma_fin, 15.865),np.percentile(gamma_fin, 84.135)-np.percentile(gamma_fin, 50))
    sampler.reset()
    return (A_mcmc, g_mcmc)




##################################################
#run FATS

def run_fats(jd, mag, err):
#function tu run fats and return the features in an array
    #list with the features to be calculated
    feature_list = [
        #'Mean',
        #'Std',
        #'Meanvariance',
        #'MedianBRP',
        #'Rcs',
        #'PeriodLS',
        #'Period_fit',
        #'Color',
        'Autocor_length',
        #'SlottedA_length',
        #'StetsonK',
        #'StetsonK_AC',
        #'Eta_e',
        #'Amplitude',
        #'PercentAmplitude',
        #'Con',
        #'LinearTrend',
        #'Beyond1Std',
        #'FluxPercentileRatioMid20',
        #'FluxPercentileRatioMid35',
        #'FluxPercentileRatioMid50',
        #'FluxPercentileRatioMid65',
        #'FluxPercentileRatioMid80',
        #'PercentDifferenceFluxPercentile',
        #'Q31',
        'CAR_sigma',
        'CAR_mean',
        'CAR_tau',
    ]


    data_array = np.array([mag, jd, err])
    data_ids = ['magnitude', 'time', 'error']
    feat_space = FATS.FeatureSpace(featureList=feature_list, Data=data_ids)
    feat_vals = feat_space.calculateFeature(data_array)
    f_results = feat_vals.result(method='array')
    f_features = feat_vals.result(method='features')

    return (f_results,f_features)




def all_single_features(jd,mag,err,ndim,nwalkers,nit,nthr):

    print "calculating var parameters"
    P,exvar , exvar_err = var_parameters(jd,mag,err)
    print "calculating SF"
    A_mcmc , g_mcmc = fitSF_mcmc(jd,mag,err,ndim,nwalkers,nit,nthr)
    print "running FATS"
    fats_features = run_fats(jd, mag, err)

    return [P, exvar, exvar_err , A_mcmc, g_mcmc, fats_features[0]]
