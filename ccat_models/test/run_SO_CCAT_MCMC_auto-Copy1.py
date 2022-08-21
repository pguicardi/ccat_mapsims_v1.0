import sys
(sys.path).insert(1,'/Users/pedroguicardi/Desktop/CMB_Analysis/MAPSIMS/directories')

import mapsims
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import healpy as hp
import matplotlib
import pysm3
from matplotlib import pyplot as plt
from pixell import enmap, enplot, reproject, utils, curvedsky
from ad_fns import *
from astropy.io import fits
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from ccat_models import ccat_noise as CCAT_noise
import scipy.optimize as op
from scipy.optimize import minimize
import fgspectra.cross as fgc
from fgspectra.frequency import _rj2cmb
import emcee
import corner


output_file = "SO+CCAT_MCMC_output_1"


NSIDE = 512
lat_lmax = 1500
sim_num = 100
SO_input_file = 'SO_sims_output_2'
CCAT_input_file = "CCAT_sims_output_2"
pysm_string = "d0,s0"

auto_only = True

lowl = 70
highl = 610 #70-600
deltal = 15

step_num = 10000
fix_rho = True

redo=True
while redo:
    try:
        os.mkdir(output_file)
        redo=False
    except:
        output_file = output_file[:-1] + str(int(output_file[-1])+1)



def get_bands(chs, noise, remove =None):
    bands = []
    for ch in chs:
        bands.append(noise.tubes[ch[5:]][0])
        bands.append(noise.tubes[ch[5:]][1])
    inds = np.ones(len(bands))
    if remove!=None:
        inds[np.array(remove)]=0
    
    return np.array(bands)[(inds.astype(int)).astype(bool)]

def read_sims(sim_num, noise, NSIDE, output_file, pysm_string, chs, apodize=False, remove=None):
    """output shape will be of the form (bands_#, sim_num , 3, n_pix)"""
    bands = get_bands(chs,noise, remove = remove)
    output = np.zeros((len(bands),sim_num, 3, 12*(NSIDE)**2))
    for j in np.arange(sim_num):
        for k in np.arange(len(bands)):
            tmp = hp.fitsfunc.read_map(output_file +"/"+bands[k].tag+"_" + bands[k].tag[:3]+"_NSIDE_" + str(NSIDE) + "_TAG_" + pysm_string + "_" + str(j) + "_" +".fits", field=(0,1,2))
            if apodize:
                for h in np.arange(tmp.shape[0]):
                    tmp[h,:] = apodize_map(tmp[h,:])
                    
            output[k,j,:,:] = tmp
    return output

def compute_cross_spectra(sim_data, lat_lmax=None, p = None,auto_only=True):
    """sim data should be of the form (freq,sim_index, pols, maps )
    returns cross spectra of the form (freq, freq, sim_index, lmax)"""
    data = sim_data
    if p ==None:
        result = np.zeros((sim_data.shape[0],sim_data.shape[0], sim_data.shape[1], 6, lat_lmax+1))
    else:
        result = np.zeros((sim_data.shape[0],sim_data.shape[0], sim_data.shape[1], lat_lmax+1))
        data = sim_data[:,:,p,:]
    if auto_only:
        for k in np.arange(sim_data.shape[1]):
            for i in np.arange(sim_data.shape[0]):
                result[i,i,k] = hp.sphtfunc.anafast(data[i,k], lmax = lat_lmax)
    else:
        for k in np.arange(sim_data.shape[1]):
            for i in np.arange(sim_data.shape[0]):
                for j in np.arange(sim_data.shape[0]):
                    if i<=j:
                        result[i,j,k] = hp.sphtfunc.anafast(data[i,k],data[j,k], lmax = lat_lmax)
                    else:
                        result[i,j,k] = result[j,i,k]
    return result

    
def bin_array(array, binl):
    shape = array.shape
    assert shape[-1]/binl == int(shape[-1]/binl)
    n = int(shape[-1]/binl)
    T = np.outer(np.ones(n),np.concatenate((np.ones(binl), np.zeros(binl*n-binl))))
    for i in np.arange(T.shape[0]):
        T[i] = np.roll(T[i],i*binl)
    T /=binl
    return np.dot(array,T.T)

def compute_PS_matrix(ps_list, diag=True):
    pspec_M = np.zeros((ps_list.shape[0],ps_list.shape[0],ps_list.shape[-1]))
    for i in np.arange(ps_list.shape[0]):
        for j in np.arange(ps_list.shape[0]):
            if i<=j:
                pspec_M[i,j] = np.sqrt(ps_list[i]*ps_list[j])
    for i in np.arange(ps_list.shape[0]):
        for j in np.arange(ps_list.shape[0]):
            if j>i:
                if diag:
                    pspec_M[i,j] = np.zeros(len(pspec_M[i,j]))
                pspec_M[i,j] = pspec_M[i,j]
                pspec_M[j,i] = pspec_M[i,j]
                
    return pspec_M

def get_noise_PS_list(chs, noise, p=1, remove = None):
    pspec_list = []
    for ch in chs:
        ell_sim, ps_T, ps_P = noise.get_fullsky_noise_spectra(ch[5:])
        if p>0:
            to_c = ps_P
        else:
            to_c = ps_T
        pspec_list.append(to_c[0])
        pspec_list.append(to_c[1])
    pspec_list = np.array(pspec_list)
    inds = np.ones(pspec_list.shape[0])
    if remove!=None:
        inds[np.array(remove)]=0
    
    return pspec_list[(inds.astype(int)).astype(bool)]

def get_noise_PS_matrix(chs, noise, p=1, remove=None, diag = True):
    
    pspec_list = get_noise_PS_list(chs, noise, p=1, remove=remove)
    
    return compute_PS_matrix(pspec_list, diag=diag)

def get_x_from_bands(bands, x = 'freq'):
    res = []
    for b in bands:
        if x=='freq':
            res.append(b.center_frequency.value)
    return np.array(res)
    
def log_likelihood(v_par, M_A, M_noise, M_std, nu, ell_c,w_SO, auto_only = auto_only):
    """v_par is of the form (A_s, A_d, alpha_s, alpha_d, beta_s, beta_d, rho_BB)"""
    dust_params = dict(nu=nu, beta= v_par[5], temp=20., nu_0=353.)
    sync_params = dict(nu=nu, beta= v_par[4], nu_0=23.)
    frequency_params = dict(kwseq=(dust_params, sync_params))
    
    try:
        rho = v_par[6]
    except:
        rho=0.045
    
    power_params = dict(
        ell= ell_c,
        alpha=np.array([v_par[3], v_par[2]]),  # +2 to (almost) get D_ell
        ell_0=84,
        amp=np.array([v_par[1]*_rj2cmb(353.)**2 , v_par[0]*_rj2cmb(23.)**2])#*(10**0.5)
        ,rho=rho
    )

    dust_sync = fgc.CorrelatedDustSynchrotron()
    T = dust_sync(frequency_params, power_params)*w_SO+M_noise
    
    coeff_mat = (np.identity(len(nu)) + (1.-auto_only)*np.ones((len(nu),len(nu)))).reshape(len(nu),len(nu),1)
    M_sig = M_std
    res_mat_1 = ((M_A-T)**2)/(M_std**2)
    res_mat = coeff_mat*(res_mat_1)
    if auto_only==False:
        res_f = -1/4*np.sum(res_mat)
    else:
        res_f = -1/4*np.sum(res_mat[np.identity(res_mat.shape[0]).astype(bool)])
    return res_f


def log_prior(v_par):
    allowed_devs = np.array([0.005, 0.001, 1. , 0.05 , 0.5 , 0.025, 0.1])[:len(v_par)]
    v_par_o =      np.array([0.012 , 0.002,-2.8, -2.35, -3.0, 1.53 , 0.045])[:len(v_par)]
    if (np.abs(v_par_o- v_par)<5*allowed_devs).all():
        return 0.0
    else:
        return -np.inf
    
def log_probability(v_par, M_A,M_noise, M_std, nu, ell_c,w_SO):
    lp = log_prior(v_par)
    ll = log_likelihood(v_par,M_A,M_noise, M_std, nu, ell_c,w_SO)
    
    if not np.isfinite(lp+ll):
        return -np.inf
    return lp + ll


chs_SO = ["tube:LT1","tube:LT3","tube:LT5","tube:LT6"]
noise_SO = mapsims.SONoiseSimulator(
        nside=NSIDE,
        return_uK_CMB = True,
        sensitivity_mode = "baseline",
        apply_beam_correction = False,
        apply_kludge_correction = False,
        homogeneous=False,
        rolloff_ell = 50,
        ell_max = lat_lmax,
        survey_efficiency = 1.0,
        full_covariance = False,
        LA_years = 5,
#        LA_noise_model = "SOLatV3point1",
        elevation = 50,
        SA_years = 5,
        SA_one_over_f_mode = "pessimistic"
    )

chs_CCAT = ["tube:LC1","tube:LC2", "tube:LC3"]
noise_CCAT = mapsims.SONoiseSimulator(
        nside=NSIDE,
        return_uK_CMB = True,
        sensitivity_mode = "baseline",
        apply_beam_correction = False,
        apply_kludge_correction = False,
        homogeneous=False,
        rolloff_ell = 50,
        ell_max = lat_lmax,
        survey_efficiency = 1.0,
        full_covariance = False,
        LA_years = 5,
        LA_noise_model = "CcatLatv2b",
        elevation = 50,
        SA_years = 5,
        SA_one_over_f_mode = "pessimistic"
    )



remove_SO = [4,3]
remove_CCAT = [5]
bands_SO = get_bands(chs_SO,noise_SO, remove = remove_SO)
bands_CCAT = get_bands(chs_CCAT,noise_CCAT, remove = remove_CCAT)

def compute_ps_from_files_auto(sim_num, noise, NSIDE, output_file, pysm_string, chs, lmax, p=2, hitmap = None,remove = None):
    bands = get_bands(chs,noise, remove = remove)
    output = np.zeros((len(bands),sim_num, lmax+1))
    for j in np.arange(sim_num):
        for k in np.arange(len(bands)):
            tmp = hp.fitsfunc.read_map(output_file +"/"+bands[k].tag+"_" + bands[k].tag[:3]+"_NSIDE_" + str(NSIDE) + "_TAG_" + pysm_string + "_" + str(j) + "_" +".fits", field=(0,1,2))
            if list(hitmap)!=None:
                for h in np.arange(tmp.shape[0]):
                    tmp[h,:] = tmp[h,:]*hitmap
            ps = hp.sphtfunc.anafast(tmp, lmax = lat_lmax)
            output[k,j] = ps[p]
    return output
    

#calibrate dust map
hitmaps = noise_SO.get_hitmaps(tube = chs_SO[0][5:])
w_1 = np.mean(hitmaps[0][0]>0)
hitmap_apodized = get_window(np.array(hitmaps[0][0]>0).astype(float),n_it=5,width = 0.1)


#Introduce galactic cutoff
hm = hp.read_map('hitmaps/SO_LA_apo_mask_nside512_cut_b10_200611.fits')
hitmap_apodized = hm*hitmap_apodized


w = np.mean(hitmap_apodized)

bands = np.append(bands_SO,bands_CCAT)

# sim_data = {"SO": read_sims(sim_num,noise_SO, NSIDE, SO_input_file, pysm_string, chs_SO, apodize = False, remove = remove_SO),
#            "CCAT": read_sims(sim_num,noise_SO, NSIDE, CCAT_input_file, pysm_string, chs_CCAT, apodize = False,remove = remove_CCAT)}


pow_specs_diag_SO = compute_ps_from_files_auto(sim_num, noise_SO, NSIDE, SO_input_file, pysm_string, chs_SO, lat_lmax, p=2, hitmap = hitmap_apodized,remove = remove_SO)
pow_specs_diag_CCAT = compute_ps_from_files_auto(sim_num, noise_CCAT, NSIDE, CCAT_input_file, pysm_string, chs_CCAT, lat_lmax, p=2, hitmap = hitmap_apodized,remove = remove_CCAT)

pow_specs_diag = np.append(pow_specs_diag_SO, pow_specs_diag_CCAT, axis=0)
pow_specs = np.zeros((pow_specs_diag.shape[0],pow_specs_diag.shape[0],pow_specs_diag.shape[1],pow_specs_diag.shape[2]))
for k in np.arange(pow_specs_diag.shape[0]):
    pow_specs[k,k] = pow_specs_diag[k]
    
pspec_list= np.append(get_noise_PS_list(chs_SO, noise_SO, p=1, remove=remove_SO),
                        get_noise_PS_list(chs_CCAT, noise_CCAT, p=1, remove=remove_CCAT), axis=0)
pspec_matrix = compute_PS_matrix(pspec_list, diag=True)

p=2
binned_ps = bin_array(pow_specs[:,:,:,lowl:highl],deltal)/w
BB_avg = np.mean(binned_ps,axis=2)
BB_std = np.std(binned_ps,axis=2)

binned_noise = bin_array(pspec_matrix[:,:,lowl:highl],deltal)

ell_bin = bin_array(np.arange(lowl,highl),deltal)

# GET THEORETICAL POWER LAW CROSS SPECTRA for s0,d0

nu = np.zeros(len(bands))
for i,band in enumerate(bands):
    nu[i] = band.center_frequency.value

v_par_o = np.array([0.012 , 0.002, -2.73, -2.32, -3.0, 1.54, 0.045])

dust_params = dict(nu=nu, beta=v_par_o[5], temp=20., nu_0=353.)
sync_params = dict(nu=nu, beta=v_par_o[4], nu_0=23.)
frequency_params = dict(kwseq=(dust_params, sync_params))
power_params = dict(
    ell=ell_bin,
    alpha=np.array([ v_par_o[3],v_par_o[2]]),  # +2 to (almost) get D_ell
    ell_0=84,
    amp=np.array([v_par_o[1]*_rj2cmb(353.)**2,v_par_o[0]*_rj2cmb(23.)**2])
    ,rho=v_par_o[6]
)

dust_sync = fgc.CorrelatedDustSynchrotron()
nu_mat = np.sqrt(np.outer(nu,nu))

cl = dust_sync(frequency_params, power_params)#in CMB units

if fix_rho:
    v_par_o = v_par_o[:-1]

ell_c = ell_bin
M_A = BB_avg
M_std = BB_std
M_noise = binned_noise*w_1
w_SO = 0.575

nll = lambda *args: -log_likelihood(*args)
initial = v_par_o
soln = minimize(nll, initial, args=(M_A, M_noise,M_std, nu, ell_c,w_SO))
sol = soln.x

pos = v_par_o + 1e-6 * np.random.randn(100,v_par_o.shape[0] )
nwalkers, ndim = pos.shape
sampler = emcee.EnsembleSampler(
    nwalkers, ndim, log_probability, args=(M_A,M_noise, M_std, nu, ell_c,w_SO)
)
sampler.run_mcmc(pos, step_num, progress=True);

fig, axes = plt.subplots(v_par_o.shape[0], figsize=(10, 7), sharex=True)
samples = sampler.get_chain()
labels_f = [r"$A_s$", r"$A_d$", r"$\alpha_s$", r"$\alpha_d$", r"$\beta_s$", r"$\beta_d$", r"$\rho$"]
labels = labels_f[:len(v_par_o)]
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("step number");

plt.savefig(output_file+'/walkers_steps')

flat_samples = sampler.get_chain(discard=100, thin=40, flat=True)

np.savetxt(output_file+'/flat_samples.txt', flat_samples, delimiter = ',' )
flat_samples_p = np.loadtxt(output_file+'/flat_samples.txt', delimiter = ',' )
flat_samples_SO_only = np.loadtxt('SO_MCMC_output_7/flat_samples.txt', delimiter = ',' )

figure = corner.corner(flat_samples, labels=labels, color = 'b')
corner.corner(flat_samples_SO_only,fig=figure, color = 'r')
plt.savefig(output_file+'/corners_together')

final_vals = []
with open(output_file+'/uncertainties.txt', 'w') as f:
    for i in range(ndim):
        mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
        final_vals.append(mcmc[1])
        q = np.diff(mcmc)
        f.write(labels[i]+': '+str(mcmc[1]) + ', ' + str(q[0])+', ' + str(q[1]))
        f.write('\n')
        
v_par = np.array(final_vals)
dust_params = dict(nu=nu, beta= v_par[5], temp=20., nu_0=353.)
sync_params = dict(nu=nu, beta= v_par[4], nu_0=23.)
frequency_params = dict(kwseq=(dust_params, sync_params))

try:
    rho = v_par[6]
except:
    rho=0.045

power_params = dict(
    ell= ell_c,
    alpha=np.array([v_par[3], v_par[2]]),  # +2 to (almost) get D_ell
    ell_0=84,
    amp=np.array([v_par[1]*_rj2cmb(353.)**2 , v_par[0]*_rj2cmb(23.)**2])#*(10**0.5)
    ,rho=rho
)

dust_sync = fgc.CorrelatedDustSynchrotron()
cl = dust_sync(frequency_params, power_params)

plt.rcParams["figure.figsize"] = (30,18)
counter = 1
p=2
color_bar = ['b','g','r','c','m','y','k','rosybrown']
for i in np.arange(BB_avg.shape[0]):
    plt.subplot(np.ceil(np.sqrt(BB_avg.shape[0])),np.ceil(np.sqrt(BB_avg.shape[0])),counter)
    counter +=1
    plt.errorbar(ell_bin, BB_avg[i,i], fmt=".k", yerr=BB_std[i,i], capsize=0)
    plt.plot(ell_bin, binned_noise[i,i]*w_1+cl[i,i]*w_SO,'--')
    #plt.plot(ell_bin, cl[i,j]*w_SO,color = color_bar[i])
    plt.title(str(bands[i].center_frequency) + " x " + str(bands[i].center_frequency))
    plt.xscale('log')
    plt.yscale('log')
#plt.show()
plt.savefig(output_file+'/agreement_plots')
plt.clf()