import sys
(sys.path).insert(1,'/Users/pedroguicardi/Desktop/CMB_Analysis/MAPSIMS/directories')

import mapsims
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import healpy as hp
import matplotlib
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

NSIDE = 512
lat_lmax = 1500
sim_num = 100
output_file = "SO_sims_output_2"
pysm_string = "d0,s0"

#chs = ["tube:LT1","tube:LT3","tube:LT5","tube:LT6"]

chs = ["tube:LT6"]

cont = True #continue adding to a previous simulations folder

sim_num_list = [[20,100]] #where sims should start and finish for each channel 

if cont==False:
    redo=True
    while redo:
        try:
            os.mkdir(output_file)
            redo=False
        except:
            output_file = output_file[:-1] + str(int(output_file[-1])+1)

        


if cont:
    sim_num = np.array(sim_num_list)
else:
    A = np.ones(len(chs))*sim_num
    sim_num=np.append(np.zeros(len(A)), A).reshape(2, len(A)).T
        


        
cmb = mapsims.SOPrecomputedCMB(
        num=2,
        nside=NSIDE,
        lensed=False,
        aberrated=False,
        has_polarization=True,
        cmb_set=0,
        cmb_dir="mapsims/tests/data",
        input_units="uK_CMB",
    )

noise = mapsims.SONoiseSimulator(
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
#            LA_noise_model = "CcatLatv2b",
            elevation = 50,
            SA_years = 5,
            SA_one_over_f_mode = "pessimistic"
        )


np.random.seed(int(np.random.random()*100000))

for i,ch in enumerate(chs):
    for j in np.arange(*tuple(sim_num[i])).astype(int):
        sim_seed = int(np.random.random()*100000)
        
        filename = ch[5:]+"_NSIDE_" + str(NSIDE) + "_TAG_" + pysm_string + "_" + str(j) + "_" +".fits"
        simulator = mapsims.MapSim(
            channels=ch,
            nside=NSIDE,
            unit="uK_CMB",
            num=sim_seed,
            pysm_output_reference_frame="C",
            pysm_components_string=pysm_string,
    #            output_filename_template = filename,
            pysm_custom_components={"cmb": cmb},
            other_components={"noise": noise},
        )
        output_map_full = simulator.execute()

    #Now Apodize
        for det in output_map_full.keys():
            for pol in np.arange(output_map_full[det].shape[0]):
                output_map_full[det][pol] = apodize_map(output_map_full[det][pol])

    #Write Files
        write_output_map(output_map_full,output_file,filename)

