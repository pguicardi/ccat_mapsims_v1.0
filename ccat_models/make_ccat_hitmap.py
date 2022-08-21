import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt


def default(nside):
    """Default hitmap written and developed by Dr. Steve Choi for the CCAT channels"""
    npix = nside**2*12
    pix = np.arange(npix)
    m = np.zeros(npix)

    dec_lo = np.pi/2-np.radians(-64.)
    dec_hi = np.pi/2-np.radians(18.)

    th, phi = hp.pix2ang(nside,pix)
    ind = np.where((th>=dec_hi)&(th<=dec_lo))

    m[ind] = 1
    m[m==0] = -0.1

    m_sm = hp.smoothing(m,fwhm=np.radians(3.))
    m_sm[m_sm<0] = 0
    m_sm/=np.max(m_sm)
    
    return m_sm

def rough_change_nside(m, nside):
    """takes any map m and roughly translates it to a corresponding map in a different nside"""
    nr = m.shape[-1]
    gr = hp.nside2npix(nside)
    nested = hp.reorder(m,inp = 'RING', out = 'NESTED')
    if nr>gr:
        step = int(nr/gr)
        B = (np.arange(gr)*step).astype(int)
        return hp.reorder(nested[B],inp='NESTED', out = 'RING')
    elif nr<gr:
        assert (gr/nr)%4==0, "bad npix side"
        mul = np.ones(int(gr/nr))
        rres = np.outer(nested,mul).flatten()
        return hp.smoothing(hp.reorder(rres,inp='NESTED', out = 'RING'),fwhm = hp.nside2resol(hp.npix2nside(nr)))
        
    else:
        return m
    
def load_and_process(nside, filename= "ccat_mapsims/hitmaps/SO_LA_apo_mask_nside512_cut_b10_200611.fits"):
    """Takes a .fits file containing a single map and roughly translates it to any desired NSIDE. Can be used as a substitute for the defaut function in generating hitmaps."""
    m = hp.read_map(filename)
    return rough_change_nside(m, nside)
    

def make_hitmap_ccat(nside, fn=default):
    """This is the function being called to generate hitmaps for any channel tag starting with 'LC' (a.k.a any of the CCAT channels added in this package). """
    return fn(nside)


