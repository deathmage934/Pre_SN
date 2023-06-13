from numpy import *
from matplotlib.pyplot import *
#%matplotlib inline
from IPython.display import Image
import scipy
from scipy.ndimage.filters import gaussian_laplace, gaussian_filter1d, gaussian_filter
from astropy.time import Time
import scipy
from scipy import integrate
# import extinction
from astropy.io import fits
import astropy.units as u
from astropy.io import ascii
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.stats import sigma_clipped_stats, sigma_clip
import subprocess
import astropy.io.fits as pyfits
import glob
from astropy.time import Time
from uncertainties import ufloat
from uncertainties.umath import *
from uncertainties import unumpy
from uncertainties import ufloat
from uncertainties.umath import *
from uncertainties import unumpy
from astropy.table import Table
import os

rc('text', usetex=False)
rc('font', family='serif')
rcParams['xtick.major.size'] = 15
rcParams['xtick.major.width'] = 2
rcParams['xtick.minor.size'] = 10
rcParams['xtick.minor.width'] = 2
rcParams['ytick.major.size'] = 15
rcParams['ytick.major.width'] = 2
rcParams['ytick.minor.size'] = 10
rcParams['ytick.minor.width'] = 2
rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'
rcParams['xtick.labelsize']=20
rcParams['ytick.labelsize']=20

def mf(m, lam):
    f = (10**(-.4*(m - 8.90))) / ((3.34e4) * lam**2)
    return f

def fm(f, lam):
    #fall = unumpy.uarray(f, ef)
    #m = -2.5*unumpy.log10(np.abs(fall)*1e-6) + 8.9
    m = -2.5*np.log10(f*((3.34e4) * lam**2)) + 8.90
    return m

eff_wvl = {'u':3546, 'B':4350.6 , 'g':4765.1 , 'c':5350., 'V':5401.4 , 'r':6223.3,\
                     'o':6900., 'i':7609.2, 'z':8917, 'J':12355, 'H':16458, 'K':21603} 

def compute_efficiency(data, bin_size=0.1, eff_target=0.5):
    
    model_mags = np.linspace(18.0, 24.0, int((24.0 - 18.0) / bin_size))

    mag_cent = []
    mag_effi = []
    for i in np.arange(len(model_mags)):
            if i==len(model_mags)-1:
                mag_cent.append(model_mags[i]+(model_mags[i]-model_mags[i-1])/2.0)
                mag_effi.append(np.nan)
            else:
                mag_lo = model_mags[i]
                mag_hi = model_mags[i+1]
                mag_cent.append((mag_lo+mag_hi)/2.0)

                mask = (data['sim_mag']<mag_hi) & (data['sim_mag']>mag_lo)

                total = len(data[mask])
                if total==0:
                    mag_effi.append(np.nan)
                else:
                    detected = 0
                    for row in data[mask]:
                        if row['snr']>3.0:
                            detected += 1
                    mag_effi.append(1.0*detected/total)

    
    return mag_cent, mag_effi

def main(sn_name, t_explosion, phot_path, diff_path):
    t_exp = t_explosion
    SNname= sn_name

    phot_lst = os.listdir(phot_path)

    g_ps1_v3 = np.genfromtxt(phot_path + "/" + phot_lst[0])
    r_ps1_v3 = np.genfromtxt(phot_path + "/" + phot_lst[1])
    i_ps1_v3 = np.genfromtxt(phot_path + "/" + phot_lst[2])
    z_ps1_v3 = np.genfromtxt(phot_path + "/" + phot_lst[3])

    #list of fake source fluxes to add in 

    f_fake = glob.glob('./FakeSources/2020tlf.g.200118_2199244_2199.067.sw_eff.dat')

    #
    ph_FS = []
    flx_FS = []
    mag_FS = []
    flx_FS_d = []
    tflx_FS_d = []
    mag_FS_d = []
    eff_FS = []
    mag_FS_s = []
    eff_FS_s = []

    for f in f_fake:
        
        date = '20'+str(f[-34:-28])
        date = date[:4]+'-'+date[4:6]+'-'+date[6:]
        t = Time(str(date), format='isot', scale='utc')
        mjd = t.mjd
        ph_FS.append(mjd)
        
        dd = Table.read(f, format='ascii', header_start=0)
        zpt = dd['det_zpt'][0]
        sim_mags, sim_eff = compute_efficiency(dd, bin_size=0.1)
        idx = (dd['det_flux'] > 0)
        
        det_flxs = dd['det_flux'][dd['det_flux']>0]
        det_mags = dd['det_mag'][dd['det_flux']>0]
        det_mags, det_flxs = zip(*sorted(zip(det_mags, det_flxs)))
        
        sm_flxs = dd['sim_flux'][dd['det_flux']>0]
        sm_mags = dd['sim_mag'][dd['det_flux']>0]
        sm_mags, sm_flxs = zip(*sorted(zip(sm_mags, sm_flxs)))
        
        #print (det_mags, zpt)
        trueflx_FS_d = 10**(-0.4*(np.asarray(det_mags)-zpt))
        
        #print (mf(np.asarray(det_mags), eff_wvl['g'])/det_flxs)

        flx_FS.append(sm_flxs) #append(flx)
        mag_FS.append(sm_mags) #append(sim_mags)
        flx_FS_d.append(det_flxs) #append(flx)
        mag_FS_d.append(det_mags) #append(sim_mags)
        eff_FS.append(sim_eff)
        mag_FS_s.append(sim_mags)
        eff_FS_s.append(sim_eff)
        tflx_FS_d.append(det_flxs*(mf(np.asarray(det_mags), eff_wvl['g'])/det_flxs))


    # g-band
    files = glob.glob(diff_path + '/g/*.diffimstats*')
    files.sort()

    ph_fake = []
    xp_fake = []
    yp_fake = []
    flx_fake = []
    eflx_fake = []

    for fname in files:
        f = np.genfromtxt(fname)
        
        date = '20'+str(fname[-88:-82])
        date = date[:4]+'-'+date[4:6]+'-'+date[6:]
        t = Time(str(date), format='isot', scale='utc')
        mjd = t.mjd
        ph_fake.append(mjd)
        xp_fake.append(f[:,0])
        yp_fake.append(f[:,1])
        flx_fake.append(f[:,2])
        eflx_fake.append(f[:,3])

        #add fake source fluxes to background aperture region fluxes and calculate the sigma
    ginj_sigma = []
    ginj_3sig = []
    ph_both = []
    for j in range(len(ph_fake)):
        sig = []
        for i in range(len(ph_FS)):
            for k in range(len(flx_FS_d[i])):

                ginj = flx_FS_d[i][k]+flx_fake[j]

                sig.append(np.asarray(ginj/(eflx_fake[j])))
    #             print ('FS flux:', flx_FS_d[i][k])
    #             print ('CL flux:', flx_fake[j])
    #             print ('CL err:', eflx_fake[j])
    #             print ('sig:', np.asarray(ginj/(eflx_fake[j])))

        ginj_sigma.append(np.asarray(sig))

        #find fraction of background regions that have 3sigma detections after fake sources are added 
    ginj_3sig = []
    ginj_3sigfrac = []
    for i in range(len(ginj_sigma)):
        sig3 = []
        s_frac = []
        for j in range(len(ginj_sigma[i])):
            fc = (ginj_sigma[i][j] >= 3).sum() / len(flx_fake[i])
            s_frac.append(fc)
            sigg = []
            fr = []
            for k in range(len(ginj_sigma[i][j])):

                fr.append(fc)

            sigg = np.asarray(sigg)
            sig3.append(sigg)  
        
            
        ginj_3sig.append(sig3)
        ginj_3sigfrac.append(s_frac)
            
    ginj_3sig = np.asarray(ginj_3sig)
    ginj_3sigfrac = np.asarray(ginj_3sigfrac)

    idx = ginj_3sigfrac[0] >= 0.997

    mag_FS = np.asarray(mag_FS)
    mag_FS_d = np.asarray(mag_FS_d)


    lim_3sig_CL = []
    lim_3sig_FS = []
    lim_80p_CL = []
    lim_80p_FS = []
    lim_50p_CL = []
    lim_50p_FS = []
    lim_80p_CL_f = []
    lim_50p_CL_f = []
    for i in range(len(ginj_3sigfrac)):
        
        idx_3sig = (ginj_3sigfrac[i] >= 0.997)
        idx_80p = (ginj_3sigfrac[i] >= 0.80)
        idx_50p = (ginj_3sigfrac[i] >= 0.50)

        gcl_80p_lim = mag_FS[0][idx_80p][-1]
        gcl_50p_lim = mag_FS[0][idx_50p][-1]
        gcl_80p_lim_f = np.asarray(flx_FS[0])[idx_80p][-1]
        gcl_50p_lim_f = np.asarray(flx_FS[0])[idx_50p][-1]

        lim_80p_CL.append(gcl_80p_lim)
        lim_50p_CL.append(gcl_50p_lim)
        lim_80p_CL_f.append(gcl_80p_lim_f)
        lim_50p_CL_f.append(gcl_50p_lim_f)
        
        figure(figsize=(15,10))

        xlabel('Magnitude', fontsize=40)
        ylabel('Recovery Fraction', fontsize=40)
        xlim(18.,24)
        title('Phase: '+str("{:.1f}".format(ph_fake[i]-t_exp))+'; Limit: '+str("{:.1f}".format(gcl_80p_lim))+'(80$\%$), '+str("{:.1f}".format(gcl_50p_lim))+'(50$\%$)', fontsize=40)
        plot(mag_FS[0], ginj_3sigfrac[i], '-', lw=5, color='r',markersize=15, markeredgecolor='k')

        #legend(loc='lower left', fontsize=40, framealpha=1)

        savefig(r'./RecoveryCurves/g/'+str(ph_fake[i])+'_RC.png', dpi = 100, bbox_inches='tight',pad_inches=0.2, facecolor='white')
        close()

        g_ps1_v3[:,1], g_ps1_v3[:,4], g_ps1_v3[:,5] = zip(*sorted(zip(g_ps1_v3[:,1], g_ps1_v3[:,4], g_ps1_v3[:,5])))

    figure(figsize=(15,10))
    ylabel('Flux (DN)', fontsize=50)
    xlabel('Phase', fontsize=50)
    title('YSE g-band '+SNname, fontsize=30)

    #xlim(58860, 58920)
    ylim(-200, 1500)
    xlim(-250, 10)
    axhline(0, linestyle='--', c='k', lw=4)
    plot(np.asarray(ph_fake[0])-t_exp, lim_80p_CL_f[0], 'o', color='dodgerblue', markersize=30, markeredgecolor='k', zorder=5, label=r'Limit (80\%)')
    plot(np.asarray(ph_fake[0])-t_exp, lim_80p_CL_f[0], 'o', color='r', markersize=30, markeredgecolor='k', zorder=5, label=r'Limit (50\%)')
    plot(g_ps1_v3[:,1][0]-t_exp, g_ps1_v3[:,4][0], '*-', markersize=50,  zorder=7, color='silver', markeredgecolor='k', label=r'Pre-SN LC (Non-Detections)' )
    plot(g_ps1_v3[:,1][0]-t_exp, g_ps1_v3[:,4][0], '*-', markersize=50,  zorder=7, color='magenta', markeredgecolor='k', label=r'Pre-SN LC (Detections, 50\%)' )
                        
    for i in range(len(g_ps1_v3[:,1])):
        for j in range(len(ph_fake)):
            if np.abs(g_ps1_v3[:,1][i] - ph_fake[j]) < 0.9:
                
                #print ('same:', g_ps1_v3[:,1][i], g_ps1_v3[:,1][i]-t_exp)
                
                plot(np.asarray(ph_fake[j])-t_exp, lim_80p_CL_f[j], 'o', color='dodgerblue', markersize=30, markeredgecolor='k', zorder=5)
                errorbar(np.asarray(ph_fake[j])-t_exp, lim_80p_CL_f[j], yerr=np.asarray(lim_80p_CL_f[j])*0.15, fmt='o',capsize=10, lw=4, capthick=1, color='dodgerblue', uplims=True)

                plot(np.asarray(ph_fake[j])-t_exp, lim_50p_CL_f[j], 'o', color='r', markersize=30, markeredgecolor='k', zorder=5)
                errorbar(np.asarray(ph_fake[j])-t_exp, lim_50p_CL_f[j], yerr=np.asarray(lim_50p_CL_f[j])*0.15, fmt='o',capsize=10, lw=4, capthick=1, color='r', uplims=True)

                plot(g_ps1_v3[:,1][i]-t_exp, g_ps1_v3[:,4][i], '*-', markersize=50,  zorder=7, color='silver', markeredgecolor='k' )
                errorbar(g_ps1_v3[:,1][i]-t_exp, g_ps1_v3[:,4][i], yerr=g_ps1_v3[:,5][i], fmt='o', markersize=1, capsize=0, lw=3, capthick=1, color='silver', zorder=5)
                
                if (lim_50p_CL_f[j] <= g_ps1_v3[:,4][i]):
                    #print (g_ps1_v3[:,1][i]-t_exp)
                    plot(g_ps1_v3[:,1][i]-t_exp, g_ps1_v3[:,4][i], '*-', markersize=50,  zorder=7, color='magenta', markeredgecolor='k' )
                    errorbar(g_ps1_v3[:,1][i]-t_exp, g_ps1_v3[:,4][i], yerr=g_ps1_v3[:,5][i], fmt='o', markersize=1, capsize=0, lw=3, capthick=1, color='magenta', zorder=5)
            
            
    legend(loc='upper left', fontsize=25, framealpha=1, ncol=2,markerscale=0.8)

    savefig(r'./FinalLCs/'+SNname+'_g_preSN.pdf', dpi = 100, bbox_inches='tight',pad_inches=0.2, facecolor='white')
    close()


    # r-band
    files_r = glob.glob(diff_path + '/r/*.diffimstats*')
    files_r.sort()

    ph_fake_r = []
    xp_fake_r = []
    yp_fake_r = []
    flx_fake_r = []
    eflx_fake_r = []

    for fname in files_r:
        f = np.genfromtxt(fname)
        
        date = '20'+str(fname[-88:-82])
        date = date[:4]+'-'+date[4:6]+'-'+date[6:]
        t = Time(str(date), format='isot', scale='utc')
        mjd = t.mjd
        ph_fake_r.append(mjd)
        xp_fake_r.append(f[:,0])
        yp_fake_r.append(f[:,1])
        flx_fake_r.append(f[:,2])
        eflx_fake_r.append(f[:,3])

    rinj_sigma = []
    rinj_3sig = []
    for j in range(len(ph_fake_r)):
        sig = []
        for i in range(len(ph_FS)):
            for k in range(len(flx_FS_d[i])):

                rinj = flx_FS_d[i][k]+flx_fake_r[j]

                sig.append(np.asarray(rinj/(eflx_fake_r[j])))

        rinj_sigma.append(np.asarray(sig))

    rinj_3sig = []
    rinj_3sigfrac = []
    for i in range(len(rinj_sigma)):
        sig3 = []
        s_frac = []
        for j in range(len(rinj_sigma[i])):
            fc = (rinj_sigma[i][j] >= 3).sum() / len(flx_fake_r[i])
            s_frac.append(fc)
            sigg = []
            fr = []
            for k in range(len(rinj_sigma[i][j])):
                fr.append(fc)

            sigg = np.asarray(sigg)
            sig3.append(sigg)  

        rinj_3sig.append(sig3)
        rinj_3sigfrac.append(s_frac)
            
    rinj_3sig = np.asarray(rinj_3sig)
    rinj_3sigfrac = np.asarray(rinj_3sigfrac)

    mag_FS = np.asarray(mag_FS)
    mag_FS_d = np.asarray(mag_FS_d)

    rlim_3sig_CL = []
    rlim_80p_CL = []
    rlim_50p_CL = []
    rlim_80p_CL_f = []
    rlim_50p_CL_f = []

    for i in range(len(rinj_3sigfrac)):
        
        idx_3sig = (rinj_3sigfrac[i] >= 0.997)
        idx_80p = (rinj_3sigfrac[i] >= 0.80)
        idx_50p = (rinj_3sigfrac[i] >= 0.50)


        rcl_80p_lim = mag_FS_d[0][idx_80p][-1]
        rcl_50p_lim = mag_FS_d[0][idx_50p][-1]
        rcl_80p_lim_f = np.asarray(flx_FS_d[0])[idx_80p][-1]
        rcl_50p_lim_f = np.asarray(flx_FS_d[0])[idx_50p][-1]

        rlim_80p_CL.append(rcl_80p_lim)
        rlim_50p_CL.append(rcl_50p_lim)
        rlim_80p_CL_f.append(rcl_80p_lim_f)
        rlim_50p_CL_f.append(rcl_50p_lim_f)
        
        
        figure(figsize=(15,10))

        xlabel('Magnitude', fontsize=40)
        ylabel('Recovery Fraction', fontsize=40)
        xlim(18.,24)
        title('Phase: '+str("{:.1f}".format(ph_fake_r[i]-t_exp))+'; Limit: '+str("{:.1f}".format(rcl_80p_lim))+'(80$\%$), '+str("{:.1f}".format(rcl_50p_lim))+'(50$\%$)', fontsize=40)
        plot(mag_FS_d[0], rinj_3sigfrac[i], '-', lw=5, color='r',markersize=15, markeredgecolor='k')
        
        #legend(loc='lower left', fontsize=40, framealpha=1)
        
        savefig(r'./RecoveryCurves/r/'+str(ph_fake_r[i])+'_RC.png', dpi = 100, bbox_inches='tight',pad_inches=0.2, facecolor='white')
        close()

    r_ps1_v3[:,1], r_ps1_v3[:,4], r_ps1_v3[:,5] = zip(*sorted(zip(r_ps1_v3[:,1], r_ps1_v3[:,4], r_ps1_v3[:,5])))

    figure(figsize=(15,10))
    ylabel('Flux (DN)', fontsize=50)
    xlabel('Phase', fontsize=50)
    title('YSE r-band '+SNname, fontsize=30)

    #xlim(58860, 58920)
    ylim(-100, 1300)
    xlim(-250, 10)
    axhline(0, linestyle='--', c='k', lw=4)
    plot(np.asarray(ph_fake_r[0])-t_exp, rlim_80p_CL_f[0], 'o', color='dodgerblue', markersize=30, markeredgecolor='k', zorder=5, label=r'Limit (80\%)')
    plot(np.asarray(ph_fake_r[0])-t_exp, rlim_80p_CL_f[0], 'o', color='r', markersize=30, markeredgecolor='k', zorder=5, label=r'Limit (50\%)')
    plot(r_ps1_v3[:,1][0]-t_exp, r_ps1_v3[:,4][0], '*-', markersize=50,  zorder=7, color='silver', markeredgecolor='k', label=r'Pre-SN LC (Non-Detections)' )
    plot(r_ps1_v3[:,1][0]-t_exp, r_ps1_v3[:,4][0], '*-', markersize=50,  zorder=7, color='magenta', markeredgecolor='k', label=r'Pre-SN LC (Detections, 50\%)' )
                        
    for i in range(len(r_ps1_v3[:,1])):
        for j in range(len(ph_fake_r)):
            if np.abs(r_ps1_v3[:,1][i] - ph_fake_r[j]) < 0.9:
                
                #print ('same:', r_ps1_v3[:,1][i], r_ps1_v3[:,1][i]-t_exp)
                
                plot(np.asarray(ph_fake_r[j])-t_exp, rlim_80p_CL_f[j], 'o', color='dodgerblue', markersize=30, markeredgecolor='k', zorder=5)
                errorbar(np.asarray(ph_fake_r[j])-t_exp, rlim_80p_CL_f[j], yerr=np.asarray(rlim_80p_CL_f[j])*0.15, fmt='o',capsize=10, lw=4, capthick=1, color='dodgerblue', uplims=True)

                plot(np.asarray(ph_fake_r[j])-t_exp, rlim_50p_CL_f[j], 'o', color='r', markersize=30, markeredgecolor='k', zorder=5)
                errorbar(np.asarray(ph_fake_r[j])-t_exp, rlim_50p_CL_f[j], yerr=np.asarray(rlim_50p_CL_f[j])*0.15, fmt='o',capsize=10, lw=4, capthick=1, color='r', uplims=True)

                plot(r_ps1_v3[:,1][i]-t_exp, r_ps1_v3[:,4][i], '*-', markersize=50,  zorder=7, color='silver', markeredgecolor='k' )
                errorbar(r_ps1_v3[:,1][i]-t_exp, r_ps1_v3[:,4][i], yerr=r_ps1_v3[:,5][i], fmt='o', markersize=1, capsize=0, lw=3, capthick=1, color='silver', zorder=5)
                
                if (rlim_50p_CL_f[j] <= r_ps1_v3[:,4][i]):
                    #print (r_ps1_v3[:,1][i])
                    plot(r_ps1_v3[:,1][i]-t_exp, r_ps1_v3[:,4][i], '*-', markersize=50,  zorder=7, color='magenta', markeredgecolor='k' )
                    errorbar(r_ps1_v3[:,1][i]-t_exp, r_ps1_v3[:,4][i], yerr=r_ps1_v3[:,5][i], fmt='o', markersize=1, capsize=0, lw=3, capthick=1, color='magenta', zorder=5)
            
            
    legend(loc='upper left', fontsize=25, framealpha=1, ncol=2,markerscale=0.8)

    savefig(r'./FinalLCs/'+SNname+'_r_preSN.pdf', dpi = 100, bbox_inches='tight',pad_inches=0.2, facecolor='white')
    close()

    # i-band
    files_i = glob.glob(diff_path + '/i/*.diffimstats*')

    files_i.sort()

    ph_fake_i = []
    xp_fake_i = []
    yp_fake_i = []
    flx_fake_i = []
    eflx_fake_i = []

    for fname in files_i:
        f = np.genfromtxt(fname)
        
        date = '20'+str(fname[-88:-82])
        date = date[:4]+'-'+date[4:6]+'-'+date[6:]
        t = Time(str(date), format='isot', scale='utc')
        mjd = t.mjd
        ph_fake_i.append(mjd)
        xp_fake_i.append(f[:,0])
        yp_fake_i.append(f[:,1])
        flx_fake_i.append(f[:,2])
        eflx_fake_i.append(f[:,3])

    iinj_sigma = []
    iinj_3sig = []
    for j in range(len(ph_fake_i)):
        sig = []
        for i in range(len(ph_FS)):
            for k in range(len(flx_FS_d[i])):

                iinj = flx_FS_d[i][k]+flx_fake_i[j]

                sig.append(np.asarray(iinj/(eflx_fake_i[j])))
                #print ('FS flux:', flx_FS_d[i][k])
                #print ('CL flux:', flx_fake[j])
                #print ('CL err:', eflx_fake[j])

        iinj_sigma.append(np.asarray(sig))

    iinj_3sig = []
    iinj_3sigfrac = []
    for i in range(len(iinj_sigma)):
        sig3 = []
        s_frac = []

        for j in range(len(iinj_sigma[i])):
            fc = (iinj_sigma[i][j] >= 3).sum() / len(flx_fake_i[i])
            s_frac.append(fc)
            sigg = []
            fr = []
            for k in range(len(iinj_sigma[i][j])):
                fr.append(fc)

            sigg = np.asarray(sigg)
            sig3.append(sigg)  

        iinj_3sig.append(sig3)
        iinj_3sigfrac.append(s_frac)
            
    iinj_3sig = np.asarray(iinj_3sig)
    iinj_3sigfrac = np.asarray(iinj_3sigfrac)

    mag_FS = np.asarray(mag_FS)
    mag_FS_d = np.asarray(mag_FS_d)

    ilim_3sig_CL = []
    ilim_80p_CL = []
    ilim_50p_CL = []
    ilim_80p_CL_f = []
    ilim_50p_CL_f = []

    for i in range(len(iinj_3sigfrac)):
        
        idx_3sig = (iinj_3sigfrac[i] >= 0.997)
        idx_80p = (iinj_3sigfrac[i] >= 0.80)
        idx_50p = (iinj_3sigfrac[i] >= 0.50)

        icl_80p_lim = mag_FS_d[0][idx_80p][-1]
        icl_50p_lim = mag_FS_d[0][idx_50p][-1]
        icl_80p_lim_f = np.asarray(flx_FS_d[0])[idx_80p][-1]
        icl_50p_lim_f = np.asarray(flx_FS_d[0])[idx_50p][-1]

        ilim_80p_CL.append(icl_80p_lim)
        ilim_50p_CL.append(icl_50p_lim)
        ilim_80p_CL_f.append(icl_80p_lim_f)
        ilim_50p_CL_f.append(icl_50p_lim_f)
        
        
        figure(figsize=(15,10))

        xlabel('Magnitude', fontsize=40)
        ylabel('Recovery Fraction', fontsize=40)
        xlim(18.,24)
        title('Phase: '+str("{:.1f}".format(ph_fake_i[i]-t_exp))+'; Limit: '+str("{:.1f}".format(icl_80p_lim))+'(80$\%$), '+str("{:.1f}".format(icl_50p_lim))+'(50$\%$)', fontsize=40)
        plot(mag_FS_d[0], iinj_3sigfrac[i], '-', lw=5, color='r',markersize=15, markeredgecolor='k')

        #plot(mag_FS_s[i], eff_FS_s[i], '-', lw=5, color='dodgerblue', markersize=20, label='Fake Source', markeredgecolor='k')

        #legend(loc='lower left', fontsize=40, framealpha=1)

        savefig(r'./RecoveryCurves/i/'+str(ph_fake_i[i])+'_RC.png', dpi = 100, bbox_inches='tight',pad_inches=0.2, facecolor='white')
        close()

    i_ps1_v3[:,1], i_ps1_v3[:,4], i_ps1_v3[:,5] = zip(*sorted(zip(i_ps1_v3[:,1], i_ps1_v3[:,4], i_ps1_v3[:,5])))

    figure(figsize=(15,10))
    ylabel('Flux (DN)', fontsize=50)
    xlabel('Phase', fontsize=50)
    title('YSE i-band '+SNname, fontsize=30)

    #xlim(58860, 58920)
    ylim(-100, 1300)
    xlim(-250, 10)
    axhline(0, linestyle='--', c='k', lw=4)
    plot(np.asarray(ph_fake_i[0])-t_exp, ilim_80p_CL_f[0], 'o', color='dodgerblue', markersize=30, markeredgecolor='k', zorder=5, label=r'Limit (80\%)')
    plot(np.asarray(ph_fake_i[0])-t_exp, ilim_50p_CL_f[0], 'o', color='r', markersize=30, markeredgecolor='k', zorder=5, label=r'Limit (50\%)')
    plot(i_ps1_v3[:,1][0]-t_exp, i_ps1_v3[:,4][0], '*-', markersize=50,  zorder=7, color='silver', markeredgecolor='k', label=r'Pre-SN LC (Non-Detections)' )
    plot(i_ps1_v3[:,1][0]-t_exp, i_ps1_v3[:,4][0], '*-', markersize=50,  zorder=7, color='magenta', markeredgecolor='k', label=r'Pre-SN LC (Detections, 50\%)' )
                        
    for i in range(len(i_ps1_v3[:,1])):
        for j in range(len(ph_fake_i)):
            if np.abs(i_ps1_v3[:,1][i] - ph_fake_i[j]) < 0.9:
                #print ('same:', r_ps1_v3[:,1][i], ph_fake_r[j])
                
                plot(np.asarray(ph_fake_i[j])-t_exp, ilim_80p_CL_f[j], 'o', color='dodgerblue', markersize=30, markeredgecolor='k', zorder=5)
                errorbar(np.asarray(ph_fake_i[j])-t_exp, ilim_80p_CL_f[j], yerr=np.asarray(ilim_80p_CL_f[j])*0.15, fmt='o',capsize=10, lw=4, capthick=1, color='dodgerblue', uplims=True)

                plot(np.asarray(ph_fake_i[j])-t_exp, ilim_50p_CL_f[j], 'o', color='r', markersize=30, markeredgecolor='k', zorder=5)
                errorbar(np.asarray(ph_fake_i[j])-t_exp, ilim_50p_CL_f[j], yerr=np.asarray(ilim_50p_CL_f[j])*0.15, fmt='o',capsize=10, lw=4, capthick=1, color='r', uplims=True)

                plot(i_ps1_v3[:,1][i]-t_exp, i_ps1_v3[:,4][i], '*-', markersize=50,  zorder=7, color='silver', markeredgecolor='k' )
                errorbar(i_ps1_v3[:,1][i]-t_exp, i_ps1_v3[:,4][i], yerr=i_ps1_v3[:,5][i], fmt='o', markersize=1, capsize=0, lw=3, capthick=1, color='silver', zorder=5)
                
                if (ilim_50p_CL_f[j] <= i_ps1_v3[:,4][i]):
                    #print (i_ps1_v3[:,1][i], i_ps1_v3[:,1][i]-t_exp)
                    plot(i_ps1_v3[:,1][i]-t_exp, i_ps1_v3[:,4][i], '*-', markersize=50,  zorder=7, color='magenta', markeredgecolor='k' )
                    errorbar(i_ps1_v3[:,1][i]-t_exp, i_ps1_v3[:,4][i], yerr=i_ps1_v3[:,5][i], fmt='o', markersize=1, capsize=0, lw=3, capthick=1, color='magenta', zorder=5)
            
            
    legend(loc='upper left', fontsize=25, framealpha=1, ncol=2,markerscale=0.8)

    savefig(r'./FinalLCs/'+SNname+'_i_preSN.pdf', dpi = 100, bbox_inches='tight',pad_inches=0.2, facecolor='white')
    close()

    # z-band
    files_z = glob.glob(diff_path + '/z/*.diffimstats*')

    files_z.sort()

    ph_fake_z = []
    xp_fake_z = []
    yp_fake_z = []
    flx_fake_z = []
    eflx_fake_z = []

    for fname in files_z:
        f = np.genfromtxt(fname)
        
        date = '20'+str(fname[-88:-82])
        date = date[:4]+'-'+date[4:6]+'-'+date[6:]
        t = Time(str(date), format='isot', scale='utc')
        mjd = t.mjd
        ph_fake_z.append(mjd)
        xp_fake_z.append(f[:,0])
        yp_fake_z.append(f[:,1])
        flx_fake_z.append(f[:,2])
        eflx_fake_z.append(f[:,3])

    zinj_sigma = []
    zinj_3sig = []
    for j in range(len(ph_fake_z)):
        sig = []
        for i in range(len(ph_FS)):
            for k in range(len(flx_FS_d[i])):

                zinj = flx_FS_d[i][k]+flx_fake_z[j]

                sig.append(np.asarray(zinj/(eflx_fake_z[j])))
                #print ('FS flux:', flx_FS_d[i][k])
                #print ('CL flux:', flx_fake[j])
                #print ('CL err:', eflx_fake[j])

        zinj_sigma.append(np.asarray(sig))

    zinj_3sig = []
    zinj_3sigfrac = []
    for i in range(len(zinj_sigma)):
        sig3 = []
        s_frac = []

        for j in range(len(zinj_sigma[i])):
            fc = (zinj_sigma[i][j] >= 3).sum() / len(flx_fake_z[i])
            s_frac.append(fc)
            sigg = []
            fr = []
            for k in range(len(zinj_sigma[i][j])):
                fr.append(fc)

            sigg = np.asarray(sigg)
            sig3.append(sigg)  

        zinj_3sig.append(sig3)
        zinj_3sigfrac.append(s_frac)
        
    zinj_3sig = np.asarray(zinj_3sig)
    zinj_3sigfrac = np.asarray(zinj_3sigfrac)

    mag_FS = np.asarray(mag_FS)
    mag_FS_d = np.asarray(mag_FS_d)

    zlim_3sig_CL = []
    zlim_80p_CL = []
    zlim_50p_CL = []
    zlim_80p_CL_f = []
    zlim_50p_CL_f = []

    for i in range(len(zinj_3sigfrac)):
        
        idx_3sig = (zinj_3sigfrac[i] >= 0.997)
        idx_80p = (zinj_3sigfrac[i] >= 0.80)
        idx_50p = (zinj_3sigfrac[i] >= 0.50)

        zcl_80p_lim = mag_FS_d[0][idx_80p][-1]
        zcl_50p_lim = mag_FS_d[0][idx_50p][-1]
        zcl_80p_lim_f = np.asarray(flx_FS_d[0])[idx_80p][-1]
        zcl_50p_lim_f = np.asarray(flx_FS_d[0])[idx_50p][-1]

        zlim_80p_CL.append(zcl_80p_lim)
        zlim_50p_CL.append(zcl_50p_lim)
        zlim_80p_CL_f.append(zcl_80p_lim_f)
        zlim_50p_CL_f.append(zcl_50p_lim_f)
        
        
        figure(figsize=(15,10))

        xlabel('Magnitude', fontsize=40)
        ylabel('Recovery Fraction', fontsize=40)
        xlim(18.,24)
        title('Phase: '+str("{:.1f}".format(ph_fake_z[i]-t_exp))+'; Limit: '+str("{:.1f}".format(zcl_80p_lim))+'(80$\%$), '+str("{:.1f}".format(zcl_50p_lim))+'(50$\%$)', fontsize=40)
        plot(mag_FS_d[0], zinj_3sigfrac[i], '-', lw=5, color='r',markersize=15, label='Control LC', markeredgecolor='k')


        #legend(loc='lower left', fontsize=40, framealpha=1)

        savefig(r'./RecoveryCurves/z/'+str(ph_fake_z[i])+'_RC.png', dpi = 100, bbox_inches='tight',pad_inches=0.2, facecolor='white')
        close()

    z_ps1_v3[:,1], z_ps1_v3[:,4], z_ps1_v3[:,5] = zip(*sorted(zip(z_ps1_v3[:,1], z_ps1_v3[:,4], z_ps1_v3[:,5])))

    figure(figsize=(15,10))
    ylabel('Flux (DN)', fontsize=50)
    xlabel('Phase', fontsize=50)
    title('YSE z-band '+SNname, fontsize=30)

    #xlim(58860, 58920)
    ylim(-100, 1300)
    xlim(-250, 10)
    axhline(0, linestyle='--', c='k', lw=4)
    plot(np.asarray(ph_fake_z[0])-t_exp, zlim_80p_CL_f[0], 'o', color='dodgerblue', markersize=30, markeredgecolor='k', zorder=5, label=r'Limit (80\%)')
    plot(np.asarray(ph_fake_z[0])-t_exp, zlim_50p_CL_f[0], 'o', color='r', markersize=30, markeredgecolor='k', zorder=5, label=r'Limit (50\%)')
    plot(z_ps1_v3[:,1][0]-t_exp, z_ps1_v3[:,4][0], '*-', markersize=50,  zorder=7, color='silver', markeredgecolor='k', label=r'Pre-SN LC (Non-Detections)' )
    plot(z_ps1_v3[:,1][0]-t_exp, z_ps1_v3[:,4][0], '*-', markersize=50,  zorder=7, color='magenta', markeredgecolor='k', label=r'Pre-SN LC (Detections, 50\%)' )
                        
    for i in range(len(z_ps1_v3[:,1])):
        for j in range(len(ph_fake_z)):
            if np.abs(z_ps1_v3[:,1][i] - ph_fake_z[j]) < 0.9:
                #print ('same:', r_ps1_v3[:,1][i], ph_fake_r[j])
                
                plot(np.asarray(ph_fake_z[j])-t_exp, zlim_80p_CL_f[j], 'o', color='dodgerblue', markersize=30, markeredgecolor='k', zorder=5)
                errorbar(np.asarray(ph_fake_z[j])-t_exp, zlim_80p_CL_f[j], yerr=np.asarray(zlim_80p_CL_f[j])*0.15, fmt='o',capsize=10, lw=4, capthick=1, color='dodgerblue', uplims=True)

                plot(np.asarray(ph_fake_z[j])-t_exp, zlim_50p_CL_f[j], 'o', color='r', markersize=30, markeredgecolor='k', zorder=5)
                errorbar(np.asarray(ph_fake_z[j])-t_exp, zlim_50p_CL_f[j], yerr=np.asarray(zlim_50p_CL_f[j])*0.15, fmt='o',capsize=10, lw=4, capthick=1, color='r', uplims=True)

                plot(z_ps1_v3[:,1][i]-t_exp, z_ps1_v3[:,4][i], '*-', markersize=50,  zorder=7, color='silver', markeredgecolor='k' )
                errorbar(z_ps1_v3[:,1][i]-t_exp, z_ps1_v3[:,4][i], yerr=z_ps1_v3[:,5][i], fmt='o', markersize=1, capsize=0, lw=3, capthick=1, color='silver', zorder=5)
                
                if (zlim_50p_CL_f[j] <= z_ps1_v3[:,4][i]):
                    #print (z_ps1_v3[:,1][i], z_ps1_v3[:,1][i]-t_exp)
                    plot(z_ps1_v3[:,1][i]-t_exp, z_ps1_v3[:,4][i], '*-', markersize=50,  zorder=7, color='magenta', markeredgecolor='k' )
                    errorbar(z_ps1_v3[:,1][i]-t_exp, z_ps1_v3[:,4][i], yerr=z_ps1_v3[:,5][i], fmt='o', markersize=1, capsize=0, lw=3, capthick=1, color='magenta', zorder=5)
            
            
    legend(loc='upper left', fontsize=25, framealpha=1, ncol=2,markerscale=0.8)

    savefig(r'./FinalLCs/'+SNname+'_z_preSN.pdf', dpi = 100, bbox_inches='tight',pad_inches=0.2, facecolor='white')
    close()








    # band_string = 'griz'

    # for band_type in band_string:


    #     #background aperture files

    #     files = glob.glob(diff_path + '/' + band_type + '/*.diffimstats*')
    #     files.sort()

    #     ph_fake = []
    #     xp_fake = []
    #     yp_fake = []
    #     flx_fake = []
    #     eflx_fake = []

    #     for fname in files:
    #         f = np.genfromtxt(fname)
            
    #         date = '20'+str(fname[-88:-82])
    #         date = date[:4]+'-'+date[4:6]+'-'+date[6:]
    #         t = Time(str(date), format='isot', scale='utc')
    #         mjd = t.mjd
    #         ph_fake.append(mjd)
    #         xp_fake.append(f[:,0])
    #         yp_fake.append(f[:,1])
    #         flx_fake.append(f[:,2])
    #         eflx_fake.append(f[:,3])

    #     #add fake source fluxes to background aperture region fluxes and calculate the sigma
    #     ginj_sigma = []
    #     ginj_3sig = []
    #     ph_both = []
    #     for j in range(len(ph_fake)):
    #         sig = []
    #         for i in range(len(ph_FS)):
    #             for k in range(len(flx_FS_d[i])):

    #                 ginj = flx_FS_d[i][k]+flx_fake[j]

    #                 sig.append(np.asarray(ginj/(eflx_fake[j])))
    #     #             print ('FS flux:', flx_FS_d[i][k])
    #     #             print ('CL flux:', flx_fake[j])
    #     #             print ('CL err:', eflx_fake[j])
    #     #             print ('sig:', np.asarray(ginj/(eflx_fake[j])))

    #         ginj_sigma.append(np.asarray(sig))

    #     #find fraction of background regions that have 3sigma detections after fake sources are added 
    #     ginj_3sig = []
    #     ginj_3sigfrac = []
    #     for i in range(len(ginj_sigma)):
    #         sig3 = []
    #         s_frac = []
    #         for j in range(len(ginj_sigma[i])):
    #             fc = (ginj_sigma[i][j] >= 3).sum() / len(flx_fake[i])
    #             s_frac.append(fc)
    #             sigg = []
    #             fr = []
    #             for k in range(len(ginj_sigma[i][j])):

    #                 fr.append(fc)

    #             sigg = np.asarray(sigg)
    #             sig3.append(sigg)  
            
                
    #         ginj_3sig.append(sig3)
    #         ginj_3sigfrac.append(s_frac)
                
    #     ginj_3sig = np.asarray(ginj_3sig)
    #     ginj_3sigfrac = np.asarray(ginj_3sigfrac)

    #     #create recovery curves 

    #     idx = ginj_3sigfrac[0] >= 0.997
    #     mag_FS = np.asarray(mag_FS)
    #     mag_FS_d = np.asarray(mag_FS_d)


    #     lim_3sig_CL = []
    #     lim_3sig_FS = []
    #     lim_80p_CL = []
    #     lim_80p_FS = []
    #     lim_50p_CL = []
    #     lim_50p_FS = []
    #     lim_80p_CL_f = []
    #     lim_50p_CL_f = []
    #     for i in range(len(ginj_3sigfrac)):
            
    #         idx_3sig = (ginj_3sigfrac[i] >= 0.997)
    #         idx_80p = (ginj_3sigfrac[i] >= 0.80)
    #         idx_50p = (ginj_3sigfrac[i] >= 0.50)

    #         gcl_80p_lim = mag_FS[0][idx_80p][-1]
    #         gcl_50p_lim = mag_FS[0][idx_50p][-1]
    #         gcl_80p_lim_f = np.asarray(flx_FS[0])[idx_80p][-1]
    #         gcl_50p_lim_f = np.asarray(flx_FS[0])[idx_50p][-1]

    #         lim_80p_CL.append(gcl_80p_lim)
    #         lim_50p_CL.append(gcl_50p_lim)
    #         lim_80p_CL_f.append(gcl_80p_lim_f)
    #         lim_50p_CL_f.append(gcl_50p_lim_f)
            
    #         figure(figsize=(15,10))

    #         xlabel('Magnitude', fontsize=40)
    #         ylabel('Recovery Fraction', fontsize=40)
    #         xlim(18.,24)
    #         title('Phase: '+str("{:.1f}".format(ph_fake[i]-t_exp))+'; Limit: '+str("{:.1f}".format(gcl_80p_lim))+'(80$\%$), '+str("{:.1f}".format(gcl_50p_lim))+'(50$\%$)', fontsize=40)
    #         plot(mag_FS[0], ginj_3sigfrac[i], '-', lw=5, color='r',markersize=15, markeredgecolor='k')

    #         #legend(loc='lower left', fontsize=40, framealpha=1)

    #         savefig(r'./RecoveryCurves/'+band_type+'/'+str(ph_fake[i])+'_RC.png', dpi = 100, bbox_inches='tight',pad_inches=0.2, facecolor='white')

    #         g_ps1_v3[:,1], g_ps1_v3[:,4], g_ps1_v3[:,5] = zip(*sorted(zip(g_ps1_v3[:,1], g_ps1_v3[:,4], g_ps1_v3[:,5])))
    #         close()

    #     figure(figsize=(15,10))
    #     ylabel('Flux (DN)', fontsize=50)
    #     xlabel('Phase', fontsize=50)
    #     title('YSE g-band '+SNname, fontsize=30)

    #     #xlim(58860, 58920)
    #     ylim(-200, 1500)
    #     xlim(-250, 10)
    #     axhline(0, linestyle='--', c='k', lw=4)
    #     plot(np.asarray(ph_fake[0])-t_exp, lim_80p_CL_f[0], 'o', color='dodgerblue', markersize=30, markeredgecolor='k', zorder=5, label=r'Limit (80\%)')
    #     plot(np.asarray(ph_fake[0])-t_exp, lim_80p_CL_f[0], 'o', color='r', markersize=30, markeredgecolor='k', zorder=5, label=r'Limit (50\%)')
    #     plot(g_ps1_v3[:,1][0]-t_exp, g_ps1_v3[:,4][0], '*-', markersize=50,  zorder=7, color='silver', markeredgecolor='k', label=r'Pre-SN LC (Non-Detections)' )
    #     plot(g_ps1_v3[:,1][0]-t_exp, g_ps1_v3[:,4][0], '*-', markersize=50,  zorder=7, color='magenta', markeredgecolor='k', label=r'Pre-SN LC (Detections, 50\%)' )
                            
    #     for i in range(len(g_ps1_v3[:,1])):
    #         for j in range(len(ph_fake)):
    #             if np.abs(g_ps1_v3[:,1][i] - ph_fake[j]) < 0.9:
                    
    #                 #print ('same:', g_ps1_v3[:,1][i], g_ps1_v3[:,1][i]-t_exp)
                    
    #                 plot(np.asarray(ph_fake[j])-t_exp, lim_80p_CL_f[j], 'o', color='dodgerblue', markersize=30, markeredgecolor='k', zorder=5)
    #                 errorbar(np.asarray(ph_fake[j])-t_exp, lim_80p_CL_f[j], yerr=np.asarray(lim_80p_CL_f[j])*0.15, fmt='o',capsize=10, lw=4, capthick=1, color='dodgerblue', uplims=True)

    #                 plot(np.asarray(ph_fake[j])-t_exp, lim_50p_CL_f[j], 'o', color='r', markersize=30, markeredgecolor='k', zorder=5)
    #                 errorbar(np.asarray(ph_fake[j])-t_exp, lim_50p_CL_f[j], yerr=np.asarray(lim_50p_CL_f[j])*0.15, fmt='o',capsize=10, lw=4, capthick=1, color='r', uplims=True)

    #                 plot(g_ps1_v3[:,1][i]-t_exp, g_ps1_v3[:,4][i], '*-', markersize=50,  zorder=7, color='silver', markeredgecolor='k' )
    #                 errorbar(g_ps1_v3[:,1][i]-t_exp, g_ps1_v3[:,4][i], yerr=g_ps1_v3[:,5][i], fmt='o', markersize=1, capsize=0, lw=3, capthick=1, color='silver', zorder=5)
                    
    #                 if (lim_50p_CL_f[j] <= g_ps1_v3[:,4][i]):
    #                     #print (g_ps1_v3[:,1][i]-t_exp)
    #                     plot(g_ps1_v3[:,1][i]-t_exp, g_ps1_v3[:,4][i], '*-', markersize=50,  zorder=7, color='magenta', markeredgecolor='k' )
    #                     errorbar(g_ps1_v3[:,1][i]-t_exp, g_ps1_v3[:,4][i], yerr=g_ps1_v3[:,5][i], fmt='o', markersize=1, capsize=0, lw=3, capthick=1, color='magenta', zorder=5)
                
                
    #     legend(loc='upper left', fontsize=25, framealpha=1, ncol=2,markerscale=0.8)

    #     savefig(r'./FinalLCs/'+SNname+'_'+band_type+'_preSN.pdf', dpi = 100, bbox_inches='tight',pad_inches=0.2, facecolor='white')
    #     close()