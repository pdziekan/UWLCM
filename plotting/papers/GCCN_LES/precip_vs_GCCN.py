from matplotlib import rc
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from scipy.optimize import curve_fit


import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../Matplotlib_common/")
from read_UWLCM_arrays import read_my_var
from latex_labels import labeldict

def power_law(x, a, b):
  return b*(x**(1./a) + 1)

def exp_law(x, a, p_0, p_max):
  return (p_max-p_0)*(1-np.exp(-a*x)) + p_0
  #return (1-np.exp(-a*x)) + b

# activate latex text rendering
rc('text', usetex=True)

series_from_it = int(sys.argv[1])
series_to_it = int(sys.argv[2])
profs_from_it = int(sys.argv[3])
profs_to_it = int(sys.argv[4])
qlimit = float(sys.argv[5])

varlabels = ["{\it clean}", "{\it standard}", "{\it polluted}"]
averaging_period = float(profs_to_it - profs_from_it) / 3600. # period over which series are averaged [h]; NOTE: we assume that series_from(to)_it = profs_from(to)_it / outfreq!

# assumed initial GCCN concentrations
GCCN_conc = [0,0.2817,5*0.2817,10*0.2817]

# init the plot
nplotx = 2 #int(nplots/6 + 0.5)
fig, axarr = plt.subplots(2, nplotx )

#prepare a list of output files, assuming the following order: prsitine, standard, polluted, for each 4 results: no GCCN, GCCN ,GCCNx5, GCCNx10
series_file_names = []
profs_file_names = []
file_no = np.arange(6, len(sys.argv)-1 , 1)
#print file_no
for no in file_no:
  series_file_names.append(sys.argv[no] + "series.dat")
  profs_file_names.append(sys.argv[no] + "profiles_"+str(profs_from_it)+"_"+str(profs_to_it)+".dat")

assert(len(series_file_names) == 12)

label_counter=0

for it in np.arange(12):
  if(it % 4 == 0):
    mean_surf_precip = []
    tot_acc_surf_precip = []
    tot_acc_surf_precip_std_dev = []
    prflux = []
    prflux_std_dev = []

  series_infile = open(series_file_names[it], "r")
  profs_infile = open(profs_file_names[it], "r")
#  print series_file_names[it]
#  print profs_file_names[it]

# calc mean surf precip
  surf_precip = read_my_var(series_infile, "surf_precip")
  mean_surf_precip.append(np.mean(surf_precip[series_from_it:series_to_it]))
# calc acc surf precip
  acc_surf_precip = read_my_var(series_infile, "acc_precip")
  acc_surf_precip_std_dev = read_my_var(series_infile, "acc_precip_std_dev")
  tot_acc_surf_precip.append(acc_surf_precip[series_to_it] - acc_surf_precip[series_from_it])
  tot_acc_surf_precip_std_dev.append(acc_surf_precip_std_dev[series_to_it] + acc_surf_precip_std_dev[series_from_it])

# find cloud base
  ql = read_my_var(profs_infile, "rliq")
  clbase = np.argmax(ql>qlimit)
#  print ql
#  print clbase
# --- get prflux at cloud base; divide by cloud fraction at this height to get an estimate of average over cloud cells only ---
#  clfrac_at_cbase = read_my_var(profs_infile, "clfrac")[clbase]
#  print clfrac_at_cbase
#  prflux_at_cbase = read_my_var(profs_infile, "prflux")[clbase]
##  print prflux_at_cbase
#  prflux.append(read_my_var(profs_infile, "prflux")[clbase] / read_my_var(profs_infile, "clfrac")[clbase])
#  prflux_std_dev.append(read_my_var(profs_infile, "prflux_std_dev")[clbase] / read_my_var(profs_infile, "clfrac")[clbase])
# --- get prflux at cloud base height ---
  prflux.append(read_my_var(profs_infile, "prflux")[clbase])
  prflux_std_dev.append(read_my_var(profs_infile, "prflux_std_dev")[clbase])
# --- get prflux at cloud base from the prflux vs cloud height profile ---
#  clb_prflux = read_my_var(profs_infile, "base_prflux_vs_clhght")
#  clb_prflux_std_dev = np.nan_to_num(read_my_var(profs_infile, "base_prflux_vs_clhght_std_dev"))
#  clb_prflux_occur = read_my_var(profs_infile, "base_prflux_vs_clhght number of occurances")
#
#  prflux.append(np.sum(clb_prflux * clb_prflux_occur) / np.sum(clb_prflux_occur))
#  prflux_std_dev.append(np.sum(clb_prflux_std_dev * clb_prflux_occur) / np.sum(clb_prflux_occur))


  if((it+1) % 4 == 0):
   # tot_acc_surf_precip_std_dev = [3 * x for x in tot_acc_surf_precip_std_dev] # we show errors bars with 3 std dev
    tot_acc_surf_precip = [(24. / averaging_period) * x for x in tot_acc_surf_precip] # turn into mm / day
    tot_acc_surf_precip_std_dev = [(24. / averaging_period) * x for x in tot_acc_surf_precip_std_dev] # turn into mm / day
    #print tot_acc_surf_precip
    #print prflux
    #axarr[0].plot(GCCN_conc, mean_surf_precip, 'o')
    axarr[0,0].errorbar(GCCN_conc, tot_acc_surf_precip, yerr = tot_acc_surf_precip_std_dev, marker='o', fmt='.', label = varlabels[(it)/4])
    axarr[0,1].errorbar(GCCN_conc, prflux, yerr = prflux_std_dev, marker='o', fmt='.')

    axarr[1,0].set_yscale('log')
    axarr[1,1].set_yscale('log')
    for i in [0,1]:
      axarr[i,0].errorbar(GCCN_conc, tot_acc_surf_precip, yerr = tot_acc_surf_precip_std_dev, marker='o', fmt='.')
      axarr[i,1].errorbar(GCCN_conc, prflux, yerr = prflux_std_dev, marker='o', fmt='.')

    # fit to prflux
    exp_law_bound = lambda GCCN_conc, expo, p_max: exp_law(GCCN_conc, expo, prflux[0], p_max)
    popt, pcov = curve_fit(exp_law_bound, GCCN_conc, prflux, bounds=([0.1,5], [2,150]))
    print '---- base prflux fit ----'
    print 'wykladnik i p_max: ', popt
    print 'p_0: ', prflux[0]
    #print 'covariance matrix: ', pcov

    line_pts = np.arange(0.00001,3,3./100)
    axarr[0,1].plot(line_pts, exp_law_bound(line_pts, *popt))
    axarr[1,1].plot(line_pts, exp_law_bound(line_pts, *popt))

    # fit to surf precip
    exp_law_bound = lambda GCCN_conc, expo, p_max: exp_law(GCCN_conc, expo, tot_acc_surf_precip[0], p_max)
    popt, pcov = curve_fit(exp_law_bound, GCCN_conc, tot_acc_surf_precip, bounds=([0.01,0], [2,5]))
    print '---- surf precip fit ----'
    print 'wykladnik i surf_max: ', popt
    print 'surf_0: ', tot_acc_surf_precip[0]
    #print 'covariance matrix: ', pcov

    line_pts = np.arange(0.00001,3,3./100)
    axarr[0,0].plot(line_pts, exp_law_bound(line_pts, *popt))
    axarr[1,0].plot(line_pts, exp_law_bound(line_pts, *popt))

for i in [0,1]:
  axarr[i,0].set_xlabel('GCCN concentration [cm$^{-3}$]')
  axarr[i,0].set_ylabel('surface precipitation rate [mm/day]')
  axarr[i,1].set_xlabel('GCCN concentration [cm$^{-3}$]')
  axarr[i,1].set_ylabel('precipitation flux at cloud base [W/m$^{2}$]')


# legend font size
#plt.rcParams.update({'font.size': 10})

# hide axes on empty plots
#if nplots % nploty == 0:
#  nemptyplots=0
#else:
#  nemptyplots = nploty - nplots % nploty
#emptyplots = np.arange(nploty - nemptyplots, nploty)
#for empty in emptyplots:
#  axarr[nplotx-1, empty].axis('off')

#axes = plt.gca()
#axes.tick_params(direction='in')
x_arr = np.arange(nplotx)
for i in [0,1]:
  for x in x_arr:
    #tics inside
    axarr[x,i].tick_params(direction='in', which='both', top=1, right=1)
    #minor tics
    axarr[x,i].xaxis.set_minor_locator(AutoMinorLocator())
    axarr[x,i].yaxis.set_minor_locator(AutoMinorLocator())
    #labels and tics font size
    for item in ([axarr[x,i].xaxis.label, axarr[x,i].yaxis.label] + axarr[x,i].get_xticklabels() + axarr[x,i].get_yticklabels()):
      item.set_fontsize(10)
    # subplot numbering
    axarr[x,i].text(0.5, 0.95, labeldict[2*x+i], fontsize=10, transform=axarr[x,i].transAxes)

## show legends
#for x in np.arange(nplotx):
#  for y in np.arange(nploty):
#    axarr[x].legend(loc="upper center")

#single legend for the whole figure
handles, labels = axarr[0,0].get_legend_handles_labels()
lgd = fig.legend(handles, labels, handlelength=4, loc='lower center', bbox_to_anchor=(0.45,0))


#figure size
#fig.set_size_inches(7.2, 4)# 5.214)#20.75,13.74)
fig.set_size_inches(7.2, 2*5)# 5.214)#20.75,13.74)
#distances between subplots and from bottom of the plot
fig.subplots_adjust(bottom=0.14 + (len(labels) - 2) * 0.10, hspace=0.25, wspace=0.4)

#fig.tight_layout(pad=0, w_pad=0, h_pad=0)

#figure size
#fig.set_size_inches(7.874, 6 + (len(labels) - 2) * 0.2)# 5.214)#20.75,13.74)
#distances between subplots and from bottom of the plot
#fig.subplots_adjust(bottom=0.15 + (len(labels) - 2) * 0.02, hspace=0.25)


#plt.show()
fig.savefig(sys.argv[len(sys.argv)-1], bbox_inches='tight', dpi=300)#, bbox_extra_artists=(lgd,))

