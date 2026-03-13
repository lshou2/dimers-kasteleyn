#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions to make the dimer-dimer and vison-vison correlator plots
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
import scipy.stats


# triangle marker for matplotlib that is CENTERED
ctverts = [[-1.5/np.sqrt(3), -.5], [1.5/np.sqrt(3), -.5], [0,1], \
         [-1.5/np.sqrt(3), -.5], [0,1]]


def setfont():
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Times" #"Computer Modern Romand"
        })
    return 0


#%% square-octagon rectangular boundary

def plot_sqoct_dimer_paths_fflu_mc(pt=16,save=False,inset=True,verbose=False):
    '''with Monte-Carlo and inset'''
    n=100
    dps=200 # which files to load
    nametag = 'rect'
    pathlabels = ['octpath','squarepath','diagpath']
    dimerlist = []
    for i in range(3):
        filepath = './data/sqoct_%s_n100_fflu/mp_dimer_%s_n100_%s'\
            %(nametag,nametag,pathlabels[i])
        dimerlist.append(np.load(filepath+'_dps%i.npy'%dps))
        if pathlabels[i]=='squarepath': # load squarepath 2
            dimerlist.append(np.load(filepath+'_2_dps%i.npy'%dps))

    labellist = ['octagon path (FFLU)', 'square path 1 (FFLU)', \
                 'square path 2 (FFLU)', 'diag. oct. path (FFLU)']
    colorlist = ['blue','tab:orange','tab:red','tab:green']
    mwidths = [1.7,1.5,1,1]
    markerlist = ["8", "s", "d", ctverts]
    pathlengths = [n//2,n//2,n//2-1,n//2-1]
    alphalist = [1,1,.9,1]
    msizes = [6,6,6,8]
    
    # start plots
    fig, ax = plt.subplots()
    for i in range(4):
        if verbose:
            print('%i, length %i'%(i,len(dimerlist[i])))
        plt.plot(range(pathlengths[i]), np.abs(np.array(dimerlist[i])),\
                 marker=markerlist[i], linestyle='',\
                 markeredgewidth=mwidths[i],ms=msizes[i],\
             color=colorlist[i],fillstyle='none',label=labellist[i],alpha=alphalist[i])
    plt.yscale('log')
    plt.xlabel(r'Distance $\ell$',fontsize=pt)
    plt.ylabel(r'$|\langle d_0d_\ell\rangle-\langle d_0\rangle\langle d_\ell\rangle|$',\
               fontsize=pt)
    plt.xticks(fontsize=pt-1)
    plt.yticks(fontsize=pt-1)
    plt.title('Square-octagon dimer correlators, $R=%i$'%(n//2), fontsize=pt)
    
    ### inset
    if inset==True:
        itruncate = 8
        xsize = .37
        ysize = .4
        x1,x2,y1,y2 = -.5,itruncate-.5,10**-6,1 # actual values
        axins = ax.inset_axes([0.07,0.07, xsize, ysize], # x0, y0, xheight, yheight (locations in figure)
                                xlim = (x1,x2), ylim=(y1,y2))
                                #xticklabels=[],yticklabels=[])
        irange=range(4)
        mwidths = [1.2,1.2,1.2,1]
        alphalist = [1,1,1,1]
        msize = [8,7,8,10]
        # plot FFLU data
        for i in irange:
            axins.plot(range(pathlengths[i])[:itruncate], \
                       np.abs(np.array(dimerlist[i]))[:itruncate], marker=markerlist[i], \
                       markeredgewidth=mwidths[i],ms=msize[i],linestyle='',\
                         color=colorlist[i],fillstyle='none',label=labellist[i],alpha=alphalist[i])
        # plot Monte-Carlo data
        mcmsize = [5,4.5,5.5,6]
        mcmarkerlist = ["8", "s", "d", ctverts]
        mccolorlist = ['dodgerblue', 'tab:olive','deeppink','lime']
        mclabellist = ['oct. path (MC)', 'sq. path 1 (MC)', \
                       'sq. path 2 (MC)', 'diag. oct. (MC)']
        mcalpha = [1,.9,.8,.7]
        mcname = ['szsz_octagon','szsz_square','szsz_square','szsz_oct_diag']
        
        irange= [0,2,3]
        for i in irange:
            mcdata = np.genfromtxt('./data/montecarlo_data/%s.csv'%mcname[i],\
                                   delimiter=",", dtype=None, encoding="utf-8", skip_header=1)
            origin = 49
            mcdata2 = mcdata[origin:]
            axins.plot(range(len(mcdata2)),mcdata2,marker=mcmarkerlist[i],\
                       color=mccolorlist[i],markeredgewidth=0,linestyle='',\
                           ms=mcmsize[i],label=mclabellist[i], alpha=mcalpha[i])
            
            if i == 2: # square -> plot second square path too
                mcdata_sq1 = mcdata[:origin+1][::-1]
                axins.plot(range(len(mcdata_sq1)),mcdata_sq1,mcmarkerlist[1],\
                           color=mccolorlist[1],markeredgewidth=0,\
                               ms=mcmsize[1],label=mclabellist[1], alpha=mcalpha[1])
            savename = 'sqoct_dimer4_n%i_inset.pdf'%n

        axins.set_yscale('log')
        axins.tick_params(axis="y", pad=-.5) # remove padding between tick labels and inset
        axins.tick_params(axis="x", pad=1)
        axins.xaxis.set_ticks(np.arange(0,itruncate,2))
        #axins.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.indicate_inset_zoom(axins, edgecolor='black')
    
        plt.legend(fontsize=pt-2,labelspacing=0., handletextpad=0, borderpad=0.1,\
               handlelength=1, loc='upper right')
            
        # inset labels for legend
        insetlegend = []
        for i in range(4):
            insetlegend.append(mpl.lines.Line2D([],[], color=mccolorlist[i], \
                                   marker=mcmarkerlist[i],linestyle='None', \
                                     markersize=mcmsize[i], label=mclabellist[i],\
                                   alpha=mcalpha[i], markeredgewidth=0))
        axins.legend(handles=insetlegend,fontsize=pt-3,labelspacing=0., handletextpad=0,\
                     borderpad=0.1,handlelength=1, loc='lower left', bbox_to_anchor=(.97,-.2))

    else:
        plt.legend(fontsize=pt-2,labelspacing=0.2, handletextpad=0.4, borderpad=.2)
        savename = 'sqoct_dimer4_n%i.pdf'%n
    
    if save == True:
        plt.savefig(savename, bbox_inches='tight')

    plt.show()
    
    # get inverse correlation lengths
    slopelist = []
    startslopes = [0,20,21,25,29,30]
    end = 47
    for start in startslopes:
        cur_slopes = []
        for i in range(4):
            cur_slopes.append(scipy.stats.linregress(range(pathlengths[i])[start:end], \
                                    np.log(np.abs(np.array(dimerlist[i][start:end]))))[0])
        slopelist.append(cur_slopes)
        print('slopes from l=%i to %i: %s'%(start,end-1,cur_slopes))
    
    return 0


def plot_sqoct_vison_paths_fflu_mc(pt=20,save=False,mccutoff=2*10**-5, \
                                   verbose=False):
    '''
    Plot square path vison correlators, with Monte-Carlo overlay
    '''
    n=100
    dps=200 # which files to load
    nametag = 'rect'
    pathlabels = ['octpath','squarepath','diagpath']
    # load vison data
    visonlist = []
    for i in range(3):
        filepath = './data/sqoct_%s_n100_fflu/mp_visons_%s_n100_%s'\
            %(nametag,nametag,pathlabels[i])
        visonlist.append(np.load(filepath+'_dps%i.npy'%dps))
        if pathlabels[i]=='squarepath': # load squarepath 2
            visonlist.append(np.load(filepath+'_2_mcmatch_dps201.npy'))

    # plot infinite limit
    limit_val = 0.774596669
    plt.axhline(y=limit_val, color='gray', label='inf. PBC limit', linewidth=1)

    # plot FFLU
    labellist = ['octagon path', 'sq. path 1 (FFLU)', 'sq. path 2 (FFLU)', 'diag. oct. path']
    colorlist = ['blue','tab:orange','tab:red','tab:green']
    mwidths = [1.7,2,1.5,1]
    markerlist = ["8", "s", "d", "^"]
    pathlengths = [n//2-1,n//2-1,n//2,n//2-1]
    alphalist = [1,1,1,.8]
    msizes = [6]*4
    
    irange = [1,2]
    for i in irange:
        if verbose:
            print('%i, vison length %i'%(i,len(visonlist[i])))
        plt.plot(range(1,pathlengths[i]+1), np.abs(np.array(visonlist[i])), markerlist[i], \
                 markeredgewidth=mwidths[i],ms=msizes[i],\
             color=colorlist[i],fillstyle='none',label=labellist[i],alpha=alphalist[i])
    
    # load Monte-Carlo
    if verbose:
        print('loading Monte-Carlo data')
    mcname = ['vis_oct.csv','vis_sq2.csv','vis_sq1.csv','vis_oct_diag.csv']
    mcvisonlist = []
    for i in range(4):
        mcvisonlist.append(np.genfromtxt('./data/montecarlo_data/%s'%mcname[i],\
                           delimiter=",", dtype=None, encoding="utf-8", skip_header=1))
        
    # plot Monte-Carlo data
    mcmsize = [3]*4
    mcmarkerlist = ["8", "s", "d", "^"]
    mccolorlist = ['dodgerblue', 'tab:olive','deeppink','lime']
    mclabellist = ['octagon path (MC)', 'sq. path 1 (MC)', \
                   'sq. path 2 (MC)', 'diag. oct. path (MC)']
    mcalpha = [.6,1,1,.6]
    for i in irange:
        v_cur = mcvisonlist[i][:pathlengths[i]]
        if verbose:
            print('%i, vison length %i'%(i,len(v_cur)))
        v_cplot = np.ma.masked_less(v_cur, mccutoff)
        plt.plot(range(1,pathlengths[i]+1), v_cplot, mcmarkerlist[i], \
             color=mccolorlist[i],label=mclabellist[i],\
                 alpha=mcalpha[i],ms=mcmsize[i],markeredgewidth=0)
    
    plt.yscale('log')
    plt.xlabel(r'Face distance $\ell$',fontsize=pt)
    plt.ylabel(r'$|\langle v_0v_\ell\rangle|$',\
               fontsize=pt)
    plt.xticks(fontsize=pt-1)
    plt.yticks(fontsize=pt-1)
    #plt.title('Square-octagon vison correlators, $R=%i$'%(n//2), fontsize=pt)
    plt.title(' ',fontsize=pt)
    
    # reorder the labels in the legend, and make Monte-Carlo legend handles larger
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [1,2,3,4,0]
    mchandles = [Line2D([0], [0], marker=mcmarkerlist[i], color=mccolorlist[i], \
                      ms=mcmsize[i]+1, linestyle='') for i in irange]
    plt.legend([handles[1], handles[2], mchandles[0], mchandles[1], handles[0]],\
               [labels[i] for i in order],\
               fontsize=pt-3, labelspacing=0.2, handletextpad=0.4, borderpad=.2,\
                   ncol=1,handlelength=1)
    
    if save == True:
        plt.savefig('sqoct_vison_square_both_n%i.pdf'%n, bbox_inches='tight')

    plt.show()

    # get inverse correlation lengths for square path 1
    if 1 in irange:
        print('square path 1 (orange square) slopes')
        startslopes = [0,5,10,20,30]
        end = 41
        for start in startslopes:
            cslope = scipy.stats.linregress(range(pathlengths[1])[start:end], \
                                        np.log(np.abs(np.array(visonlist[1][start:end]))))[0]
            print('slope from l=%i to %i inclusive: %s'%(start,end-1,cslope))


    return 0



def plot_sqoct_octpath_mc(ival, pt=20,save=False,cverbose=False,\
                          verbose=False):
    '''
    Make vison correlator plot for sqoct rectangular bdy conditions along 
    either octagon path (ival=0) or diagonal octagon path (ival=3)
    cverbose: whether to print condition number or not
    '''
    assert ival in [0,3], 'receieved ival=%i, should 0 or 3'%ival
    
    # load FFLU vison data
    n = 100
    dps = 200 # which files to load
    nametag = 'rect'
    pathlabels = ['octpath','squarepath','squarepath','diagpath']
    filepath = './data/sqoct_%s_n100_fflu/mp_visons_%s_n100_%s'\
            %(nametag,nametag,pathlabels[ival])
    v=np.load(filepath+'_dps%i.npy'%dps)
    if verbose:
        print(len(v))
    
    limit_val = 0.774596669
    plt.axhline(y=limit_val, color='gray', label='inf. PBC limit 0.7745966...',\
                linewidth=1)
    
    # plot FFLU data
    pathlength = n//2-1
    labellist = ['octagon path (FFLU)', 'square path 1 (FFLU)', \
                 'square path 2 (FFLU)', 'diag. oct. path (FFLU)']
    colorlist = ['blue','tab:orange','tab:red','tab:green']
    mwidths = [1.3,1.5,1,1.3]
    msizes = [6,6,6,8]
    
    markerlist = ["8", "s", "d", ctverts]
    plt.plot(range(1,pathlength+1), np.abs(np.array(v)), marker=markerlist[ival],\
             markeredgewidth=mwidths[ival],ms=msizes[ival],linestyle='',\
             fillstyle='none',label=labellist[ival],color=colorlist[ival])
    
    # Monte-Carlo data
    # load data
    mcname = ['vis_oct.csv','vis_sq1.csv','vis_sq2.csv','vis_oct_diag.csv']
    v_mc = np.genfromtxt('./data/montecarlo_data/%s'%mcname[ival],\
                    delimiter=",", dtype=None, encoding="utf-8", skip_header=1)
    if verbose:
        print(len(v_mc))
    # plot Monte-Carlo data
    mcmsize = [3,3,3,4]
    mcmarkerlist = ["8", "s", "d", ctverts]
    mccolorlist = ['dodgerblue', 'tab:olive','deeppink','lime']
    mclabellist = ['octagon path (MC)', 'sq. path 1 (MC)', \
                   'sq. path 2 (MC)', 'diag. oct. path (MC)']
    mcalpha = [1,1,1,1]
    mcname = ['szsz_octagon','szsz_square','szsz_square','szsz_oct_diag']
    plt.plot(range(1,pathlength+1), v_mc[:pathlength], marker=mcmarkerlist[ival], \
             color=mccolorlist[ival],label=mclabellist[ival], linestyle='',\
                 alpha=mcalpha[ival],ms=mcmsize[ival],markeredgewidth=0)
    
    #plt.yscale('log')
    plt.xlabel(r'Face distance $\ell$',fontsize=pt)
    plt.ylabel(r'$|\langle v_0v_\ell\rangle|$',\
               fontsize=pt)
    plt.xticks(fontsize=pt-1)
    plt.yticks(fontsize=pt-1)
    #plt.title('Square-octagon vison correlator, $R=%i$'%(n//2), fontsize=pt)
    plt.title(' ',fontsize=pt)
    
    # reorder the labels in the legend
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [1,2,0]
    # make Monte-Carlo legend handles larger
    mchandle = Line2D([0], [0], marker=mcmarkerlist[ival], color=mccolorlist[ival], \
                      ms=mcmsize[ival]+1, linestyle='')
    plt.legend([handles[1], mchandle, handles[0]],[labels[i] for i in order],\
               fontsize=pt-2, labelspacing=0.2, handletextpad=0.4, borderpad=.2,\
                  loc='lower center',markerscale=1, handlelength=1)
    
    if save == True:
        plt.savefig('sqoct_vison_octagonpath_both_i%i_n%i.pdf'%(ival,n), \
                    bbox_inches='tight')

    plt.show()

    return 0




#%% square-octagon fortress

def plot_sqoctfortress_dimer_paths_fflu(irange=range(1,3),pt=18,save=False, \
                            truncate=100,ylabel=True,title='',mwidths=[],\
                                inset=False, verbose=False):
    n=200
    dps=200 # which files to load
    nametag = 'fortress'
    pathlabels = ['octpath','squarepath','diagpath']
    dimerlist = []
    
    # load all the saved data
    for i in range(3):
        filepath = './data/sqoct_%s_n%i_fflu/mp_dimer_%s_n%i_%s'\
            %(nametag,n,nametag,n,pathlabels[i])
        dimerlist.append(np.load(filepath+'_dps%i.npy'%dps))
        if pathlabels[i]=='squarepath': # load squarepath 2
            dimerlist.append(np.load(filepath+'_2_dps%i.npy'%dps))

    labellist = ['octagon path', 'square path 1', 'square path 2', 'diagonal octagon path']
    if inset == True:
        labellist[-1] = 'diag. oct. path'
    colorlist = ['blue','tab:orange','tab:red','tab:green']
    if len(mwidths) == 0:
        mwidths = [1.7,1.5,1.5,1.5]
    markerlist = ["8", "s", "d", ctverts]
    pathlengths = [n//2-1,n//2-1,n//2-2,n//4]
    alphalist = [1,1,.8,1]
    
    # start plots
    fig, ax = plt.subplots()
    for i in irange:
        if verbose:
            print('plotting %s'%labellist[i])
        plt.plot(range(pathlengths[i])[:truncate], np.abs(np.array(dimerlist[i]))[:truncate], \
                 marker = markerlist[i], linestyle='',\
                 markeredgewidth=mwidths[i],\
             color=colorlist[i],fillstyle='none',label=labellist[i],alpha=alphalist[i])
    
    phase_bdy1 = 1/np.sqrt(5) / 2. * 100
    phase_bdy2 = 3/np.sqrt(5) / 2. * 100
    dphase_bdy1 = 3/5 / 2. * 50
    dphase_bdy2 = 50
    if 0 in irange or  1 in irange or 2 in irange:
        plt.axvline(x=phase_bdy1, color='gray', linestyle='dashed')
        if phase_bdy2 <= truncate:
            plt.axvline(x=phase_bdy2, color='gray', linestyle='dashed')
    if 3 in irange:
        plt.axvline(x=dphase_bdy1, color='gray', linestyle='dashed')
        plt.axvline(x=dphase_bdy2, color='gray', linestyle='dashed')

    plt.yscale('log')

    plt.xlabel(r'Distance $\ell$',fontsize=pt)
    if ylabel:
        plt.ylabel(r'$|\langle d_0d_\ell\rangle-\langle d_0\rangle\langle d_\ell\rangle|$',\
                   fontsize=pt)
    plt.xticks(fontsize=pt-1)
    plt.yticks(fontsize=pt-1)
    
    plt.title(title, fontsize=pt)
    
    ### inset
    if inset==True:
        itruncate = 25
        xsize = .52
        ysize = .52
        x1,x2,y1,y2 = -1,itruncate,10**-11,1 # actual values
        axins = ax.inset_axes([0.1,0.1, xsize, ysize], # x0, y0, xheight, yheight (locations in figure)
                                xlim = (x1,x2), ylim=(y1,y2))
                                #xticklabels=[],yticklabels=[])
        for i in irange:
            axins.plot(range(pathlengths[i])[:itruncate], \
                       np.abs(np.array(dimerlist[i]))[:itruncate], marker=markerlist[i], \
                       markeredgewidth=mwidths[i], linestyle='',\
                         color=colorlist[i],fillstyle='none',label=labellist[i],alpha=alphalist[i])
        axins.axvline(x=phase_bdy1, color='gray', linestyle='dashed')
        axins.set_yscale('log')
        ax.indicate_inset_zoom(axins, edgecolor='black')
    
        plt.legend(fontsize=pt-2,labelspacing=0., handletextpad=0, borderpad=0.1,\
               handlelength=1, loc='upper right')
        savename = 'sqoct_fortress_dimer%i_n%i_%i_inset.pdf'%(max(irange),n,truncate)
    else:
        plt.legend(fontsize=pt-2,labelspacing=0.2, handletextpad=0.4, borderpad=.2)
        savename = 'sqoct_fortress_dimer%i_n%i_%i.pdf'%(max(irange),n,truncate)
    
    if save == True:
        print('saving as %s'%savename)
        plt.savefig(savename, bbox_inches='tight')

    plt.show()
    
    # get inverse correlation lengths
    slopelist = []
    startslopes = [0,2,3,5,8]
    end = 17
    for start in startslopes:
        cur_slopes = []
        for i in irange:
            if i == 3: # diagonal octagon path
                end = 10
            cur_slopes.append(scipy.stats.linregress(range(pathlengths[i])[start:end], \
                                    np.log(np.abs(np.array(dimerlist[i][start:end]))))[0])
        slopelist.append(cur_slopes)
        print('slopes from l=%i to %i inclusive: %s'%(start,end-1,cur_slopes))
    

def plot_sqoctfortress_vison_paths_fflu(irange=range(1,3),pt=18,save=False,\
                        log=True, truncate=98,inflimit=True,legendloc='',\
                        shortlabel=False, ylabel=True, title='', inset=False):
    '''
    Creates and saves vison plot.

    Parameters
    ----------
    irange : a subset of [0,1,2,3] indicating which paths to plot
    pt : fontsize
    save : whether to save the image (pdf)
    log : y-axis scaling
    truncate : max \ell value
    inflimit : whether to plot the horizontal line infinite limit or not
    legendloc : optional to specify where the legend should go. 
                loc=legendloc is called when creating the legend
    shortlabel : whether to shorten 'infinite PBC limit' to 'inf. PBC limit'
    ylabel : whether to label the y-axis with text or not
    title : to give an alternate title

    '''
    n=200
    dps=200 # which files to load
    nametag = 'fortress'
    pathlabels = ['octpath','squarepath','diagpath']
    visonlist = []
    for i in range(3):
        filepath = './data/sqoct_%s_n%i_fflu/mp_visons_%s_n%i_%s'\
            %(nametag,n,nametag,n,pathlabels[i])
        if i==2: # diag path -> the data goes to l=50 but we stop at 49
            visonlist.append(np.load(filepath+'_dps%i.npy'%dps)[:-1])
        else:
            visonlist.append(np.load(filepath+'_dps%i.npy'%dps))
        if pathlabels[i]=='squarepath': # load squarepath 2
            visonlist.append(np.load(filepath+'_2_dps%i.npy'%dps))

    labellist = ['octagon path', 'square path 1', 'square path 2', 'diagonal octagon path']
    colorlist = ['blue','tab:orange','tab:red','tab:green']
    mwidths = [1.7,1.5,1.5,1.5]
    markerlist = ["8", "s", "d", ctverts]
    pathlengths = [n//2-2,n//2-2,n//2-3,n//4-1]
    alphalist = [1,1,1,1]
    
    # start plotting
    fig, ax = plt.subplots()

    limit_val = 0.774596669
    if inflimit:
        labelname = 'infinite PBC limit' if not shortlabel else 'inf. PBC limit'
        plt.axhline(y=limit_val, color='gray', label=labelname)
    
    for i in irange:
        #print(i)
        plt.plot(range(1,pathlengths[i]+1)[:truncate], np.abs(np.array(visonlist[i]))[:truncate], \
                 marker=markerlist[i], linestyle='', \
                 markeredgewidth=mwidths[i],\
             color=colorlist[i],fillstyle='none',label=labellist[i],alpha=alphalist[i])
    

    
    phase_bdy1 = 1/np.sqrt(5) / 2. * 100
    phase_bdy2 = 3/np.sqrt(5) / 2. * 100
    dphase_bdy1 = 3/5 / 2. * 50
    dphase_bdy2 = 50
    if 0 in irange or  1 in irange or 2 in irange:
        plt.axvline(x=phase_bdy1, color='gray', linestyle='dashed')
        if phase_bdy2 <= truncate:
            plt.axvline(x=phase_bdy2, color='gray', linestyle='dashed')
    if 3 in irange:
        plt.axvline(x=dphase_bdy1, color='gray', linestyle='dashed')
        plt.axvline(x=dphase_bdy2, color='gray', linestyle='dashed')
    
    if log:
        plt.yscale('log')
    
    plt.xlabel(r'Face distance $\ell$',fontsize=pt)
    if ylabel:
        plt.ylabel(r'$|\langle v_0v_\ell\rangle|$',fontsize=pt)
    plt.xticks(fontsize=pt-1)
    plt.yticks(fontsize=pt-1)
    plural = 'correlators' if len(irange)>=2 else 'correlator'
    
    if title == '':
        title = 'Square-octagon fortress vison %s, $R=%i$'%(plural,n//2)
    plt.title(title, fontsize=pt)
    
    ### inset (really only for octagon path, no log scale)
    if inset==True:
        itruncate = 24
        xsize = .55
        ysize = .45
        x1,x2,y1,y2 = 0,itruncate+1,.67,.8 # actual values
        axins = ax.inset_axes([0.43,0.53, xsize, ysize], # x0, y0, xheight, yheight (locations in figure)
                                xlim = (x1,x2), ylim=(y1,y2))
                                #xticklabels=[],yticklabels=[])
        axins.axhline(y=limit_val, color='gray', label=labelname)
        for i in irange:
            axins.plot(range(1,pathlengths[i]+1)[:itruncate], \
                       np.abs(np.array(visonlist[i]))[:itruncate], marker=markerlist[i], \
                       markeredgewidth=mwidths[i], linestyle='',\
                         color=colorlist[i],fillstyle='none',label=labellist[i],alpha=alphalist[i])
        axins.axvline(x=phase_bdy1, color='gray', linestyle='dashed')
        ax.indicate_inset_zoom(axins, edgecolor='black')
    
        ax.legend(fontsize=pt-3,labelspacing=0.2, handletextpad=0.4,\
                  handlelength=1, borderpad=.2,bbox_to_anchor=(.61,.23))
        savename = 'sqoct_fortress_vison%i_n%i_%i_inset.pdf'%(max(irange),n,truncate)
    else:
        savename = 'sqoct_fortress_vison%i_n%i_%i.pdf'%(max(irange),n,truncate)    
    
        if legendloc != '':
            plt.legend(fontsize=pt-2,labelspacing=0.2, handletextpad=0.4,\
                       borderpad=.2, handlelength=1, loc=legendloc)
        else:
            plt.legend(fontsize=pt-2,labelspacing=0.2, handletextpad=0.4,\
                       borderpad=.2,handlelength=1)
    
    if save == True:
        print('saving as %s'%savename)
        plt.savefig(savename, bbox_inches='tight')

    plt.show()
    
    # get inverse correlation lengths for square path 1
    if 1 in irange:
        print('square path 1 (orange square) slopes')
        startslopes = [0,2,3,5,8]
        end = 13
        for start in startslopes:
            cslope = scipy.stats.linregress(range(pathlengths[1])[start:end], \
                                        np.log(np.abs(np.array(visonlist[1][start:end]))))[0]
            print('slope from l=%i to %i inclusive: %s'%(start,end-1,cslope))

    return 0

    