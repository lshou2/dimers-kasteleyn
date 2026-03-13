#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Functions involving K^{-1}. Generally only standard floating point functions here.
'''

import numpy as np
import time
import scipy as sp
import scipy.sparse as sps
import mpmath as mp
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patheffects as path_effects
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.collections import LineCollection


from functions.kasteleyn_shapes import kasteleyn_shape, is_it_squareoctagon, \
    get_default_string, sqoct_vert_string_bylength, plot_shape,\
        realoct_coords, get_dimerlist_matrixcoords



#%% Kasteleyn matrix functions

def kinv_lu_sparse(n, kmatsparse, shape_coord, string=None, squares=False,squarestart=1,\
                   pathlength='',  verbose=True, bwmatcoords=''):
    ''' 
    Returns the submatrix of the inverse Kasteleyn matrix K^{-1} corresponding
    to the dimers/vertices in 'string', using sparse floating point LU 
    (scipy.sparse.linalg.splu) to solve for the inverse matrix elements
    
    Input:
        n: system size (diameter)
        kmatsparse: the Kasteleyn matrix in a scipy sparse format, prefably csc
        shape_coord: the shape coordinate function (aztec_coord, square_coord, etc.)
        
        string (optional): list of dimers using lattice coordinates. string should use the
            the format: [[(i1,j1),(i2,j2)], [(k1,l1),(k2,l2)], ...]
            (recall: first coordinate of pair (i1,j1) = i coordinate = vertical/y coordinate,
                     second coordinate of pair (i1,j1) = j coordinate = horizontal/x coordinate)
            If no string is specified, a `default_string' will be chosen using
            'get_default_string'
        squares (optional): If 'string' is not specified, this is used to
                            generate the default string (ignored if 'string' is specified)
        squarestart (optional): If 'string' is not specified, this is used to
                                generate the default string (ignored if 'string' is specified)
        pathlength (optional): If 'string' is not specified, this is used to
                                generate the default string (ignored if 'string' is specified)            
        bwmatcoords (optional): tuple of lists 
            (matrix indices of black vertices in string, matrix indices of 
             white vertices in string)
            If 'bwmatcoords' is specified, this overrides all optional parameters 
            concerning 'string', and the function returns the submatrix of K^{-1}
            corresponding to the coordinates in 'bwmatcoords'
    
    Output:
        Submatrix of K^{-1} as numpy array.

    '''
    ksize = np.shape(kmatsparse)[0]
    
    if len(bwmatcoords) == 0:
        if string is None:
            string = get_default_string(n, shape_coord, squares,\
                                        pathlength=pathlength, squarestart=squarestart)
            print('Using default string/path')
            if verbose==True:
                print(string)
        
        bcoords, wcoords = get_dimerlist_matrixcoords(n, shape_coord, string)
        
    else:
        bcoords, wcoords = bwmatcoords
    
    Asize = len(bcoords)
    
    t0 = time.time()
    lusolver = sps.linalg.splu(kmatsparse.tocsc())
    t1 = time.time()
    print("Computed sparse LU factorization in %.2f seconds"% (t1-t0))
    
    kinvsmall = np.zeros((Asize,Asize),dtype=float)
    for index in range(Asize):
        standard_basis_v = np.transpose(np.eye(1, ksize, bcoords[index]))
        kinvsmall[:,index] = (lusolver.solve(standard_basis_v))[wcoords,0]
    
    return kinvsmall


def shrink(n,shape_coord, k, kinv, string=None, squares=False, plot=False, \
           verbose=False, pathlength=''):
    '''
    Return submatrices of K and K^{-1} corresponding to dimers/vertices in 
    'string' (list of dimers in lattice coordinates)
    '''
    
    if string is None:
        string = get_default_string(n, shape_coord, squares, pathlength)
        print('Using default string/path')
        if verbose==True:
            print(string)
        
    if plot==True:
        plot_shape(n, shape_coord, string,numbering=True)
        
    bcoords, wcoords = get_dimerlist_matrixcoords(n, shape_coord, string)
    
    k_small = k[np.ix_(bcoords, wcoords)]
    kinv_small = kinv[np.ix_(wcoords, bcoords)]
    
    return k_small, kinv_small


def shrinkone(n,shape_coord, k, string=None, squares=False, squarestart=1,\
              plot=False, bwmatcoords='',\
           verbose=False, pathlength=''):
    '''
    Similar function as 'shrink' but only requires K and only returns submatrix
    of K (no K^{-1}) corresponding to the string
    '''
    
    if len(bwmatcoords) == 0:
        if string is None:
            if pathlength == '':
                pathlength = n//2-1
            if is_it_squareoctagon(shape_coord) == True:
                if verbose:
                    print('square octagon')
                string = sqoct_vert_string_bylength(n, length=pathlength, \
                                                    squares=squares, squarestart=squarestart)
            else: # square lattice
                if verbose:
                    print('square')
                string = [[(n//2-1, n//2+1+l), (n//2, n//2+1+l)] for l in range(pathlength)]
            print('Using default string/path')
            if verbose==True:
                print(string)
            
        if plot==True:
            plot_shape(n, shape_coord, string,numbering=False)
            
        wcoords = []
        bcoords = []
        for pos in range(len(string)):
            dimer = string[pos]
            v0 = dimer[0]
            v1 = dimer[1]
            if shape_coord(n,*v0)[1] == -1: # swap
                v0, v1 = v1, v0 # swap
            bcoords.append(shape_coord(n,*v0)[0])
            wcoords.append(shape_coord(n,*v1)[0])
    else:
        bcoords, wcoords = bwmatcoords
        
    k_small = k[np.ix_(bcoords, wcoords)]
    
    return k_small


#%% Dimers and visons fast (floating point)

# Calculation of vison correlator using floating point sparse LU

def vison_fast(n, shape_coord, string, cverbose=False):
    '''
    Return vison correlator at every up point up to length 'length-1',
    along a path specified by 'string'.
    Does not require precomputation of K or K^{-1}; calculates 'ksmall'
    and 'kinvs', the latter using (floating point) sparse LU decomposition.
    
    cverbose = verbose about condition number or not
    '''
    kmat = kasteleyn_shape(n, shape_coord, sparse=True)
    kinvs = kinv_lu_sparse(n, kmat, shape_coord,string=string,verbose=False)
    ksmall = shrinkone(n, shape_coord, kmat,string=string, verbose=False).toarray()
    pathlength = len(string)
    v = vison_fast_fromsmall(ksmall, kinvs, length=pathlength, verbose=cverbose)
    return v

def vison(ksmall, kinvsmall, length='', verbose=False):
    '''
    Return the vison correlator at length 'length'. 
    If 'length' is not specified, uses the dimensions of 'ksmall'.
    
    If verbose=True, print the condition number of the small matrix involved 
    if it is > 10**6.
    '''
    if length == '':
        size = np.shape(ksmall)[0]
    else:
        size = length
    ksmalledges = np.diag(np.diag(ksmall)) #ksmall but only the vertices involved in the string
    
    mat = np.eye(size) - 2*kinvsmall[:size,:size] @ ksmalledges[:size,:size]
    if verbose:
        cond_num = np.linalg.cond(mat)
        if cond_num > 10**6:
            print('l=%i, condition number %.10f'%(length,cond_num))
    
    return np.linalg.det(mat)

def vison_fast_fromsmall(ksmall, kinvsmall, length=10,verbose=False):
    '''
    Given matrices 'ksmall' and 'kinvsmall' (K and K^{-1} restricted to the 
                                             string vertices, respectively),
    return the vison correlator at every point up to length 'length-1'.
    '''
    return [vison(ksmall, kinvsmall,l,verbose=verbose) for l in range(1,length)]

def vison_fast_defaultpath(n, shape_coord, pathlength=10,squares=False,squarestart=1,\
                cverbose=False):
    '''
    Return vison correlator at every point up to length 'length-1'.
    The path is determined using default path options.
        
    cverbose = verbose about condition number or not
    '''
    kmat = kasteleyn_shape(n, shape_coord, sparse=True)
    kinvs = kinv_lu_sparse(n, kmat, shape_coord, pathlength=pathlength, \
                           squares=squares,squarestart=squarestart, verbose=False)
    ksmall = shrinkone(n, shape_coord, kmat, pathlength=pathlength,\
                       squares=squares, squarestart=squarestart, verbose=False).toarray()
    v = vison_fast_fromsmall(ksmall, kinvs, length=pathlength, verbose=cverbose)
    return v

# Calculation of dimer-dimer correlator using floating point sparse LU

def dimer_fast(n, shape_coord, string):
    '''
    Dimer correlator at every point up to 1 before the end of the string.
    '''
    kmat = kasteleyn_shape(n, shape_coord, sparse=True)
    kinvs = kinv_lu_sparse(n, kmat, shape_coord,string=string,verbose=False)
    ksmall = shrinkone(n, shape_coord, kmat, string=string, verbose=False).toarray()
    pathlength=np.shape(ksmall)[0]
    d = [dimer_fast_fromsmall(ksmall, kinvs, distance) for distance in range(1,pathlength)]
    return d


def dimer_fast_fromsmall(ksmall, kinvsmall, distance=10):
    '''Dimer correlator at distance 'distance' from Ksmall and K^{-1}small.'''
    if distance > 0:
        return -ksmall[0,0]*ksmall[distance,distance]*kinvsmall[0,distance]*kinvsmall[distance,0]
    p = ksmall[0,0] * kinvsmall[0,0]
    assert p>0, p
    return p - p**2

def dimer_fast_defaultpath(n, shape_coord, pathlength=10,squares=False,squarestart=1):
    '''
    Dimer correlator at every point up to length 'pathlength-1'.
    Uses default path arguments.
    '''
    kmat = kasteleyn_shape(n, shape_coord, sparse=True)
    kinvs = kinv_lu_sparse(n, kmat, shape_coord, pathlength=pathlength, \
                           squares=squares,squarestart=squarestart, verbose=False)
    ksmall = shrinkone(n, shape_coord, kmat, pathlength=pathlength,\
                       squares=squares, squarestart=squarestart, verbose=False).toarray()
    d = [dimer_fast_fromsmall(ksmall, kinvs, distance) for distance in range(1,pathlength)]
    return d



#%% Octic curve

def ocurve(x, y):
    '''
    Evaluate points on the octic curve from https://arxiv.org/pdf/math/0111034
    '''
    return 400*x**8 +400*y**8 +3400*x**2 *y**6 +3400*y**2*x**6 +8025*x**4 *y**4 \
        +1000*x**6 +1000*y**6 - 17250*x**4 * y**2 -17250*x**2* y**4 -1431*x**4\
            -1431*y**4+25812*x**2* y**2 -3402*x**2-3402*y**2 +729 

def arcticcurve(x, y):
    '''Arctic circle'''
    return x**2 + y**2 - 2

def plot_ocurve(axs, istart, jstart, size, resolution=1000, \
                linewidth=1, loneplot=False,axes=False, arctic=False):
    '''
    Plot the octic curve in the region from (0,0) to (size,size).
    istart and jstart give optional lower or left cut-offs. (Put 0,0 for full plot.)
    
    Input:
        axs: the axes to attach the plot to. If not drawing on top of another
            plot, can just take axs='plt', and loneplot=True

        resolution (optional): the number of sample points used for x and y coords
        linewidth (optional): Line width of contour
        loneplot (optional): If True, set the aspect ratio and axes to look nice.
        axes (optiontal): Used if loneplot=True: If axes=False, don't plot the
                     coordinate axes.
    '''
    x = np.linspace(jstart, size, resolution)
    y = np.linspace(istart, size, resolution)

    xvals, yvals = np.meshgrid(x, y) 
    scaled_xvals = (xvals-jstart-size/2.)/size*4 # scale to put xvals in [-2,2]
    scaled_yvals = (yvals-jstart-size/2.)/size*4 # scale to put yvals in [-2,2]
    if not arctic:
        zvals = ocurve(scaled_xvals, scaled_yvals) # evaluate function on grid
    else:
        zvals = arcticcurve(scaled_xvals, scaled_yvals)
    # plot contour where ocurve(x,y)=0
    axs.contour(xvals, yvals, zvals, levels=[0], colors='black', linewidths=linewidth)
    if loneplot==True:
        plt.gca().set_aspect('equal')
        if axes == False:
            plt.axis('off')
    return 0

#%% Conditional probabilities of dimers

def are_dimers_the_same(d1, d2):
    '''
    Return True if dimers d1 and d2 are the same, False otherwise.
    Dimers are represented as [(vx,vy), (wx,wy)], which is the same dimer
    as [(wx,wy), (vx,vy)].
    '''
    if d1 == d2:
        return True
    if d1[0] == d2[1] and d1[1] == d2[0]:
        return True
    return False


def cond_prob(d0list, dimer, ksmall, kinvsmall, return_detcount=False):
    '''
    Return the conditional probability
    Pr['dimer'=1 | all dimers in 'd0list'=1 (are present)].
    
    'ksmall' and 'kinvsmall' are the small version of the Kasteleyn matrix
    and its inverse, size mxm where m-1 = len(d0list).
    They must be indexed as follows:
        first m-1 rows and columns for dimers in 'd0list'
        Last row/column for 'dimer'
    For example they can be generated using the function 'shrink'
    
    'dimer' is represented in the form  [(v1,v2), (w1,w2)], and 'd0list'
    is a list of such representations.
    '''
    for d in d0list:
        if are_dimers_the_same(d,dimer):
            return 1 # if the dimer
    
    # probability of all the dimers in {dimer, 'd0list'} appearing
    prob_all = np.prod(np.diag(ksmall)) * np.linalg.det(kinvsmall)
    
    # probability of dimers in 'd0list' occuring
    prob_d0list = np.prod(np.diag(ksmall)[:-1]) * \
                  np.linalg.det(kinvsmall[:-1,:-1])
    
    assert prob_all > -10**-12 and prob_d0list > -10**-12, \
        'Computed a negative probability: %.4f, %.4f'%(prob_all, prob_d0list)
    
    cprob = prob_all/prob_d0list
    if return_detcount == False:
        return cprob
    return cprob, prob_d0list
    

def edges_in_box(n, shape_coord, dimerlist, radius, firstonly=True):
    '''
    Return all the edges in a box of roughly radius 'radius' around the first
    vertex in dimerlist. 
    
    If 'firstonly'=True, exactly radius 'radius' around first vertex.
    Otherwise tries to look at all the vertices in 'dimerlist'.
    
    Edges in 'dimerlist' are given in lattice coordinates [(v1,v2),(w1,w2)]
    '''
    if is_it_squareoctagon(shape_coord) == True:
        rwidth = radius // 2
    else:
        rwidth = radius 
        
    numdimers = len(dimerlist)
    
    if firstonly == True:
        firstvertex = dimerlist[0][0]
        leftmost = firstvertex[1]
        topmost = firstvertex[0]
        rightmost = firstvertex[1]
        botmost = firstvertex[0]
    
    else: # consider all the dimers in the list
        firstvertex = [dimerlist[i][0] for i in range(numdimers)]
        leftmost = min([firstvertex[i][1] for i in range(numdimers)])
        topmost = min([firstvertex[i][0] for i in range(numdimers)])
        rightmost = max([firstvertex[i][1] for i in range(numdimers)])
        botmost = max([firstvertex[i][0] for i in range(numdimers)])
    
    hordist = rightmost - leftmost
    vertdist = botmost - topmost
    
    # choose all vertices to include
    vertex_list = [(topmost+i, leftmost+j) for i in range(-rwidth,vertdist+rwidth+1)\
                   for j in range(-radius, hordist+radius+1)]
    kmat = kasteleyn_shape(n, shape_coord)
    
    edge_list = []
    for v in vertex_list: # loops through vertex list and add all edges
        try:
            v_matcoord, v_color = shape_coord(n, *v)
            for direction in [(0,1),(1,0),(0,-1),(-1,0)]:
                try:
                    possible_neighbor = v[0]+direction[0], v[1]+direction[1]
                    pn_matcoord = shape_coord(n, *possible_neighbor)[0]
                    if v_color == 1 and ([v, possible_neighbor] not in edge_list\
                                            and [possible_neighbor,v] not in edge_list):
                        indices = v_matcoord, pn_matcoord
                        if kmat[*indices] != 0:
                            edge_list.append([v, possible_neighbor])
                    else:
                        indices = pn_matcoord, v_matcoord
                        if kmat[*indices] != 0 and ([v, possible_neighbor] not in edge_list\
                                                and [possible_neighbor,v] not in edge_list):
                            edge_list.append([possible_neighbor, v])
                except AssertionError:
                    pass
        except AssertionError:
            #print('skipping %s'%(str(v)))
            pass
    
    return edge_list


def conditional_prob_plot(n, shape_coord, k, kinv, dimer0list, radius, \
                          colors='coolwarm', pt=16, savename='',\
                          title = '', axes=False, gapwidth=.2, edgewidth=10,\
                        colors2='coolwarm', firstonly=True, compare_infinite=False,\
                            colorbar=True, ocurve=False,\
                    inset=False, inset_radius=5, insetedgewidth=10, pltaxes=None,\
                        showplot=True,bdybox=True, makefigure=True, verbose=False):
    '''
    Plot conditional probabilities of edges.
    handles a list of dimers set = 1
    
    'k' is the Kasteleyn matrix, 'kinv' is its inverse.
    'dimer0list' and 'radius' are the same as parameters in 'edges_in_box'
    
    'colors' is the colors for square edges.
    'colors2' is the colors for diagonal/octagon edges (only applies for 
                                                        square-octagon lattice).
    
    'compare_infinite' changes the center point of the color scheme to compare 
    the dimer probabilities to the infinite limit dimer probabilities on the 
    periodic square or square-octagon lattice.
    
    'inset' is only intended for square-octagon with real octagon coords
    '''
    if is_it_squareoctagon(shape_coord) == True:   
        if makefigure:
            fig = plt.figure(figsize=(8,7.75))
        inf_prob_square = 0.441
    else:
        if makefigure:
            fig = plt.figure(figsize=(4,4))
        inf_prob_square = .25
    
    ax = plt.subplot() if pltaxes is None else pltaxes
    
    if axes==False: # remove axis/labels
        ax.set_xticks([])
        ax.set_yticks([])
    if bdybox==False:
        plt.axis('off')
    
    list_of_edges = edges_in_box(n, shape_coord, dimer0list, radius, firstonly)
    
    # dictionaries of the probabilities (only 1 is needed; both computed
    # here in case we want to change the code to give one or both)
    probs = {}
    probsdiff = {}

    # loops through each edge and compute the conditional probability
    # this is not the most efficient, but there isn't really a speed issue
    # since these plots don't have many edges generally
    for dimer in list_of_edges:
        alldimers = dimer0list+[dimer] # dimer needs [] to add to python list
                
        ksmall, kinvsmall = shrink(n, shape_coord, k, kinv, alldimers, verbose=False)
       
        cp = cond_prob(dimer0list, dimer, ksmall, kinvsmall)
        probs[(dimer[0],dimer[1])] = cp
        
        v1_matcoord, v2_matcoord = shape_coord(n, *dimer[0]), shape_coord(n,*dimer[1])
        if v1_matcoord[1] == 1:
            dimer_prob = k[v1_matcoord[0], v2_matcoord[0]] *\
                        kinv[v2_matcoord[0], v1_matcoord[0]]
        else:
            dimer_prob = k[v2_matcoord[0], v1_matcoord[0]] *\
                        kinv[v1_matcoord[0], v2_matcoord[0]]
        pdiff = (cp - dimer_prob)
        denom = 1 - dimer_prob if pdiff > 0 else dimer_prob
        probsdiff[(dimer[0],dimer[1])] = pdiff / denom

    cmap = mpl.cm.get_cmap(colors)
    cmap2 = mpl.cm.get_cmap(colors2)    
    if compare_infinite == True:
        norm1 = mpl.colors.TwoSlopeNorm(vmin=0, vcenter=inf_prob_square, vmax=1)
        norm2 = mpl.colors.TwoSlopeNorm(vmin=0, vcenter=0.118, vmax=1) # for octagon
        probval = probs
    else:
        probval = probsdiff
        norm1 = mpl.colors.Normalize(vmin=-1, vmax=1)
        norm2 = norm1
    
    # gap width to leave at ends of edges
    endpt_width = gapwidth
    
    starting_dimer_coords = []
    starting_dimer_colorvals = []
    
    # loop through each edge and plot the conditional probability
    # this is done separately from computing the probability in case
    for dimer in list_of_edges:
        vi, vj = dimer[0]
        wi, wj = dimer[1]
        if is_it_squareoctagon(shape_coord) == True:
            rvi, rvj = realoct_coords(n, vi, vj, shape_coord)
            rwi, rwj = realoct_coords(n, wi, wj, shape_coord)
        else:
            rvi, rvj, rwi, rwj = vi, vj, wi, wj
        
        # get the coordinates for drawing the edges,
        # with some adjustments to prevent overlapping edges
        # firstcoords = x-coords (j), secondcoords = y-coords (i)
        edgetype = 'square'
        if rvj == rwj: # vertical edge
            ysign = 1 if rvi < rwi else -1
            xsign = 0
        elif rvi == rwi: # horizontal edge
            ysign = 0
            xsign = 1 if rvj < rwj else -1
        else: # diagonal edge
            diaggap = 1/1.4
            ysign = diaggap if rvi < rwi else -diaggap
            xsign = diaggap if rvj < rwj else -diaggap
            edgetype = 'oct'
        firstcoords = [rvj + xsign*endpt_width, rwj - xsign*endpt_width]
        secondcoords = [rvi + ysign*endpt_width, rwi - ysign*endpt_width]

        # plot in color the edge
        if edgetype == 'square':
            ecmap = cmap
            norm = norm1
        else:
            ecmap = cmap2
            norm = norm2
        ecmap = cmap if edgetype == 'square' else cmap2
        
        ax.plot(firstcoords, secondcoords,'-', \
                 color=ecmap(norm(probval[(dimer[0],dimer[1])])), linewidth=edgewidth)
    
        # get the info needed to outline the starting dimers
        # starting dimers could be inputted into dictionary backwards 
        for dimer0 in dimer0list:
            if are_dimers_the_same(dimer, dimer0):
                starting_dimer_coords.append((firstcoords, secondcoords))
                starting_dimer_colorvals.append(cmap(norm(probval[(dimer[0],dimer[1])])))
                break

    # outline for starting dimer (do this last)
    pe = [path_effects.Stroke(linewidth=1.5*edgewidth, foreground='black'),
                           path_effects.Normal()] 
    
    if verbose:
        print(len(starting_dimer_colorvals), len(dimer0list)) # dimer0 could be entered twice in the dictionary
    
    # draw/outline the starting dimers
    for i in range(len(starting_dimer_colorvals)):
        ax.plot(*starting_dimer_coords[i],'-', \
             color=starting_dimer_colorvals[i], linewidth=.7*edgewidth, path_effects=pe)
    
    # set title
    if title != '':
        plt.title(title, fontsize=pt)
    
    # other plot parameters
    ax.set_aspect(1)
    
    # make colorbar legend
    if colorbar == True:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size=.18, pad=0.05)
        cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm,cmap=cmap), cax=cax)
        cbar.ax.tick_params(labelsize=pt-2)
        cax.set_aspect(8) 
        
    # plot octal curve
    if ocurve == True:
        # handles a lower y-cutoff (2* because the coordinates are stretched by 2 for the octagons)
        lower_cutoff = 2 * min([edge[0][0] for edge in list_of_edges])
        if verbose:
            print(lower_cutoff)
        plot_ocurve(ax, -.5 + lower_cutoff, -.5, 2*n)
    
    # INSET
    if inset == True:
        xsize = .45
        ysize = .45
        rad_inset = inset_radius
        # actual coordinate values to zoom in on
        x1, y1 = dimer0list[0][0][1] - rad_inset, 2*dimer0list[0][0][0] - rad_inset
        if verbose:
            print(x1,y1)
        x2, y2 = x1 + 2*rad_inset, y1 + 2*rad_inset
        axins = ax.inset_axes([0,0,xsize,ysize], # x0, y0, xheight, yheight (locations in figure, between 0 and 1)
                                xlim = (x1,x2), ylim=(y1,y2))    

        ax.indicate_inset_zoom(axins, edgecolor='black')
        conditional_prob_plot(n, shape_coord, k, kinv, dimer0list, radius=rad_inset,\
                               pltaxes=axins,colorbar=False,edgewidth=insetedgewidth,\
                                   inset=False, showplot=False, makefigure=False)  
    
    if savename != '':
        print('saving to %s'%savename)
        plt.savefig(savename, bbox_inches='tight')
    
    if showplot:
        plt.show()
    
    print('Pattern probability')
    print(cond_prob(dimer0list, list_of_edges[0], ksmall, kinvsmall,return_detcount=True)[1])
    
    return probs



#%% Dimer probability plots 

def is_edge_sq(n, edge):
    '''
    For square octagon lattice/fortress.
    Return True if the edge is a "square" edge, False otherwise
    '''
    v, w = edge
    # Square edges:
        # vertical edges
        # every horizontal edge starting (left to right) at odd column
    if abs(v[0] - w[0]) == 1:
        return True
    assert v[0] == w[0], 'is it a valid edge? %s'%str(edge)
    j = min(v[1],w[1])
    if j%2 == 1:
        return True
    return False

def type_of_sq_edge(n, edge):
    '''
    For square octagon lattice/fortress.
    Return whether the square edge is a left edge, top edge, right edge,
    or bottom edge.
    
    Does NOT really check that the edge is a square edge; therefore one should
    make sure to only pass it square edges and not octagon/diagonal ones.
    '''
    v, w = edge
    j = min(v[1], w[1])
    if abs(v[0] - w[0]) == 1: # vertical edge
        if j % 4 in [1,3]:
            return 'left'
        elif j % 4 in [0,2]:
            return 'right'
    assert v[0] == w[0], 'is it a valid edge? %s'%str(edge)
    i = v[0]
    if (i % 2 == 0 and j%4 == 1) or (i%2 == 1 and j%4 == 3):
        return 'bottom'
    elif (i%2 == 0 and j%4==3) or (i%2==1 and j%4==1):
        return 'top'
    return 1 



def rc_cmap(center, original_cmap_name):
    '''
    Return a new colormap instance which is a rescaling/recentering of
    'original_cmap_name' to have center 'center' instead of 0.5
    
    essentially makes a new colormap by implementing TwoSlopeNorm inherently
    into the colormap
    '''
    def scale(x, center):
        if x >= center:
            return (x - center) / (1 - center) * 128 + 128
        else:
            return 128 - (x - center) / -center * 128
    colorlist = [mpl.colormaps[original_cmap_name](int(scale(x/256.,center))) for x in range(256)]
    return mpl.colors.ListedColormap(colorlist)


def edge_prob_plot(n, shape_coord, k, kinv, startdimer=0, radius=0, \
                          colors='PuOr_r', pt=16, savename='',\
                          title = '', axes=False, gapwidth=.05, edgewidth=2,\
                            toprightquadrant=True, verbose=False,
                            adjust_squares=.1, colors2='RdGy_r', curve=False):
    '''
    Plot the lattice with edges colored according to their probabilities.
    
    Options:
        startdimer, radius- set these to restrict the plot to a certain region
                            of the lattice, centered near 'startdimer' and with
                            radius 'radius'. Otherwise the entire lattice is used.
        colors- colormap for the square edges
        colors2- colormap for the diagonal/octagon edges, used for square-octagon
                 lattice
        gapwidth- gap width to leave at the ends of edges to avoid overlap
        edgewidth- thickness of edges
        toprightquadrant- If True, only uses the top right quadrant K^{-1} 
                          values, and reflects the rest
        adjust_squares- draw the squares a bit shifted from where they usually 
                        are so they are more visible
    '''
    if is_it_squareoctagon(shape_coord) == True:    
        cbarsize = .4 #.18
        fig = plt.figure(figsize=(8+cbarsize,8))
        width = 2*n
        inf_prob_square = 0.440537
        
        if type(startdimer)==int and startdimer == 0: # no value given
            startdimer = [(n//2, n), (n//2, n+1)]
            if radius == 0:
                radius = n
    else:
        cbarsize = .2 #.09
        pt = pt*.7
        fig = plt.figure(figsize=(4+cbarsize,4))
        width = n
        adjust_squares = 0
        inf_prob_square = .25
        
        if startdimer == 0: # no value given
            startdimer = [(n//2, n//2), (n//2, n//2+1)]
            if radius == 0:
                radius = n//2
    
    gs = fig.add_gridspec(2,2, width_ratios=[30,1],\
                          wspace=0,hspace=0)
    ax = fig.add_subplot(gs[:,0])
    
    if axes==False: # remove axis/labels
        plt.xticks([])
        plt.yticks([])
        plt.axis('off')
    
    list_of_edges = edges_in_box(n, shape_coord, [startdimer], radius)
    if verbose:
        print('Got list of edges')
    
    # dictionaries of the probabilities (only 1 is needed; both computed
    # here in case we want to change the code to give one or both)
    probs = {}
    
    # get probability of each edge
    for dimer in list_of_edges:
        v1, v2 = dimer
        if toprightquadrant==True: # reflect the other quadrants
            if v1[0] < n//2: # bottom half
                v1, v2 = (n-1-v1[0], v1[1]), (n-1-v2[0], v2[1])
            if v1[1] < width//2: # left half
                v1, v2 = (v1[0], width-1-v1[1]), (v2[0], width-1-v2[1])
        v1_matcoord, v2_matcoord = shape_coord(n, *v1), shape_coord(n,*v2)
        if v1_matcoord[1] == 1:
            dimer_prob = k[v1_matcoord[0], v2_matcoord[0]] *\
                        kinv[v2_matcoord[0], v1_matcoord[0]]
        else:
            dimer_prob = k[v2_matcoord[0], v1_matcoord[0]] *\
                        kinv[v1_matcoord[0], v2_matcoord[0]]
        probs[(dimer[0],dimer[1])] = dimer_prob
    
    cmap = rc_cmap(inf_prob_square, colors)
    cmap2 = rc_cmap(0.118925, colors2)
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    
    # gap width to leave at ends of edges
    endpt_width = gapwidth
    
    if verbose:
        print('Starting to draw edges')
    edges = []
    ecolorlist = []
    # loop through each edge and plot the probability
    # this is done separately from computing the probability in case
    # we want to change the norm vmin and vmax based on the probabilities
    for dimer in list_of_edges:
        vi, vj = dimer[0]
        wi, wj = dimer[1]
        if is_it_squareoctagon(shape_coord) == True:
            rvi, rvj = realoct_coords(n, vi, vj, shape_coord)
            rwi, rwj = realoct_coords(n, wi, wj, shape_coord)
        else:
            rvi, rvj, rwi, rwj = vi, vj, wi, wj
        
        # get the coordinates for drawing the edges,
        # with some adjustments to prevent overlapping edges
        # firstcoords = x-coords (j), secondcoords = y-coords (i)            
        sqtype = 1
        edgetype = 'square' 
        if rvj == rwj: # vertical edge (square edge)
            ysign = .5 if rvi < rwi else -.5
            xsign = 0
            sqtype = type_of_sq_edge(n, dimer)
        elif rvi == rwi: # horizontal edge (square edge)
            ysign = 0
            xsign = .5 if rvj < rwj else -.5
            sqtype = type_of_sq_edge(n, dimer)
        else: # diagonal edge (not square edge)
            diaggap = 1/1.4
            ysign = diaggap if rvi < rwi else -diaggap
            xsign = diaggap if rvj < rwj else -diaggap
            edgetype = 'oct'

        firstcoords = np.array([rvj + xsign*endpt_width, rwj - xsign*endpt_width])
        secondcoords = np.array([rvi + ysign*endpt_width, rwi - ysign*endpt_width])
        
        match sqtype:
            case 'left':
                firstcoords -= adjust_squares
            case 'right':
                firstcoords += adjust_squares
            case 'top':
                secondcoords += adjust_squares
            case 'bottom':
                secondcoords -= adjust_squares
        # Add the edge, with its color, to the list of edges to plot in LineCollection
        if edgetype == 'square':
            ecmap = cmap
        else:
            ecmap = cmap2
        ecmap = cmap if edgetype == 'square' else cmap2
        edges.append([(firstcoords[0],secondcoords[0]),\
                      (firstcoords[1],secondcoords[1])])
        ecolorlist.append(ecmap(norm(probs[(dimer[0],dimer[1])])))
        
    line_collection = LineCollection(edges, colors=ecolorlist, linewidths=edgewidth)
    ax.add_collection(line_collection)

    plt.margins(x=0.01,y=.01) # less padding around plot

    # set title
    if title != '':
        plt.title(title, fontsize=pt)

    # make colorbar legend
    cax = fig.add_subplot(gs[0,1])
    cbar = fig.colorbar(mpl.cm.ScalarMappable(cmap=cmap), \
                        ticks=[0,inf_prob_square,1],\
                        orientation="vertical", cax=cax,\
                            format=lambda x, _: '%.3f'%x) #f"{x:.3%}")
    cbar.ax.tick_params(labelsize=pt-2)

    # make second colorbar legend if square octagon
    if is_it_squareoctagon(shape_coord) == True: 
        cax2 = fig.add_subplot(gs[1,1])
        cbar2 = fig.colorbar(mpl.cm.ScalarMappable(cmap=cmap2),\
                     ticks=[0,0.118925,1],\
                     orientation="vertical",cax=cax2, \
                         format=lambda x, _: '%.3f'%x)
        cbar2.ax.tick_params(labelsize=pt-2)
        
        # label two different colorbars
        cax.text(-1, 0.5, "square edge prob.", va='center', fontsize=pt-2, rotation='vertical')
        cax2.text(-1, 0.5, "diagonal edge prob.", va='center', fontsize=pt-2, rotation='vertical')
        
    cbar_aspect = 10 # can change the size (aspect ratio) of the colorbar
    cax.set_aspect(cbar_aspect) 
    if is_it_squareoctagon(shape_coord) == True: 
        cax2.set_aspect(cbar_aspect)
    ax.set_aspect(1)

    if savename != '':
        print('saving to %s'%savename)
        plt.savefig(savename, bbox_inches='tight')
    
    # plot the octal curve
    if curve == True:
        if is_it_squareoctagon(shape_coord) == True:
            plot_ocurve(ax, -.5, -.5, 2*n)
            cname = 'ocurve'
        else:
            plot_ocurve(ax, -.5,-.5,n,linewidth=.7,arctic=True)
            cname = 'acurve'
        if savename != '':
            savename_curve = savename[:-4]+'_%s.pdf'%cname
            print('saving plot with curve to %s'%savename_curve)
            plt.savefig(savename_curve, bbox_inches='tight')
        
    plt.show()
    
    return probs
