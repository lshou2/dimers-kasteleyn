#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Fraction-free LU methods
'''

import numpy as np
import scipy.sparse as sps
import time
import multiprocessing as mup
import multiprocessing.pool
import os
from os import cpu_count
import itertools

import mpmath as mp
from mpmath import fadd, fsub, fmul, fdiv, fneg

from functions.kasteleyn_shapes import get_default_string, get_dimerlist_matrixcoords,\
    kasteleyn_shape, get_octdiag_string, is_it_squareoctagon, num_pairs_of_vertices,\
    square_coord, octagon_square_coord, aztec_coord, octagon_fortress_coord
    
from functions.kasteleyn_inverse import shrinkone


#%% Fraction-free LU (FFLU)

def lu_sparse(mat_sparse, verbose=True):
    '''
    Computes the LU matrix of the sparse matrix 'mat_sparse' which is given in
    lil_matrix format.
    
    Warning: row-swapping (due to zero pivot) is NOT implemented.

    Using 
    [1] Lee and Saunders, Fraction Free Gaussian Elimination for Sparse Matrices,
        J. Symb. Comput. 19, 393 (1995).
    [2] Nakos, Turner, and Williams, Fraction-free Gaussian algorithms for 
        linear and polynomial equations, ACM SIGSAM Bull. 31, 11 (1997).

    Note: This function converts everything to python (non numpy) lists 
    to allow for arbitrary precision integers.
    '''
    if not isinstance(mat_sparse, sps.lil_matrix):
        raise ValueError("requires il_matrix format -- use .tolil() first")    
    
    # make the matrix B a python list of lists so it can hold python (non-numpy) ints
    B = mat_sparse.toarray().tolist()
    N = mat_sparse.shape[0]

    # make copies of the lists in mat_sparse.rows so we don't modify mat_sparse!
    nz_rows = [rowlist[:] for rowlist in mat_sparse.rows] 
    nz_cols = [collist[:] for collist in mat_sparse.transpose().rows] 
        
    h = np.zeros((N,N), dtype=int) # the history matrix

    # list of P_m = A^{(m)}_{m+1,m+1} (must be python list to take python ints)
    # the index is k in [1], which is k+1 in the for loops below
    p = [0]*(N+1)
    p[-1] = 1 # initialize P

    if verbose == True:
        print("matrix size N=%i"%N)
    
    t0 = time.time()
    # loop through the iteration counter k, updating B in place to B^{(k+1)}
    # Note: This is NOT the same k as in [1]. It is 1 LESS than the k in [1].
    for k in range(N-1):
        nz_rows_k = nz_rows[k] # get the nonzero entries in row k
        nz_cols_k = nz_cols[k] # get the nonzero entries in column k
        
        # this value can be very large and will be used more than once, 
        # so pull it now so we don't have to keep looking it up in p
        pk_m1 = p[k-1]
        
        if B[k][k] == 0: # would want to swap with bottom-most row with nonzero pivot
            print('Warning: row swapping not implemented')
            return 1
        
        bk = B[k]
        # update p
        l = h[k,k]
        p[k] = bk[k] * pk_m1 // p[l-1]

        s = h[k, k]  # the variable s in [1]

        # get the list of rows i where B[i,k] != 0,
        # and the list of columns j where B[k,j] != 0
        icoords_to_update = [x for x in nz_cols_k if x>=k+1]
        jcoords_to_update = [x for x in nz_rows_k if x>=k+1]
        
        for i in icoords_to_update:
            u = h[i, k]  # the variable u in [1]
            bi = B[i]

            for j in jcoords_to_update:

                # the variables v and t in [1]
                v = h[i, j]
                t = h[k, j]
                
                # Compute the new B value
                # B[k,k]*p[k-1]*B[i,j]/(p[s-1]p[v-1]) - B[i,k]*B[k,j]*p[k-1]/(p[u-1]*p[t-1])
                # but make simplifications if possible to reduce arithmetic cost
                # (which is large for very large integers)
                if s==k:
                    num1 = bk[k]*bi[j] // p[v-1]
                elif v==k:
                    num1 = bk[k]*bi[j] // p[s-1]
                else:
                    num1 = (((bk[k]*pk_m1) // p[s-1]) * bi[j]) // p[v-1]
                
                if u==k:
                    num2 = bi[k]*bk[j] // p[t-1]
                elif t==k:
                    num2 = bi[k]*bk[j] // p[u-1]
                else:
                    num2 = (((bi[k]*pk_m1)//p[u-1]) * bk[j]) // p[t-1]
                    
                newval = num1 - num2
                bi[j] = newval

                # if we set B[i,j]=0, remove it from the lists of nonzero entries
                if newval == 0:
                    nz_rows[i] = [x for x in nz_rows[i] if x != j]
                    nz_cols[j] = [x for x in nz_cols[j] if x != i]
                
                # if we set B[i,j]!=0, add it to the lists of nonzero entries
                else:
                    if i not in nz_cols[j]:
                        nz_cols[j].append(i)
                    if j not in nz_rows[i]:
                        nz_rows[i].append(j)
                
                # update h^(k-1) to h^(k)
                h[i,j] = k+1 # h[i,j] is not called again for this k
    t1 = time.time()
    print('Calculated sparse B matrix in %.2f seconds'%(t1-t0))
    
    # update B to A (in place)
    t2 = time.time()
    for layer in range(N):
        coords = [(row,col) for row in range(layer, N) \
                  for col in nz_rows[row] if (col >= layer) and \
                      (row==layer or col==layer) ]
        # if B[row,col] is not up to date, then update it 
        # otherwise do nothing
        pklayer = p[layer-1]
        for row,col in coords:
            l = h[row,col]
            #min(row,col) is the last time that A would have been updated in non-sparse Bareiss
            if l < layer: 
                B[row][col] = B[row][col] * pklayer // p[l-1]
    t3 = time.time()
    print('Calculated LU matrix A from B in %.2f seconds'%(t3-t2))
    return B


def forward_sub(lu, b):
    '''Forward substitution from [2]. Modifies b'''
    n = len(lu)
    for i in range(n-1):
        luii = lu[i][i]
        for j in range(i+1, n):
            b[j] = luii * b[j] - lu[j][i] * b[i]
            if i > 0:
                b[j] = b[j] // lu[i-1][i-1]
    return b

def back_sub(lu, r):
    '''Backwards substitution from [2].'''
    n = len(lu)
    s = r
    det = lu[n-1][n-1]
    for i in range(n-2,-1,-1):
        s[i] = det*s[i]
        for j in range(i+1,n):
            s[i] = s[i] - lu[i][j]*s[j]
        s[i] = s[i] // lu[i][i]
    return s, det


def kinv_ff(n, kmat, shape_coord, string=[], squares=False,squarestart=1,\
                   pathlength='',  bwmatcoords='',\
                       lu='', save=True, nametag='fortress', divdps=200):
    '''
    Return fraction-free kinvsmall, single thread.
    '''
    if len(string)==0:
        string = get_default_string(n, shape_coord, squares,pathlength=pathlength, \
                                    squarestart=squarestart)
        
    bcoords, wcoords = get_dimerlist_matrixcoords(n, shape_coord, string)
    
    if type(lu) == str:
        #t0 = time.time()
        lu = lu_sparse(kmat)
        #t1 = time.time()
        #print('Computed sparse LU decomposition in %.2f seconds'%(t1-t0))
    ksize = kmat.shape[0]
    submatsize = len(bcoords)
    kinvsmall = mp.matrix(submatsize)

    mp.mp.dps = divdps
    t2 = time.time()
    for index in range(submatsize):
        standard_basis_v = [0]*ksize
        standard_basis_v[bcoords[index]] = 1
        rvec = forward_sub(lu, standard_basis_v)
        soln, det = back_sub(lu, rvec)
        if save:
            np.save('sqoct_%s_n%i_fflu/lusoln_matsize%i_bindex%i_s.npy'\
                    %(nametag,n,ksize, bcoords[index]), soln[0])
            np.save('sqoct_%s_n%i_fflu/lusoln_matsize%i_bindex%i_det.npy'\
                    %(nametag,n,ksize, bcoords[index]), soln[1])
        
        solnsmall = [soln[w] for w in wcoords]
        for j in range(submatsize):
            kinvsmall[j,index] = fdiv(solnsmall[j], det)
        
    t3 = time.time()
    print('Computed kinvsmall from LU decomp. in %.2f seconds'%(t3-t2))
    return kinvsmall


def det_sparse(mat_sparse, verbose=True):
    '''
    Computes determinant of the matrix 'mat_sparse' given in sparse format.
    Using 
    [1] Lee and Saunders, Fraction Free Gaussian Elimination for Sparse Matrices,
        J. Symb. Comput. 19, 393 (1995).

    This function takes lil_matrix format
    but has to convert everything to python (non numpy) int.
    
    It is the same as the first part of the function 'lu_sparse', but with
    row swapping implemented to handle zero pivots.
    '''
    if not isinstance(mat_sparse, sps.lil_matrix):
        raise ValueError("requires il_matrix format -- use .tolil() first")    
    
    # make the matrix B a python list of lists so it can hold python (non-numpy) ints
    B = mat_sparse.toarray().tolist()
    N = mat_sparse.shape[0]

    # make copies of the lists in mat_sparse.rows so we don't modify mat_sparse!
    nz_rows = [rowlist[:] for rowlist in mat_sparse.rows] 
    nz_cols = [collist[:] for collist in mat_sparse.transpose().rows] 
        
    h = np.zeros((N,N), dtype=int) # the history matrix

    # list of P_m = A^{(m)}_{m+1,m+1} (must be python list to take python ints)
    # the index is k in [1], which is k+1 in the for loops below
    p = [0]*(N+1)
    p[-1] = 1 # initialize P

    if verbose == True:
        print("matrix size N=%i"%N)
    
    sign = 1 # keeps track of any row swaps which flip the determinant sign
    # loop through the iteration counter k, updating B in place to B^{(k+1)}
    # Note: This is NOT the same k as in [1]. It is 1 LESS than the k in [1].
    for k in range(N-1):
        nz_rows_k = nz_rows[k] # get the nonzero entries in row k
        nz_cols_k = nz_cols[k] # get the nonzero entries in column k
        
        # this value can be very large and will be used more than once, 
        # so pull it now so we don't have to keep looking it up in p
        pk_m1 = p[k-1]
        
        if B[k][k] == 0: # then swap with bottom-most row with nonzero pivot
            if len(nz_cols_k) == 0:
                return 0 # zero column => determinant = 0
            swapto = max(nz_cols_k)
            if swapto <= k:
                return 0 # all remaining pivots 0
            if verbose == True:
                print('swapping row %i with row %i'%(k,swapto))

            nz_rows_newk = nz_rows[swapto]
            
            # update the columns first
            for kcol in nz_rows_k:
                # update nz_cols[kcol]: replace row index k with swapto (do nothing if both nonzero)
                if swapto not in nz_cols[kcol]:
                    nz_cols[kcol] = [index if index != k else swapto for index in nz_cols[kcol]]
                    
            for newcol in nz_rows_newk:
                # update nz_cols[newcol]: replace row index swapto with k (do nothing if both nonzero)
                if k not in nz_cols[newcol]:
                    nz_cols[newcol] = [index if index != swapto else k for index in nz_cols[newcol]]
            
            # swap the rows
            B[k], B[swapto] = B[swapto][:], B[k][:]
            
            # update the nonzero list in the rows, and the history matrix
            nz_rows[k], nz_rows[swapto] = nz_rows_newk[:], nz_rows_k[:]
            h[[k,swapto],:] = h[[swapto,k],:]
            
            # flip sign of determinant since we swapped rows
            sign = -sign
            
            # Update these variables too
            nz_rows_k = nz_rows[k]
            nz_cols_k = nz_cols[k]
            
        bk = B[k]
        # update p
        l = h[k,k]
        p[k] = bk[k] * pk_m1 // p[l-1]

        s = h[k, k]  # the variable s in [1]

        # get the list of rows i where B[i,k] != 0,
        # and the list of columns j where B[k,j] != 0
        icoords_to_update = [x for x in nz_cols_k if x>=k+1]
        jcoords_to_update = [x for x in nz_rows_k if x>=k+1]
        
        for i in icoords_to_update:
            u = h[i, k]  # the variable u in [1]
            bi = B[i]

            for j in jcoords_to_update:

                # the variables v and t in [1]
                v = h[i, j]
                t = h[k, j]
                
                # Compute the new B value
                # B[k,k]*p[k-1]*B[i,j]/(p[s-1]p[v-1]) - B[i,k]*B[k,j]*p[k-1]/(p[u-1]*p[t-1])
                # but make simplifications if possible to reduce arithmetic cost
                # (which is large for very large integers)
                if s==k:
                    num1 = bk[k]*bi[j] // p[v-1]
                elif v==k:
                    num1 = bk[k]*bi[j] // p[s-1]
                else:
                    num1 = (((bk[k]*pk_m1) // p[s-1]) * bi[j]) // p[v-1]
                
                if u==k:
                    num2 = bi[k]*bk[j] // p[t-1]
                elif t==k:
                    num2 = bi[k]*bk[j] // p[u-1]
                else:
                    num2 = (((bi[k]*pk_m1)//p[u-1]) * bk[j]) // p[t-1]
                    
                newval = num1 - num2
                bi[j] = newval

                # if we set B[i,j]=0, remove it from the lists of nonzero entries
                if newval == 0:
                    # if verbose==True:
                    #     print('Note: B[%i][%i]=0, step kvar=%i'%(i,j,k))
                    #     print('Deleting (i,j) from appropriate row/col. lists')
                    nz_rows[i] = [x for x in nz_rows[i] if x != j]
                    nz_cols[j] = [x for x in nz_cols[j] if x != i]
                
                # if we set B[i,j]!=0, add it to the lists of nonzero entries
                else:
                    if i not in nz_cols[j]:
                        nz_cols[j].append(i)
                    if j not in nz_rows[i]:
                        nz_rows[i].append(j)
                
                # update h^(k-1) to h^(k)
                h[i,j] = k+1 # h[i,j] is not called again for this k

    # calculate the last entry for p, that is p[N-1]
    l = h[N-1,N-1]
    pn_m1 = B[N-1][N-1] * p[N-2] // p[l-1] 

    return sign * pn_m1

#%% FFLU mutithreading

def kinv_mup_helper(bindex, lu, wcoords, n=100, nametag='rect', divdps=200):
    '''
    Used for multithreading
    
    Perform forward and back substitution to solve LU x = e_{bindex}
    Then restrict x to vertices in 'wcoords' and return resulting vector.
    Also saves the intermediate steps (before float division)
    
    Set 'divdps' manually before running.
    '''
    ksize = len(lu)
    standard_basis_v = [0]*ksize
    standard_basis_v[bindex] = 1
    rvec = forward_sub(lu, standard_basis_v)
    soln = back_sub(lu, rvec)
    np.save('sqoct_%s_n%i_fflu/lusoln_matsize%i_bindex%i_s.npy'\
            %(nametag,n,ksize, bindex), soln[0])
    np.save('sqoct_%s_n%i_fflu/lusoln_matsize%i_bindex%i_det.npy'\
            %(nametag,n,ksize, bindex), soln[1])
    
    return np.array([fdiv(soln[0][i],soln[1]) for i in wcoords])

def kinv_ff_mup(n, kmat, shape_coord, string=[], squares=False,squarestart=1,\
                   pathlength='',  bwmatcoords='',\
                       numthreads=0, nametag='fortress', dps=200):
    '''
    Return kinvsmall, computed using fraction-free sparse LU decomposition
    and multithreading. Uses 'numthreads' threads if given, otherwise uses
    0.5*total number of threads.
    
    Save location for intermediate calculations is the folder 
    'sqoct_'nametag'_n%i_fflu/'.
    
    Saves LU matrix in root dir.
    '''
    mp.mp.dps = dps
    
    saveloc = 'sqoct_%s_n%i_fflu/'%(nametag, n)
    if not os.path.exists(saveloc):
        os.makedirs(saveloc)
        print('Created folder %s'%saveloc)
    
    if len(string) == 0:
        string = get_default_string(n, shape_coord, squares,pathlength=pathlength, \
                                squarestart=squarestart)
        
    bcoords, wcoords = get_dimerlist_matrixcoords(n, shape_coord, string)
    
    t0 = time.time()
    lu = lu_sparse(kmat)
    t1 = time.time()
    print('Computed sparse LU decomposition in %.2f seconds'%(t1-t0))
    np.save('lu_sparse_n%i_%s.npy'%(n,shape_coord.__name__), lu)
    
    submatsize = len(bcoords)
    
    if numthreads == 0:
        num_workers = int(0.5 * cpu_count())
    else:
        num_workers = numthreads
    print('using %i threads to calculate kinvsmall from LU'%num_workers)

    t2 = time.time()
    bindex = [bcoords[index] for index in range(submatsize)]
    with mup.Pool(num_workers) as p:
        # note: requires too much RAM to repeat lu many times!
        kinvsmall = p.starmap(kinv_mup_helper, \
                              zip(bindex, itertools.repeat(lu), \
                                  itertools.repeat(wcoords), itertools.repeat(n),\
                                      itertools.repeat(nametag)))
    kinvsmall = np.transpose(kinvsmall)
    t3 = time.time()
    print('Computed kinvsmall from LU decomp. in %.2f seconds'%(t3-t2))
    return kinvsmall

def kinvs_from_lu(lu, bcoords, wcoords, numthreads=0, dps=200):
    '''
    Second half of kinv_ff_mup, if we saved the LU decomposition but didn't 
    run the rest.
    '''
    mp.mp.dps = dps
    
    if numthreads == 0:
        num_workers = int(0.5 * cpu_count())
    else:
        num_workers = numthreads
    print('using %i threads to calculate kinvsmall from LU'%num_workers)

    submatsize = len(bcoords)

    t2 = time.time()
    bindex = [bcoords[index] for index in range(submatsize)]
    
    with mup.Pool(num_workers) as p:
        # note: requires too much RAM to repeat lu many times!
        kinvsmall = p.starmap(kinv_mup_helper, \
                              zip(bindex, itertools.repeat(lu), itertools.repeat(wcoords)))
    kinvsmall = np.transpose(kinvsmall)
    t3 = time.time()
    print('Computed kinvsmall from LU decomp. in %.2f seconds'%(t3-t2))
    return kinvsmall


#%% mpmath to calculate kinvsmall

def mp_kinv_from_lusol(n, shape_coord, string, filepath, dps=200, divdps=2000):
    '''
    Return kinvsmall as an mpmath matrix, calculated using all the saved values
    of lusoln in 'filepath/'
    '''
    mp.mp.dps = dps
    matsize = num_pairs_of_vertices(n, shape_coord)
    bcoords, wcoords = get_dimerlist_matrixcoords(n, shape_coord, string)
    stringlength = len(bcoords)
    kinvs = mp.matrix(stringlength)
    for i in range(stringlength):
        bindex = bcoords[i]
        soln = np.load(filepath+'/lusoln_matsize%i_bindex%i_s.npy'\
                       %(matsize,bindex),allow_pickle=True)[wcoords].tolist()
        det = int(np.load(filepath+'/lusoln_matsize%i_bindex%i_det.npy'\
                          %(matsize,bindex),allow_pickle=True))
        
        mp.mp.dps = divdps
        for j in range(stringlength):
            kinvs[j,i] = fdiv(soln[j], det)
        mp.mp.dps = dps
    return kinvs


def mp_vison(ksmall, kinvsmall, length, verbose=False):
    '''
    Calculate single vison value via det(I-2K_E K^{-1}_E),
    using mpmath det.
    Uses whatever mp.mp.dps was set last.
    '''
    size = length
    ksmalledges = mp.diag([ksmall[i,i] for i in range(len(ksmall))])

    mat = mp.eye(size) - 2*kinvsmall[:size,:size] * ksmalledges[:size,:size]
    
    if verbose:
        cn = np.linalg.cond(np.array(mat.tolist(),dtype=float))
        if cn > 10**6:
            print(size, cn)
    
    return float(mp.det(mat))

def mp_vison_fast_fromsmall(ksmall, kinvsmall, length=10, verbose=False):
    '''
    Calculate vison correlator along path associated with 'ksmall' and 
    'kinvsmall', which should be K and K^{-1} restricted to the path.
    '''
    return [mp_vison(ksmall, kinvsmall,l, verbose) for l in range(1,length+1)]

def mp_dimer(ksmall, kinvsmall, distance):
    '''
    Calculate dimer-dimer correlator at 'distance'
    '''
    if distance > 0:
        return float(fneg(mp.fprod([ksmall[0,0],ksmall[distance,distance],\
                            kinvsmall[0,distance],kinvsmall[distance,0]])))
    p = fmul(ksmall[0,0], kinvsmall[0,0])
    assert p>0
    return float(fsub(p, fmul(p,p)))

def mp_dimer_fast_fromsmall(ksmall, kinvsmall, length=10):
    '''
    Calculate dimer-dimer correlator along path.
    '''
    return [mp_dimer(ksmall, kinvsmall, distance) \
            for distance in range(length)]


def mp_save_visons(n, shape_coord, nametag,verbose=True,dps=50):
    '''
    Calculate and save vison correlators along the 4 paths
    (octpath, squarepath 1 and 2, diagpath), using mpmath
    
    must be in filepath ./sqoct_'nametag'_n'n'_fflu_'pathlabel' where 
    'pathlabel' is 'octpath', 'squarepath', or 'diagpath'
    For exapmle, ./sqoct_fortress_n200_fflu_octpath/
    
    if verbose=True, prints large condition numbers of vison determinant matrix
    '''
    kmat = kasteleyn_shape(n,shape_coord)
    
    pathlength = n//2-1

    pathlabels = ['octpath', 'squarepath', 'diagpath']
    strings = [get_default_string(n,shape_coord,squares=False,pathlength=pathlength),\
               get_default_string(n,shape_coord,squares=True,squarestart=0,pathlength=pathlength),\
               get_octdiag_string(n,shape_coord,pathlength=n)]
    pathlengths = [len(string) for string in strings]
    if verbose:
        print(pathlengths)

    for i in range(3):
        filepath = './sqoct_%s_n%i_fflu_%s/'%(nametag,n,pathlabels[i])
        
        # kinvs sets the dps
        kinvs = mp_kinv_from_lusol(n, shape_coord, strings[i], filepath, dps=dps)
        ksmall = shrinkone(n,shape_coord,kmat,string=strings[i],verbose=verbose).toarray()
        
        np.save(filepath+'mp_visons_%s_n%i_%s_dps%i.npy'%(nametag,n,pathlabels[i],dps),\
                mp_vison_fast_fromsmall(ksmall,kinvs,length=pathlengths[i],verbose=verbose))

        if pathlabels[i] == 'squarepath': # also do square path 2
            ksmall2 = ksmall[1:,:][:,1:]
            kinvs2 = kinvs[1:,:][:,1:]
            
            np.save(filepath+'mp_visons_%s_n%i_%s_2_dps%i.npy'%(nametag,n,pathlabels[i],dps),\
                    mp_vison_fast_fromsmall(ksmall2,kinvs2,length=pathlengths[i]-1,verbose=verbose))

    return 0


def mp_save_dimer(n, shape_coord, nametag,verbose=True,dps=50):
    '''
    Calculate and save vison correlators along the 4 paths
    (octpath, squarepath 1 and 2, diagpath), using mpmath
    
    must be in filepath ./sqoct_'nametag'_n'n'_fflu_'pathlabel' where 
    'pathlabel' is 'octpath', 'squarepath', or 'diagpath'
    For exapmle, ./sqoct_fortress_n200_fflu_octpath/
    
    if verbose=True, prints large condition numbers of vison determinant matrix
    '''
    kmat = kasteleyn_shape(n,shape_coord)
    
    pathlength = n//2-1

    pathlabels = ['octpath', 'squarepath', 'diagpath']
    strings = [get_default_string(n,shape_coord,squares=False,pathlength=pathlength),\
               get_default_string(n,shape_coord,squares=True,squarestart=0,pathlength=pathlength),\
               get_octdiag_string(n,shape_coord,pathlength=n)]
    pathlengths = [len(string) for string in strings]
    if verbose:
        print(pathlengths)

    for i in range(3):
        filepath = './sqoct_%s_n%i_fflu_%s/'%(nametag,n,pathlabels[i])
        
        # kinvs sets the dps
        kinvs = mp_kinv_from_lusol(n, shape_coord, strings[i], filepath, dps=dps)
        ksmall = shrinkone(n,shape_coord,kmat,string=strings[i],verbose=verbose).toarray()
        
        np.save(filepath+'mp_dimer_%s_n%i_%s_dps%i.npy'%(nametag,n,pathlabels[i],dps),\
                mp_dimer_fast_fromsmall(ksmall,kinvs,length=pathlengths[i]))

        if pathlabels[i] == 'squarepath': # also do square path 2
            ksmall2 = ksmall[1:,:][:,1:]
            kinvs2 = kinvs[1:,:][:,1:]
            
            np.save(filepath+'mp_dimer_%s_n%i_%s_2_dps%i.npy'%(nametag,n,pathlabels[i],dps),\
                    mp_dimer_fast_fromsmall(ksmall2,kinvs2,length=pathlengths[i]-1))

    return 0


#%% LU and mpmath to calculate the entire K^{-1}

def top_quadrant_vertices(n, shape_coord):
    '''
    Return list of all row/black vertices (in matrix coordinates) in the 
    top right quadrant + boundary
    '''
    vlist = []
    width = 2*n if is_it_squareoctagon(shape_coord) == True else n
    center = n//2-1, width//2-1 
    for i in range(center[0], n):
        for j in range(center[1], width):
            try: 
                matcoord, color = shape_coord(n, i, j)
                if color == 1:
                    vlist.append(matcoord)
            except AssertionError:
                pass
    return vlist

def kinv_mup_helper2(bindex, lu, n=140,nametag='fortress',lattice='sqoct'):
    '''
    Perform forward and back substitution to solve LU x = e_{bindex}
    Saves the intermediate steps (before float division).
    Does not return anything
    
    Set n, nametag, and lattice MANUALLY EACH TIME 
    (or change 'solve_kinvsfull_from_lu' p.starmap to not require this)
    
    examples: n=100, nametag='fortress',lattice='sqoct'
    n=100, nametag='coord', lattice='aztec'
    '''
    #print('saving in folder %s_%s_n%i_fflu/'%(lattice,nametag,n))
    ksize = len(lu)
    standard_basis_v = [0]*ksize
    standard_basis_v[bindex] = 1
    rvec = forward_sub(lu, standard_basis_v)
    soln = back_sub(lu, rvec)
    np.save('%s_%s_n%i_fflu/lusoln_matsize%i_bindex%i_s.npy'\
            %(lattice,nametag,n,ksize, bindex), soln[0])
    np.save('%s_%s_n%i_fflu/lusoln_matsize%i_bindex%i_det.npy'\
            %(lattice,nametag,n,ksize, bindex), soln[1])
    return 0

def solve_kinvsfull_from_lu(n, shape_coord, lu, numthreads=2, nametag='fortress'):
    '''
    Solve the entire top right quadrant for K^{-1} and save
    Requires a folder like 'sqoct_nametag_n100_fflu'
    '''
    if numthreads == 0:
        num_workers = int(0.5 * cpu_count())
    else:
        num_workers = numthreads
    print('will use %i threads to calculate kinvsmall from LU'%num_workers)

    vlist = top_quadrant_vertices(n, shape_coord)
    bindices = vlist
    print('Got list of top right quadrant vertices, there are %i'%len(vlist))

    t2 = time.time()
    with mup.Pool(num_workers) as p:
        # requires too much RAM to repeat lu many times!
        p.starmap(kinv_mup_helper2, zip(bindices, itertools.repeat(lu)))

    t3 = time.time()
    print('Saved K^{-1} columns from LU decomp. in %.2f seconds'%(t3-t2))
    return 0

def kinvs_full_from_lusol(n, shape_coord, filepath, divdps=2000, dps=50):
    '''
    Return top right quadrant of full K^{-1} matrix from LU decomposition 
    and solution, calculated using all the saved values of lusoln in 'filepath/'
    
    Note: for speed reasons the other quadrants of K^{-1} are NOT filled in
    This function is just so we can plot the edge probabilities easily,
    or calculate probabilities in the top right quadrant only.
    '''
    matsize = num_pairs_of_vertices(n, shape_coord)
    kinv = np.zeros((matsize,matsize))
    
    width = 2*n if is_it_squareoctagon(shape_coord) == True else n
    center = n//2-1, width//2-1

    for i in range(center[0], n):
        for j in range(center[1], width):
            try: 
                matcoord, color = shape_coord(n, i, j)
                if color == 1:
                    bindex = matcoord
                    soln = np.load(filepath+'lusoln_matsize%i_bindex%i_s.npy'\
                                   %(matsize,bindex),allow_pickle=True).tolist()
                    det = int(np.load(filepath+'lusoln_matsize%i_bindex%i_det.npy'\
                                      %(matsize,bindex),allow_pickle=True))
                    
                    mp.mp.dps = divdps
                    kinv[:,bindex] = [float(fdiv(soln[y], det)) for y in range(matsize)]
                    
                    end = fdiv(soln[j],det)
                    if end > 10**10:
                        print(bindex, end)
            except AssertionError:
                pass

    mp.mp.dps=dps
    return kinv

