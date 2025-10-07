#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Definitions of Kasteleyn matrices for bipartite lattices and shapes.
Functions to plot lattices and paths.
'''

import numpy as np
import scipy.sparse as sps

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

def setfont():
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Times"
        })
    return 0

#%% Shapes
# This section defines the different shapes, 
# e.g. square shape, fortress shape, etc. for square lattice and square octagon lattice
# The lattices are bipartite, meaning the vertices can be partitioned into two
# disjoint sublattices A and B so that all edges connect a vertex in A to a 
# vertex in B. The sets A and B can be viewed as a vertex coloring of the graph.
# This allows the Kasteleyn matrix to be represented as a "half-size" matrix,
# with row indices corresponding to one set of vertices, and column indices
# corresponding to the other.

'''
    In general, all coordinate functions behave in the following way:

    Given the size 'n' and lattice coordinates (i,j), where i = row and j = col., 
    returns the tuple (matrix_index, color),
    
    where matrix_index is the row or column index for the half-size bipartite
    adjacency/Kasteleyn matrix (1 set of vertices = rows, the other = columns),
    and color indicates the vertex color (1 = rows, black, -1 = columns, white).
        
    Throws an error if the lattice coordinate is not in the shape.
    (Changed to boolean on newer versions in 'kasteleyn_large.py')
'''


def aztec_coord(n,i,j):
    '''
    Coordinate function for Aztec diamond (square lattice).
    '''
    assert n%2 == 0, 'n must be even, receieved n=%i'%n
    assert 0 <= i <= n-1, 'i must be between 0 and %i, receieved i=%i'%(n-1,i)
    
    matrixcoord = 0
    numvertexpairs = n//2 * (n//2 + 1)
    
    # For top half of diamond, i<n/2
    # j ranges from n/2-1-i,...,n/2-1,n/2,...,n/2+i
    if i < n//2:
        
        # check j is in the diamond
        assert n//2-1-i <= j <= n//2+i, \
            'In top half of diamond. For row %i, j must be between %i and %i, receieved j=%i'\
                %(i, n//2-1-i,n//2+i,j)
        
        # the matrix coordinate for the first vertex in row i is \sum_{k=1}^i k = i(i+1)/2
        rowstart_coord = i*(i+1)//2
        
        # the lattice coordinate for the first vertex in row i is (i, n/2-1-i)
        # divide by 2 since we count in pairs of vertices
        matrixcoord = rowstart_coord + (j - (n//2-1-i)) // 2
    
    # For bottom half of diamong, i>=n/2
    # j ranges from n/2-1-(n-1-i),...n/2-1,n/2,...n/2+(n-1-i)
    if i >= n//2:
        
        # check j is in the diamond
        assert n//2-1-(n-1-i) <= j <= n//2+(n-1-i), 'j not in diamond'
    
        # the matrix coordinate for the first vertex in row i is 
        # numvertexpairs - \sum_{k=1}^{n-i} k = numvertexpairs - (n-i)(n-i+1)/2
        rowstart_coord = numvertexpairs - (n-i)*(n-i+1) // 2
    
        # the lattice coordinate for the first vertex is row i is (i, n/2-1-(n-1-i))
        matrixcoord = rowstart_coord + (j - (n//2-1-(n-1-i))) // 2
    
    # determine vertex color
    if ((i+j)%2 == 0 and (n//2)%2==1) or ((n//2)%2==0 and (i+j)%2==1):
        color = 1
    else:
        color = -1
    
    return matrixcoord, color

def square_coord(n, i, j):
    '''
    Coordinate function for the normal nxn square lattice.
    '''
    assert 0 <= i <= n-1
    assert 0 <= j <= n-1
    
    rowlength = n // 2
    
    if (i+j)%2 == 0:
        color = 1
    else:
        color = -1
    return i*rowlength + j//2, color


def octagon_square_coord(n, i, j):
    '''
    Coordinate function for the square-octagon, drawn within a square lattice.
    (basically same as nx2n lattice, but 1st and last rows have corners removed)
    '''
    assert 0 <= i <= n-1, 'received %i but n-1=%i'%(i,n-1)
    assert 0 <= j <= 2*n-1
    
    if (i+j)%2 == 0:
        color = 1
    else:
        color = -1
    
    # the top and bottom rows are missing the left and right-most vertices
    if i == 0 or i == n-1:
        assert 1 <= j <= 2*n-2, 'top and bottom rows are missing corner vertices'
    
    # length of the rows (in terms of pairs of verticies), except for 1st and last row
    rowlength = n
    
    # handle top and bottom row coordinates a bit differently (j-1 since no j=0 vertex)
    if i == 0:
        return i*rowlength + (j-1)//2, color
    elif i == n-1:
        return i*rowlength + (j-1)//2 - 1, color
    
    # otherwise for all other rows, subtract 1 for missing top row vertices
    return i*rowlength + j//2 - 1, color 

def octagon_fortress_coord(n, i, j):
    '''
    Coordinate function for the square-octagon fortress
    '''
    assert n%4 == 0, 'n must be a multiple of 4, receieved n=%i'%n
    assert 0 <= i <= n-1, 'i must be between 0 and %i, receieved i=%i'%(n-1,i)
    
    matrixcoord = 0
    numvertexpairs = sum([6+2*i+4*(i//2) for i in range(n//2)])
    
    # For top half of diamond, i<n/2
    # j ranges from n/2-1-i,...,n/2-1,n/2,...,n/2+i
    if i < n//2:
        
        # check j is in the diamond
        extrawidth = i + 2*(i//2)
        assert n - 3 - extrawidth <= j <= n + 2 + extrawidth, \
            'In top half of diamond. For row %i, j must be between %i and %i, receieved j=%i'\
                %(i, n-3-extrawidth-i,n+2+extrawidth,j)
        
        # the number of vertices in each row is 6 + i*(i+1) + 4*i//2
        # the matrix coordinate for the first vertex in row i is
        # \sum_{k=0}^i (6 + k*(k+1) + 4*k//2)
        rowstart_coord = sum([3+k+2*(k//2) for k in range(i)])
        
        # the lattice coordinate for the first vertex in row i is (i, n - 3 - extrawidth)
        # divide by 2 since we count in pairs of vertices
        matrixcoord = rowstart_coord + (j - (n - 3 - extrawidth)) // 2
    
    # For bottom half of diamong, i>=n/2
    # Essentially copy top half, but replace i -> n-i
    if i >= n//2:
        
        # check j is in the diamond
        extrawidth = (n-i-1) + 2*((n-i-1)//2)
        assert n - 3 - extrawidth <= j <= n + 2 + extrawidth, 'j not in diamond'
    
        # the matrix coordinate for the first vertex in row i is 
        # numvertexpairs/2 + 
        rowstart_coord = numvertexpairs//2 + sum([3+(n//2-1-k)+2*((n//2-1-k)//2) \
                                               for k in range(i-n//2)])
    
        # the lattice coordinate for the first vertex is row i is (i, n//2-3-extrawidth)
        matrixcoord = rowstart_coord + (j - (n-3-extrawidth)) // 2
    
    # determine vertex color
    if (i+j)%2 == 0:
        color = -1 
    else:
        color = 1 
        
    return matrixcoord, color

def octagon_fakefortress_coord(n, i, j):
    '''
    Coordinate function for the square-octagon diamond shape with extra square 
    faces added along the boundaries. It is NOT the same as the square-octagon
    fortress.
    '''
    assert n%2 == 0, 'n must be a multiple of 2, receieved n=%i'%n
    assert 0 <= i <= n-1, 'i must be between 0 and %i, receieved i=%i'%(n-1,i)
    #assert 0 <= j <= 2*n-1, 'j must be between 0 and %i, receieved j=%i'%(2*n-1,i)
    matrixcoord = 0
    numvertexpairs = sum([6+4*i for i in range(n//2)]) - 2 # -2 for no extra squares
    
    # For bottom half of diamond, i<n/2
    # j ranges from n/2-1-i,...,n/2-1,n/2,...,n/2+i
    if i < n//2-1:
        
        # check j is in the diamond
        extrawidth = 2*i
        assert n - 3 - extrawidth <= j <= n + 2 + extrawidth, \
            'In top half of diamond. For row %i, j must be between %i and %i, receieved j=%i'\
                %(i, n-3-extrawidth-i,n+2+extrawidth,j)
        
        # the number of vertices in each row is 6 + i*(i+1) + 2*i
        # the matrix coordinate for the first vertex in row i is
        # \sum_{k=0}^i (6 + k*(k+1) + 2*i)
        rowstart_coord = sum([3+2*k for k in range(i)])
        
        # the lattice coordinate for the first vertex in row i is (i, n - 2 - extrawidth)
        # divide by 2 since we count in pairs of vertices
        matrixcoord = rowstart_coord + (j - (n - 3 - extrawidth)) // 2
    
    if i == n//2-1:
        assert 0 <= j <= 2*n-1
        rowstart_coord = sum([3+2*k for k in range(i)]) # -1 do not add a square
        matrixcoord = rowstart_coord + j // 2
    
    # For top half of diamond, i>=n/2
    # Essentially copy bottom half, but replace i -> n-i
    if i == n//2:
        assert 0 <= j <= 2*n-1
        rowstart_coord = numvertexpairs//2
        matrixcoord = rowstart_coord + j // 2
            
    if i > n//2:
        # check j is in the diamond
        extrawidth = 2*(n-i-1)
        assert n - 3 - extrawidth <= j <= n + 2 + extrawidth, 'j not in diamond'
    
        # the matrix coordinate for the first vertex in row i is 
        # numvertexpairs/2 + 
        rowstart_coord = numvertexpairs//2 + sum([3+2*(n//2-1-k) \
                                               for k in range(i-n//2)])
    
        # the lattice coordinate for the first vertex is row i is (i, n//2-3-extrawidth)
        matrixcoord = rowstart_coord + (j - (n-3-extrawidth)) // 2 - 1 # do not add square at the end of row n/2
    
    # determine vertex color
    if (i+j)%2 == 0:
        color = -1
    else:
        color = 1
    
    return matrixcoord, color


def is_in_shape(n,i,j, shape_coord):
    '''
    Return True if lattice coordinate (i,j) is in the shape described by 
    'shape_coord', False if not.
    '''
    try:
        shape_coord(n,i,j)
        return True
    except AssertionError:
        return False

def is_it_squareoctagon(shape_coord):
    '''
    Return True if the shape comes from square octagon grid.
    This is useful because the grid is then nx2n, not nxn.
    '''
    if shape_coord.__name__ in ['octagon_fortress_coord','octagon_square_coord',\
                                'octagon_fakefortress_coord']:
        return True
    return False

def num_pairs_of_vertices(n, shape_coord):
    '''
    Return the number of pairs of vertices in the region determined by 'shape_coord'
    '''
    count = 0
    
    # square octagon is double width, nx2n
    if is_it_squareoctagon(shape_coord) == True:
        width = 2*n
    else:
        width = n
    for i in range(n):
        for j in range(width):
            try:
                shape_coord(n, i, j)
                count += 1
            except AssertionError:
                pass
    return count // 2

#%% Create Kasteleyn matrices

def kasteleyn_shape(n, shape_coord, horweights=[1,1], vertweights=[1,1], \
                    abweights = [1,1], dtype=int, sparse=True):
    '''
    Returns Kasteleyn matrix (w/ zero boundary conditions) for the 
    nxn lattice inside the shape given by 'shape_coord'.
    This is the "half-size" Kasteleyn matrix used for bipartite graphs.
    
    Optional arguments:
        horweights - horizontal edge weights for square lattice
        vertweights - vertical edge weights for square lattice
        abweights - square and octagon edge weights for square octagon lattice
        dtype - type of entries in Kasteleyn matrix- must be manually set
                    to float if you use decimals in the edge weights
        sparse - use scipy.sparselil_array if True, otherwise standard numpy array
    '''

    
    if is_it_squareoctagon(shape_coord) == True:
        assert n%4 == 0, 'n must be mult. of 4 for square octagon grids, received %i'%n
        width = 2*n # square octagon is nx2n grid
        squareoct = True
    else:
        assert n%2 == 0, 'n must be even, received %i'%n
        # 'keep' identifies which edges to keep or remove in square octagon grid
        # for square lattice, we keep all edges:
        keep = True     
        width = n
        squareoct = False
    
    # the number of vertex pairs, and size of the Kasteleyn matrix
    size = num_pairs_of_vertices(n, shape_coord)
    
    # create the blank matrices
    if sparse == True:
        mat = sps.lil_array((size,size),dtype=dtype)
    else:
        mat = np.zeros((size,size),dtype=dtype)
    
    # loop over lattice coordinates
    for i in range(n):
        for j in range(width):
            if is_in_shape(n,i,j,shape_coord) == True:
                # get the matrix coordinate
                coord, color = shape_coord(n,i,j)

                if color == 1: 
                    # check if its neighbors are in the shape too
                    # the i,j coordinates are REVERSED (i = vertical movement, j = horizontal)
                    # direction coordinates are usual xy, e.g. (1,0) = right
                    for direction in [(1,0),(0,1),(-1,0),(0,-1)]:
                        neighbor = i + direction[1], j + direction[0]
                        if is_in_shape(n,neighbor[0],neighbor[1],shape_coord) == True:
                            #print(coord,neighbor,aztec_coord(n,neighbor[0],neighbor[1])[0])
                            
                            # if square octagon, not all edges in \Z^2 are kept
                            if squareoct == True and \
                                is_in_shape(n,neighbor[0],neighbor[1],shape_coord) == True:
                                
                                # check which neighbors are allowed to be connected
                                # i even: Below: j = 1,2 mod 4
                                #         Above: j = 0,3 mod 4
                                # i odd: Below: j = 0,3 mod 4.
                                #        Above: j = 1,2 mod 4
                                # to see the above, note that i=n//2-1, which is the last top row,
                                # always starts with octagon => vertical edges when j=0,3 mod 4
                                # and the rows always alternate starting with square vs octagon
                                
                                if (direction == (0,1) and i%2==0) or \
                                    (direction == (0,-1) and i%2==1):
                                    if j%4 == 1 or j%4 == 2:
                                        keep = True
                                    else:
                                        keep = False
                                elif (direction == (0,1) and i%2==1) or\
                                    (direction == (0,-1) and i%2==0): 
                                    if j%4 == 0 or j%4 == 3:
                                        keep = True
                                    else:
                                        keep = False
                                else:
                                    keep =True
                            
                            if squareoct == True:
                                # Determine the sign and weighting for the edge
                                a, b = abweights[0], abweights[1]
                                match direction:
                                    case (0,1): #up
                                        weight = (-1)**j * b # vertical is always b (square edge)
                                    case (0,-1): # down
                                        weight = (-1)**(j+1) * b # vertical is always b
                                    
                                    case (1,0): # right
                                        if i%2 == 0: 
                                            weight = a # octagon edge
                                        else:
                                            weight = b
                                    
                                    case (-1,0): # left
                                        if i%2 == 0:
                                            weight = -b
                                        else:
                                            weight = -a
                            else:
                                # Determine the sign and weighting for the edge
                                if direction == (1,0):
                                    weight = horweights[0]
                                elif direction == (0,1):
                                    weight = (-1)**j * vertweights[0]
                                elif direction == (-1,0):
                                    weight = -horweights[1]
                                elif direction == (0,-1):
                                    weight = (-1)**(j+1) * vertweights[1]
                            
                            # Add the neighbor
                            if keep == True: 
                                mat[coord, \
                                    shape_coord(n,neighbor[0],neighbor[1])[0]] = weight
                            
    return mat


def get_dimerlist_matrixcoords(n, shape_coord, string):
    '''
    Given a list of dimers (given by lattice coordinates [(i1,j1),(i2,j2)])
    in 'string', return the list of matrix coordinates of all the vertices 
    of each color. Does not check for duplicate entries/vertices.
    
    Order is preserved, so the ith entries in the lists correspond to the ith
    dimer [(v1,v2),(w1,w2)] in 'string'.
    
    Input:
        n: system size (diameter)
        shape_coord: shape coordinate function like 'aztec_coord', 'octagon_square_coord', etc
        string: list of dimers using lattice coordinates. string should use the
            the format: [[(i1,j1),(i2,j2)], [(k1,l1),(k2,l2)], ...]
            (recall: first coordinate of pair (i1,j1) = i coordinate = vertical/y coordinate,
                     second coordinate of pair (i1,j1) = j coordinate = horizontal/x coordinate)
    '''
    bcoords = []
    wcoords = []
    for pos in range(len(string)):
        dimer = string[pos]
        v0 = dimer[0]
        v1 = dimer[1]
        if shape_coord(n,*v0)[1] == -1: 
            v0, v1 = v1, v0
        bcoords.append(shape_coord(n,*v0)[0])
        wcoords.append(shape_coord(n,*v1)[0])
    return bcoords, wcoords



#%% Define paths in lattices

def get_default_string(n, shape_coord, squares=False, squarestart=1, \
                       pathlength='', verbose=False):
    '''
    Return a `default string' for calculating vison or dimer correlators.
    For square coordinates (square lattice + Aztec), gives the horizontal string
    from center to right edge.
    
    For square-octagon lattice/fortress, gives the vertical string from the center
    that is either the octagon path ('squares'=False), 
    or one of the two square paths
    ('squarestart'=1 -> square path 2, the shorter path, 
     'squarestart'=0 -> square path 1, the longer path)
    '''
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
        string = [[(n//2-1, n//2+l), (n//2, n//2+l)] for l in range(pathlength)]
        
    return string

def sqoct_vert_string_bylength(n, length=3, squares=True, squarestart=1):
    '''
    Returns the string for square octagon fortress (vertically up from center)
    of length 'length'
    
    squarestart: whether to start in the inside of a square or octagonface
    '''
    if squares == True:
        jstart = n-1 #jstart is the column
        if squarestart == 1:
            istart = n//2-2
        else:
            istart = n//2-1
    else:
        jstart = n-2
        istart = n//2-1
    
    return [[(istart-l, jstart), (istart-l, jstart+1)] for l in range(length)]

def sqoct_square2_mcmatch(n=100):
    '''
    A longer (+2 edges) square path 2 string to match the Monte Carlo data.
    '''
    length = n//2
    istart = n//2
    jstart = n-1
    return [[(istart-l, jstart), (istart-l, jstart+1)] for l in range(length)]

def get_octdiag_string(n, shape_coord, pathlength=10, flip=False):
    '''
    diagonal path through octagons, automatically shortens
    pathlength to max length (to edge) if it is too long
    '''
    assert is_it_squareoctagon(shape_coord)
    if shape_coord.__name__ == 'octagon_fortress_coord':
        pathlength = min(n//4, pathlength)
    pathlength = min(n//2-1,pathlength)
    if flip == False:
        istart = n//2
        jstart = n
        idir = 1
    else:
        istart = n//2 -1
        jstart = n
        idir = -1
    string =  [[(istart-idir*l, jstart+2*l), (istart-idir*l, jstart+2*l+1)] \
                   for l in range(pathlength)]
    return string

#%% Plot shapes and paths

def plot_shape(n, shape_coord, string=[],numbering=False, weights=False, \
               kmat='', axes=True, pt=16, vertices=True, edgecolor='k',\
               vertex_list=[], vsize=60, del_vertex_edges=True, vlist_loop=False, \
               vlist_color='r', vedge_thickness=3):
    '''
    Draw the lattice and edges determined by 'shape_coord'.
    
    'string': a list of edges [[(v1,v2),(v3,v4)], [(w1,w2),(w3,w4)], ...]
    that will be highlighted in the drawing
    
    'numbering': If True, will number every vertex with its matrix coordinate
    
    'weights': If True, label each edge with its weight from the Kasteleyn 
               matrix. (If 'kmat' is not supplied, the weights will be the
                        uniform \pm 1 in the Kastleyn matrix)
        
    'kmat': Optional argument- If passed, this will be used for the Kasteleyn
            matrix. This is only useful with 'weights'=True, to plot non-uniform
            weights.
            
    'vertex_list': Optional- If passed, will highlight those vertices in 'vlist_color'
    'del_vertex_edges': Optional- If True, will delete all edges corresponding
                                  to the vertices in 'vertices'
    
    Remember all plotting is backwards...i = vertical, j=horizontal,
    so for plt.plot, reverse the coords
    '''
    if type(kmat) != str:
        kast = kmat
    else:
        kast = kasteleyn_shape(n, shape_coord)
    if is_it_squareoctagon(shape_coord) == True:
        width = 2*n
    else:
        width = n
    
    fig,ax = plt.subplots(figsize=(n,width))
    
    # line segments for the lattice
    line_segments = []
    
    # line segments for 'vertex_list'
    if len(vertex_list) > 0:
        remove_line_segments = []
        loop_line_segments = []
        
    thinlinewidth = 1.5 # lattice edges
    
    # list of points (by x vs y coord) for the lattice
    bpts_x = []
    bpts_y = []
    wpts_x = []
    wpts_y = []
    
    for i in range(n):
        for j in range(width):
            try: 
                coord, color = shape_coord(n,i,j)
                
                if color == 1:
                    bpts_x.append([j])
                    bpts_y.append([i])
                    
                    # check which neighboring vertices are in the shape, 
                    # and which are connected
                    for neighbor in [(i+1,j),(i,j+1),(i-1,j),(i,j-1)]:
                        if is_in_shape(n, neighbor[0], neighbor[1], shape_coord):
                            kval = kast[coord, shape_coord(n,neighbor[0],neighbor[1])[0]]
                            if kval != 0 and (del_vertex_edges==False or \
                                              ((i,j) not in vertex_list and neighbor not in vertex_list)):
                                line_segments.append([(j,i),(neighbor[1],neighbor[0])])
                                if weights == True:
                                    plt.text(np.mean([j,neighbor[1]]), \
                                             np.mean([i,neighbor[0]]), \
                                        "%.1f"%kval, ha="center", fontsize=pt)
                            else: 
                                # deleted edges associated with vertices in 'vertex_list'
                                if len(vertex_list)>0:
                                    remove_line_segments.append([(j, i),\
                                                           (neighbor[1],neighbor[0])])
                            if vlist_loop: # if we want to draw the segments for 'vertex_list'
                                if (i,j) in vertex_list and neighbor in vertex_list:
                                    # edge between 2 vertices in 'vertex_list' -> add to loop list
                                    loop_line_segments.append([(j, i),\
                                                               (neighbor[1],neighbor[0])])
                else:
                    # plot gray points (edges were already drawn)
                    wpts_x.append([j])
                    wpts_y.append([i])
                    
                # if we want to number every vertex with its matrix coordinate
                if numbering == True:
                    plt.text(j, i, "%d" %coord, ha="center", fontsize=pt)
            except AssertionError:
                pass
    
    # DRAW
    ax.set_xlim(-1, width)
    ax.set_ylim(-1, n)
    ax.set_aspect("equal")

    # draw lattice edges
    line_collection = LineCollection(line_segments, colors=edgecolor, linewidths=.7)
    ax.add_collection(line_collection)

    # draw lattice points
    dotsize = 10
    if vertices:
        plt.scatter(bpts_x,bpts_y,color='black',s=dotsize)
        plt.scatter(wpts_x,wpts_y,color='lightgray',s=dotsize)

    # draw the edges indicated in 'string'
    if len(string) > 0:
        string_segments = []
        for dimer in string:
            v1, v2 = dimer
            string_segments.append([(v2[1],v2[0]),(v1[1],v1[0])])
            #plt.plot([v1[1],v2[1]],[v1[0],v2[0]],'r-', linewidth=4)
        ax.add_collection(LineCollection(string_segments, colors='r', linewidths=4))
    
    # draw vertices indicated in 'vertex_list'
    if len(vertex_list) > 0:
        if del_vertex_edges:
            # draw lattice edges in path as gray and dotted
            remove_line_collection = LineCollection(remove_line_segments, colors='lightgray',\
                                                  linestyle='dotted',linewidth = thinlinewidth)
            ax.add_collection(remove_line_collection)
        if vlist_loop:
            # draw lattice edges in loop as 'vlist_color'
            path_line_collection = LineCollection(loop_line_segments, colors=vlist_color,\
                                                  linewidth = vedge_thickness)
            ax.add_collection(path_line_collection)
                
        v_x = []
        v_y = []
        for v in vertex_list:
            v_x.append([v[1]])
            v_y.append([v[0]])
        ax.scatter(v_x,v_y,color=vlist_color,s=vsize,zorder=10)
    
    # make (0,0) the top left corner to agree with lattice numbering
    #plt.gca().invert_yaxis()
    plt.xticks(fontsize=pt-2)
    plt.yticks(fontsize=pt-2)
    if axes == False:
        plt.axis('off')

    plt.show()
    return 0

def plot_shape_octagons(n, shape_coord, strings=[],numbering=False, weights=False, \
               kmat='', axes=False, pt=20, colorlist=['blue','orange','purple'],\
            labels=['path1', 'path2', 'path3'], savename='',paths3=False,\
                vertices=True, edgecolor='black', \
            vertex_list=[], vsize=60, del_vertex_edges=True, vlist_loop=False, \
            vlist_color='r', vedge_thickness=3, verbose=False):
    '''
    Draw the lattice and edges determined by 'shape_coord'.
    
    'string': a list of edges [[(v1,v2),(v3,v4)], [(w1,w2),(w3,w4)], ...]
    that will be highlighted in the drawing
    
    'numbering': If True, will number every vertex with its matrix coordinate
    
    'weights': If True, label each edge with its weight from the Kasteleyn 
               matrix. (If 'kmat' is not supplied, the weights will be the
                        uniform \pm 1 in the Kastleyn matrix)
        
    'kmat': Optional argument- If passed, this will be used for the Kasteleyn
            matrix. This is only useful with 'weights'=True, to plot non-uniform
            weights.
    
    'vertices': whether to draw vertices in lattice or not
    
    'vertex_list': List of vertices to draw in 'vlist_color'.
    'del_vertex_edges': Delete all edges associated to vertices in 
                        'vertex_list', and draw them in light gray dotted instead
    'vlist_loop': Draw 'vlist_color' edges between neighboring vertices in 'vertex_list'.
                 Useful when the vertices in 'vertex_list' form a path and
                 we want to draw the path.
    
    Remember all plotting is backwards...i = vertical, j=horizontal,
    so for plt.plot, reverse the coords
    '''
    if type(kmat) != str:
        kast = kmat
    else:
        kast = kasteleyn_shape(n, shape_coord)
    if is_it_squareoctagon(shape_coord) == True:
        stretch = 1
        scale = 1+stretch
        vadd = stretch # should be the same as vstretch...
        width = 2*n
    else:
        width = n
        
    fig,ax = plt.subplots(figsize=(width,width))
    thinlinewidth = 1.5 # lattice edges
    thicklinewidth = 6 # highlighted edges in 'string'
    
    # line segments for the lattice
    line_segments = []
    # line segments for the lattice associated with 'vertex_list'
    if len(vertex_list) > 0:
        remove_line_segments = []
        loop_line_segments = []
    
    # list of points (by x vs y coord) for the lattice
    bpts_x = []
    bpts_y = []
    wpts_x = []
    wpts_y = []
    
    for i in range(n):
        for j in range(width):
            try:
                coord, color = shape_coord(n,i,j)
                octi,octj = realoct_coords(n,i,j,shape_coord,scale,vadd)

                if color == 1: # loop through black vertices
                    bpts_x.append([octj])
                    bpts_y.append([octi])
                    
                    # check which neighboring vertices are in the shape, 
                    # and which are connected
                    for neighbor in [(i+1,j),(i,j+1),(i-1,j),(i,j-1)]:
                        if is_in_shape(n, neighbor[0], neighbor[1], shape_coord):
                            kval = kast[coord, shape_coord(n,neighbor[0],neighbor[1])[0]]
                            if kval != 0:                                    
                                octni,octnj = realoct_coords(n,*neighbor,shape_coord,scale,vadd)
                                if del_vertex_edges==False or \
                                        ((i,j) not in vertex_list and neighbor not in vertex_list):
                                    line_segments.append([(octj,octi),\
                                                          (octnj,octni)])
                                    if weights == True:
                                        plt.text(np.mean([octj,octnj]), \
                                                 np.mean([octi,octni]), \
                                            "%.1f"%kval, ha="center", fontsize=pt)
                                else: 
                                    # deleted edges associated with vertices in 'vertex_list'
                                    remove_line_segments.append([(octj, octi),\
                                                               (octnj,octni)])
                                if vlist_loop: # if we want to draw the segments for 'vertex_list'
                                    if (i,j) in vertex_list and neighbor in vertex_list:
                                        # edge between 2 vertices in 'vertex_list' -> add to loop list
                                        loop_line_segments.append([(octj, octi),\
                                                                   (octnj,octni)])
                else:
                    # add to list of white vertices (edges already drawn)
                    wpts_x.append([octj])
                    wpts_y.append([octi])
                    
                # if we want to number every vertex with its matrix coordinate
                if numbering == True:
                    plt.text(octj, octi,\
                             "%d" %coord, ha="center", fontsize=pt)
            except AssertionError:
                pass
    
    # DRAW
    ax.set_xlim(-1, width)
    ax.set_ylim(-1, width)
    ax.set_aspect("equal")

    # draw lattice edges
    if verbose:
        print('drawing lattice edges')
    line_collection = LineCollection(line_segments, colors=edgecolor, linewidths=thinlinewidth)
    ax.add_collection(line_collection)

    # draw vertices
    dotsize = 20
    if vertices:
        if verbose:
            print('drawing lattice points')
        plt.scatter(bpts_x,bpts_y,color='k',s=dotsize)
        plt.scatter(wpts_x,wpts_y,facecolors='white', edgecolor='k',s=dotsize)
    
    # draw vertices indicated in 'vertex_list'
    if len(vertex_list) > 0:
        if del_vertex_edges:
            # draw lattice edges in path as gray and dotted
            remove_line_collection = LineCollection(remove_line_segments, colors='lightgray',\
                                                  linestyle='dotted',linewidth = thinlinewidth)
            ax.add_collection(remove_line_collection)
        if vlist_loop:
            # draw lattice edges in loop as 'vlist_color'
            path_line_collection = LineCollection(loop_line_segments, colors=vlist_color,\
                                                  linewidth = vedge_thickness)
            ax.add_collection(path_line_collection)
                
        v_x = []
        v_y = []
        for v in vertex_list:
            octi, octj = realoct_coords(n,v[0],v[1],shape_coord,scale,vadd)
            v_x.append([octj])
            v_y.append([octi])
        ax.scatter(v_x,v_y,color=vlist_color,s=vsize,zorder=10)
    
    if verbose:
        print('starting string drawing')
    # draw the edges indicated in 'string'
    if paths3 == True: # draw specific paths
        pathlinewidth = 3
        
        strings=[get_default_string(n, shape_coord),\
                 get_default_string(n,shape_coord,squares=True,squarestart=0),\
                 get_octdiag_string(n,shape_coord,n)]
        labels = ['octagon path', 'square path 1', 'square path 2', 'diagonal oct. path']
        colorlist4 = ['blue', 'tab:orange', 'tab:red', 'tab:green']
        # draw the square paths
        squarestart = (n//2-1, n) #i,j lattice coords
        squareend = (0,n)
        octs = realoct_coords(n,*squarestart,shape_coord,scale,vadd)
        octe = realoct_coords(n,*squareend,shape_coord,scale,vadd)
        oshift=-.5 # vertical shift
        sqxshift=.15
        jsqstring = octs[1]-.5
        isqstart = octs[0]-oshift
        plt.plot([jsqstring-sqxshift]*2,[isqstart,octe[0]+1.5],color=colorlist4[1],\
                 linewidth = pathlinewidth)
        plt.plot([jsqstring+sqxshift]*2,[isqstart-2,octe[0]+1.5],color=colorlist4[2],\
                 linewidth = pathlinewidth)
            
        # draw octagon path
        octstart = (n//2-1,n-1)
        octend = (0,n-1)
        octs = realoct_coords(n,*octstart,shape_coord,scale,vadd)
        octe = realoct_coords(n,*octend,shape_coord,scale,vadd)
        joctstring = octs[1]-.5
        ioctstart = octs[0]+.5
        plt.plot([joctstring]*2,[ioctstart,octe[0]+1.5],color=colorlist4[0],\
                 linewidth = pathlinewidth)
        
        # draw diagonal path
        diagstart = (n//2,n)
        diagend = strings[-1][-1][0]
        octs = realoct_coords(n,*diagstart,shape_coord,scale,vadd)
        octe = realoct_coords(n,*diagend,shape_coord,scale,vadd)
        jdiagstart = octs[1]-.5
        idiagstart = octs[0]+1.5
        plt.plot([jdiagstart, octe[1]+1.5],[idiagstart,octe[0]-.5],\
                 color=colorlist4[3],linewidth = pathlinewidth)
 
        # draw path markers
        for sign in [1,-1]:
            plt.plot([jsqstring-sign*sqxshift], [isqstart+(sign-1)], marker='x',\
                     markersize=30, markeredgewidth=3, color=colorlist4[-sign//2+2])
        plt.plot([joctstring], [ioctstart], marker='x', markersize=30, \
                 markeredgewidth=5, color=colorlist4[0])
        plt.plot([jdiagstart], [idiagstart], marker='x', \
                 markersize=30, markeredgewidth=5, color=colorlist4[3])

        colorlist = ['blue', 'tab:orange', 'tab:green']# skip square 2 color
    
    # color edges of string
    if len(strings) > 0:
        legendlines = []
        
        if not isinstance(strings[0][0],list):
            strings = [strings]
        for i in range(len(strings)):
            string = strings[i]
            stringcolor = 'r' if len(colorlist)==0 else colorlist[i]
            string_segments = []
            for dimer in string:
                v1, v2 = dimer
                octi1, octj1 = realoct_coords(n,*v1,shape_coord,scale,vadd)
                octi2, octj2 = realoct_coords(n,*v2,shape_coord,scale,vadd)
                string_segments.append([(octj1,octi1),(octj2,octi2)])
                #plt.plot([v1[1],v2[1]],[v1[0],v2[0]],'r-', linewidth=4)
            ax.add_collection(LineCollection(string_segments, colors=stringcolor,\
                                             linewidths=thicklinewidth))
            legendlines.append(Line2D([0],[0],color=colorlist[i],linewidth=thicklinewidth)) # for the legend
            if i == 1: # square path -> add square path 2
                legendlines.append(Line2D([0],[0],color=colorlist4[2],linewidth=thicklinewidth)) 
        plt.legend(legendlines, labels,fontsize=pt-2)        
    # make (0,0) the top left corner to agree with lattice numbering
    #plt.gca().invert_yaxis()
    plt.xticks(fontsize=pt-2)
    plt.yticks(fontsize=pt-2)
    if axes == False:
        plt.axis('off')
    
    #plot_ocurve(plt, -.5,-.5, 2*n)
    
    if savename != '':
        plt.savefig(savename, bbox_inches='tight')    
    
    plt.show()
    return 0

def is_dep_square(n, i, j):
    '''
    Return True if the coordinate is one of the depressed squares,
    i.e. is in the bottom of an octagon
    This is identified by whether it has an edge to the vertex below it 
    (in the square grid formulation)
    '''
    shape_coord = octagon_square_coord
    coord, color = shape_coord(n,i,j)
    
    if i == 0: # bottom row
        i += 2
    
    if (i%2==0 and j%4 in [0,3]) or (i%2==1 and j%4 in [1,2]):
        return True
    return False

def realoct_coords(n,i,j,shape_coord, scale=2,vadd=1):
    '''
    Return the lattice coordinate for (i,j) when represented
    using actual octagons.
    Only for square octagon lattice/fortress.
    '''
    if is_dep_square(n,i,j):
        octi = i*scale
    else:
        octi = i*scale+vadd
    return octi, j


def plot3paths(n,shape_coord, colorlist=['blue','orange','purple'],\
               labels=['octagon path', 'square path 1', 'diagonal path'],\
                   pt=45, savename=''):
    '''
    plot3paths(12,octagon_fortress_coord, savename='sqoct_fortress_paths.pdf',pt=45)
    plot3paths(12,octagon_square_coord, savename='sqoct_paths.pdf',pt=45)
    '''
    plot_shape_octagons(n,shape_coord,\
        strings=[get_default_string(n, shape_coord),\
                 get_default_string(n,shape_coord,squares=True,squarestart=0),\
                 get_octdiag_string(n,shape_coord,n)],\
            pt=pt, axes=False,savename=savename,\
            paths3=True, vertices=False, edgecolor='k')
    return 0

