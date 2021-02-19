"""
Eric Gelphman
University of California Irvine Department of Mathematics

Program using various Python packages to solve the inhomogeneous two dimensional wave equation in with natural boundary 
conditions over the domain of my house using the linear finite element method. Section 9.4 of Kincaid and Cheney, Section 
12.5 of Burden and Faires, and Chapters 6 and 8 of Reddy were the three main resources consulted in the construction of 
this program.

February 19, 2020
v0.7.0
"""

import numpy as np
import triangle
from scipy.interpolate import griddata
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.widgets import Slider
from shapely.geometry import Point, Polygon

def evaluateMInt(x_coords, y_coords, tri_coefs_j, tri_coefs_k):
    """
    Function to evaluate integral over a triangular elements that does not involve the source term f
    using Gaussian quadrature

    Parameters:
    x_coords: Length 3 numpy array pf x-coordinates [x_0, x_1, x_2] of verticies of triangle
    y_coords: Length 3 numpy array of y-coordinates [y_0, y_1, y_2] of verticies of triangle
    tri_coefs_j: Length 3 numpy array of triangle coefficents [a_j, b_j, c_j]
    tri_coefs_k: Length 3 numpy array of triangle coefficents [a_k, b_k, c_k]
    """
    x_0 = x_coords[0]
    x_1 = x_coords[1]
    x_2 = x_coords[2]
    y_0 = y_coords[0]
    y_1 = y_coords[1]
    y_2 = y_coords[2]
    a_j_i = tri_coefs_j[0]
    b_j_i = tri_coefs_j[1]
    c_j_i = tri_coefs_j[2]
    a_k_i = tri_coefs_k[0]
    b_k_i = tri_coefs_k[1]
    c_k_i = tri_coefs_k[2]
    gauss_c = np.array([0.3626837833783620,0.3626837833783620,0.3137066458778873,0.3137066458778873,0.2223810344533745,0.2223810344533745,0.1012285362903763,0.1012285362903763])
    gauss_r = np.array([-0.1834346424956498,0.1834346424956498,-0.5255324099163290,0.5255324099163290,-0.7966664774136267,0.7966664774136267,-0.9602898564975363,0.9602898564975363])
    #Change coordinates so double integral is over the triangle with verticies (0,0),(1,0),(0,1)
    J_abs = np.abs((x_1 - x_0)*(y_2 - y_0) - (x_2 - x_0)*(y_1 - y_0))#Absolute value of determinant of Jacobian matrix
    #Apply two-variable Gaussian quadrature, this involves another change of variables-multiply Jacobians
    h = 0.5
    J2 = 0.0
    for m in np.arange(len(gauss_r)):
        u = h*gauss_r[m] + h
        d = 1.0 - u
        k = d/2.0
        J2X = 0.0
        for n in np.arange(len(gauss_r)):
            v = k*gauss_r[n] + k
            phi = (x_1 - x_0)*u + (x_2 - x_0)*v + x_0
            psi = (y_1 - y_0)*u + (y_2 - y_0)*v + y_0
            val = (a_j_i*a_k_i) + (a_j_i*b_k_i + a_k_i*b_j_i)*phi + (a_j_i*c_k_i + a_k_i*c_j_i)*psi
            val = val + (b_j_i*b_k_i)*np.power(phi,2) + (b_j_i*c_k_i + c_j_i*b_k_i)*phi*psi + (c_j_i*c_k_i)*np.power(psi,2)
            val = J_abs*val#Jacobian is a constant, as the derivatives are of linear functions
            J2X = J2X + gauss_c[n]*val
            J2 = J2 + gauss_c[m]*h*J2X
    z_int = h*J2#Approximation of double integral
    return z_int

def triangulationFEM(gp):
    """
    Function to compute the Delaunay triangulation of the simulation region

    Parameters:
    gp: L x 2 array of xy-coordinates of grid points to be enforced in the triangulation

    Return:
    tri_FEM_R: Delaunay FEM triangulation after one refinement
    """
    grid_dict = {hash((p[0], p[1])): p for p in gp }
    grid_points = np.array(list(grid_dict.values()))
    tri_dict = dict(vertices=grid_points)
    tri_FEM = triangle.triangulate(tri_dict, 'qD')#Initial triangulation 
    tri_FEM_R = triangle.triangulate(tri_FEM, 'ra0.15')#Refinement
    del tri_dict
    del grid_dict
    del tri_FEM#Triangulation takes up a lot of memory, so delete the unrefined triangulation object
    return tri_FEM_R

def generateFEM():
    """
    Function to generate points on the boundary of the downstairs floor of the house. The boundary points are ordered 
    in a counter clockwise manner, as they will later be used to evaluate line integrals. 

    Return:
    t_FEM: Dictionary of xy-coordinates of FEM grid points, node indices of FEM triangles, and node indices of segments that make up boundary 
           of the simulation region
    lr_inner, lr_outer, br, ovr_bdy: Numpy arrays of points that make up inner boundary of living room, outer boundary of living room, boundary of bathroom,
                                     and overall boundary of simulation region, respectively
    """
    #Enter coordinates of points along inner wall of house
    bd_1 = np.array([[np.linspace(0.0,3.6,num=18,endpoint=False),0.3*np.ones(18)]]).reshape((2,-1)).T
    bd_2 = np.array([[3.6*np.ones(20),np.linspace(0.3,4.3,num=20,endpoint=False)]]).reshape((2,-1)).T
    bd_3 = np.array([[np.linspace(3.6,6.1,num=13,endpoint=False),4.3*np.ones(13)]]).reshape((2,-1)).T
    bd_4 = np.array([[6.1*np.ones(8),np.linspace(4.3,5.6,num=8,endpoint=False)]]).reshape((2,-1)).T
    bd_5 = np.array([[np.linspace(6.1,8.6,num=13,endpoint=False),5.6*np.ones(13)]]).reshape((2,-1)).T
    bd_6 = np.array([[8.6*np.ones(12),np.linspace(5.6,3.3,num=12,endpoint=False)]]).reshape((2,-1)).T
    bd_7 = np.array([[np.linspace(8.6,3.9,num=15,endpoint=False),3.3*np.ones(15)]]).reshape((2,-1)).T
    bd_8 = np.array([[3.9*np.ones(16),np.linspace(3.3,0.0,num=16,endpoint=False)]]).reshape((2,-1)).T
    bd_9 = np.array([[np.linspace(3.9,6.9,num=15,endpoint=False),np.zeros(15)]]).reshape((2,-1)).T
    bd_10 = np.array([[6.9*np.ones(10),np.linspace(0.0,2.3,num=10,endpoint=False)]]).reshape((2,-1)).T
    bd_11 = np.array([[np.linspace(6.9,9.6,num=10,endpoint=False),2.3*np.ones(10)]]).reshape((2,-1)).T
    bd_12 = np.array([[9.6*np.ones(10),np.linspace(2.3,0.0,num=10,endpoint=False)]]).reshape((2,-1)).T
    bd_13 = np.array([[np.linspace(9.6,12.9,num=16,endpoint=False),np.zeros(16)]]).reshape((2,-1)).T
    bd_14 = np.array([[12.9*np.ones(16),np.linspace(0.0,3.3,num=16,endpoint=False)]]).reshape((2,-1)).T
    bd_15 = np.array([[np.linspace(12.9,9.6,num=16,endpoint=False),3.3*np.ones(16)]]).reshape((2,-1)).T
    bd_16 = np.array([[9.6*np.ones(20),np.linspace(3.3,7.3,num=20,endpoint=False)]]).reshape((2,-1)).T
    bd_17 = np.array([[np.linspace(9.6,3.6,num=25,endpoint=False),7.3*np.ones(25)]]).reshape((2,-1)).T
    bd_18 = np.array([[3.6*np.ones(7),np.linspace(7.3,5.8,num=7,endpoint=False)]]).reshape((2,-1)).T
    bd_19 = np.array([[np.linspace(3.6,0.0,num=16,endpoint=False),5.8*np.ones(16)]]).reshape((2,-1)).T
    bd_20 = np.array([[np.zeros(27),np.linspace(5.8,0.3,num=27,endpoint=False)]]).reshape((2,-1)).T
    #Enter coordinates along outer wall of house - don't include garage
    o_1 = np.array([[np.linspace(0.0,3.6,num=19,endpoint=True),0.15*np.ones(19)]]).reshape((2,-1)).T
    o_2 = np.array([[3.6,0.0]])
    o_3 = np.array([[np.linspace(3.6,12.9,num=47,endpoint=True),-0.15*np.ones(47)]]).reshape((2,-1)).T
    o_4 = np.array([[13.05*np.ones(17), np.linspace(-0.15,3.3,num=17,endpoint=True)]]).reshape((2,-1)).T
    o_5 = np.array([[np.linspace(13.05,9.75,num=17,endpoint=True),3.45*np.ones(17)]]).reshape((2,-1)).T
    o_6 = np.array([[9.75*np.ones(19), np.linspace(3.45,7.3,num=19,endpoint=True)]]).reshape((2,-1)).T
    o_7 = np.array([[np.linspace(9.75,3.6,num=31,endpoint=True),7.45*np.ones(31)]]).reshape((2,-1)).T
    o_8 = np.array([[3.45*np.ones(7), np.linspace(7.45,5.95,num=7,endpoint=True)]]).reshape((2,-1)).T
    o_9 = np.array([[np.linspace(3.3,-0.15,num=17,endpoint=True),5.95*np.ones(17)]]).reshape((2,-1)).T
    o_10 = np.array([[-0.15*np.ones(30), np.linspace(5.95,0.15,num=30,endpoint=True)]]).reshape((2,-1)).T
    #Enter coordinates along inner wall of bathroom
    bath1 = np.array([[8.45*np.ones(5), np.linspace(3.45,4.4,num=5,endpoint=False)]]).reshape((2,-1)).T
    bath2 = np.array([[np.linspace(8.45,6.3,num=10,endpoint=False),4.4*np.ones(10)]]).reshape((2,-1)).T
    bath3 = np.array([[6.3*np.ones(5), np.linspace(4.4,3.45,num=5,endpoint=False)]]).reshape((2,-1)).T
    bath4 = np.array([[np.linspace(6.3,8.45,num=10,endpoint=False),3.45*np.ones(10)]]).reshape((2,-1)).T
    #Enter coordinates that form rectangular bpundary of overall simulation region D
    D_lower =  np.array([[np.linspace(-3.1,16.1,num=15,endpoint=False),-3.1*np.ones(15)]]).reshape((2,-1)).T
    D_right =  np.array([[16.1*np.ones(15), np.linspace(-3.1,10.5,num=15,endpoint=False)]]).reshape((2,-1)).T
    D_upper =  np.array([[np.linspace(16.1,-3.1,num=15,endpoint=False),10.5*np.ones(15)]]).reshape((2,-1)).T
    D_left =  np.array([[-3.1*np.ones(15), np.linspace(10.5,-3.1,num=15,endpoint=False)]]).reshape((2,-1)).T
    #Concatenate lists of points that we want to appear as nodes for the FEM triangulation
    lr_inner = np.concatenate((bd_1,bd_2,bd_3,bd_4,bd_5,bd_6,bd_7,bd_8,bd_9,bd_10,bd_11,bd_12,bd_13,bd_14,bd_15,bd_16,bd_17,bd_18,bd_19,bd_20))
    lr_outer = np.concatenate((o_1, o_2, o_3, o_4, o_5, o_6, o_7, o_8, o_9, o_10))
    br = np.concatenate((bath1, bath2, bath3, bath4))
    ovr_bdy = np.concatenate((D_lower, D_right, D_upper, D_left))
    enforceable_gp = np.concatenate((lr_inner, lr_outer, br,  np.array([[0.5,1.0]]), ovr_bdy))#Include grid point exactly where source is located
    ax = plt.axes()
    t_FEM = triangulationFEM(enforceable_gp)
    triangle.plot(ax, **t_FEM)
    plt.show()
    return t_FEM, lr_inner, lr_outer, br, ovr_bdy

def calcPermitivitty(gp, lr_inner, lr_outer, br, ovr_b):
    """
    Function to calculate the relative electric permitivitty coefficient at each grid point

    Parameters:
    gp: N x 2 Numpy array of xy-coordinates of grid points
    lr_inner, lr_outer, br, ovr_bdy: Numpy arrays of points that make up inner boundary of living room, outer boundary of living room, boundary of bathroom,
                                     and overall boundary of simulation region, respectively

    Return: Array eps_r of permitivitty coefficients at each point in gp. gp[i] corresponds to eps_r[i]
    """
    N = gp.shape[0]
    p_inner = [Point(g[0],g[1]) for g in lr_inner]
    lr_inner_bdy = Polygon([(p.x,p.y) for p in p_inner])
    p_outer = [Point(g[0],g[1]) for g in lr_outer]
    lr_outer_bdy = Polygon([(p.x,p.y) for p in p_outer])
    p_b = [Point(g[0],g[1]) for g in br]
    b_bdy = Polygon([(p.x,p.y) for p in p_b])
    p_ovr = [Point(g[0],g[1]) for g in ovr_b]
    ovr_bdy = Polygon([(p.x,p.y) for p in p_ovr])
    eps_r = np.zeros(N)
    for ii in np.arange(N):
        p = Point(gp[ii][0], gp[ii][1])
        if lr_inner_bdy.contains(p) or b_bdy.contains(p):
            eps_r[ii] = 1.0#Point is in the interior of the house
        elif gp[ii][0] == -3.1 or gp[ii][0] == 16.1 or gp[ii][1] == -3.1 or gp[ii][1] == 10.5:
            eps_r[ii] = 1.0#Point is on boundary of simulation region
        elif ovr_bdy.contains(p) == True and lr_outer_bdy.contains(p) == False:
            eps_r[ii] = 1.0#Point is inside simulation region but outside the house
        else:#Point is on or inside a wall
            eps_r[ii] = 2.7
    return eps_r
    

def extractVertexCoordinates(tri_boi, grid_points):
    """
    Finds the xy-coordinates of the verticies, given as a length 3 array of node indices, in the
    triangle tri_boi 
    """
    x_0 = grid_points[tri_boi[0]][0]
    y_0 = grid_points[tri_boi[0]][1]
    x_1 = grid_points[tri_boi[1]][0]
    y_1 = grid_points[tri_boi[1]][1]
    x_2 = grid_points[tri_boi[2]][0]
    y_2 = grid_points[tri_boi[2]][1]
    return x_0,y_0,x_1,y_1,x_2,y_2
    

def generateLinearPolynomials(tri_bois, grid_points, gp_markers):
    """
    Function to generate the linear polynomials in the variables x and y at each node/vertex of
    the finite element grid

    Parameters:
    tri_bois: N x 3 Numpy array of triangles. Each row represnts a triangle T_i and the 3 entries in each row represent the 
              node indices of the 3 verticies if T_i
    grid_points: N x 2 Numpy array of xy-coordinates of each vertex/node. grid_points[i] corresponds to node i
    gp_markers: Numpy logical array that is 1 if the grid point at a particular index is on the boundary, and 0 if it is not on the boundary

    Return:
    N_eq: N x 3 x 3 Numpy array of the piecewise linear polynomials. The element N_eq[i][j][k] 
          for k = 0,1,2 corresponds to the linear 2-variable polynomial N_j^i(x,y) = a_j^i + (b_j^i)x + (c_j^i)y 
          where the doubly-indexed coefficients a_j^i,b_j^i,c_j^i correspond to k = 0,1,2 respectively
    """
    N = tri_bois.shape[0]
    N_eq = np.zeros((N, 3, 3))
    for ii in np.arange(N):
        x_0,y_0,x_1,y_1,x_2,y_2 = extractVertexCoordinates(tri_bois[ii], grid_points)
        mat = np.array([[1.0, x_0, y_0], [1.0,x_1,y_1], [1.0,x_2,y_2]])
        del_i = np.linalg.det(mat)
        N_eq[ii][0][0] = (x_1*y_2 - y_1*x_2)/del_i#a_0^i
        N_eq[ii][0][1] = (y_1 - y_2)/del_i#b_0^i
        N_eq[ii][0][2] = (x_2 - x_1)/del_i#c_0^i
        N_eq[ii][1][0] = (x_2*y_0 - y_2*x_0)/del_i#a_1^i
        N_eq[ii][1][1] = (y_2 - y_0)/del_i#b_1^i
        N_eq[ii][1][2] = (x_0 - x_2)/del_i#c_1^i
        N_eq[ii][2][0] = (x_0*y_1 - y_0*x_1)/del_i#a_2^i
        N_eq[ii][2][1] = (y_0 - y_1)/del_i#b_2^i
        N_eq[ii][2][2] = (x_1 - x_0)/del_i#c_2^i
    return N_eq

def extractTriangleCoefs(linear_poly):
    """
    Function to extract the coefficients of the two-variable linear polynomial that represents a triangular element

    Parameters:
    linear_poly: Slice of 3d matrix that stores the coefficients

    Return:
    Coefficients a_j_i,b_j_i,c_j_i in that format
    """
    a_j_i = linear_poly[0]
    b_j_i = linear_poly[1]
    c_j_i = linear_poly[2]
    return a_j_i,b_j_i,c_j_i

def calcTriangleIntegrals(linear_polynomials, tri_bois, grid_points, eps_r_arr):
    """
    Function to calculate the integrals over each triangle to generate the matrix in the finite element method

    Parameters: 
    linear_polynomials: N x 3 x 3 Numpy array of linear polynomials at each node of the finite element grid
    tri_bois: N x 6 Numpy array representing the xy-coordinates of the verticies of each triangular element
    grid_points: N x 2 Numpy array of grid points
    eps_r_arr: Numpy array of length N that holds the value of the relative electrical permitivitty coefficient epsilon_r at each node

    Return:
    M_arr: N x 3 x 3 Numpy array of double integrals over triangular elements representing time dependent term
    H_arr: N x 3 x 3 Numpy array of double integrals over triangular elements representing representing spatial coordinate dependent terms only
    """
    N = linear_polynomials.shape[0]
    M_arr = np.zeros((N,3,3))
    K_arr = np.zeros((N,3,3))
    f_arr = np.zeros((N,3))
    for i in np.arange(N):
        x_0,y_0,x_1,y_1,x_2,y_2 = extractVertexCoordinates(tri_bois[i], grid_points)
        x_coords = np.array([x_0, x_1, x_2])
        y_coords = np.array([y_0, y_1, y_2])
        Area_triangle = 0.5*np.linalg.det(np.array([[x_0,y_0,1],[x_1,y_1,1],[x_2,y_2,1]]))
        for j in np.arange(3):
            a_j_i,b_j_i,c_j_i = extractTriangleCoefs(linear_polynomials[i][j])
            t_j_coefs = np.array([a_j_i, b_j_i, c_j_i])
            for k in np.arange(j+1):
                a_k_i,b_k_i,c_k_i = extractTriangleCoefs(linear_polynomials[i][k])
                t_k_coefs = np.array([a_k_i, b_k_i, c_k_i])
                eps_r = 1.0#Relative permitivitty
                eps_r_V0 = eps_r_arr[tri_bois[i][0]]
                eps_r_V1 = eps_r_arr[tri_bois[i][1]]
                eps_r_V2 = eps_r_arr[tri_bois[i][2]]
                if (eps_r_V0 == 2.7 and (eps_r_V1 == 2.7 or eps_r_V2 == 2.7)) or (eps_r_V1 == 2.7 and eps_r_V2 == 2.7):#Triangles have eps_r=2.7 if they have at least two vertices on or inside a wall
                    eps_r = 2.7
                K_arr[i][j][k] = (b_j_i*b_k_i*Area_triangle) + (c_j_i*c_k_i*Area_triangle)
                u_0 = 12.5663706144e-7
                e_0 = 8.8541878176e-12
                M_arr[i][j][k] = eps_r*e_0*u_0*evaluateMInt(x_coords, y_coords, t_j_coefs, t_k_coefs)
    return M_arr, K_arr

def solveFEMtSystemElectric(M_arr, K_arr, tri_bois, grid_points, node_markers, linear_poly, eps_r_arr):
    """
    Function to solve the system of equation that results from  the linear FEM method for
    the electric field for the time-dependent node coefficients

    Parameters:
    M_arr: N x 3 x 3 Numpy array of values of inner product integrals over each triangular element representing time dependent terms
    K_arr: N x 3 x 3 Numpy array of values of inner product integrals over each triangular element representing spatial coordinate dependent terms only 
    tri_bois: N x 3 array of node indices verticies of each triangle
    grid_points: M x 2 Numpy array represrnting xy-coordinates of verticies
    node_markers: Length M Numpy array indicating whether or not a node lies on the boundary
    linear_poly: N x 3 x 3 Numpy array of coefficients of the polynomials in x and y that represent the triangular elements
    eps_r_arr: Numpy array of length M that holds the electrical permitivitty coefficient at each node

    Return:
    E_vectors: M x 1001 Numpy matrix representing node coefficients at each time step, 1001 in total including the initial condition
               that results from the transient simulation 
    """
    N = tri_bois.shape[0]
    M = grid_points.shape[0]
    source_idx = 0
    for i in np.arange(M):
        if grid_points[i][0] == 0.5 and grid_points[i][1] == 1.0:
            print("Source found")
            source_idx = i

    #Store sparse matrices in (i,j,V) format
    i_M = []
    j_M = []
    V_M = []
    i_K = []
    j_K = []
    V_K = []

    #Assemble integrals over triangles with natural boundary conditions
    for i in np.arange(N):
        original_node_idx = np.array([tri_bois[i][0],tri_bois[i][1],tri_bois[i][2]])#Original node indices
        for k in np.arange(3):
            l = original_node_idx[k]
            if k > 0:
                for j in np.arange(k):
                    t = original_node_idx[j]
                    i_M.append(l)
                    j_M.append(t)
                    V_M.append(M_arr[i][k][j])
                    i_M.append(t)#Matrix M is symmetric
                    j_M.append(l)
                    V_M.append(M_arr[i][k][j])
                    i_K.append(l)
                    j_K.append(t)
                    V_K.append(K_arr[i][k][j])
                    i_K.append(t)#Matrix K is symmetric
                    j_K.append(l)
                    V_K.append(K_arr[i][k][j])
            i_M.append(l)
            j_M.append(l)
            V_M.append(M_arr[i][k][k])
            i_K.append(l)
            j_K.append(l)
            V_K.append(K_arr[i][k][k])

    #Setup and solve the sparse linear system for interior points
    sp_M = csc_matrix((np.array(V_M), (np.array(i_M, dtype=int), np.array(j_M, dtype=int))), shape=(M,M))
    sp_K = csc_matrix((np.array(V_K), (np.array(i_K, dtype=int), np.array(j_K, dtype=int))), shape=(M,M))
    del i_M#Free memory
    del j_M
    del V_M
    del i_K#Free memory
    del j_K
    del V_K
    
    del_t = 6.0e-11#Time step, chosen to meat stability criterion
    NUM_TIME_STEPS = int(1e5)
    E_s = np.zeros(M)
    E_sm = np.zeros(M)#Zero initial condition on partial derivative of E with respect to t
    E_sp = np.zeros(M)
    i_E = []#Create sparse matrix of E vectors at each time step
    j_E = []
    V_E = []
    i_E.append(source_idx)
    j_E.append(0)
    V_E.append(2.0)
    
    #Run transient simulation
    sp_A = ((1.0/del_t)**2)*sp_M
    inv_A = splu(sp_A, permc_spec='MMD_ATA')
    PI = np.pi
    t = 0
    for j in np.arange(NUM_TIME_STEPS):
        f = np.zeros(M)
        t = t + del_t
        f[source_idx] = 2.0*(np.cos(2.0*PI*2.4e9*t) + np.cos(2.0*PI*5.0e9*t))
        b = f - (sp_K - (2.0/(del_t**2))*sp_M)@E_s- (((1.0/del_t)**2)*sp_M)@E_sm
        E_sp = inv_A.solve(b)
        for ii in np.arange(M):
            i_E.append(ii)
            j_E.append(t+1)
            V_E.append(E_sp[ii])
        E_s = E_sp#Update
        E_sm = E_s
            
    E_vectors = csc_matrix((np.array(V_E), (np.array(i_E, dtype=int),np.array(j_E, dtype=int))), shape=(M, NUM_TIME_STEPS + 1))
    return E_vectors

def graphFEMSol(grid_points, tri_bois, E_coefs, eps_arr, linear_poly):
    """
    Function to generate a 3D plot of the solution obtained using the finite element method

    Parameters:
    grid_points: M x 2 representing xy-coordinates of grid points/nodes
    tri_bois: N x 3 numpy array of node indices of the three vertices of each triangle 
    E_coefs: Sparse matrix holding node coefficients column-wise at each time step
    eps_arr: Numpy array of length n that holds the values of the electric permitivitty coefficient(e_r*e_0) at each node
    linear_poly: N x 3 x 3 Numpy array of two-variable linear polynomial coefficients
    """
    print("Graphing started")
    #Generate and plot numerical values of solution at the centroid of each triangle
    N = grid_points.shape[0]
    x_vals = np.zeros(N)
    y_vals = np.zeros(N)
    E_t_vals = np.zeros(N)
    E_coefs_t = np.array(E_coefs[:,0].todense()).flatten()#Load intial data
    for ii in np.arange(N):
        x_0,y_0,x_1,y_1,x_2,y_2 = extractVertexCoordinates(tri_bois[ii], grid_points)
        #Calculate centroid
        centroid_x = (x_0 + x_1 + x_2)/3.0
        x_vals[ii] = centroid_x
        centroid_y = (y_0 + y_1 + y_2)/3.0
        y_vals[ii] = centroid_y
        for jj in np.arange(3):
            E_t_vals[ii] = E_t_vals[ii] + E_coefs_t[tri_bois[ii][jj]]*(linear_poly[ii][jj][0] + linear_poly[ii][jj][1]*centroid_x + linear_poly[ii][jj][2]*centroid_y)
                
    #Set up plot and slider
    fig, ax = plt.subplots()
    xi = np.linspace(-3.1, 16.1, 1000)
    yi = np.linspace(-3.1, 10.5, 1000)
    X,Y = np.meshgrid(xi,yi)
    u_0 = 12.5663706144e-7
    Z = griddata((x_vals, y_vals), np.dot(E_t_vals,E_t_vals)/(2.0*np.sqrt(u_0/eps_arr)), (X, Y), method='linear')
    #Contour the gridded electric field data, plotting dots at the nonuniform data points.
    cs = ax.contourf(X, Y, Z, locator=ticker.LogLocator(), cmap = 'PuBu_r')#Plot interpolated electric field data
    cbar = fig.colorbar(cs)
    ax.scatter(grid_points[:,0],grid_points[:,1], marker='x', linewidths=1, c='g')#Plot grid points for reference
    ax.set_title("Time-dependent Power Density")
    ax_t = plt.axes([0.25, 0.0, 0.65, 0.03])
    S_t = Slider(ax=ax_t, label='Time, us', valmin=0.0, valmax=6.0, valinit=0.0, valfmt='%1.5f us', dragging=True, valstep=6.0e-5)

    #Update inner function
    def update(val):
        ax.clear()
        t = S_t.val
        coef_idx = int(t/(6e-5 + 1))
        E_coefs_updated = np.array(E_coefs[:,coef_idx].todense()).flatten()#Extract column
        E_updated = np.zeros(N)
        for ii in np.arange(N):
            x_0,y_0,x_1,y_1,x_2,y_2 = extractVertexCoordinates(tri_bois[ii], grid_points)
            #Calculate centroid
            centroid_x = (x_0 + x_1 + x_2)/3.0
            centroid_y = (y_0 + y_1 + y_2)/3.0
            for jj in np.arange(3):
                E_updated[ii] = E_updated[ii] + E_coefs_updated[tri_bois[ii][jj]]*(linear_poly[ii][jj][0] + linear_poly[ii][jj][1]*centroid_x + linear_poly[ii][jj][2]*centroid_y)
        #Plot new data
        Z1 = griddata((x_vals, y_vals), np.dot(E_updated,E_updated)/(2.0*np.sqrt(u_0/eps_arr)), (X, Y), method='linear')
        #Contour the gridded electric field data, plotting dots at the nonuniform data points.
        cs = ax.contourf(X, Y, Z1, locator=ticker.LogLocator(), cmap = 'PuBu_r')#Plot interpolated electric field data
        ax.scatter(grid_points[:,0],grid_points[:,1], marker='x', linewidths=1, c='g')#Plot grid points for reference
        ax.set_title("Time-dependent Power Density")
        fig.canvas.draw_idle()
    S_t.on_changed(update)
    plt.show()
    
def main():
    #Note: Scale all quantities for meters and Hz
    #Parameters not dependent on omega or alpha
    t_FEM1, lr_inner, lr_outer, br, ovr_boundary = generateFEM()
    FEM_triangles = np.array(t_FEM1['triangles'].tolist())
    nodes = np.array(t_FEM1['vertices'].tolist())
    M = len(nodes)
    print("Number of FEM Nodes: " + str(M))
    node_markers = np.array(t_FEM1['vertex_markers'].tolist()).flatten()#Marker is 1 if node is on the boundary, 0 otherwise

    eps_r_arr = calcPermitivitty(nodes, lr_inner, lr_outer, br, ovr_boundary)#Calculate permitivitty at each node
    linear_polynomials = generateLinearPolynomials(FEM_triangles, nodes, node_markers)#Calculate linear polynomials
    M_arr, K_arr = calcTriangleIntegrals(linear_polynomials, FEM_triangles, nodes, eps_r_arr)#Calculate triangle integrals
    E_coefs = solveFEMtSystemElectric(M_arr, K_arr, FEM_triangles, nodes, node_markers, linear_polynomials, eps_r_arr)#Solve for electric field coefficients

    #Plot Results of Simulation
    e_0 = 8.8541878176e-12
    eps_arr = e_0*eps_r_arr
    graphFEMSol(nodes, FEM_triangles, E_coefs, eps_arr, linear_polynomials)#Graph the solution

if __name__ == "__main__":
    main()