"""
Eric Gelphman
University of California Irvine Department of Mathematics

Program using various Python packages to solve the linear Helmholtz equation in the phasor domain 
with inverse-square source term with natural boundary conditions(set 3.0m from the house, over the domain 
of my house using the linear finite element method. Section 9.4 of Kincaid and Cheney, Section 12.5 of Burden 
and Faires, and Chapter 8 of Reddy were the three main resources consulted in the constructionof this program.

February 19, 2020
v1.5.0
"""

import numpy as np
import triangle
from scipy.integrate import nquad, simpson
from scipy.interpolate import griddata
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from matplotlib import ticker
from shapely.geometry import Point, Polygon

def evaluateZInt(x_coords, y_coords, tri_coefs_j, tri_coefs_k):
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

def evaluateHInt(x_coords, y_coords, tri_coefs_j):
    """
    Function to evaluate integral over a triangular elements that does involve the source term f
    using Gaussian quadrature

    Parameters:
    x_coords: Length 3 numpy array pf x-coordinates [x_0, x_1, x_2] of verticies of triangle
    y_coords: Length 3 numpy array of y-coordinates [y_0, y_1, y_2] of verticies of triangle
    tri_coefs_j: Length 3 numpy array of triangle coefficents [a_j, b_j, c_j]

    Return:
    Approximation to integral involving f over triangle
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
            f = J_abs*(1.7/8.8541878176e-9)#Jacobian is a constant, as the derivatives are of linear functions, source is a point charge
            J2X = J2X + gauss_c[n]*f
            J2 = J2 + gauss_c[m]*h*J2X
    H_int = h*J2#Approximation of double integral
    return H_int


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
    enforceable_gp = np.concatenate((lr_inner, lr_outer, br, np.array([[0.5, 1.0]]), ovr_bdy))#Include source term as a grid point
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

def calcTriangleIntegrals(linear_polynomials, tri_bois, grid_points, eps_r_arr, omega):
    """
    Function to calculate the integrals over each triangle to generate the matrix in the finite element method

    Parameters: 
    linear_polynomials: N x 3 x 3 Numpy array of linear polynomials at each node of the finite element grid
    tri_bois: N x 6 Numpy array representing the xy-coordinates of the verticies of each triangular element
    grid_points: N x 2 Numpy array of grid points
    eps_r_arr: Numpy array of length N that holds the value of the relative electrical permitivitty coefficient at each node 
    omega: Angular frequency


    Return:
    z_arr: N x 3 x 3 Numpy array of double integrals
    H_arr: N x 3 Numpy array of double integrals involving inhomogeneous term
    """
    N = linear_polynomials.shape[0]
    z_arr = np.zeros((N,3,3))
    H_arr = np.zeros((N,3))
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
                eps_r = 1.0
                eps_r_V0 = eps_r_arr[tri_bois[i][0]]
                eps_r_V1 = eps_r_arr[tri_bois[i][1]]
                eps_r_V2 = eps_r_arr[tri_bois[i][2]]
                if (eps_r_V0 == 2.7 and (eps_r_V1 == 2.7 or eps_r_V2 == 2.7)) or (eps_r_V1 == 2.7 and eps_r_V2 == 2.7):#Triangles have eps_r=2.7 if they have at least two vertices on or inside a wall
                    eps_r = 2.7
                e_0 = 8.8541878176e-12
                u_0 = 12.5663706144e-7
                k_sq = (omega**2)*eps_r*e_0*u_0
                double_int_z = evaluateZInt(x_coords, y_coords, t_j_coefs, t_k_coefs)
                z_arr[i][j][k] = (b_j_i*b_k_i*Area_triangle) + (c_j_i*c_k_i*Area_triangle) - (k_sq*double_int_z)
            double_integral_H = 0
            if (np.sqrt((x_0 - 0.5)**2 + (y_0 - 1.0)**2) < 1e-12) or (np.sqrt((x_1 - 0.5)**2 + (y_1 - 1.0)**2) < 1e-12) or (np.sqrt((x_2 - 0.5)**2 + (y_2 - 1.0)**2) < 1e-12):#Model source term as point charge
                double_integral_H = evaluateHInt(x_coords, y_coords, t_j_coefs)
            H_arr[i][j] = -1.0*double_integral_H
    return z_arr, H_arr

def onBoundary(tri_boi, node_markers):
    """
    Function to determine if a triangle has at least one edge along the boundary
    """
    V_0 = tri_boi[0]
    V_1 = tri_boi[1]
    V_2 = tri_boi[2]
    if (node_markers[V_0] == 1 and (node_markers[V_1] == 1 or node_markers[V_2] == 1)) or (node_markers[V_1] == 1 and node_markers[V_2] == 1):
        return True
    else:
        return False

def solveFEMSystemElectric(z_arr, H_arr, tri_bois, grid_points, node_markers):
    """
    Function to solve the system of equation that results from  the linear FEM method for
    the electric field in phasor form

    Parameters:
    z_arr: N x 3 x 3 Numpy array of values of double integrals over each triangular element
    H_arr: N x 3 Numpy array of values of double integrals involving inhomogeneous term over each triangle
    J_arr: N x 3 x 3 Numpy array of values of line lintegrals along the boundary of each triangle
    tri_bois: N x 6 array of xy-coordinates of verticies of each triangle
    grid_points: Array of arrays of length 2 represrnting xy-coordinates of verticies
    node_markers: Array of arrays indicating whether or not a node lies on the boundary

    Return:
    x: Solution vector of the matrix equation Ax = b that results from the finite element method. This vector holds
       the coefficients of the finite elements
    """
    N = tri_bois.shape[0]
    M = grid_points.shape[0]
    #Store sparse matrix in (i,j,V) format
    i_A = []
    j_A = []
    V_A = []
    b = np.zeros(M)

    #Assemble integrals over interior triangular elements into linear system
    for i in np.arange(N):
        #if (tri_bois[i][0] in int_points_dict) or (tri_bois[i][1] in int_points_dict) or (tri_bois[i][2] in int_points_dict):#If at least one vertex is in the interior
        original_node_idx = np.array([tri_bois[i][0],tri_bois[i][1],tri_bois[i][2]])#Original node indices
        for k in np.arange(3):
            l = original_node_idx[k]
            #if l in int_points_dict:
            #l_int = int_points_dict.get(l)
            if k > 0:
                for j in np.arange(k):
                    t = original_node_idx[j]
                    #if t in int_points_dict:
                    #t_int = int_points_dict.get(t)#Interior node index
                    i_A.append(l)
                    j_A.append(t)
                    V_A.append(z_arr[i][k][j])
                    i_A.append(t)#Matrix is symmetric
                    j_A.append(l)
                    V_A.append(z_arr[i][k][j])
            i_A.append(l)
            j_A.append(l)
            V_A.append(z_arr[i][k][k])
            b[l] = b[l] + H_arr[i][k]
    #Solve the sparse linear system 
    sp_A = csr_matrix((np.array(V_A), (np.array(i_A, dtype=int), np.array(j_A, dtype=int))), shape=(M,M))
    del i_A#Free memory
    del j_A
    del V_A
    print("Solving Electric System")
    x_E = spsolve(sp_A, b)
    print("End solve electric system")
    return x_E

def graphFEMSol(grid_points, tri_bois, E_coefs_24, E_coefs_5, eps_arr, linear_poly):
    """
    Function to generate a 3D plot of the solution obtained using the finite element method

    Parameters:
    grid_points: M x 2 representing xy-coordinates of grid points/nodes
    tri_bois: N x 3 numpy array of node indices of the three vertices of each triangle 
    E_coefs_24: Numpy array of length M of node coefficients that solve the FEM equations for the electric field at 2.4GHz
    E_coefs_5: Numpy array of length M of node coefficients that solve the FEM equations for the electric field at 5.0GHz
    eps_arr: Numpy array of length n that holds the values of the electric permitivitty coefficient(e_r*e_0) at each node
    linear_poly: N x 3 x 3 Numpy array of two-variable linear polynomial coefficients
    """
    print("Graphing started")
    #Generate and plot numerical values of solution at the centroid of each triangle
    N = grid_points.shape[0]
    x_vals = np.zeros(N)
    y_vals = np.zeros(N)
    E_phasor_24 = np.zeros(N)
    E_phasor_5 = np.zeros(N)
    for ii in np.arange(N):
        x_0,y_0,x_1,y_1,x_2,y_2 = extractVertexCoordinates(tri_bois[ii], grid_points)
        #Calculate centroid
        centroid_x = (x_0 + x_1 + x_2)/3.0
        x_vals[ii] = centroid_x
        centroid_y = (y_0 + y_1 + y_2)/3.0
        y_vals[ii] = centroid_y
        for jj in np.arange(3):
            E_phasor_24[ii] = E_phasor_24[ii] + E_coefs_24[tri_bois[ii][jj]]*(linear_poly[ii][jj][0] + linear_poly[ii][jj][1]*centroid_x + linear_poly[ii][jj][2]*centroid_y)
            E_phasor_5[ii] = E_phasor_5[ii] + E_coefs_5[tri_bois[ii][jj]]*(linear_poly[ii][jj][0] + linear_poly[ii][jj][1]*centroid_x + linear_poly[ii][jj][2]*centroid_y)
    u_0 = 12.5663706144e-7

    #Power density
    power_density = np.power(E_phasor_24 + E_phasor_5,2)/(2.0*np.sqrt(u_0/eps_arr)) 
    del E_phasor_24
    del E_phasor_5
    xi = np.linspace(-3.1, 16.1, 1000)
    yi = np.linspace(-3.1, 10.5, 1000)
    X,Y = np.meshgrid(xi,yi)
    Z = griddata((x_vals, y_vals), power_density, (X, Y), method='linear')
    # contour the gridded power density data, plotting dots at the nonuniform data points.
    fig, ax = plt.subplots()
    cs = ax.contourf(X, Y, Z, locator=ticker.LogLocator(), cmap = 'PuBu_r')#Plot interpolated power density data
    cbar = fig.colorbar(cs)
    ax.scatter(grid_points[:,0],grid_points[:,1], marker='x', linewidths=1, c='g')#Plot grid points for reference
    ax.set_title("Superimposed WiFi Power Density (Gray is High, Blue is Low)")
    plt.show()

def main():
    #Note: Scale all quantities for meters and Hz
    #Parameters not dependent on omega or alpha
    PI = np.pi
    t_FEM1, lr_inner, lr_outer, br, ovr_boundary = generateFEM()
    FEM_triangles = np.array(t_FEM1['triangles'].tolist())
    nodes = np.array(t_FEM1['vertices'].tolist())
    node_markers = np.array(t_FEM1['vertex_markers'].tolist()).flatten()#Marker is 1 if node is on the boundary, 0 otherwise
    eps_r_arr = calcPermitivitty(nodes, lr_inner, lr_outer, br, ovr_boundary)#Calculate permitivitty at each node
    linear_polynomials = generateLinearPolynomials(FEM_triangles, nodes, node_markers)#Calculate linear polynomials

    #Solve using FEM aout each frequency spike, then combine the two solutions using superposition - this is a linear PDE with linear BCs
    #Solve at 2.4GHz
    omega_24 = 2*PI*2.4e9
    z_arr_24, H_arr_24 = calcTriangleIntegrals(linear_polynomials, FEM_triangles, nodes, eps_r_arr, omega_24)#Calculate triangle integrals
    E_coefs_24 = solveFEMSystemElectric(z_arr_24, H_arr_24, FEM_triangles, nodes, node_markers)#Solve for electric field coefficients

    #Solve at 5.0GHz
    omega_5 = 2*PI*5.0e9
    z_arr_5, H_arr_5 = calcTriangleIntegrals(linear_polynomials, FEM_triangles, nodes, eps_r_arr, omega_5)#Calculate triangle integrals
    E_coefs_5 = solveFEMSystemElectric(z_arr_5, H_arr_5, FEM_triangles, nodes, node_markers)#Solve for electric field coefficients

    #Plor Results of Simulation
    e_0 = 8.8541878176e-12
    eps_arr = e_0*eps_r_arr
    graphFEMSol(nodes, FEM_triangles, E_coefs_24, E_coefs_5, eps_arr, linear_polynomials)#Graph the solution

if __name__ == "__main__":
    main()