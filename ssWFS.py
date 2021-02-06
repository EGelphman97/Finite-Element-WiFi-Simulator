"""
Eric Gelphman
University of California Irvine Department of Mathematics

Program using various Python packages to solve the linear Helholtz equation in the phasor domain 
with constant phasor source term for arbitrary Robin boundary conditions over 
arbitrary piecewise-polygonal domains in R^2 using the linear finite element method. Section 9.4 of
Kincaid and Cheney and Section 12.5 of Burden and Faires were the two main resources consulted in the 
creation of this program.

February 6, 2020
v1.0.0
"""

import numpy as np
import triangle
from scipy.integrate import nquad
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from descartes import PolygonPatch
from shapely.geometry import Point, Polygon

def triangulationFEM(bdp):
    """
    Function to compute the constrained conformal Delaunay triangulation of the simulation region

    Parameters:
    bdp: L x 2 array of xy-coordinates of points on the boundary of the simulation region

    Return:
    tri_FEM3: Constrained confromal FEM triangulation after three refinements of the initial triangulation from the boundary
    """
    #Generate line segements that make up boundary
    L = bdp.shape[0]
    segs = np.zeros((L,2), dtype=int)
    for ii in np.arange(L):
        segs[ii][0] = ii
        if ii == L-1:#Connect first and last boubdary points to full enclose the boundary of the region
            segs[ii][1] = 0
        else:
            segs[ii][1] = ii + 1
    tri_dict = dict(vertices=bdp, segments=segs)
    tri_FEM = triangle.triangulate(tri_dict, 'pqD')#Initial triangulation from boundary points only
    tri_FEM2 = triangle.triangulate(tri_FEM, 'rpa0.2')#Refinement 1
    tri_FEM3 = triangle.triangulate(tri_FEM2, 'rpa0.1')#Refinement 2
    tri_FEM4 = triangle.triangulate(tri_FEM3, 'rpa0.05')#Refinement 3
    return tri_FEM4

def generateFEM():
    """
    Function to generate points on the boundary of the downstairs floor of the house. The boundary points are ordered 
    in a counter clockwise manner, as they will later be used to evaluate line integrals. 

    Return:
    t_FEM: Dictionary of xy-coordinates of FEM grid points, node indices of FEM triangles, and node indices of segments that make up boundary 
           of the simulation region
    """
    #Enter coordinates of points along the walls in correct order
    bd_1 = np.array([[np.linspace(0.0,3.6,num=18,endpoint=False),0.3*np.ones(18)]]).reshape((2,-1)).T
    bd_2 = np.array([[3.6*np.ones(20),np.linspace(0.3,4.3,num=20,endpoint=False)]]).reshape((2,-1)).T
    bd_3 = np.array([[np.linspace(3.6,6.3,num=13,endpoint=False),4.3*np.ones(13)]]).reshape((2,-1)).T
    bd_4 = np.array([[6.3*np.ones(8),np.linspace(4.3,5.6,num=8,endpoint=False)]]).reshape((2,-1)).T
    bd_5 = np.array([[np.linspace(6.3,8.9,num=13,endpoint=False),5.6*np.ones(13)]]).reshape((2,-1)).T
    bd_6 = np.array([[8.9*np.ones(12),np.linspace(5.6,3.3,num=12,endpoint=False)]]).reshape((2,-1)).T
    bd_7 = np.array([[np.linspace(8.9,4.2,num=15,endpoint=False),3.3*np.ones(15)]]).reshape((2,-1)).T
    bd_8 = np.array([[4.2*np.ones(16),np.linspace(3.3,0.0,num=16,endpoint=False)]]).reshape((2,-1)).T
    bd_9 = np.array([[np.linspace(4.2,7.2,num=15,endpoint=False),np.zeros(15)]]).reshape((2,-1)).T
    bd_10 = np.array([[7.2*np.ones(7),np.linspace(0.0,2.3,num=7,endpoint=False)]]).reshape((2,-1)).T
    bd_11 = np.array([[np.linspace(7.2,9.9,num=10,endpoint=False),2.3*np.ones(10)]]).reshape((2,-1)).T
    bd_12 = np.array([[9.9*np.ones(11),np.linspace(2.3,0.0,num=11,endpoint=False)]]).reshape((2,-1)).T
    bd_13 = np.array([[np.linspace(9.9,13.2,num=16,endpoint=False),np.zeros(16)]]).reshape((2,-1)).T
    bd_14 = np.array([[13.2*np.ones(16),np.linspace(0.0,3.3,num=16,endpoint=False)]]).reshape((2,-1)).T
    bd_15 = np.array([[np.linspace(13.2,9.9,num=16,endpoint=False),3.3*np.ones(16)]]).reshape((2,-1)).T
    bd_16 = np.array([[9.9*np.ones(20),np.linspace(3.3,7.3,num=20,endpoint=False)]]).reshape((2,-1)).T
    bd_17 = np.array([[np.linspace(9.9,3.6,num=25,endpoint=False),7.3*np.ones(25)]]).reshape((2,-1)).T
    bd_18 = np.array([[3.6*np.ones(7),np.linspace(7.3,5.8,num=7,endpoint=False)]]).reshape((2,-1)).T
    bd_19 = np.array([[np.linspace(3.6,0.0,num=16,endpoint=False),5.8*np.ones(16)]]).reshape((2,-1)).T
    bd_20 = np.array([[np.zeros(27),np.linspace(5.8,0.3,num=27,endpoint=False)]]).reshape((2,-1)).T
    #Concatenate line segments that form boundary
    bd_ordered = np.concatenate((bd_1,bd_2,bd_3,bd_4,bd_5,bd_6,bd_7,bd_8,bd_9,bd_10,bd_11,bd_12,bd_13,bd_14,bd_15,bd_16,bd_17,bd_18,bd_19,bd_20))
    t_FEM = triangulationFEM(bd_ordered)
    return t_FEM

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

def calcTriangleIntegrals(linear_polynomials, tri_bois, grid_points, k_sq, f):
    """
    Function to calculate the integrals over each triangle to generate the matrix in the finite element method

    Parameters: 
    linear_polynomials: N x 3 x 3 Numpy array of linear polynomials at each node of the finite element grid
    tri_bois: N x 6 Numpy array representing the xy-coordinates of the verticies of each triangular element
    grid_points: N x 2 Numpy array of grid points
    k_sq: The parameter k^2 in the Helmholtz equation 
    f: Source term in the PDE

    Return:
    z_arr: N x 3 x 3 Numpy array of double integrals
    H_arr: N x 3 Numpy array of double integrals
    """
    N = linear_polynomials.shape[0]
    z_arr = np.zeros((N,3,3))
    H_arr = np.zeros((N,3))
    for i in np.arange(N):
        x_0,y_0,x_1,y_1,x_2,y_2 = extractVertexCoordinates(tri_bois[i], grid_points)
        Area_triangle = 0.5*np.linalg.det(np.array([[x_0,y_0,1],[x_1,y_1,1],[x_2,y_2,1]]))
        for j in np.arange(3):
            for k in np.arange(j+1):
                a_j_i,b_j_i,c_j_i = extractTriangleCoefs(linear_polynomials[i][j])
                a_k_i,b_k_i,c_k_i = extractTriangleCoefs(linear_polynomials[i][k])
                #Change coordinates so double integral is over the triangle with verticies (0,0),(1,0),(0,1)
                J_abs = np.abs((x_1 - x_0)*(y_2 - y_0) - (x_2 - x_0)*(y_1 - y_0))#Absolute value of determinant of Jacobian matrix

                #Evaluate linear polynomial at a point (u,v)
                def h(u,v):
                    phi = (x_1 - x_0)*u + (x_2 - x_0)*v + x_0
                    psi = (y_1 - y_0)*u + (y_2 - y_0)*v + y_0
                    val = a_j_i + (b_j_i*phi) + (c_j_i*psi)
                    return val

                #Evaluate composition of change of coordinates map and product of two linear polynomials at point (u,v)
                def h_1(u,v):
                    phi = (x_1 - x_0)*u + (x_2 - x_0)*v + x_0
                    psi = (y_1 - y_0)*u + (y_2 - y_0)*v + y_0
                    val = (a_j_i*a_k_i) + (a_j_i*b_k_i + a_k_i*b_j_i)*phi + (a_j_i*c_k_i + a_k_i*c_j_i)*psi
                    val = val + (b_j_i*b_k_i)*np.power(phi,2) + (b_j_i*c_k_i + c_j_i*b_k_i)*phi*psi + (c_j_i*c_k_i)*np.power(psi,2)
                    return val

                #Bounds for double integration
                def bounds_u():
                    return [0.0,1.0]

                def bounds_v(u):
                    return [0.0, 1.0 - u]

                double_integral_z,error_est_z = nquad(h_1, [bounds_v, bounds_u])#Calculate double integral of product of linear polynomials (N_j^i)(N_k^i) over triangle with verticies (0,0),(1,0),(0,1)
                z_arr[i][j][k] = (b_j_i*b_k_i*Area_triangle) + (c_j_i*c_k_i*Area_triangle) - (k_sq*double_integral_z)
            double_integral_H,error_est_H = nquad(h, [bounds_v, bounds_u])
            H_arr[i][j] = -1.0*f*double_integral_H
    return z_arr, H_arr

def onBoundary(tri_boi, node_markers):
    """
    Function to check if the triangle tri_boi is on the boundary
    """
    V0 = tri_boi[0]
    V1 = tri_boi[1]
    V2 = tri_boi[2]
    if (node_markers[V0] == 1 and (node_markers[V1] == 1 or node_markers[V2] == 1)) or (node_markers[V1] == 1 and node_markers[V2] == 1):
        return True
    else:
        return False
            

def calcLineIntegrals(tri_bois, linear_polynomials, boundary_points, node_markers, alpha):
    """
    Function to calculate the line integrals used in the system of linear equations that results from the finite element method

    Parameters:
    tri_bois: N x 3 Numpy array of node indices of the three vertices of each triangle
    linear_polynomials: N x 3 x 3 array of the linear polynomials that are the basis functions used in finite element method
    boundary_points: List of points(in counter-clockwise order) of points on boundary of simulation region
    alpha: Coefficient that represents how much energy is absorbed by the wall when an incident wave hits it, is set to be
           1/(1-Tau) where Tau = transmission coefficient at air/wall interface

    Return:
    J_ints: N x 3 x 3 array of values of line integrals
    """
    N = linear_polynomials.shape[0]
    bp = np.concatenate((boundary_points, [boundary_points[0]]))#Boundary must be closed
    L = bp.shape[0]
    J_ints = np.zeros((N,3,3))
    for i in np.arange(N):
        #Check if triangle has at least two points on boundary. If this is true, then the triangle has at least one side along the boundary
        if onBoundary(tri_bois[i], node_markers):
            for j in np.arange(3):
                for k in np.arange(j+1):
                    #Approximate line integral along boundary of simulation region using variation of midpoint rule
                    del_x = np.sqrt((bp[1:L][0] - bp[0:(L-1)][0])**2 + (bp[1:L][1] - bp[0:(L-1)][1])**2)
                    a_j_i,b_j_i,c_j_i = extractTriangleCoefs(linear_polynomials[i][j])
                    a_k_i,b_k_i,c_k_i = extractTriangleCoefs(linear_polynomials[i][k])

                    #Evaluate product of two linear polynomials at a point (x,y)
                    def h_2(x,y):
                        val = (a_j_i*a_k_i) + (a_j_i*b_k_i + a_k_i*b_j_i)*x + (a_j_i*c_k_i + a_k_i*c_j_i)*y
                        val = val + (b_j_i*b_k_i)*np.power(x,2) + (b_j_i*c_k_i + c_j_i*b_k_i)*x*y + (c_j_i*c_k_i)*np.power(y,2)
                        return val

                    #Compute line integrals along boundary by modified version of Simpson's rule
                    eval_f = np.zeros(len(del_x))
                    for ii in np.arange(len(del_x)):
                        if ii == 0:#Trapezoid rule
                            eval_f[ii] = 0.5*(h_2(bp[ii+1][0],bp[ii+1][1]) + h_2(bp[ii][0], bp[ii][1]))
                        elif ii == len(del_x) - 1:#Trapezoid rule
                            eval_f[ii] = 0.5*(h_2(bp[ii-1][0],bp[ii-1][1]) + h_2(bp[ii][0], bp[ii][1]))
                        else:
                            eval_f[ii] = (1.0/6)*(h_2(bp[ii-1][0],bp[ii-1][1]) + 4.0*h_2(bp[ii][0],bp[ii][1]) + h_2(bp[ii+1][0], bp[ii+1][1]))
                    J_ints[i][j][k] = alpha*np.dot(eval_f,del_x)
    return J_ints
  
def solveFEMSystem(z_arr, H_arr, J_arr, tri_bois, grid_points):
    """
    Function to solve the system of equation that results from  the linear FEM method

    Parameters:
    z_arr: N x 3 x 3 Numpy array of values of inner product integrals over each triangular element
    H_arr: N x 3 Numpy array of values of integrals involving source term over each triangular element
    J_arr: N x 3 x 3 Numpy array of line integrals along boundary of simulation region
    tri_bois: N x 6 array of xy-coordinates of verticies of each triangle
    grid_points: Array of arrays of length 2 represrnting xy-coordinates of verticies

    Return:
    x: Solution vector of the matrix equation Ax = b that results from the finite element method. This vector holds
       the coefficients of the finite elements
    nodesToTriangles: Index map from node index to triangle index needed to evaluate the solution. Node i is mapped to
                      triangle nodesToTriangles[i]
    """
    N = tri_bois.shape[0]
    M = grid_points.shape[0]
    A = np.zeros((M,M))
    b = np.zeros(M)
    #Assemble integrals over triangular elements into linear system
    for i in np.arange(N):
        node_idx = np.array([tri_bois[i][0],tri_bois[i][1],tri_bois[i][2]])#xy-coordinates of verticies
        for k in np.arange(3):
            r = node_idx[k]#Node index corresponding to row r of matrix
            if k > 0:
                for j in np.arange(k):
                    A[r][j] = A[r][j] + z_arr[i][k][j] + J_arr[i][k][j]
                    A[j][r] = A[j][r] + z_arr[i][k][j] + J_arr[i][k][j]#Matrix is symmetric
            A[r][r] = A[r][r] + z_arr[i][k][k] + J_arr[i][k][k]
            b[r] = b[r] + H_arr[i][k]
    #Solve the sparse linear system 
    sp_A = csr_matrix(A)
    x_sol = spsolve(sp_A,b)
    return x_sol

def findGridIdx(x, y, gp):
    """
    Function to find the index in the array gp of the point (x,y)
    """
    for ii in np.arange(gp.shape[0]):
        if gp[ii][0] == x and gp[ii][1] == y:
            return ii
    return -1

def graphFEMSol(grid_points, tri_bois, coefs, linear_polys, alpha, bd_path):
    """
    Function to generate a 3D plot of the solution obtained using the finite element method

    Parameters:
    grid_points: M x 2 representing xy-coordinates of grid points/nodes
    tri_bois: N x 3 numpy array of node indices of the three vertices of each triangle 
    coefs: Numpy array of length M of two-variable polynomial coefficients that solve the linear system of equations that results from the finite element method
    linear_polys: N x 3 x 3 array of the linear polynomials used in the finite element method
    alpha: Fraction of incident wave power that is aborbed by the wall
    bd_path: Numpy array of length 2 arrays that represents a path around the boundary of the shape
    """
    print("Graphing started")
    M = grid_points.shape[0]
    E_phasor = np.zeros(M)
    x_d = grid_points[:,0]
    y_d = grid_points[:,1]
    for ii in np.arange(M):
        #Evaluate function obtained by FEM at grid point (x,y)
        t_ii,t_jj = np.where(tri_bois == ii)
        r_ii = t_ii[0]
        r_jj = t_jj[0]
        E_phasor[ii] = (1.0e-6)*coefs[ii]*(linear_polys[r_ii][r_jj][0] + linear_polys[r_ii][r_jj][1]*grid_points[ii][0] + linear_polys[r_ii][r_jj][2]*grid_points[ii][1])#Scale
    print(E_phasor)
    p_density = np.zeros((M,M))
    for ii in np.arange(M):
        for jj in np.arange(M):
            k = findGridIdx(x_d[ii], y_d[jj], grid_points)
            p_density[ii][jj] = 0.5*(E_phasor[k]**2)
    print(p_density)
    #Contour plot of the power density
    plt.figure()
    fig, ax = plt.subplots()
    cs = ax.contourf(x_d, y_d, p_density, cmap="cool")
    cbar = fig.colorbar(cs)
    plt.scatter(x_d,y_d, marker='x', linewidths=1, c='g')

    #Clip graph so it only plots the contours within the simulation region
    ax.set_xlim(-0.5, 13.5)
    ax.set_ylim(-0.5, 7.5)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    #### 'Universe polygon': 
    ext_bound = Polygon([(xlim[0], ylim[0]), (xlim[0], ylim[1]), (xlim[1], ylim[1]), (xlim[1], ylim[0]), (xlim[0], ylim[0])])
    #### Clipping mask as polygon:
    pointList = []
    for ii in np.arange(bd_path.shape[0]):
            pointList.append(Point(bd_path[ii][0], bd_path[ii][1]))
    inner_bound = Polygon([ (p.x, p.y) for p in pointList])
    #### Mask as the symmetric difference of both polygons:
    mask = ext_bound.symmetric_difference(inner_bound)
    ax.add_patch(PolygonPatch(mask, facecolor='white', zorder=1, edgecolor='white'))  

    plt.title("WiFi Power Density, Alpha = %f " % alpha)
    plt.show()
    
def main():
    #Note: Scale all quantities for "whole SI units" i.e. meters
    #Parameters not dependent on omega or alpha
    alpha_arr = np.array([0.05,0.2,0.5,1.0,1.5,2.0])
    t_FEM1 = generateFEM()#Generate FEM grid and triangles
    FEM_triangles = np.array(t_FEM1['triangles'].tolist())
    nodes = np.array(t_FEM1['vertices'].tolist())
    node_markers = np.array(t_FEM1['vertex_markers'].tolist()).flatten()#Marker is 1 if node is on the boundary, 0 otherwise
    #Identify boundary points
    bd_p = []
    for ii in np.arange(len(node_markers)):
        if node_markers[ii] == 1:
            bd_p.append(nodes[ii])
    boundary_points_ordered = np.array(bd_p)
    linear_polynomials = generateLinearPolynomials(FEM_triangles, nodes, node_markers)
    f = 0.8e-6#Constant source term in PDE, account for scaling in frequency

    #Parameter sweep for radiation coefficient alpha
    for alpha in alpha_arr:
        PI = np.pi
        f_arr = np.array([2.4e6,5.0e6])#Scale by 10e-3 in frequency, so scale k^2 by 10e-6
        u_0 = 12.5663706144e-7
        eps_0 = 8.8541878176e-12
        coefs = np.zeros(nodes.shape[0])
        J_int = calcLineIntegrals(FEM_triangles, linear_polynomials, boundary_points_ordered, node_markers, alpha)
        for f in f_arr:
            omega = 2*PI*f
            k_sq = (omega**2)*u_0*eps_0#Parameter in Helmholtz equation
            z_arr, H_arr = calcTriangleIntegrals(linear_polynomials, FEM_triangles, nodes, k_sq, f)
            coefs = coefs + solveFEMSystem(z_arr, H_arr, J_int, FEM_triangles, nodes)
        coefs = coefs
        graphFEMSol(nodes, FEM_triangles, coefs, linear_polynomials, alpha, boundary_points_ordered)
   
if __name__ == "__main__":
    main()