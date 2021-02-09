"""
Eric Gelphman
University of California Irvine Department of Mathematics

Program using various Python packages to solve the linear Helholtz equation in the phasor domain 
with inverse-square source term with homogeneous Dirichlet boundary conditions(set 3.0m from the house, 
a crude PML boundary) over the domain of my house using the linear finite element method. Section 9.4 of
Kincaid and Cheney, Section 12.5 of Burden and Faires, and Chapter 9 of Reddy were the three main 
resources consulted in the creation of this program.

February 9, 2020
v1.1.1
"""

import numpy as np
import triangle
from scipy.integrate import nquad
from scipy.interpolate import griddata
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import gmres
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from shapely.geometry import Point, Polygon

def triangulationFEM(gp):
    """
    Function to compute the Delaunay triangulation of the simulation region

    Parameters:
    gp: L x 2 array of xy-coordinates of grid points to be enforced in the triangulation

    Return:
    tri_FEM: Delaunay FEM triangulation 
    """
    tri_dict = dict(vertices=gp)
    tri_FEM = triangle.triangulate(tri_dict, 'qD')#Initial triangulation 
    return tri_FEM

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
    enforceable_gp = np.concatenate((lr_inner, lr_outer, br, ovr_bdy))
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
            print("inside")
            eps_r[ii] = 1.0#Point is in the interior of the house
        elif gp[ii][0] == -3.1 or gp[ii][0] == 16.1 or gp[ii][1] == -3.1 or gp[ii][1] == 10.5:
            print("on boundary")
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

def calcTriangleIntegrals(linear_polynomials, tri_bois, grid_points, k2_arr):
    """
    Function to calculate the integrals over each triangle to generate the matrix in the finite element method

    Parameters: 
    linear_polynomials: N x 3 x 3 Numpy array of linear polynomials at each node of the finite element grid
    tri_bois: N x 6 Numpy array representing the xy-coordinates of the verticies of each triangular element
    grid_points: N x 2 Numpy array of grid points
    k2_arr: Numpy array of length N that holds the value of the parameter k^2 in the Helmholtz equation at each grid point

    Return:
    z_arr: N x 3 x 3 Numpy array of double integrals
    H_arr: N x 3 Numpy array of double integrals involving inhomogeneous term
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

                #Evaluate linear polynomial times the source term f at a point (u,v)
                def h(u,v):
                    phi = (x_1 - x_0)*u + (x_2 - x_0)*v + x_0
                    psi = (y_1 - y_0)*u + (y_2 - y_0)*v + y_0
                    #Control magnitude of source term - point sources will cause the integration subroutine to fail
                    denom = ((phi - 0.5)**2 + (psi - 0.5)**2)
                    if denom < 1:
                        val = (2 - denom)*(a_j_i + (b_j_i*phi) + (c_j_i*psi))#Ensure the continuity
                    else:
                        val = (a_j_i + (b_j_i*phi) + (c_j_i*psi))/denom
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
                k_sq = 1.0
                if k2_arr[tri_bois[i][0]] == 2.7 or k2_arr[tri_bois[i][1]] == 2.7 or k2_arr[tri_bois[i][2]] == 2.7:#Triangles have eps_r=2.7 if they have at least one vertex on a wall
                    k_sq = 1.0
                z_arr[i][j][k] = (b_j_i*b_k_i*Area_triangle) + (c_j_i*c_k_i*Area_triangle) - (k_sq*double_integral_z)
            double_integral_H, error_est_H = nquad(h, [bounds_v, bounds_u])
            H_arr[i][j] = -1.0*double_integral_H
    return z_arr, H_arr

  
def solveFEMSystem(z_arr, H_arr, tri_bois, grid_points, node_markers):
    """
    Function to solve the system of equation that results from  the linear FEM method

    Parameters:
    z_arr: N x 3 x 3 Numpy array of values of inner product integrals over each triangular element
    H_arr: N x 3 Numpy array of values of inner product integrals involving inhomogeneous term
    tri_bois: N x 6 array of xy-coordinates of verticies of each triangle
    grid_points: Array of arrays of length 2 represrnting xy-coordinates of verticies
    node_markers: Array of arrays indicating whether or not a node lies on the boundary

    Return:
    x: Solution vector of the matrix equation Ax = b that results from the finite element method. This vector holds
       the coefficients of the finite elements
    """
    N = tri_bois.shape[0]
    M = grid_points.shape[0]
    int_points_idx = []#Original grid indices of the interior points
    for ii in np.arange(M):
        if node_markers[ii] == 0:
            int_points_idx.append(ii)
    int_points = np.array(int_points_idx, dtype=int)
    L = len(int_points)#Linear system is for the coefficients corresponding to the interior points
    A = np.zeros((L,L))
    b = np.zeros(L)
    int_triangle_idx = []#Indices, in FEM_triangles, of the triangles with interior points only as verticies
    for ii in np.arange(N):
        if not (node_markers[tri_bois[ii][0]] == 1 or node_markers[tri_bois[ii][1]] == 1 or node_markers[tri_bois[ii][2]] == 1):
            int_triangle_idx.append(ii)

    int_triangles = np.array(int_triangle_idx, dtype=int)
    #Assemble integrals over interior triangular elements into linear system
    for i in np.arange(len(int_triangles)):
        original_tri_idx = int_triangles[i]#Original index of triangle in FEM_triangles
        original_node_idx = np.array([tri_bois[original_tri_idx][0],tri_bois[original_tri_idx][1],tri_bois[original_tri_idx][2]])#Original node indices
        for k in np.arange(3):
            l = original_node_idx[k]
            if k > 0:
                for j in np.arange(k):
                    t = original_node_idx[j]
                    if l in int_points:#Node l is an interior point
                        l_int = int(np.where(int_points == l)[0])#Interior node index
                        if t in int_points:#Node t is an interior point
                            t_int = int(np.where(int_points == t)[0])#Interior node index
                            A[l_int][t_int] = A[l_int][t_int] + z_arr[original_tri_idx][k][j] 
                            A[t_int][l_int] = A[t_int][l_int] + z_arr[original_tri_idx][k][j]#Matrix is symmetric
            if l in int_points:
                l_int = int(np.where(int_points == l)[0])
                A[l_int][l_int] = A[l_int][l_int] + z_arr[original_tri_idx][k][k]
                b[l_int] = b[l_int] + H_arr[i][k]

    #Solve the sparse linear system 
    sp_A = csr_matrix(A)
    x_sol_interior, int_info = gmres(sp_A, b)
    x = np.zeros(M)#Build overall solution vector
    for ii in np.arange(M):
        if node_markers[ii] == 0:#Interior point
            int_idx = int(np.where(int_points == ii)[0])#Interior index
            x[ii] = x_sol_interior[int_idx]
        else:#Boundary point
            x[ii] = 0.0
    return x

def writeToFile(x_coords, y_coords, z_coords, filename):
    """
    Output the simulation results to file in (x,y,z) format
    """
    file1 = open(filename, 'r+')#open file
    N = len(z_coords)
    for ii in np.arange(N):
        file1.write("(" + str(x_coords[ii]) + "," + str(x_coords[ii]) + "," + str(z_coords[ii]) + ")" + "\n")
    file1.close()#close file


def graphFEMSol(grid_points, tri_bois, coefs, linear_poly):
    """
    Function to generate a 3D plot of the solution obtained using the finite element method

    Parameters:
    grid_points: M x 2 representing xy-coordinates of grid points/nodes
    tri_bois: N x 3 numpy array of node indices of the three vertices of each triangle 
    coefs: Numpy array of length M of two-variable polynomial coefficients that solve the linear system of equations that results from the finite element method
    """
    print("Graphing started")
    #Generate and plot numerical values of solution at the centroid of each triangle
    N = grid_points.shape[0]
    x_vals = np.zeros(N)
    y_vals = np.zeros(N)
    E_phasor = np.zeros(N)
    for ii in np.arange(N):
        x_0,y_0,x_1,y_1,x_2,y_2 = extractVertexCoordinates(tri_bois[ii], grid_points)
        #Calculate centroid
        centroid_x = (x_0 + x_1 + x_2)/3.0
        x_vals[ii] = centroid_x
        centroid_y = (y_0 + y_1 + y_2)/3.0
        y_vals[ii] = centroid_y
        coef_0 = coefs[tri_bois[ii][0]]
        coef_1 = coefs[tri_bois[ii][1]]
        coef_2 = coefs[tri_bois[ii][2]]
        E_phasor[ii] = coef_0*(linear_poly[ii][0][0] + linear_poly[ii][0][1]*centroid_x + linear_poly[ii][0][2]*centroid_y) + coef_1*(linear_poly[ii][1][0] + linear_poly[ii][1][1]*centroid_x + linear_poly[ii][1][2]*centroid_y) + coef_2*(linear_poly[ii][2][0] + linear_poly[ii][2][1]*centroid_x + linear_poly[ii][2][2]*centroid_y)

    power_density = 0.5*np.power(E_phasor,2)
    xi = np.linspace(-3.1, 16.1, 750)
    yi = np.linspace(-3.1, 10.5, 750)
    X,Y = np.meshgrid(xi,yi)
    Z = griddata((x_vals, y_vals), power_density, (X, Y), method='nearest')
    # contour the gridded data, plotting dots at the nonuniform data points.
    fig, ax = plt.subplots()
    cs = ax.contourf(X, Y, Z, cmap = 'PuBu_r')#Plot interpolated data
    cbar = fig.colorbar(cs)
    plt.scatter(grid_points[:,0],grid_points[:,1], marker='x', linewidths=1, c='g')#Plot grid points for reference

    #Clip graph so it only plots the contours within the simulation region
    plt.title("Steady-State WiFi Power Density")
    plt.show()
    writeToFile(x_vals, y_vals, power_density, "WiFiPowerDensity.txt")
    
def main():
    #Note: Scale all quantities for meters and Hz
    #Parameters not dependent on omega or alpha
    u_0 = 12.5663706144e-7
    e_0 = 8.8541878176e-12
    PI = np.pi
    t_FEM1, lr_inner, lr_outer, br, ovr_boundary = generateFEM()
    FEM_triangles = np.array(t_FEM1['triangles'].tolist())
    nodes = np.array(t_FEM1['vertices'].tolist())
    node_markers = np.array(t_FEM1['vertex_markers'].tolist()).flatten()#Marker is 1 if node is on the boundary, 0 otherwise
    eps_r_arr = calcPermitivitty(nodes, lr_inner, lr_outer, br, ovr_boundary)#Calculate permitivitty at each node
    print(eps_r_arr)
    linear_polynomials = generateLinearPolynomials(FEM_triangles, nodes, node_markers)#Calculate linear polynomials
    #Solve using FEM aout each frequency spike, then combine the two solutions using superposition - this is a linear PDE with linear BCs
    f_arr = np.array([2.4e9,5.0e9])
    coefs = np.zeros(nodes.shape[0])
    for f in f_arr:
        omega = 2*PI*f
        k2_arr = (omega**2)*u_0*e_0*eps_r_arr
        z_arr, H_arr = calcTriangleIntegrals(linear_polynomials, FEM_triangles, nodes, k2_arr)#Calculate triangle integrals
        coefs = coefs + solveFEMSystem(z_arr, H_arr, FEM_triangles, nodes, node_markers)
    graphFEMSol(nodes, FEM_triangles, coefs, linear_polynomials)

   
if __name__ == "__main__":
    main()