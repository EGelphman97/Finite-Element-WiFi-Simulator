# Finite-Element-WiFi-Simulator
ssWFS is a Python program that calculates the (relative) power density of the wifi signal over the downstairs of 
my house. The program solves the inhomogeneous Helmholtz equation with inverse-square source and with PML boundary
conditions at the boundary using the finite element method and then generates a filled contour plot of the power density. 
The script is configured for automatic documentation generation using Python's built-in command line tools. 

To run the script, simply enter 

$ python ssWFS.py

into any command line interface. The script can also be run from any Python IDE such as IDLE or VS Code. There are
dependencies on the following Python libraries, all of which are easily installed from the command line using pip:

Numpy
SciPy
Matplotlib
Triangle: https://rufat.be/triangle/
Shapely: https://pypi.org/project/Shapely/

Possible extensions of this program include modifiying the code such that it solves general second order linear 
elliptic PDEs with Robin boundary conditions as well as including a function that can read in parameters 
desribing the domain of the problem using a file

Eric Gelphman
University of California Irvine Department of Mathematics
