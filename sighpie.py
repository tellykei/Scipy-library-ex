from scipy import linalg, stats, ndimage,misc
from scipy.special import comb
from scipy.special import perm
from scipy.optimize import minimize
from scipy.ndimage import geometric_transform
import matplotlib.pyplot as plt
import numpy as np
import imageio
from scipy.interpolate import interp1d

def menu():
    
    x = int(input("Select which example to run\n1.Image Processing\n2. Linear Algebra"
        "\n3. Statistics\n4. Optimization\n5. Interpolation\n"))
    if(x==0):
        return
    if(x== 1):
        imgprocess()
    if(x == 2):
        maths()
    if(x == 3):
        statss()
    if(x == 4):
        optimized()
    if(x == 5):
        interpol()
    

def imgprocess():
    p = misc.face()
    imageio.imwrite('face.png', p)
    #plot or show image of face
    plt.imshow( p )
    plt.show()
    #rotate the face
    protate = ndimage.rotate(p, 135)
    plt.imshow(protate)
    plt.show()
    # retrieve a grayscale image
    p = misc.face(gray=True)
    plt.imshow(p, cmap=plt.cm.gray) 
    plt.show()
    return menu()

def maths():
    #eigenvalues
    arr1= np.array([[2,6],[6,3]]) #creating a 2-D matrix array
    #eg_val returns the eigen value, eg_vect returns the right or left eigen vectors
    eg_val, eg_vect = linalg.eig(arr1)
    print("eigen values:" , eg_val , "\neigen vectors:\n" , eg_vect)

    #put a [1] or [0] to return either the vectors or the values
    egval = print("\nspecified eigen values:",linalg.eig(arr1)[0])
    return menu()

def statss():
    #calculates the T-test for the means of two independent scores
    rvs1 = stats.norm.rvs(loc = 5,scale = 10,size = 500)
    rvs2 = stats.norm.rvs(loc = 5,scale = 10,size = 500)
    print (stats.ttest_ind(rvs1,rvs2), "\n")
    #Permutations and Combinations
    com = print("Combinations",(comb(6, 3, exact = False, repetition=True)))
    #exact gives an integer
    per = print("Permutations", (perm(2,6, exact = True)))
    return menu()

def optimized():
    #nelder-mead method
    def f(x):   
        return .4*(1 - x[0])**2
    print("Nelder- Mead\n", minimize(f, [2, -1], method="Nelder-Mead"))
    return menu()

def interpol():
    x = np.linspace(0, 10, num=11, endpoint=True)
    y = np.cos(-x**2/6.0)
    f = interp1d(x, y)
    f2 = interp1d(x, y, kind='cubic')
    xnew = np.linspace(0, 10, num=41, endpoint=True)
    plt.plot(x, y, 'o', xnew, f(xnew), '-', xnew, f2(xnew), '--')
    plt.legend(['data', 'linear', 'cubic'], loc='best')
    plt.show()
    return menu()

menu()