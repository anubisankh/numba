#Compare the performance between GPU and CPU.

import numpy
import time
from numba import double, jit

def filter2d(image, filt):
    
    M, N = image.shape
    Mf, Nf = filt.shape
    print(M,N,Mf,Nf) #print image and filter sizes
    Mf2 = Mf // 2 #divide to integer 
    Nf2 = Nf // 2 #divide to integer
    
    result = numpy.zeros_like(image)
    for i in range(Mf2, M-Mf2):
        for j in range(Nf2, N-Nf2):
            num = 0.0
            for ii in range(Mf):
                for jj in range(Nf):
                    num += (filt[Mf-1-ii, Nf-1-jj] * image[i-Mf2+ii, j-Nf2+jj])
            result[i, j] = num
    return result

def main():
    
    image = numpy.random.random((100, 100)) #Define image size
    filt = numpy.random.random((50, 50)) #Deinf filter size
    
    #Using cpu for regular 2D filter
    start = time.time()
    res = filter2d(image, filt)
    filter2d_time = time.time()-start
    print(res)
    print("2D filter took %s seconds for cpu." %filter2d_time)

    #Implement gpu (cuda) jit for faster 2D filter
    fastfilter_2d = jit(double[:,:](double[:,:], double[:,:]))(filter2d)
    start = time.time()
    res2 = fastfilter_2d(image, filt)
    fastfilter2d_time = time.time()-start
    print(res2)
    print("2D filter took %s seconds for gpu." %fastfilter2d_time)

    speed=filter2d_time/fastfilter2d_time

    print("GPU is %s times faster than CPU" %speed)

if __name__ == "__main__":
    main()
