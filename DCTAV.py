import cv2 as cv
from cv2 import COLOR_BGR2GRAY
import numpy as np
#import matplotlib.pyplot as plt

pi=3.142857
m = 2
n = 2

img1 = cv.imread("3.jpg")

img2 = cv.imread("4.jpg")

cv.imshow("img1",img1)
cv.waitKey(0)
cv.imshow("img2",img2)
cv.waitKey(0)

img2 = cv.resize(img2,(304,304))
img1 = cv.resize(img1,(304,304))
B=8

a=img1.shape[0]%8
b=img1.shape[1]%8
x = img1.shape[0]-a
y =  img1.shape[1]-b
for i in range(0,2):
    if((img1.shape[i]%8)!=0):
        img1 = cv.resize(img1,(y,x))

#print(img1.shape)

ims=np.hsplit(img1,4)

image = gray_frame=cv.cvtColor(img1,COLOR_BGR2GRAY)

sub_shape = (2,2)

#divide the image1 into sub_matrices of subshape
view_shape = tuple(np.subtract(image.shape, sub_shape) + 1) + sub_shape
strides = image.strides + image.strides

print(view_shape)
print("888")
print(strides)
sub_matrices1 = np.squeeze(np.lib.stride_tricks.as_strided(image,view_shape,strides)[::sub_shape[0],::sub_shape[1],:])
#divide the image2 into sub_matrices of subshape
image2=cv.cvtColor(img2,COLOR_BGR2GRAY)
view_shape2 = tuple(np.subtract(image2.shape, sub_shape) + 1) + sub_shape
strides2 = image2.strides + image2.strides
sub_matrices2 = np.squeeze(np.lib.stride_tricks.as_strided(image2,view_shape2,strides2)[::sub_shape[0],::sub_shape[1],:])
#print(sub_matrices)
 
#print(np.shape(sub_matrices1))
#print(np.shape(sub_matrices2))


#print("***********")
#a=  cv.dct(sub_matrices[0][1])



def dctTransform(inp_matrix):
  #dct=np.full((8, 8),0)
  dct=np.arange(0,4).astype(float).reshape(2,2)

  for i in range(0,m):
    for j in range(0,n):


      
      if (i == 0):
        ci = 1 / np.sqrt(m)
      else:
        ci = np.sqrt(2) / np.sqrt(m)
      if (j == 0):
        cj = 1 / np.sqrt(n)
      else:
        cj = np.sqrt(2) / np.sqrt(n)

      
      
      sum = 0
      for k in range(0,m): 
        for l in range(0,n): 
          dct1 = inp_matrix[k][l] *np.cos((2 * k + 1) * i * pi / (2 * m)) *np.cos((2 * l + 1) * j * pi / (2 * n))
          sum = sum + dct1
          
      
      dct[i][j] = (ci * cj * sum)
      #print(dct[0][0])
  return(dct)

after_dct=np.arange(0,23104).reshape(152,152)
after_dct=after_dct.tolist()
#temp = [0 for x in range(38)]
#transform_1 = [temp for x in range(38)]
#print(dctTransform(sub_matrices1[0][0]))
#print("-------------------")
#print(dctTransform(sub_matrices2[0][0]))
x=0.5*(dctTransform(sub_matrices1[0][0])+dctTransform(sub_matrices2[0][0]))


#print(x)

for i in range(0,152):
  for j in range(0,152):
    after_dct[i][j]= 0.5*(dctTransform(sub_matrices1[i][j])+dctTransform(sub_matrices2[i][j]))
    
final=np.arange(0,23104).reshape(152,152)
final=final.tolist()
#print("***************************************************************")
for i in range(0,152):
  for j in range(0,152):
    final[i][j]=(dctTransform(after_dct[i][j]))

#print(final[0][0])
#print(">>>>>>>>>>>>>>>")
#print(final[0][1])
#print(">>>>>>>>>>>>>>")

im=[0 for i in range(152)]


for k in range(0,152):
  im[k] = cv.hconcat([final[k][0],final[k][1]])
  for i in range(2,152):
    im[k] = cv.hconcat([im[k],final[k][i]])
final2 = cv.vconcat([im[0],im[1]])
for j in range(2,152):
  final2 = cv.vconcat([final2,im[j]])
final2 = cv.convertScaleAbs(final2, alpha=1, beta=0)
#final2 = cv.resize(final2,(500,500))
cv.imshow("final",final2)
cv.waitKey(0)
print(final)
cv.destroyAllWindows()
