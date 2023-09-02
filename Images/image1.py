# this is to import scipy to deal with images 

# from scipy.misc import imread,imsave
import imageio
from imageio import imread,imsave

# this is to read image  
img=imread("/home/edouard/Desktop/Programming/codingWithPython/MachineLearning/UpdatedTutorial/Images/imagefiles/IMG_20230416_095707_2.jpg")
print(img.dtype,img.shape)
print(img)

# let me tint the image
tintImage=img*[1,0.45,0.3]

# let me save the the tinted image
imsave("/home/edouard/Desktop/Programming/codingWithPython/MachineLearning/UpdatedTutorial/Images/imagefiles/tintedImage1.jpg",tintImage)

# let me resize the image 
# resizedImage=imresize(img,(400,400))

# # let me save the resized image
# imsave("/home/edouard/Desktop/Programming/codingWithPython/MachineLearning/UpdatedTutorial/Images/imagefiles/resiedImage.jpg",resizedImage)
