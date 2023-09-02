import imageio 
from imageio import imread,imsave
im=imread("/home/edouard/Desktop/Programming/codingWithPython/MachineLearning/UpdatedTutorial/Images/imagefiles/my photo.jpg")

# let me tint the image
tint_img=im*[3,5,10]

# let me save the image
imsave("/home/edouard/Desktop/Programming/codingWithPython/MachineLearning/UpdatedTutorial/Images/imagefiles/my photot.jpg",tint_img)
