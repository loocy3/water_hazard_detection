import math

def logTransform(c, f):
    g = c * math.log(float(1 + f), math.e)
    return g

def logTransformImage(image, outputMax=255, inputMax=255):
    c = outputMax / math.log(inputMax + 1, math.e)

    for i in range(0, image.shape[0] - 1):
        for j in range(0, image.shape[1] - 1):
            # Get pixel value at (x,y) position of the image

            # Do log transformation of the pixel

            image[i, j, 0] = round(logTransform(c, image[i, j, 0]))

            image[i, j, 1] = round(logTransform(c, image[i, j, 1]))

            image[i, j, 2] = round(logTransform(c, image[i, j, 2]))

    return image
