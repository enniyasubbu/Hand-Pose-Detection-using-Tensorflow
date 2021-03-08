from PIL import Image

def resizeImage(imageName):
    basewidth = 100
    img = Image.open(imageName)
    #wpercent = (basewidth/float(img.size[0]))
    hsize = 100
    img = img.resize((basewidth,hsize), Image.ANTIALIAS)
    img.save(imageName)

for i in range(1000, 1101):
    # Mention the directory in which you wanna resize the images followed by the image name
    try:  
     resizeImage("Dataset/Why_test/why_" + str(i) + '.png')
    except IOError:
     pass
