from PIL import Image
import os

def filedirectory(filedir):
    """

    :param filedir:
    :return:
    """
    filelist = []
    for filename in os.listdir(filedir):
        f = os.path.join(filedir, filename)
        filelist.append(f)
    return filelist

def convert_image(picture):
    """

    :param picture:
    :return:
    """
    im = Image.open(picture)
    new_image = im.resize((256, 256))
    new_image.save(picture)

def main():
    allfilelists = []
    fileswithpicturepath = []
    # Kijkt naar de eerste volgende path (test, train, validatie)
    filelist = filedirectory(r'./beans_data')

    # Kijkt naar de eerst volgende path (bean, conan, doraemon, naruto, shinchan)
    for i in filelist:
        filelist = filedirectory(i)
        allfilelists.append(filelist)

    # Kijkt naar images in de hele path (1.jpg, 2.jpg, 3.jpg, 4.jpg, 5.jpg etc...)
    for i in allfilelists:
        for i2 in i:
            filewithpicturelist = filedirectory(i2)
            fileswithpicturepath.append(filewithpicturelist)

    # Pakt de hele path van de images
    for dir in fileswithpicturepath:
        for picture in dir:
            # Zet de images om van jpg naar png
            convert_image(picture)
main()
