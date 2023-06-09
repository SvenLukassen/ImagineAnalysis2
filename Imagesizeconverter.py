from PIL import Image
import os


def filedirectory(filedir):
    """ Opens a folder with multiple files in it and puts the
    filenames into a list

    :return: List of filenames - list
    """
    try:
        filelist = []
        # Gets all the filenames from a certain folder
        for filename in os.listdir(filedir):
            # Makes a value of the path for the input files
            f = os.path.join(filedir, filename)
            filelist.append(f)
    except NotADirectoryError:
        print("Directory not found")
    except FileExistsError:
        print("File already exists")
    except FileNotFoundError:
        print("File not found")

    return filelist


def convert_image(picture):
    """Converts image sizes to 256,256

    :param picture: picture from a directory
    :return: picture with the size 256x256
    """
    try:
        im = Image.open(picture)
        # Resizes the image to 256x256
        new_image = im.resize((256, 256))
        # Saves the new picture
        new_image.save(picture)
    except FileNotFoundError:
        print("File not found")


def main():
    allfilelists = []
    fileswithpicturepath = []
    # Checks a folder and gives back the test, validation, training folders
    filelist = filedirectory(r'./beans_data')

    # Checks the folders test, validation, training and returns folders
    # angular_leaf_spot, bean_rust, healthy
    for i in filelist:
        filelist = filedirectory(i)
        allfilelists.append(filelist)

    # Gives the whole path of the images in the folder angular_leaf_spot,
    # bean_rust, healthy
    for i in allfilelists:
        for i2 in i:
            filewithpicturelist = filedirectory(i2)
            fileswithpicturepath.append(filewithpicturelist)

    for dir in fileswithpicturepath:
        for picture in dir:
            # Converts image sizes
            convert_image(picture)
main()
