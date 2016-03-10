import random
from loaders import lfwgcloader

gcloader = lfwgcloader.LFWGenderClassificationLoader('lfw/LFW-gender-folds.dat','lfw', shrink_by=2, use_greyscale=True)

dict_list = gcloader.load_data_file()


subdict = random.sample(dict_list, 5)

# Load up the images
subdict_wimages = gcloader.load_images( subdict )

# And visualize what we have
gcloader.show_image_from_dict( subdict_wimages[0])
