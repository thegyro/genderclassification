from .tfimageloader import TFImageLoader
from PIL import Image as PIL_Image
import numpy as np

class LFWGenderClassificationLoader:

	LFW_IMAGE_SHAPE = [250,250]

	def __init__(self, data_filename, base_dir='.', 
		use_greyscale = False, flatten=False, shrink_by=1
		):
		""" 
		Sets the base_dir for images and the data file to use
		self.data has the keys:
			filename: 	Path to the image, relative to base_dir
			fold	: 	Which fold the sample belongs to
			gender	:	Which gender the sample is of
			image 	: 	Decoded contents of the image ( numpy array )

		Args:
		- base_dir: Base directory of the LFW images dataset
		- data_filename: The text file with the info relevant to the task
		"""
		self.base_dir = base_dir
		self.data_file = data_filename
		
		self.options = {'use_greyscale': use_greyscale, 'flatten':flatten, 'shrink_by': shrink_by}
		self.data = None

	def get_image_shape(self):
		""" Returns the post-shrink image size """
		
		if self.options['use_greyscale']:
			return (
				self.__class__.LFW_IMAGE_SHAPE[0]/self.options['shrink_by'],
				self.__class__.LFW_IMAGE_SHAPE[1]/self.options['shrink_by'],
			)
		else:
			return (
				self.__class__.LFW_IMAGE_SHAPE[0]/self.options['shrink_by'],
				self.__class__.LFW_IMAGE_SHAPE[1]/self.options['shrink_by'],
				3
			)

	def get_PIL_image_mode(self):
		return 'L' if self.options['use_greyscale'] else 'RGB'

	def load(self):
		""" Single point to load everything. 
		Supports shrinking and flattening of images as in TFImageLoader
		"""
		self.load_data_file()
		self.load_images()

	
	def load_data_file(self):
		""" Loads the data from the file and populates self.data """
		def _make_data_from_line(l):
			field = l.split('\t')
			
			return { 
				"filename": field[0],
				"fold": field[1],
				"gender": field[2],
				"image": None,
			}
	
	
		f = open(self.data_file,"r")
		lines = f.read().split('\n')
		f.close()
		# Prune empty lines
		lines = [l for l in lines if l]
		self.data = [ _make_data_from_line(l) for l in lines ]
		return self.data


	def load_images(self, data=None ):
		""" loads the image for each item in `data`.
		`data` defaults to self.data.
		Supports shrinking and flattening of images as in TFImageLoader
		"""
		tfil = TFImageLoader(self.base_dir, self.__class__.LFW_IMAGE_SHAPE)

		if data is None:
			use_data = self.data
		else:
			use_data = data 
		
		use_data = tfil.load_into_dict_list(
			use_data,
			flatten=self.options['flatten'],
			shrink_by=self.options['shrink_by'],
			use_greyscale = self.options['use_greyscale']
		)

		if data is None:
			self.data = use_data

		return use_data
	
	def show_image(self, raw_image):
		image_mode = self.get_PIL_image_mode()
		to_shape = self.get_image_shape()
		
		shaped_image = np.reshape( raw_image, to_shape )
		
		PIL_Image.fromarray(shaped_image, image_mode).show()

	def show_image_from_dict(self, data_dict ):
		self.show_image( data_dict['image'] )
		