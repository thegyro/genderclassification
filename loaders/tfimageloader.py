
import tensorflow as tf
import os
class TFImageLoader:
	
	def __init__(self, base_dir, image_shape):
		"""
		Args:
		- base_dir : The directory where the LFW set is.
		- image_shape: a list of 3 elements [height, width, channels]

		"""
		if not os.path.isdir(base_dir):
			raise Exception( "%s is not a directory"%base_dir )

		self.base_dir = base_dir
		self.image_shape = image_shape


	def load_into_dict_list(self, dict_list, use_greyscale=False, shrink_by = 1, flatten=False):
		""" 
		Loads the images with key 'image' into each dict in dict_list
		For each `item` in dict_list,
		Stores the image specified by item['filename'] into item['image']
		The image will be stored as a numpy array representation
		
		The images can be shrunk or flattened.
		* A flattened image will be a 1D array of size 
			 (height*width*channels).
		* A non-flattened image will be a 3D array of shape
			  [ height, width, channels] 
		
		Flattened images are useful for regular models.
		Whereas, unflattened 3D representations are useful for convnets


		Args:
		- dict_list : A list of dictionaries of the form
		 		[ {'filename': 'path/to/image.jpg', ...}, ...] 
		- shrink_by : The factor to shrink the image by.
		- flatten	: Whether or not to flatten the image
		
		Returns:
		- dict_list	with the 'image' key set
		
		"""
		# PREPARE THE TENSORFLOWGRAPH!
		single_image = tf.placeholder(tf.string)
		channels = 1 if use_greyscale else 3
		image_decoder = tf.image.decode_jpeg( single_image, channels=channels )
		
		

		if shrink_by != 1:
			final_dim0 = self.image_shape[0]/shrink_by
			final_dim1 = self.image_shape[1]/shrink_by
			image_resized = tf.cast( 
					tf.image.resize_images( 
						image_decoder, final_dim0, final_dim1
					),
					tf.uint8
				)
		else:
			image_resized = image_decoder

		if flatten:
			final_size = (self.image_shape[0] * self.image_shape[1] * channels/ shrink_by**2)
			image_final = tf.reshape( image_resized,[final_size] )
		else:
			image_final = image_resized
		
		with tf.Session() as sess:
			for item in dict_list:
				try:
					f = open( os.path.join(self.base_dir, item['filename']) )
					img = f.read()
					f.close()
					item['image'] = sess.run(
						image_final, 
						feed_dict =  { single_image: img } 
					)
				except:
					print "Could not open %s"%item['filename']
					
		return dict_list
