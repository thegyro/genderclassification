from loaders import tfimageloader
tfil = tfimageloader.TFImageLoader('./lfw', [250,250,3])

test_dict = [ {'image':None, 'filename': "Zinedine_Zidane/Zinedine_Zidane_0001.jpg"}] 

res_dict = tfil.load_into_dict_list( test_dict )
flat_dict = tfil.load_into_dict_list( test_dict, flatten=True )
shrunk_dict = tfil.load_into_dict_list( test_dict, shrink_by=2 )
flat_shrunk_dict = tfil.load_into_dict_list( test_dict, shrink_by=2, flatten=True )