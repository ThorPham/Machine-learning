

def model_cnn(inputs):
	#inputs = Input(name='the_inputs', shape=(120,32,3), dtype='float32')
	x = layers.Conv2D(64, (3, 3),
                  activation='relu',
                  padding='same',
                  name='block1_conv1')(inputs)
	x = layers.Conv2D(64, (3, 3),
	                  activation='relu',
	                  padding='same',
	                  name='block1_conv2')(x)
	x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

	# Block 2
	x = layers.Conv2D(128, (3, 3),
	                  activation='relu',
	                  padding='same',
	                  name='block2_conv1')(x)
	x = layers.Conv2D(128, (3, 3),
	                  activation='relu',
	                  padding='same',
	                  name='block2_conv2')(x)
	x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
	skip_1 = x

	# Block 3
	x = layers.Conv2D(256, (3, 3),
	                  activation='relu',
	                  padding='same',
	                  name='block3_conv1')(x)
	x = layers.Conv2D(256, (3, 3),
	                  activation='relu',
	                  padding='same',
	                  name='block3_conv2')(x)
	y_1 = Concatenate(axis=-1)([skip_1,x])
	x = layers.MaxPooling2D((2,2), strides=(2, 2), name='block3_pool')(y_1)
	skip_2 = x
	# Block 4
	x = layers.Conv2D(512, (3, 3),
	                  activation='relu',
	                  padding='same',
	                  name='block4_conv1')(x)
	x = layers.Conv2D(512, (3, 3),
	                  activation='relu',
	                  padding='same',
	                  name='block4_conv2')(x)
	x = layers.Conv2D(512, (3, 3),
	                  activation='relu',
	                  padding='same',
	                  name='block4_conv3')(x)
	y_2 =  Concatenate(axis=-1)([skip_2,x])
	x = layers.MaxPooling2D((1, 2), strides=(1, 2), name='block4_pool')(y_2)

	# Block 5
	x = layers.Conv2D(512, kernel_size=(1, 2),strides=(1,2),
	                  activation='relu',
	                  padding='valid',
	                  name='block5_conv1')(x)
# 	x = layers.Conv2D(256, (3, 3),
# 	                  activation='relu',
# 	                  padding='same',
# 	                  name='block5_conv2')(x)
# 	x = layers.Conv2D(256, (3, 3),
# 	                  activation='relu',
# 	                  padding='same',
# 	                  name='block5_conv3')(x)
# 	x = layers.MaxPooling2D((2, 2), strides=(2, 1), name='block5_pool')(x)


	model = models.Model(inputs, x, name='cnn')
	return model
