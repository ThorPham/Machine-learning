def conv_block(tensor, f ,k):
#     x = conv_bn_rl(tensor, f1)
#     x = conv_bn_rl(x, f1, 3, s=s)
    #x = keras.layers.ZeroPadding2D(padding=(1, 1))(tensor)
#     x = Conv2D(filters=f, kernel_size=k,padding="valid")(tensor)
    x = Conv2D(filters=f, kernel_size=k,padding="same")(tensor)
    x = BatchNormalization()(x)
    if x.shape[-1] != tensor.shape[-1]:
        tensor = Conv2D(filters=f, kernel_size=2,padding="same")(tensor)
#       shortcut = Conv2D(f2, 1, strides=s, padding='same')(tensor)
        #shortcut = BatchNormalization()(shortcut)

    x = add([tensor, x])
    output = ReLU()(x)

    return output
inputs = Input(name='the_inputs', shape=(128,32,3), dtype='float32')
x = layers.Conv2D(64, (7, 7),
                  padding='same',strides=(2,2),
                  name='block1_conv1')(inputs)
x = BatchNormalization()(x)
x = layers.ReLU()(x)
x = layers.MaxPooling2D((3, 3), strides=(2, 2),padding='same', name='max_poo_1')(x)
x = conv_block(x, 64 ,3)
x = conv_block(x, 64 ,3)
x = conv_block(x, 128 ,3)
x = conv_block(x, 128 ,3)
x = conv_block(x, 256 ,3)
x = conv_block(x, 256 ,3)
x = conv_block(x, 512 ,3)
x = conv_block(x, 512 ,3)
x = Conv2D(filters=512, kernel_size=(2,3),padding="same")(x)
model = Model(inputs, x)
model.summary()