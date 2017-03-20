from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.layers import Dense, AveragePooling2D, Flatten, Dropout, Input
from keras import backend as K

train_datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

validation_datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

train_img = 'train/' 
validation_img = 'validation/'

if(K.image_dim_ordering() == 'th'):
    input_tensor = Input(shape=(3, 299, 299))
else:
    input_tensor = Input(shape=(299, 299, 3))
base_model = InceptionV3(input_tensor=input_tensor, weights='imagenet', include_top=False)

x = base_model.output
x = AveragePooling2D((8, 8), border_mode='valid', name='avg_pool')(x)
x = Dropout(0.4)(x)
x = Flatten()(x)
predictions = Dense(2, activation='softmax')(x)

model = Model(input=base_model.input, output=predictions)

for layer in base_model.layers:
    layer.trainable = False

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
train_generator = train_datagen.flow_from_directory(
			train_img,
			target_size=(299, 299),
			batch_size=32,
			class_mode='categorical'
			)
validation_generator = validation_datagen.flow_from_directory(
			validation_img,
			target_size=(299, 299),
			batch_size=32,
			class_mode='categorical'
			)
model.fit_generator(train_generator, samples_per_epoch=320, nb_epoch=5, validation_data=validation_generator, nb_val_samples=64)
model.save('dog_vs_catTensorflow.h5')