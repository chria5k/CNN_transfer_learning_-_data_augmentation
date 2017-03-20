from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

img = 'train/' 
i = 0
for batch in datagen.flow_from_directory(img,
                          save_to_dir='previewNew', save_prefix='train', batch_size=1, save_format='jpeg', target_size=(299, 299)):
    i += 1
    if i > 50:
        break