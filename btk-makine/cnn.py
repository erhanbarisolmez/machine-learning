from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense

# ilkleme
classifier = Sequential()

# Adım 1 - Convolution
classifier.add(Convolution2D(32, 3, 3, input_shape = (64,64, 3), activation='relu'))

# Adım 2 - Pooling
classifier.add(MaxPool2D(pool_size=(2,2)))

# 2. convolution katmanı
classifier.add(Convolution2D(32,3,3, activation='relu'))
classifier.add(MaxPool2D(pool_size=(2,2)))

# Adım 3 - Flattening
classifier.add(Flatten())

# Adım 4 - YSA
classifier.add(Dense(units = 128, activation='relu'))
classifier.add(Dense(units = 1, activation='sigmoid'))

# CNN 
classifier.compile(optimizer= 'adam', loss='binary_crossentropy', metrics= ['accuracy'])

#CNN ve Resimler
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255,
                                  shear_range = 0.2,
                                  zoom_range = 0.2,
                                  horizontal_flip = True)

training_set = train_datagen.flow_from_directory('btk-makine/data/veriler/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 1,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('btk-makine/data/veriler/test_set',
                                            target_size = (64, 64),
                                            batch_size = 1,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         samples_per_epoch = 8000,
                         nb_epoch = 1,
                         validation_data= test_set,
                         nb_val_samples = 2000)