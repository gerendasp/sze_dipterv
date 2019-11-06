from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.applications.densenet import preprocess_input
from keras.applications import DenseNet121
import os

os.environ["KERAS_BACKEND"] = "theano"

# Load dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()


## Resize dataset

#import cv2

#WIDTH = 224
#HEIGHT = 224
#    
#x_test_resized = []
#for img in range(0, len(x_test)):
#
#    full_size_image = x_test[img]
#    x_test_resized.append(cv2.resize(full_size_image, (WIDTH,HEIGHT), interpolation=cv2.INTER_CUBIC))
#
#x_test_resized = np.reshape(x_test_resized,(len(x_test),WIDTH,HEIGHT,3))
#
#
#x_train_resized = []
#for img in range(0, len(x_train)):
#
#    full_size_image = x_train[img]
#    x_train_resized.append(cv2.resize(full_size_image, (WIDTH,HEIGHT), interpolation=cv2.INTER_CUBIC))
#
#x_train_resized = np.reshape(x_train_resized,(len(x_train),WIDTH,HEIGHT,3))


# Preprocess images
x_train = preprocess_input(x_train)
x_test = preprocess_input(x_test)


# Build network

batch_size = 32
epochs = 100
data_augmentation = True
num_classes = 10


input_shape = x_train.shape[1:]
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print('y_train shape:', y_train.shape)

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = DenseNet121(
    weights=None, 
    include_top=True, 
    classes=10,
    input_shape=(32,32,3)
)


model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])


model.summary()

# Train model

model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True)


# Evaluate model

scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])