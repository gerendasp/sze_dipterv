import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg16 import VGG16


# Load dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()


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


# Build network
batch_size = 32
epochs = 100
data_augmentation = True
num_classes = 10


# Preprocess images
x_train = preprocess_input(x_train)
x_test = preprocess_input(x_test)


# Preprocess data
input_shape = x_train.shape[1:]
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)


# Create model
model = VGG16(include_top=True,
                 weights=None,
                 input_tensor=None,
                 input_shape=(32,32,3),
                 pooling='avg',
                 classes=10)
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