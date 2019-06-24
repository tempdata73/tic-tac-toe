"""Train model to detect X's, O's or None"""


import os
import cv2
import numpy as np

from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Flatten, Dense, Dropout
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator


# Replicate results
np.random.seed(42)

ROOT_DIR = '../data/images'
TRAIN_DIR = os.path.join(ROOT_DIR, 'train')
TEST_DIR = os.path.join(ROOT_DIR, 'test')
input_shape = (32, 32, 1)
batch_size = 32
epochs = 30

# Build and compile model
model = Sequential()
model.add(Conv2D(64, (3, 3), input_shape=input_shape, padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())

model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.4))

# model.add(Dense(32))
# model.add(Activation('relu'))
# model.add(Dropout(0.3))

model.add(Dense(3, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


# Load data
def load_img(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    # Fit to input size
    img = cv2.resize(img, (32, 32))
    img = np.expand_dims(img, axis=-1)
    return img.astype(np.float32)


def one_hot_encode(labels):
    mapper = {'blank': 0, 'nough': 1, 'circle': 2}
    encoded = [mapper[label] for label in labels]
    target = to_categorical(encoded)
    return target


def load_data(root_dir, shuffle=True):
    X, y = [], []
    for root, dirs, files in os.walk(root_dir):
        for class_dir in dirs:
            image_dir = os.path.join(root_dir, class_dir)
            y.extend(np.tile(class_dir, len(os.listdir(image_dir))))
            for img_fn in os.listdir(image_dir):
                img = load_img(os.path.join(image_dir, img_fn))
                X.append(img)
    y = one_hot_encode(y)
    if shuffle:
        # Combine to maintain order
        data = list(zip(X, y))
        np.random.shuffle(data)
        X, y = zip(*data)
    return np.asarray(X), np.asarray(y)


print('Loading data...')
X_train, y_train = load_data(TRAIN_DIR)
X_test, y_test = load_data(TEST_DIR)
print('{} instances for training'.format(len(X_train)))
print('{} instances for evaluation'.format(len(X_test)))


# Create more instances since our dataset is VERY limited
train_val_datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=90,
    shear_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    rescale=1 / 255,
    validation_split=0.2)

train_generator = train_val_datagen.flow(
    X_train, y_train, batch_size=batch_size, subset='training')

val_generator = train_val_datagen.flow(
    X_train, y_train, batch_size=batch_size, subset='validation')

# Train and evaluate
callbacks = [
    EarlyStopping(patience=5, verbose=1, restore_best_weights=True)]

print('Training model...')
history = model.fit_generator(
    train_generator,
    steps_per_epoch=128,
    epochs=epochs,
    validation_data=val_generator,
    validation_steps=32,
    callbacks=callbacks)

print('Evaluating model...')
# Load data
test_datagen = ImageDataGenerator(rescale=1 / 255)
X_test, y_test = next(test_datagen.flow(X_test, y_test, batch_size=len(X_test)))

loss, acc = model.evaluate(X_test, y_test, batch_size=batch_size)
print('Crossentropy loss: {:0.3f}'.format(loss))
print('Accuracy: {:0.3f}'.format(acc))

# Save model
# model.save('../data/model.h5')
# print('Saved model to disk')
