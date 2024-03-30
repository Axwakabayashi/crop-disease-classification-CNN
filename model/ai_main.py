from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
from tensorflow.keras.models import load_model



gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

data_dir = "/sorted_images"
image_exts = ['jpeg', 'jpg', 'png', 'bmp']

for image_class in os.listdir(data_dir):
    class_dir = os.path.join(data_dir, image_class)

    # Skip non-directories
    if not os.path.isdir(class_dir):
        continue

    for image in os.listdir(class_dir):
        image_path = os.path.join(class_dir, image)
        try:
            # Skip if the current path is a directory
            if os.path.isdir(image_path):
                continue

            # Open image using PIL
            img = Image.open(image_path)

            # Check if the file extension is in the allowed extensions
            _, ext = os.path.splitext(image)
            if ext[1:].lower() not in image_exts:
                print('Image not in ext list {}'.format(image_path))
                os.remove(image_path)
        except IsADirectoryError:
            print('Skipping directory {}'.format(image_path))
        except Exception as e:
            print('Issue with image {}: {}'.format(image_path, e))

            # Check if the user has permission to remove the file
            if os.access(image_path, os.W_OK):
                os.remove(image_path)
            else:
                print(f"Permission denied: {image_path}")


data = tf.keras.utils.image_dataset_from_directory(data_dir)
data_iterator = data.as_numpy_iterator()
batch = data_iterator.next()


fig, ax = plt.subplots(ncols=4, figsize=(20,20))
for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img.astype(int))
    ax[idx].title.set_text(batch[1][idx])

data = data.map(lambda x,y: (x/255, y))
data.as_numpy_iterator().next()
train_size = int(len(data)*.7)
val_size = int(len(data)*.2)
test_size = int(len(data)*.1)
train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size+val_size).take(test_size)

#Building model
train
model = Sequential()
model.add(Conv2D(16, (3,3), 1, activation='relu', input_shape=(256,256,3)))
model.add(MaxPooling2D())
model.add(Conv2D(32, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(16, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])
model.summary()

logdir='logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
hist = model.fit(train, epochs=20, validation_data=val, callbacks=[tensorboard_callback])



fig = plt.figure()
plt.plot(hist.history['loss'], color='teal', label='loss')
plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc="upper left")
plt.show()



fig = plt.figure()
plt.plot(hist.history['accuracy'], color='teal', label='accuracy')
plt.plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')
fig.suptitle('Accuracy', fontsize=20)
plt.legend(loc="upper left")
plt.show()


#Evaluating model
pre = Precision()
re = Recall()
acc = BinaryAccuracy()
for batch in test.as_numpy_iterator():
    X, y = batch
    yhat = model.predict(X)
    pre.update_state(y, yhat)
    re.update_state(y, yhat)
    acc.update_state(y, yhat)

print(pre.result(), re.result(), acc.result())



#Testing Model
#image_path = '/Users/axwakabayashi/Desktop/CNN-Breast-Cancer-Classifier/15902_idx5_x2751_y751_class1.png'
#img = Image.open(image_path)
#plt.imshow(img)
#plt.show()

resize = tf.image.resize(img, (256,256))
plt.imshow(resize.numpy().astype(int))
plt.show()

yhat = model.predict(np.expand_dims(resize/255, 0))
if yhat > 0.5:
    print(f'Cancer is detected')
else:
    print(f'No cancer is detected')


#Saving model
model.save('/Users/axwakabayashi/Desktop/CNN-Breast-Cancer-Classifier/models/imageclassifier.Keras')
new_model = load_model('/Users/axwakabayashi/Desktop/CNN-Breast-Cancer-Classifier/models/imageclassifier.Keras')
model.save(os.path.join('/Users/axwakabayashi/Desktop/CNN-Breast-Cancer-Classifier/models/imageclassifier.Keras'))
new_model.predict(np.expand_dims(resize/255, 0))