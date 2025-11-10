import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

tf.random.set_seed(42)


train_gen = ImageDataGenerator(rescale = 1./255,
                            validation_split = 0.3)

test_gen = ImageDataGenerator(rescale=1./255)

train_ds = train_gen.flow_from_directory(
    '/root/ssd/Yeonseo/ESS/data/spectrogram_image_1024/Train',
    target_size = (400, 300),
    batch_size = 32,
    class_mode = 'binary',
    subset = 'training',
    seed = 42
)

val_ds = train_gen.flow_from_directory(
    '/root/ssd/Yeonseo/ESS/data/spectrogram_image_1024/Train',
    target_size = (400, 300),
    batch_size = 32,
    class_mode = 'binary',
    subset = 'validation',
    seed = 42
)

test_ds = test_gen.flow_from_directory(
    '/root/ssd/Yeonseo/ESS/data/spectrogram_image_1024/Test',
    target_size = (400, 300),
    batch_size = 32,
    class_mode = 'binary',
)

train_model = models.Sequential()
train_model.add(layers.Conv2D(16, (44, 33), activation='relu', input_shape = (400, 300, 3)))
train_model.add(layers.MaxPooling2D((3, 3)))
train_model.add(layers.Conv2D(32, (40, 30), activation='relu'))
train_model.add(layers.MaxPooling2D((3, 3)))
train_model.add(layers.Conv2D(64, (12, 9), activation='relu'))
train_model.add(layers.MaxPooling2D((3, 3)))
train_model.add(layers.Conv2D(128, (4, 3), activation='relu'))
train_model.add(layers.MaxPooling2D((3, 3)))
train_model.add(layers.Flatten()) 
train_model.add(layers.Dense(64, activation='relu'))
train_model.add(layers.Dropout(0.7))
train_model.add(layers.Dense(1, activation='sigmoid'))

train_model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['Accuracy', 'Recall', 'Precision'])

train_model.fit(
    train_ds,
    steps_per_epoch=len(train_ds),
    epochs=30,
    validation_data=val_ds,
    validation_steps=len(val_ds)
)



test_loss, test_acc, test_recall, test_precision = train_model.evaluate(test_ds, steps=test_ds.n // 32, verbose=1)

print('Test loss:', test_loss)
print('Test accuracy:', test_acc)
print('Test Recall: ', test_recall)
print('Test Precision: ', test_precision)

predicted_labels_test= train_model.predict(test_ds, steps=test_ds.n // 32, verbose=1)
print("label: ", predicted_labels_test)

train_model.save('/root/ssd/Yeonseo/ESS/model.h5')