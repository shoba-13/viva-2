# ===== Paste each block into the matching Colab cell =====
# (Cells 1, 2, 4, 6, 9 and 11 are given to you - leave them unchanged.)
# Variables from cell 2 are assumed: PATH, train_dir, validation_dir, test_dir,
# total_train, total_val, batch_size, epochs, IMG_HEIGHT, IMG_WIDTH

# ---------------------------------------------------------------- # 3
train_image_generator = ImageDataGenerator(rescale=1./255)
validation_image_generator = ImageDataGenerator(rescale=1./255)
test_image_generator = ImageDataGenerator(rescale=1./255)

train_data_gen = train_image_generator.flow_from_directory(
    batch_size=batch_size,
    directory=train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    class_mode='binary')

val_data_gen = validation_image_generator.flow_from_directory(
    batch_size=batch_size,
    directory=validation_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    class_mode='binary')

# test folder has no class subfolders, so point at PATH and name the 'test' folder as the class
test_data_gen = test_image_generator.flow_from_directory(
    PATH,
    classes=['test'],
    batch_size=batch_size,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    class_mode=None,
    shuffle=False)

# ---------------------------------------------------------------- # 5
train_image_generator = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

# ---------------------------------------------------------------- # 7
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dropout(0.5),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

# ---------------------------------------------------------------- # 8
history = model.fit(
    train_data_gen,
    steps_per_epoch=total_train // batch_size,
    epochs=epochs,                      # set epochs = 20 or more in cell 2 for ~70%
    validation_data=val_data_gen,
    validation_steps=total_val // batch_size)

# ---------------------------------------------------------------- # 10
test_data_gen.reset()
test_images = next(test_data_gen)           # batch_size (128) > 50, so this is all test images
probabilities = model.predict(test_data_gen).flatten().tolist()   # 0 = cat, 1 = dog

plotImages(test_images, probabilities=probabilities)
