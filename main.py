import numpy as np
import pandas as pd
from keras import layers, models

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

X_train = train_data.drop(columns=['label']).values.reshape(-1, 28, 28, 1)
y_train = train_data['label'].values
X_test = test_data.values.reshape(-1,28,28,1)

X_train = X_train.astype('float32')  / 255.0
X_test = X_test.astype('float32') / 255.0

model = models.Sequential([
    layers.Conv2D(32, (3,3), activation= 'relu', input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2,)),
    layers.Conv2D(64, (3,3), activation= 'relu'),
    layers.MaxPooling2D((2,2,)),
    layers.Conv2D(64, (3,3), activation= 'relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


history = model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.2)

predictions = model.predict(X_test)

submission = pd.DataFrame({'ImageId': range(1, len(predictions) + 1), 'Label': np.argmax(predictions, axis=1)})
submission.to_csv('submission.csv', index=False)