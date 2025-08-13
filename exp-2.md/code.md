from keras.datasets import mnist 
from keras.models import Sequential 
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense 
from keras.utils import to_categorical 
(X_train, y_train), (X_test, y_test) = mnist.load_data() 
X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255 
X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255 
y_train = to_categorical(y_train) 
y_test = to_categorical(y_test) 
model = Sequential([ 
Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)), 
MaxPooling2D(pool_size=(2, 2)), 
Flatten(), 
Dense(128, activation='relu'), 
Dense(10, activation='softmax') 
]) 
model.compile(optimizer='adam', loss='categorical_crossentropy', 
metrics=['accuracy']) 
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, 
y_test))
  Output<img width="1525" height="292" alt="Screenshot 2025-08-13 103557" src="https://github.com/user-attachments/assets/c36ae37a-8823-4e3b-bb82-6bff49db5fd2" />
