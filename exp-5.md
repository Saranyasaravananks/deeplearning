from keras.datasets import imdb 
from keras.models import Sequential 
from keras.layers import Embedding, LSTM, Dense 
from keras.preprocessing.sequence import pad_sequences 
# Load dataset 
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=10000) 
X_train = pad_sequences(X_train, maxlen=100) 
X_test = pad_sequences(X_test, maxlen=100) 
# Model 
model = Sequential([ 
Embedding(10000, 32, input_length=100), 
LSTM(100), 
Dense(1, activation='sigmoid') 
]) 
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) 
model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.2)<img width="1412" height="619" alt="Screenshot 2025-09-24 102942" src="https://github.com/user-attachments/assets/c37d73db-e5c1-4911-9819-8d50c7595b5e" />
<img width="1244" height="410" alt="Screenshot 2025-09-24 103034" src="https://github.com/user-attachments/assets/abc1e50c-b103-4679-b0b7-9c6c4918c7d5" />
<img width="1226" height="273" alt="Screenshot 2025-09-24 103750" src="https://github.com/user-attachments/assets/38e205b6-06af-4250-8865-990d0ad9abcd" />

