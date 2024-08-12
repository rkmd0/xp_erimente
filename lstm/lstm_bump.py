import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout # dropout gegen mögl. overfitting
from tensorflow.keras.utils import to_categorical
import time


# alternativ: magnitude berechnen -> grenzwert setzen -> bump für jede prediction setzen
# asphalt 17 - paving stone 25
start_time = time.time()
print("starten des skripts...")

# definiere den Pfad zu daten
base_path = r'C:\Users\easyd\Documents\GitHub\bumpy_data'

# funktion um daten zu laden (ohne labeln)
def load_data(path):
    print(f"laden vom path {path}...")
    files = [f for f in os.listdir(path) if f.endswith('.csv')]
    df_list = [pd.read_csv(os.path.join(path, f)) for f in files]
    df = pd.concat(df_list, ignore_index=True)
    return df

# laden der daten
asphalt_df = load_data(os.path.join(base_path, 'asphalt_bumpy'))
cobblestone_df = load_data(os.path.join(base_path, 'cobblestone_processed'))
paving_stone_df = load_data(os.path.join(base_path, 'paving_stone_bumpy'))

# konkatiniere alle daten
data = pd.concat([asphalt_df, cobblestone_df, paving_stone_df], ignore_index=True)

print("alle daten geladen und kombiniert.")

# verteilung der labels überprüfen
print("\nlabel verteilung:")
print(data['Label'].value_counts())

# dynamische anzahl an samples, um schneller zu arbeiten
sample_percentage = 100
data = data.sample(frac=sample_percentage/100, random_state=42).reset_index(drop=True)

print(f"verwende {len(data)} samples ({sample_percentage}%) für die analyse.")

# seperiere features und labels
X = data[['AccX', 'AccY', 'AccZ', 'GyrX', 'GyrY', 'GyrZ']]
y = data['Label']

# update label mapping
label_mapping = {'asphalt': 0, 'asphalt_bumpy': 1, 'cobblestone': 2, 'paving_stone': 3, 'paving_stone_bumpy': 4}
y = y.map(label_mapping)

# splitten der daten in trainings- und testdaten
# test_and_train_split()
'''split_index = int(len(X) * 0.8)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]'''

# Splitten der Daten in Trainings- und Testdaten
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# skalieren der daten
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# konvertiere labels in kategorische werte
num_classes = len(label_mapping)
y_train_cat = to_categorical(y_train, num_classes=num_classes)
y_test_cat = to_categorical(y_test, num_classes=num_classes)

print("daten vorbereitet und in trainings- und testdaten aufgeteilt.")

# fenstergröße für lstm
window_size = 30

# Funktion zum erstellen von LSTM-Datensätzen, schaut sich die letzten 30 Werte an und sagt den nächsten Wert
def create_lstm_data(X, window_size):
    lstm_data = []
    for i in range(len(X) - window_size + 1):
        lstm_data.append(X[i:i+window_size])
    return np.array(lstm_data)

X_train_lstm = create_lstm_data(X_train_scaled, window_size)
X_test_lstm = create_lstm_data(X_test_scaled, window_size)

y_train_lstm = y_train_cat[window_size-1:]
y_test_lstm = y_test_cat[window_size-1:]

# definiere das LSTM-Modell
model = Sequential()
model.add(LSTM(64, input_shape=(window_size, X_train_lstm.shape[2]))) # 64 neuronen in der ersten LSTM-Schicht
model.add(Dense(num_classes, activation='softmax'))  # softmax aktivierungsfunktion für die ausgabeschicht

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("training des modells")
history = model.fit(X_train_lstm, y_train_lstm, epochs=20, batch_size=64, validation_data=(X_test_lstm, y_test_lstm))

# evaluaieren des modells
y_pred_prob = model.predict(X_test_lstm)
y_pred = np.argmax(y_pred_prob, axis=1)
y_test_labels = np.argmax(y_test_lstm, axis=1)

print("\nLSTM klassifikations report:")
print(classification_report(y_test_labels, y_pred))
print("\nLSTM confusion matrix:")
print(confusion_matrix(y_test_labels, y_pred))

end_time = time.time()

print(f"skript beendet. Laufzeit: {end_time - start_time} sekunden.")

import joblib
import tensorflow as tf
print("\nspeichere modell und scaler...")
model.save('road_surface_lstm_model_bump.keras')
joblib.dump(scaler, 'road_surface_scaler_bump.joblib')

# cryptische flags die notwendig sind um modell zu konvertieren
# lstm supportet von tflite?
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
converter._experimental_lower_tensor_list_ops = False
converter.experimental_enable_resource_variables = True
tflite_model = converter.convert()

# speichere das tflite modell
with open('road_surface_lstm_model_bump.tflite', 'wb') as f:
    f.write(tflite_model)

# anschließend: zu c array konvertieren (bit_creater.py)

print("\nmodel wurde konvertiert und gespeichert.")
print("skript beendet.")
