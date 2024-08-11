'''
modell zur erkennung von straßenoberflächen mit den 3 klassifizierungen asphalt, kopfsteinpflaster und pflasterstein
(anschließend bumpy stufen)
skript beinhaltet alle schritte von der datenverarbeitung bis zum speichern des modells und umwandlung in ein header file (mit nötigen anpassungen)
ursprünglich war ein lstm modell geplant, allerdings gab es hier einige probleme wodurch ich einfachheitshalber 'last second' auf ein dense modell umgestiegen bin
dense modell inspiriert von: https://github.com/PaulaScharf/bicycle-surface-detection-edge/blob/main/training/train_dense.py
die meisten schritten werden geprintet wodurch der user den fortschritt des skripts verfolgen kann
anschließend wird das modell in ein tflite modell umgewandelt und als c array gespeichert
equivalent zu -xdd?

fragestellung:
hauptsächlich arduino code, der das modell ausführt und die daten verarbeitet

tester: path zu den daten anpassen und dann ausführen
'''


# importiere benötigte bibliotheken
import os
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import joblib # nicht verwendet
import tensorflow as tf
import time
import random

# einer der probleme die beim lstm modell auftraten wurden hierdurch 'gefixt'
# s. https://github.com/tensorflow/tensorflow/issues/63867
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# um die laufzeit des skripts zu messen
start_time = time.time()
print("starte skript...")

# path zu den daten
base_path = r'C:\your\path\to\road-surface-data'

# praktische funktion zum laden und gleichzeitigem labeln der daten
# dadurch muss der nutzer nicht unbedingt labelfiller.py verwenden
# bei bump analyse aber nicht verwendbar (bump label sind im gleichen file)
def load_data(path, label):
    print(f"laden der {label} daten...")
    files = [f for f in os.listdir(path) if f.endswith('.csv')]
    
    # randomisiert die files und splittet 20%
    random.shuffle(files)
    split_index = int(len(files) * 0.2)
    test_files = files[:split_index]
    train_files = files[split_index:]

    # laden der trainingsdaten
    train_df_list = [pd.read_csv(os.path.join(path, f)) for f in train_files]
    train_df = pd.concat(train_df_list, ignore_index=True)
    train_df['Label'] = label

    # laden der testdaten
    test_df_list = [pd.read_csv(os.path.join(path, f)) for f in test_files]
    test_df = pd.concat(test_df_list, ignore_index=True)
    test_df['Label'] = label

    return train_df, test_df

# laden der daten
asphalt_train_df, asphalt_test_df = load_data(os.path.join(base_path, 'aphalt_processed'), 'asphalt')
cobblestone_train_df, cobblestone_test_df = load_data(os.path.join(base_path, 'cobblestone_processed'), 'cobblestone')
paving_stone_train_df, paving_stone_test_df = load_data(os.path.join(base_path, 'paving_stone_processed'), 'paving_stone')

# konkatiniere alle daten in ein dataframe
train_data = pd.concat([asphalt_train_df, cobblestone_train_df, paving_stone_train_df], ignore_index=True)
test_data = pd.concat([asphalt_test_df, cobblestone_test_df, paving_stone_test_df], ignore_index=True)
print("alle daten wurden geladen und kombiniert.")

# label verteilung anzeigen
print("\ntrainings label verteilung:")
print(train_data['Label'].value_counts())
print("\ntest label verteilung:")
print(test_data['Label'].value_counts())

# seperiere features und labels
X_train = train_data[['AccX', 'AccY', 'AccZ', 'GyrX', 'GyrY', 'GyrZ']]
y_train = train_data['Label']

X_test = test_data[['AccX', 'AccY', 'AccZ', 'GyrX', 'GyrY', 'GyrZ']]
y_test = test_data['Label']

# konvertiere labels in numerische wertel
label_mapping = {'asphalt': 0, 'cobblestone': 1, 'paving_stone': 2}
y_train = y_train.map(label_mapping)
y_test = y_test.map(label_mapping)

print("daten wurden in trainings und testdaten aufgeteilt...")

# definieren des modells
# evtl. anpassen der anzahl der neuronen und schichten?
# weniger -> mehr oder umgekehrt?
# hyperparameter tuning
model = Sequential()
model.add(Dense(20, activation='relu', input_shape=(X_train.shape[1],))) # relu aktivierungsfunktion, 20 neuronen, 6 features
model.add(Dense(10, activation='relu'))
model.add(Dense(3, activation='softmax'))  # softmax aktivierungsfunktion für die ausgabeschicht (wahrcheinlichkeiten der 3 klassen)

# kompilieren des modells
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# praktisches feature um das training zu stoppen wenn keine verbesserung mehr stattfindet
# model kann quasi 
early_stopping = EarlyStopping(
    monitor='val_loss', 
    patience=3,
    restore_best_weights=True
)

print("training des modells...")
history = model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=64,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping]
)

# evaluation des modells
y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)

print("\nklassifikationsbericht:")
print(classification_report(y_test, y_pred))
print("\nconfusion matrix:")
print(confusion_matrix(y_test, y_pred))

end_time = time.time()

print(f"skript wurde in {end_time - start_time:.2f} sekunden abgeschlossen.")

# speichern des modells
print("\nspeichere modell...")
model.save('benjamin_results_test.keras')

# konvertiere das modell in ein tflite modell
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# speichere das tflite modell
tflite_model_path = 'benjamin_results_test.tflite'
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)

# printe die größe des tflite modells
tflite_model_size = os.path.getsize(tflite_model_path) / 1024  # KB
print(f"\nTFLite model größe: {tflite_model_size:.2f} KB")

print("modell wurde gespeichert...")

# funktion um das tflite modell in ein c array zu konvertieren(?)
def convert_to_c_array(file_path, output_file):
    with open(file_path, "rb") as f:
        data = f.read()
    
    data_length = len(data)
    
    with open(output_file, "w") as f:
        f.write(f"unsigned char model_tflite[] = {{\n")
        for i, byte in enumerate(data):
            f.write(f"0x{byte:02x}, ")
            if (i + 1) % 12 == 0:
                f.write("\n")
        f.write(f"\n}};\n\n")
        f.write(f"unsigned int model_tflite_len = {data_length};\n")

# Convert the TFLite model to a C array
convert_to_c_array(tflite_model_path, "benjamin_results_test.h")

print("model wurde als c array gespeichert...")
print("skript abgeschlossen.")
