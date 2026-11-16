import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# 1. Load Dataset
df = pd.read_csv("test_tiered.csv")

# 2. Features and Target
X = df[["battery_power", "clock_speed", "int_memory", "n_cores", "ram"]]
y = df["tier"]

# Encode classes (Budget=0, Mid-Range=1, Flagship=2)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 3. Train-Test Split & Scaling
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Simple Sequential ANN Model
model = Sequential([
    Dense(32, activation="relu", input_shape=(5,)),
    Dropout(0.2),
    Dense(16, activation="relu"),
    Dense(3, activation="softmax")  # 3 output classes
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# 5. Backtracking Mechanism (Restores best weights if model regresses)
backtracking = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
    verbose=1
)

# 6. Train Model
history = model.fit(
    X_train_scaled, y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    callbacks=[backtracking],
    verbose=1
)

# 7. Evaluate on Test Set
loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"\nFinal Test Accuracy: {accuracy * 100:.2f}%")

# 8. Save Model and Preprocessors for Streamlit Frontend
model.save("model.keras")
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)
print("Saved model.keras, scaler.pkl, and label_encoder.pkl successfully!")
