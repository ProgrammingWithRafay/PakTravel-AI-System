# Part 4: Bus Delay Prediction using ANN
# required libraries: pip install numpy pandas scikit-learn

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix)
import warnings
warnings.filterwarnings('ignore')  # suppress sklearn convergence warnings for cleaner output

# fix random seed so results are reproducible every run
np.random.seed(42)

print("=" * 65)
print("   PakTravel AI System - Part 4: Delay Predictor (ANN)")
print("=" * 65)

# ============================================================
# TASK 1: Data Preparation
# ============================================================
print("\n--- TASK 1: Data Preparation ---\n")

def generate_dataset(n=500):
    """
    Generates n synthetic bus journey records.
    Features: Distance, Weather_Score, Traffic_Score, Road_Quality, Bus_Age
    Label rule (from project spec):
      IF weather>=4 OR traffic>=4 OR (distance>400 AND road_quality<=2) -> Delayed (1)
      ELSE -> On Time (0)
    """
    data = {
        'Distance':      np.random.randint(80, 701, size=n),
        'Weather_Score': np.random.randint(1, 6, size=n),
        'Traffic_Score': np.random.randint(1, 6, size=n),
        'Road_Quality':  np.random.randint(1, 6, size=n),
        'Bus_Age':       np.random.randint(1, 16, size=n),
    }

    df = pd.DataFrame(data)

    # apply label rule from the project spec
    def label(row):
        if row['Weather_Score'] >= 4:
            return 1  # delayed - bad weather
        if row['Traffic_Score'] >= 4:
            return 1  # delayed - heavy traffic
        if row['Distance'] > 400 and row['Road_Quality'] <= 2:
            return 1  # delayed - long route on poor road
        return 0  # on time

    df['Delayed'] = df.apply(label, axis=1)
    return df


df = generate_dataset(500)

print("First 10 Rows of Dataset:")
print(df.head(10).to_string(index=False))

print(f"\nDataset Shape: {df.shape}")

# class distribution
on_time_count  = (df['Delayed'] == 0).sum()
delayed_count  = (df['Delayed'] == 1).sum()
print(f"\nClass Distribution:")
print(f"  On Time (0) : {on_time_count}  ({on_time_count/len(df)*100:.1f}%)")
print(f"  Delayed (1) : {delayed_count}  ({delayed_count/len(df)*100:.1f}%)")

# separate features and target
X = df[['Distance', 'Weather_Score', 'Traffic_Score', 'Road_Quality', 'Bus_Age']].values
y = df['Delayed'].values

# split 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTrain size: {X_train.shape[0]} samples")
print(f"Test size : {X_test.shape[0]} samples")

# normalize using Min-Max scaling (all features to 0-1 range)
# important: fit scaler only on training data to avoid data leakage!
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print(f"\nFeature ranges after Min-Max scaling (first row):")
feature_names = ['Distance', 'Weather', 'Traffic', 'Road_Quality', 'Bus_Age']
for i, name in enumerate(feature_names):
    print(f"  {name:<15}: {X_train_scaled[0][i]:.4f}")


# ============================================================
# TASK 2: ANN using sklearn MLPClassifier
# ============================================================
print("\n" + "=" * 65)
print("--- TASK 2: ANN Model (sklearn MLPClassifier) ---\n")

# architecture as per project spec:
# Input: 5 neurons
# Hidden Layer 1: 10 neurons, ReLU
# Hidden Layer 2: 8 neurons, ReLU
# Output: 1 neuron, Sigmoid (handled internally by sklearn for binary)

print("Building ANN with architecture: 5 -> 10 -> 8 -> 1")
print("Activation: ReLU (hidden), Sigmoid (output)")
print("Training...")

sklearn_model = MLPClassifier(
    hidden_layer_sizes=(10, 8),  # two hidden layers
    activation='relu',           # ReLU for hidden layers
    solver='adam',               # Adam optimizer works well
    max_iter=500,
    random_state=42,
    learning_rate_init=0.01,
)

sklearn_model.fit(X_train_scaled, y_train)

y_pred = sklearn_model.predict(X_test_scaled)
y_pred_prob = sklearn_model.predict_proba(X_test_scaled)[:, 1]  # probability of class 1

# metrics
acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec  = recall_score(y_test, y_pred, zero_division=0)
f1   = f1_score(y_test, y_pred, zero_division=0)

print(f"\nModel Performance on Test Set:")
print(f"  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
print(f"  Precision : {prec:.4f}")
print(f"  Recall    : {rec:.4f}")
print(f"  F1-Score  : {f1:.4f}")

# confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(f"\nConfusion Matrix:")
print(f"  {'':20} Predicted On-Time  Predicted Delayed")
print(f"  {'Actual On-Time':<20} {cm[0][0]:>17}  {cm[0][1]:>16}")
print(f"  {'Actual Delayed':<20} {cm[1][0]:>17}  {cm[1][1]:>16}")
print(f"\n  True Positives (TP)  : {cm[1][1]}  (correctly predicted Delayed)")
print(f"  True Negatives (TN)  : {cm[0][0]}  (correctly predicted On Time)")
print(f"  False Positives (FP) : {cm[0][1]}  (said Delayed but was On Time)")
print(f"  False Negatives (FN) : {cm[1][0]}  (said On Time but was Delayed)")


# ============================================================
# TASK 3: Test on 5 Specific Bus Journeys
# ============================================================
print("\n" + "=" * 65)
print("--- TASK 3: Testing on 5 Real Journeys ---\n")

# the 5 test journeys from the project spec
test_journeys = [
    # Distance, Weather, Traffic, Road_Quality, Bus_Age
    [320, 2, 2, 4, 3],   # Journey 1 - Ali's bus (Sukkur->Multan)
    [650, 5, 4, 2, 12],  # Journey 2
    [160, 1, 1, 5, 2],   # Journey 3
    [380, 4, 3, 3, 8],   # Journey 4
    [570, 3, 5, 2, 10],  # Journey 5
]

journey_labels = [
    "Ali's bus (Sukkur->Multan)",
    "Journey 2",
    "Journey 3",
    "Journey 4",
    "Journey 5",
]

test_arr = np.array(test_journeys)
test_scaled = scaler.transform(test_arr)

preds = sklearn_model.predict(test_scaled)
probs = sklearn_model.predict_proba(test_scaled)[:, 1]

print(f"{'Journey':<6} {'Distance':>9} {'Weather':>8} {'Traffic':>8} {'Road':>5} {'Age':>4}  {'Prediction':<12} {'Confidence'}")
print("-" * 75)

for i, (journey, pred, prob) in enumerate(zip(test_journeys, preds, probs)):
    result    = "DELAYED" if pred == 1 else "ON TIME"
    conf      = prob * 100 if pred == 1 else (1 - prob) * 100
    name_tag  = f"J{i+1}"
    print(f"{name_tag:<6} {journey[0]:>9} {journey[1]:>8} {journey[2]:>8} {journey[3]:>5} {journey[4]:>4}  {result:<12} {conf:.1f}%")

# special message for Ali's journey
ali_pred = "ON TIME" if preds[0] == 0 else "DELAYED"
ali_conf = (1 - probs[0]) * 100 if preds[0] == 0 else probs[0] * 100
print(f"\n  *** Ali's bus prediction: {ali_pred}! (confidence: {ali_conf:.1f}%) ***")

print("\nJourney Analysis:")
for i, (journey, pred, prob) in enumerate(zip(test_journeys, preds, probs)):
    result = "DELAYED" if pred == 1 else "ON TIME"
    print(f"\n  Journey {i+1} - {journey_labels[i]}")
    print(f"    Features: Distance={journey[0]}km, Weather={journey[1]}, "
          f"Traffic={journey[2]}, Road={journey[3]}, Age={journey[4]}yrs")
    conf = prob * 100 if pred == 1 else (1 - prob) * 100
    print(f"    Prediction: {result} ({conf:.1f}% confidence)")
    if journey[1] >= 4:
        print(f"    Reason: Bad weather score ({journey[1]}/5)")
    elif journey[2] >= 4:
        print(f"    Reason: Heavy traffic score ({journey[2]}/5)")
    elif journey[0] > 400 and journey[3] <= 2:
        print(f"    Reason: Long distance ({journey[0]}km) on poor road (quality={journey[3]})")
    else:
        print(f"    Reason: All conditions are acceptable")


# ============================================================
# TASK 4: ANN from Scratch using NumPy
# ============================================================
print("\n" + "=" * 65)
print("--- TASK 4: ANN from Scratch (NumPy only) ---\n")

class NumpyANN:
    """
    A 3-layer artificial neural network built from scratch using only NumPy.
    Architecture: 5 -> 10 -> 8 -> 1
    Uses sigmoid activation throughout + binary cross-entropy loss.
    """

    def __init__(self, input_size=5, h1=10, h2=8, output_size=1, lr=0.05):
        # initialize weights using small random values
        # using 0.1 scale - too large causes exploding gradients, too small is slow
        np.random.seed(42)
        self.lr = lr  # learning rate

        self.W1 = np.random.randn(input_size, h1) * 0.1
        self.b1 = np.zeros((1, h1))

        self.W2 = np.random.randn(h1, h2) * 0.1
        self.b2 = np.zeros((1, h2))

        self.W3 = np.random.randn(h2, output_size) * 0.1
        self.b3 = np.zeros((1, output_size))

    def sigmoid(self, z):
        # sigmoid activation: f(z) = 1 / (1 + e^-z)
        # clip to avoid overflow in exp for very large/small values
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    def sigmoid_derivative(self, a):
        # derivative of sigmoid: f'(z) = f(z) * (1 - f(z))
        # using the already-computed activation value 'a'
        return a * (1.0 - a)

    def forward(self, X):
        # forward pass through all 3 layers
        # layer 1
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)

        # layer 2
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)

        # output layer
        self.z3 = np.dot(self.a2, self.W3) + self.b3
        self.a3 = self.sigmoid(self.z3)

        return self.a3

    def compute_loss(self, y_true, y_pred):
        """
        Binary Cross Entropy Loss:
        L = -1/m * sum(y * log(y_hat) + (1-y) * log(1-y_hat))
        """
        m   = len(y_true)
        eps = 1e-8  # tiny value to prevent log(0)
        y_true = y_true.reshape(-1, 1)
        loss = -np.mean(
            y_true * np.log(y_pred + eps) +
            (1 - y_true) * np.log(1 - y_pred + eps)
        )
        return loss

    def backward(self, X, y):
        """
        Backpropagation - computes gradients and updates weights.
        Uses chain rule to propagate error from output back to input.
        """
        m = X.shape[0]
        y = y.reshape(-1, 1)

        # output layer gradient: dL/dz3
        dz3 = self.a3 - y
        dW3 = np.dot(self.a2.T, dz3) / m
        db3 = np.sum(dz3, axis=0, keepdims=True) / m

        # hidden layer 2 gradient
        da2 = np.dot(dz3, self.W3.T)
        dz2 = da2 * self.sigmoid_derivative(self.a2)
        dW2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m

        # hidden layer 1 gradient
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * self.sigmoid_derivative(self.a1)
        dW1 = np.dot(X.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        # update weights using gradient descent
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W3 -= self.lr * dW3
        self.b3 -= self.lr * db3

    def train(self, X, y, epochs=100):
        """Trains the network for given number of epochs."""
        print("  Training NumPy ANN (100 epochs):")
        for epoch in range(1, epochs + 1):
            y_pred = self.forward(X)
            loss   = self.compute_loss(y, y_pred)
            self.backward(X, y)

            if epoch % 20 == 0:
                # compute training accuracy to show progress
                preds_bin = (y_pred > 0.5).astype(int).flatten()
                train_acc = np.mean(preds_bin == y)
                print(f"    Epoch {epoch:3d}/100  |  Loss: {loss:.4f}  |  Train Acc: {train_acc:.4f}")

    def predict(self, X):
        probs = self.forward(X).flatten()
        return (probs > 0.5).astype(int)

    def predict_proba(self, X):
        return self.forward(X).flatten()


print("Building and training ANN from scratch with NumPy...")
numpy_ann = NumpyANN(input_size=5, h1=10, h2=8, output_size=1, lr=0.05)
numpy_ann.train(X_train_scaled, y_train, epochs=100)

# evaluate on test set
np_preds = numpy_ann.predict(X_test_scaled)
np_acc   = np.mean(np_preds == y_test)
np_prec  = precision_score(y_test, np_preds, zero_division=0)
np_rec   = recall_score(y_test, np_preds, zero_division=0)
np_f1    = f1_score(y_test, np_preds, zero_division=0)

print(f"\n  NumPy ANN Test Accuracy : {np_acc:.4f}  ({np_acc*100:.2f}%)")
print(f"  NumPy ANN Precision     : {np_prec:.4f}")
print(f"  NumPy ANN Recall        : {np_rec:.4f}")
print(f"  NumPy ANN F1-Score      : {np_f1:.4f}")

# test 5 journeys with NumPy ANN
print("\n  Testing 5 Journeys with NumPy ANN:")
np_probs = numpy_ann.predict_proba(test_scaled)
np_journey_preds = (np_probs > 0.5).astype(int)

for i, (journey, pred, prob) in enumerate(zip(test_journeys, np_journey_preds, np_probs)):
    result = "DELAYED" if pred == 1 else "ON TIME"
    conf   = prob * 100 if pred == 1 else (1 - prob) * 100
    print(f"    J{i+1}: {result:<10} (Confidence: {conf:.1f}%)")

# final comparison
print("\n" + "=" * 65)
print("Final Comparison: sklearn ANN vs NumPy ANN")
print(f"  {'Model':<20} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
print("  " + "-" * 55)
print(f"  {'sklearn MLP':<20} {acc:>10.4f} {prec:>10.4f} {rec:>10.4f} {f1:>10.4f}")
print(f"  {'NumPy ANN':<20} {np_acc:>10.4f} {np_prec:>10.4f} {np_rec:>10.4f} {np_f1:>10.4f}")

if acc > np_acc:
    diff = abs(acc - np_acc) * 100
    print(f"\n  sklearn MLP is {diff:.2f}% more accurate.")
    print("  sklearn uses adam optimizer and ReLU which are better tuned.")
elif np_acc > acc:
    diff = abs(np_acc - acc) * 100
    print(f"\n  NumPy ANN is {diff:.2f}% more accurate in this run.")
else:
    print("\n  Both models achieved the same accuracy!")

print("""
  Analysis:
  sklearn MLPClassifier is generally better because it uses:
  - Adam optimizer (adaptive learning rate, faster convergence)
  - Better weight initialization
  - More epochs and internal validation
  
  Our NumPy ANN is built from scratch to understand the internals.
  It uses basic gradient descent which converges slower but still
  learns the patterns in the data correctly.
""")