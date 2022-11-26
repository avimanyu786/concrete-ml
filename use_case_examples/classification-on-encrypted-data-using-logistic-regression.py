# Example of classification on encrypted data using logistic regression mentioned at https://docs.zama.ai/concrete-ml
import numpy
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from concrete.ml.sklearn import LogisticRegression

# Lets create a synthetic data-set
x, y = make_classification(n_samples=100,
    class_sep=2, n_features=4, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# Now we train in plaintext using quantization
model = LogisticRegression(n_bits=2)
model.fit(X_train, y_train)

y_pred_clear = model.predict(X_test)

# Finally we compile and run inference on encrypted inputs!
model.compile(x)
y_pred_fhe = model.predict(X_test, execute_in_fhe=True)

print("In clear  :", y_pred_clear)
print("In FHE    :", y_pred_fhe)
print("Comparison:", (y_pred_fhe == y_pred_clear))

# Output:
#   In clear  : [0 1 0 1 0 1 0 1 1 1 0 1 1 0 1 0 0 1 1 1]
#   In FHE    : [0 1 0 1 0 1 0 1 1 1 0 1 1 0 1 0 0 1 1 1]
#   Comparison: [ True  True  True  True  True  True  True  True  True  True  True  True
#   True  True  True  True  True  True  True  True]
