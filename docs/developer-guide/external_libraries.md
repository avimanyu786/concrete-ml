# External Libraries

## Hummingbird

[Hummingbird](https://microsoft.github.io/hummingbird/) is a third-party, open-source library that converts machine learning models into tensor computations, and it can export these models to ONNX. The list of supported models can be found in [the Hummingbird documentation](https://microsoft.github.io/hummingbird/api/hummingbird.ml.supported.html).

Concrete-ML allows the conversion of an ONNX inference to NumPy inference (note that NumPy is always the entry point to run models in FHE with Concrete ML).

Hummingbird exposes a `convert` function that can be imported as follows from the `hummingbird.ml` package:

```python
# Disable Hummingbird warnings for pytest.
import warnings
warnings.filterwarnings("ignore")
from hummingbird.ml import convert
```

This function can be used to convert a machine learning model to an ONNX as follows:

<!--pytest-codeblocks:cont-->

```python
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

# Instantiate the logistic regression from sklearn
lr = LogisticRegression()

# Create synthetic data
X, y = make_classification(
    n_samples=100, n_features=20, n_classes=2
)

# Fit the model
lr.fit(X, y)

# Convert the model to ONNX
onnx_model = convert(lr, backend="onnx", test_input=X).model
```

In theory, the resulting `onnx_model` could be used directly within Concrete-ML's `get_equivalent_numpy_forward` method (as long as all operators present in the ONNX model are implemented in NumPy) and get the NumPy inference.

In practice, there are some steps needed to clean the ONNX output and make the graph compatible with Concrete-ML, such as applying quantization where needed or deleting/replacing non-FHE friendly ONNX operators (such as _Softmax_ and _ArgMax)._

## Skorch

Concrete-ML uses [Skorch](https://skorch.readthedocs.io/en/stable/) to implement multi-layer, fully-connected PyTorch neural networks in a way that is compatible with the Scikit-learn API.

This wrapper implements Torch training boilerplate code, alleviating the work that needs to be done by the user. It is possible to add hooks during the training phase, for example once an epoch is finished.

Skorch allows the user to easily create a classifier or regressor around a neural network (NN), implemented in Torch as a `nn.Module`, which is used by Concrete-ML to provide a fully-connected multi-layer NN with a configurable number of layers and optional pruning (see [pruning](../advanced-topics/pruning.md) and the [neural network documentation](../built-in-models/neural-networks.md) for more information).

Under the hood, Concrete-ML uses a Skorch wrapper around a single PyTorch module, `SparseQuantNeuralNetImpl`. More information can be found [in the API guide](../developer-guide/api/concrete.ml.sklearn.qnn.md#class-sparsequantneuralnetimpl).

```
class SparseQuantNeuralNetImpl(nn.Module):
    """Sparse Quantized Neural Network classifier.
```

## Brevitas

[Brevitas](https://github.com/Xilinx/brevitas) is a quantization aware learning toolkit built on top of PyTorch. It provides quantization layers that are one-to-one equivalents to PyTorch layers, but also contain operations that perform the quantization during training.

While Brevitas provides many types of quantization, for Concrete-ML, a custom _"mixed integer"_ quantization applies. This _"mixed integer"_ quantization is much simpler than the _"integer only"_ mode of Brevitas. The _"mixed integer"_ network design is defined as:

- all weights and activations of convolutional, linear and pooling layers must be quantized
  (e.g. using Brevitas layers, `QuantConv2D`, `QuantAvgPool2D`, `QuantLinear`)
- PyTorch floating point versions of univariate functions can be used. E.g. `torch.relu`, `nn.BatchNormalization2D`, `torch.max` (encrypted vs. constant), `torch.add`, `torch.exp`. See the [PyTorch supported layers page](../deep-learning/torch_support.md) for a full list.

The _"mixed integer"_ mode used in Concrete-ML neural networks is based on the [_"integer only"_ Brevitas quantization](https://github.com/Xilinx/brevitas#low-precision-integer-only-lenet) that makes both weights and activations representable as integers during training. However, through the use of lookup tables in Concrete-ML, floating point univariate PyTorch functions are supported.

For _"mixed integer"_ quantization to work, the first layer of a Brevitas `nn.Module` must be a `QuantIdentity` layer. However, you can then use functions such as `torch.sigmoid` on the result of such a quantizing operation.

```python
import torch.nn as nn

class QATnetwork(nn.Module):
    def __init__(self):
        super(QATnetwork, self).__init__()
        self.quant_inp = qnn.QuantIdentity(
            bit_width=4, return_quant_tensor=True)
        # ...

    def forward(self, x):
        out = self.quant_inp(x)
        return torch.sigmoid(out)
        # ...
```

For examples of such a _"mixed integer"_ network design, please see the Quantization Aware Training examples:

- [QuantizationAwareTraining.ipynb](https://github.com/zama-ai/concrete-ml/blob/release/0.5.x/docs/advanced_examples/QuantizationAwareTraining.ipynb)
- [ConvolutionalNeuralNetwork.ipynb](https://github.com/zama-ai/concrete-ml/blob/release/0.5.x/docs/advanced_examples/ConvolutionalNeuralNetwork.ipynb)

or go to the [MNIST use-case example](https://github.com/zama-ai/concrete-ml/blob/release/0.5.x/use_case_examples/mnist/mnist_in_fhe.ipynb).

You can also refer to the [`SparseQuantNeuralNetImpl`](../developer-guide/api/concrete.ml.sklearn.qnn.md#class-sparsequantneuralnetimpl) class which is the basis of the built-in `NeuralNetworkClassifier`.
