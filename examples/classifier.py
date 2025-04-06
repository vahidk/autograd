import numpy as np

import autograd

from .mnist import MNIST


class Classifier(autograd.Sequential):
    def __init__(self):
        super().__init__(
            autograd.Conv2D(in_channels=1, out_channels=32, kernel_size=3, stride=2),
            autograd.ReLU(),
            autograd.Conv2D(in_channels=32, out_channels=64, kernel_size=3, stride=2),
            autograd.ReLU(),
            autograd.Reshape(shape=(-1, 64 * 6 * 6)),
            autograd.Dense(in_channels=64 * 6 * 6, out_channels=512),
            autograd.ReLU(),
            autograd.Dense(in_channels=512, out_channels=10),
        )


train_dataset = MNIST(train=True)
test_dataset = MNIST(train=False)
train_loader = iter(autograd.DataLoader(train_dataset, batch_size=64, repeat=True))
test_loader = iter(autograd.DataLoader(test_dataset, batch_size=64, repeat=False))

classifier = Classifier()
optim = autograd.SGD(params=classifier.parameters(), lr=0.01, momentum=0.9)

# Train the classifier
for i in range(1000):
    images, labels = next(train_loader)
    x = autograd.Tensor(images.astype(np.float32), dtype=np.float32)
    y = autograd.Tensor(labels.astype(np.int32), dtype=np.int32)

    yhat = classifier(x)
    loss = yhat.sparse_softmax_cross_entropy(y)

    optim.zero_grad()
    loss.backward()
    optim.step()

# Compute accuracy
correct = 0
total = 0
for images, labels in test_loader:
    x = autograd.Tensor(images.astype(np.float32), dtype=np.float32)
    y = autograd.Tensor(labels.astype(np.int32), dtype=np.int32)

    yhat = classifier(x)
    pred = yhat.argmax(axis=1)

    correct += np.sum(pred.data == y.data)
    total += len(y.data)

accuracy = correct / total
print(f"Test accuracy: {accuracy:.4f}")
