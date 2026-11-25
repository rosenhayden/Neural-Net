A configurable multilayer perceptron for classification and regression, including backpropagation,
Glorot initialization and a command-line layer-spec parser

## Regression

"" depth 0:
Best dev loss: 6.355864860659855 and r2 is 0.23363416073635412

"32x1" width 32, depth 1
Best dev loss: 0.07335927636747497 and r2 is 0.9911546194524821

"32x3"
Best dev loss: 0.05205512639285659 and r2 is 0.9937233922525699

"32x12"
Best dev loss: 0.08826653242475492 and r2 is 0.9893571596181536

"256x1"
Best dev loss: 0.05135510488688661 and r2 is 0.993807798164383
Increasing the depth seemed to improve performance. Linear did really bad which tells us there is something nonlinear about the data. Adding a single 32 wide layer improved accuracy a lot. The improvement from a width of 32 to 256 did not have a huge impact on loss. Increasing depth did get more performance going from 1 deep to 3, but 3 to 12 did worse in terms of loss and R2. The best result came from the 32x3 network.

## Classification (4 classes)

""
Best dev loss: 0.09092867585300518 and acc is 0.966

"32x1"
Best dev loss: 0.014828596084282886 and acc is 0.992

"256x1"
Best dev loss: 0.017057639225133023 and acc is 0.994

"32x3"
Best dev loss: 0.028509542044865156 and acc is 0.992

"32x12"
Best dev loss: 1.3859402507142038 and acc is 0.26
We again saw performance suffer when adding a lot of depth to the network. The widest network performed best in terms of accuracy and the deepest did worst. Linear did pretty good this time so the data may close to linearly sepreable but not perfect.
