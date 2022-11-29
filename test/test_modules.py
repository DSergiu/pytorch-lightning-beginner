import sys

sys.path.append('src')
sys.path.append('01-Linear-Regression')
sys.path.append('02-Logistic-Regression')
sys.path.append('03-Neural-Network')
sys.path.append('04-Convolutional-Network')
sys.path.append('05-01-Simple-Autoencoder')
sys.path.append('05-02-Convolutional-Autoencoder')
sys.path.append('05-03-Variational-Autoencoder')

from pytorch_lightning import Trainer
from LinearRegression import LinearRegression
from LogisticRegression import LogisticRegression
from NeuralNetwork import NeuralNetwork
from ConvolutionalNetwork import ConvolutionalNetwork
from SimpleAutoEncoder import SimpleAutoEncoder
from ConvolutionalAutoEncoder import ConvolutionalAutoEncoder
from VariationalAutoEncoder import VariationalAutoEncoder


def run_fast_dev_train(model_class):
    parser = model_class.add_model_specific_args()
    args = parser.parse_args()
    model = model_class(**vars(args))
    trainer = Trainer(fast_dev_run=True, accelerator='cpu', devices=1)
    trainer.fit(model)


def test_linear_regression():
    run_fast_dev_train(LinearRegression)


def test_logistic_regression():
    run_fast_dev_train(LogisticRegression)


def test_neural_network():
    run_fast_dev_train(NeuralNetwork)


def test_convolutional_network():
    run_fast_dev_train(ConvolutionalNetwork)


def test_simple_autoencoder():
    run_fast_dev_train(SimpleAutoEncoder)


def test_convolutional_autoencoder():
    run_fast_dev_train(ConvolutionalAutoEncoder)


def test_variational_autoencoder():
    run_fast_dev_train(VariationalAutoEncoder)
