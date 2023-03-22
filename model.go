package gonn

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"math"
	"math/rand"
	"os"
	"time"
)

type NeuralNetwork struct {
	inputSize  int
	hiddenSize int
	outputSize int
	weights1   [][]float64
	weights2   [][]float64
	bias1      []float64
	bias2      []float64
	activationFunction1 func(float64)float64
	activationFunction1Derivative func(float64)float64
	activationFunction2 func(float64)float64
	activationFunction2Derivative func(float64)float64
}

type Weights struct {
	InputSize  int       `json:"inputSize"`
	HiddenSize int       `json:"hiddenSize"`
	OutputSize int       `json:"outputSize"`
	Weights1   [][]float64 `json:"wi"`
	Weights2   [][]float64 `json:"wo"`
	Bias1      []float64   `json:"biasI"`
	Bias2      []float64   `json:"biasO"`
}

func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func sigmoidDerivative(x float64) float64 {
	return sigmoid(x) * (1 - sigmoid(x))
}

func relu(x float64) float64 {
	if x >= 0 {
		return x
	} else {
		return 0
	}
}

func reluDerivative(x float64) float64 {
	if x >= 0 {
		return 1
	} else {
		return 0
	}
}

func NewNeuralNetwork(inputSize, hiddenSize, outputSize int, activationFunction string) *NeuralNetwork {
	nn := &NeuralNetwork{
		inputSize:  inputSize,
		hiddenSize: hiddenSize,
		outputSize: outputSize,
		weights1:   make([][]float64, inputSize),
		weights2:   make([][]float64, hiddenSize),
		bias1:      make([]float64, hiddenSize),
		bias2:      make([]float64, outputSize),
	}

	if activationFunction == "relu-sigmoid" {
		nn.activationFunction1 = relu
		nn.activationFunction1Derivative = reluDerivative
		nn.activationFunction2 = sigmoid
		nn.activationFunction2Derivative = sigmoidDerivative
	} else if activationFunction == "sigmoid-sigmoid" {
		nn.activationFunction1 = sigmoid
		nn.activationFunction1Derivative = sigmoidDerivative
		nn.activationFunction2 = sigmoid
		nn.activationFunction2Derivative = sigmoidDerivative
	}

	rand.Seed(time.Now().UnixNano())

	for i := range nn.weights1 {
		nn.weights1[i] = make([]float64, nn.hiddenSize)
		for j := range nn.weights1[i] {
			nn.weights1[i][j] = rand.Float64()
		}
	}

	for i := range nn.weights2 {
		nn.weights2[i] = make([]float64, nn.outputSize)
		for j := range nn.weights2[i] {
			nn.weights2[i][j] = rand.Float64()
		}
	}

	for i := range nn.bias1 {
		nn.bias1[i] = rand.Float64()
	}

	for i := range nn.bias2 {
		nn.bias2[i] = rand.Float64()
	}

	return nn
}

func (nn *NeuralNetwork) Forward(input []float64) []float64 {
	hidden := make([]float64, nn.hiddenSize)
	output := make([]float64, nn.outputSize)

	for i := range hidden {
		for j := range input {
			hidden[i] += input[j] * nn.weights1[j][i]
		}
		hidden[i] = nn.activationFunction1(hidden[i] + nn.bias1[i])
	}

	for i := range output {
		for j := range hidden {
			output[i] += hidden[j] * nn.weights2[j][i]
		}
		output[i] = nn.activationFunction2(output[i] + nn.bias2[i])
	}

	return output
}

func (nn *NeuralNetwork) TrainNeuralNetwork(inputs [][]float64, outputs [][]float64, learningRate float64, epochs int) {
	for epoch := 0; epoch < epochs; epoch++ {
		correct := 0 // 正解数をカウントするための変数
		for i := range inputs {
			input := inputs[i]
			output := outputs[i]
			hidden := make([]float64, nn.hiddenSize)
			outputLayer := make([]float64, nn.outputSize)

			// Forward propagation
			for j := range hidden {
				for k := range input {
					hidden[j] += input[k] * nn.weights1[k][j]
				}
				hidden[j] = nn.activationFunction1(hidden[j] + nn.bias1[j])
			}

			for j := range outputLayer {
				for k := range hidden {
					outputLayer[j] += hidden[k] * nn.weights2[k][j]
				}
				outputLayer[j] = nn.activationFunction2(outputLayer[j] + nn.bias2[j])
			}

			// 正解数をカウントする
			prediction := 0
			for j, val := range outputLayer {
				if val > outputLayer[prediction] {
					prediction = j
				}
			}
			if output[prediction] == 1 {
				correct++
			}

			// Backpropagation
			outputLayerError := make([]float64, nn.outputSize)
			for j := range outputLayer {
				outputLayerError[j] = output[j] - outputLayer[j]
			}

			outputLayerDelta := make([]float64, nn.outputSize)
			for j := range outputLayerDelta {
				outputLayerDelta[j] = outputLayerError[j] * nn.activationFunction2Derivative(outputLayer[j])
			}

			hiddenError := make([]float64, nn.hiddenSize)
			for j := range hidden {
				for k := range outputLayerDelta {
					hiddenError[j] += outputLayerDelta[k] * nn.weights2[j][k]
				}
			}

			hiddenDelta := make([]float64, nn.hiddenSize)
			for j := range hiddenDelta {
				hiddenDelta[j] = hiddenError[j] * nn.activationFunction1Derivative(hidden[j])
			}

			// Update weights and biases
			for j := range nn.bias2 {
				nn.bias2[j] += learningRate * outputLayerDelta[j]
				for k := range hidden {
					nn.weights2[k][j] += learningRate * outputLayerDelta[j] * hidden[k]
				}
			}

			for j := range nn.bias1 {
				nn.bias1[j] += learningRate * hiddenDelta[j]
				for k := range input {
					nn.weights1[k][j] += learningRate * hiddenDelta[j] * input[k]
				}
			}
		}

		// トレーニングセット全体に対する正答率を出力する
		accuracy := float64(correct) / float64(len(inputs)) * 100.0
		fmt.Printf("epoch: %d, accuracy: %f\n", epoch, accuracy)
	}
}

func (nn *NeuralNetwork) SaveWeights(filepath string) error {
	weights := Weights{
		InputSize:  nn.inputSize,
		HiddenSize: nn.hiddenSize,
		OutputSize: nn.outputSize,
		Weights1:   nn.weights1,
		Weights2:   nn.weights2,
		Bias1:      nn.bias1,
		Bias2:      nn.bias2,
	}

	file, err := os.Create(filepath)
	if err != nil {
		return err
	}
	defer file.Close()

	encoder := json.NewEncoder(file)
	err = encoder.Encode(weights)
	if err != nil {
		return err
	}

	return nil
}

func (nn *NeuralNetwork) LoadWeights(filepath string) error {
	weights := Weights{}

	data, err := ioutil.ReadFile(filepath)
	if err != nil {
		return err
	}

	err = json.Unmarshal(data, &weights)
	if err != nil {
		return err
	}

	nn.inputSize = weights.InputSize
	nn.hiddenSize = weights.HiddenSize
	nn.outputSize = weights.OutputSize
	nn.weights1 = weights.Weights1
	nn.weights2 = weights.Weights2
	nn.bias1 = weights.Bias1
	nn.bias2 = weights.Bias2

	return nil
}
