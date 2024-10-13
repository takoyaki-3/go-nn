package gonn

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"math"
	"math/rand"
	"os"
	"time"
	"encoding/gob"
	"bytes"
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
	Score			 float64
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

	nn.SetActivationFunction(activationFunction)

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

func (nn *NeuralNetwork)SetActivationFunction(activationFunction string){
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
}

func (nn *NeuralNetwork)PrintSize(){
	fmt.Println("---------------------")
	fmt.Println("input size:",nn.inputSize)
	fmt.Println("hidden size:",nn.hiddenSize)
	fmt.Println("output size:",nn.outputSize)
	fmt.Println("nn.bias1 len:",len(nn.bias1))
	fmt.Println("nn.bias2 len:",len(nn.bias2))
	fmt.Println("---------------------")
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

	if len(nn.bias1) == 0 {
		nn.bias1 = make([]float64, nn.hiddenSize)
	}
	if len(nn.bias2) == 0 {
		nn.bias2 = make([]float64, nn.outputSize)
	}

	return nil
}

// 遺伝的アルゴリズムにおける交配
func Crossover(parents []*NeuralNetwork, numChildren int, mutationRate float64) []*NeuralNetwork {
	children := make([]*NeuralNetwork, numChildren)
	rand.Seed(time.Now().UnixNano())
	for i := 0; i < numChildren; i++ {
		child := &NeuralNetwork{
			inputSize:                   parents[0].inputSize,
			hiddenSize:                  parents[0].hiddenSize,
			outputSize:                  parents[0].outputSize,
			weights1:                    make([][]float64, parents[0].inputSize),
			weights2:                    make([][]float64, parents[0].hiddenSize),
			bias1:                       make([]float64, parents[0].hiddenSize),
			bias2:                       make([]float64, parents[0].outputSize),
			activationFunction1:         parents[0].activationFunction1,
			activationFunction1Derivative: parents[0].activationFunction1Derivative,
			activationFunction2:         parents[0].activationFunction2,
			activationFunction2Derivative: parents[0].activationFunction2Derivative,
		}

		for j := range child.weights1 {
			child.weights1[j] = make([]float64, child.hiddenSize)
			for k := range child.weights1[j] {
				if rand.Float64() < 0.5 {
					child.weights1[j][k] = parents[0].weights1[j][k]
				} else {
					child.weights1[j][k] = parents[1].weights1[j][k]
				}

				if rand.Float64() < mutationRate {
					child.weights1[j][k] += rand.Float64() - 0.5
				}
			}
		}

		for j := range child.weights2 {
			child.weights2[j] = make([]float64, child.outputSize)
			for k := range child.weights2[j] {
				if rand.Float64() < 0.5 {
					child.weights2[j][k] = parents[0].weights2[j][k]
				} else {
					child.weights2[j][k] = parents[1].weights2[j][k]
				}

				if rand.Float64() < mutationRate {
					child.weights2[j][k] += rand.Float64() - 0.5
				}
			}
		}

		for j := range child.bias1 {
			if rand.Float64() < 0.5 {
				child.bias1[j] = parents[0].bias1[j]
			} else {
				child.bias1[j] = parents[1].bias1[j]
			}

			if rand.Float64() < mutationRate {
				child.bias1[j] += rand.Float64() - 0.5
			}
		}

		for j := range child.bias2 {
			if rand.Float64() < 0.5 {
				child.bias2[j] = parents[0].bias2[j]
			} else {
				child.bias2[j] = parents[1].bias2[j]
			}
			if rand.Float64() < mutationRate {
				child.bias2[j] += rand.Float64() - 0.5
			}
		}
	
		children[i] = child
	}
	
	return children
}

func (nn *NeuralNetwork) SaveWeightsBinary(filepath string) error {
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

	// Save Weights in binary format
	encoder := gob.NewEncoder(file)
	err = encoder.Encode(weights)
	if err != nil {
		return err
	}

	return nil
}

func (nn *NeuralNetwork) LoadWeightsBinary(filepath string) error {
	weights := Weights{}

	data, err := ioutil.ReadFile(filepath)
	if err != nil {
		return err
	}

	// Load Weights in binary format
	decoder := gob.NewDecoder(bytes.NewReader(data))
	err = decoder.Decode(&weights)
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

	if len(nn.bias1) == 0 {
		nn.bias1 = make([]float64, nn.hiddenSize)
	}
	if len(nn.bias2) == 0 {
		nn.bias2 = make([]float64, nn.outputSize)
	}

	return nil
}

// GetWeight1 returns the weight from input layer i to hidden layer j
func (nn *NeuralNetwork) GetWeight1(i, j int) float64 {
	if i >= 0 && i < nn.inputSize && j >= 0 && j < nn.hiddenSize {
		return nn.weights1[i][j]
	}
	fmt.Println("Invalid index")
	return 0
}

// GetWeight2 returns the weight from hidden layer i to output layer j
func (nn *NeuralNetwork) GetWeight2(i, j int) float64 {
	if i >= 0 && i < nn.hiddenSize && j >= 0 && j < nn.outputSize {
		return nn.weights2[i][j]
	}
	fmt.Println("Invalid index")
	return 0
}
