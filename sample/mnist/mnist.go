package main

import (
	"bufio"
	"encoding/csv"
	"fmt"
	"io"
	"log"
	"os"
	"strconv"
	"strings"
	. "github.com/takoyaki-3/go-nn/v2"
)


func ReadCSVFile(filename string) ([][]float64, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	reader := csv.NewReader(bufio.NewReader(file))
	var data [][]float64
	for {
		record, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, err
		}
		row := make([]float64, len(record))
		for i, v := range record {
			val, err := strconv.ParseFloat(strings.TrimSpace(v), 64)
			if err != nil {
				return nil, err
			}
			if i != 0 {
				row[i] = val/256.0
			} else {
				row[0] = val
			}
		}
		data = append(data, row)
	}
	return data, nil
}

func main() {
	// Load training data
	inputs, err := ReadCSVFile("data/mnist_train.csv")
	if err != nil {
		log.Fatal(err)
	}
	// Create one-hot encoded labels
	labels := [][]float64{
		{1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
		{0, 1, 0, 0, 0, 0, 0, 0, 0, 0},
		{0, 0, 1, 0, 0, 0, 0, 0, 0, 0},
		{0, 0, 0, 1, 0, 0, 0, 0, 0, 0},
		{0, 0, 0, 0, 1, 0, 0, 0, 0, 0},
		{0, 0, 0, 0, 0, 1, 0, 0, 0, 0},
		{0, 0, 0, 0, 0, 0, 1, 0, 0, 0},
		{0, 0, 0, 0, 0, 0, 0, 1, 0, 0},
		{0, 0, 0, 0, 0, 0, 0, 0, 1, 0},
		{0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
	}

	// make outputs
	outputs := [][]float64{}
	for _, input := range inputs {
		outputs = append(outputs, labels[int(input[0])])
	}

	// Train neural network
	nn := NewNeuralNetwork(len(inputs[0]), 64, len(labels[0]), "relu-sigmoid")
	nn.TrainNeuralNetwork(inputs, outputs, 0.01, 50)

	// Save weights
	nn.SaveWeights("weights.json")

	// Load test data
	testInputs, err := ReadCSVFile("data/mnist_test.csv")
	if err != nil {
		log.Fatal(err)
	}

	// Test neural network
	ca := 0
	for i := range testInputs {
		outputs := nn.Forward(testInputs[i])
		bestIndex := 0
		bestValue := 0.0

		for j := range outputs {
			if outputs[j] > bestValue {
				bestValue = outputs[j]
				bestIndex = j
			}
		}
		fmt.Printf("Predicted: %d ,Correct answer: %d \n", bestIndex, int(testInputs[i][0]))
		if bestIndex == int(testInputs[i][0]) {
			ca++
		}
	}
	fmt.Printf("Correct answer rate: %d / %d", ca, len(testInputs))
}