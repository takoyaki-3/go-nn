package main

import (
	"fmt"
	"sort"
	"strconv"

	gonn "github.com/takoyaki-3/go-nn"
	"github.com/takoyaki-3/goc"
)

func loadFile(fname string, inputNodeSize, outputNodeSize int) ([]*[]float64, []*[]float64) {
	inputs := []*[]float64{}
	outputs := []*[]float64{}

	df := goc.Read2DStr(fname)

	for _, line := range df {
		input := make([]float64,len(line)-1)
		output := make([]float64,10)
		x, _ := strconv.Atoi(line[0])
		output[x] = 0.999

		for i, n := range line[1:] {
			x, _ := strconv.Atoi(n)
			input[i] = float64(x)/255.0+0.001
		}
		inputs = append(inputs, &input)
		outputs = append(outputs, &output)
	}

	return inputs, outputs
}

const NumCore = 8

func main() {
	fmt.Println("hello")

	inputs, outputs := loadFile("./sample_data/mnist_train_small.csv", 784, 10)

	nns := gonn.MakeNewNNs(100,784,10,200)

	// 現状出力
	fmt.Println("-----------------")
	for _, nn := range nns {
		fmt.Print(nn.Score, " ")
	}
	fmt.Println()

	for e := 0; e < 50000; e += 1 {

		if e>0{
			nns = gonn.Parents2Children(nns,10,100,0.001,0)
		}

		goc.Parallel(NumCore,len(nns),func(i1, i2 int) {
			nns[i1].Score = Try(nns[i1],inputs[:1000],outputs[:1000])
		})
		sort.Slice(nns,func(i, j int) bool {
			return nns[i].Score > nns[j].Score
		})

		// 現状出力
		fmt.Println("-----------------")
		fmt.Println(e, ":")
		for _, nn := range nns {
			fmt.Print(nn.Score, " ")
		}
		fmt.Println()
	}

	if false {
		fmt.Println(nns)
	}
}

func Try(nn *gonn.NeuralNetwork, inputs, outputs []*[]float64) float64 {
	// 評価部分
	var ok, ng int
	for i := 0; i < len(inputs); i++ {
		ans := 0
		for j := 1; j < 10; j++ {
			if (*outputs[i])[j] > (*outputs[i])[ans] {
				ans = j
			}
		}
		output := nn.Query(inputs[i])
		most := 0
		for j := 1; j < 10; j++ {
			if (*output)[j] > (*output)[most] {
				most = j
			}
		}
		if ans == most {
			ok++
		} else {
			ng++
		}
	}
	return float64(ok) / float64(ok+ng)
}
