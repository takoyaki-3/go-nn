package main

import (
	"fmt"
	// "math/rand"
	"sort"
	"strconv"
	"sync"

	gonn "github.com/takoyaki-3/go-nn"
	"github.com/takoyaki-3/goc"

	"gonum.org/v1/gonum/mat"
)

func loadFile(fname string, inputNodeSize, outputNodeSize int) ([]*mat.Dense, []*mat.Dense) {
	inputs := make([]*mat.Dense, 0)
	outputs := make([]*mat.Dense, 0)

	df := goc.Read2DStr(fname)

	for _, line := range df {
		input := mat.NewDense(len(line)-1, 1, nil)
		output := mat.NewDense(10, 1, nil)
		x, _ := strconv.Atoi(line[0])
		output.Set(x, 0, 0.999)

		for i, n := range line[1:] {
			x, _ := strconv.Atoi(n)
			input.Set(i, 0, float64(x)/255.0+0.001)
		}
		inputs = append(inputs, input)
		outputs = append(outputs, output)
	}

	return inputs, outputs
}

func matPrint(X mat.Matrix) {
	fa := mat.Formatted(X, mat.Prefix(""), mat.Squeeze())
	fmt.Printf("%v\n", fa)
}

func NewDense(r, c int, x float64) *mat.Dense {
	data := []float64{}
	for i := 0; i < r*c; i++ {
		data = append(data, x)
	}

	return mat.NewDense(r, c, data)
}

const NumParent = 20

const NumCore = 8

func main() {
	fmt.Println("hello")

	inputs, outputs := loadFile("./sample_data/mnist_train_small.csv", 784, 10)

	nns := []*gonn.NeuralNetwork{}
	for i := 0; i < NumParent; i++ {
		nn := gonn.NewNeuralNetwork(784, 10, 200)
		nns = append(nns, nn)
	}

	// 現状出力
	fmt.Println("-----------------")
	for _, nn := range nns {
		fmt.Print(nn.Score, " ")
	}
	fmt.Println()

	for e := 0; e < 50000; e += 100 {

		in := []*mat.Dense{}
		out := []*mat.Dense{}

		p := nns[:NumParent]

		NumChild := 200
		er := 0.01
		if e%10 == 9 {
			NumChild = 500
			er = 0.02
		}

		nnsSub := make([][]*gonn.NeuralNetwork, NumCore)

		n := e
		if n > len(inputs) {
			n = len(inputs)
		}
		in = inputs[:n]
		out = outputs[:n]

		wg := sync.WaitGroup{}
		wg.Add(NumCore)
		for rank := 0; rank < NumCore; rank++ {
			go func(rank int) {
				defer wg.Done()
				for i := rank; i < NumChild; i += NumCore {
					nn := gonn.Parent2Child(p, er)
					nn.Score = Try(nn, in, out)
					nnsSub[rank] = append(nnsSub[rank], nn)
				}
			}(rank)
		}
		wg.Wait()
		nns = []*gonn.NeuralNetwork{}
		for rank := 0; rank < NumCore; rank++ {
			nns = append(nns, nnsSub[rank]...)
		}

		for i, _ := range p[:2] {
			p[i].Score = Try(p[i], in, out)
		}
		nns = append(nns, p[:2]...)
		sort.Slice(nns, func(i, j int) bool {
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

func Try(nn *gonn.NeuralNetwork, inputs, outputs []*mat.Dense) float64 {
	// 評価部分
	var ok, ng int
	for ii := 0; ii < len(inputs); ii++ {
		i := ii
		ans := 0
		for j := 1; j < 10; j++ {
			if outputs[i].At(j, 0) > outputs[i].At(ans, 0) {
				ans = j
			}
		}
		output := nn.Query(inputs[i])
		most := 0
		for j := 1; j < 10; j++ {
			if output.At(j, 0) > output.At(most, 0) {
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
