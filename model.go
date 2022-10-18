package gonn

import (
	// "fmt"
	"math"
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

type NeuralNetwork struct {
	inputNodes int
	hiddenNodes int
	outputNodes int
	wi *mat.Dense
	wo *mat.Dense
	Score float64
}

const RAND = 10000

func NewRandom05to05Dense(r,c int) *mat.Dense {
	data := []float64{}
	for i:=0;i<r*c;i++{
		data = append(data, float64(rand.Intn(RAND)) / float64(RAND) - 0.5)
	}

	return mat.NewDense(r, c, data)
}

func NewNeuralNetwork(inputNodes, outputNodes, hiddenNodes int)*NeuralNetwork{
	nn := new(NeuralNetwork)
	nn.inputNodes = inputNodes
	nn.hiddenNodes = hiddenNodes
	nn.outputNodes = outputNodes

	nn.wi = NewRandom05to05Dense(hiddenNodes,inputNodes)
	nn.wo = NewRandom05to05Dense(outputNodes,hiddenNodes)

	return nn
}

func Parent2Child(p []*NeuralNetwork,mutation float64)*NeuralNetwork{
	nn := new(NeuralNetwork)
	nn.inputNodes = p[0].inputNodes
	nn.hiddenNodes = p[0].hiddenNodes
	nn.outputNodes = p[0].outputNodes

	nn.wi = mat.NewDense(nn.hiddenNodes,nn.inputNodes,nil)
	nn.wo = mat.NewDense(nn.outputNodes,nn.hiddenNodes,nil)
	// fmt.Println(nn.wo.ColView(0).Len())
	// fmt.Println(nn.wo.RowView(0).Len())

	ps := []int{}
	ps = append(ps, rand.Intn(len(p)))
	ps = append(ps, rand.Intn(len(p)))

	for i:=0;i<nn.hiddenNodes;i++{
		for j:=0;j<nn.inputNodes;j++{
			// fmt.Println(i,j,"inputNodes")
			// fmt.Println(nn.wi.ColView(0).Len())
			// fmt.Println(nn.wi.RowView(0).Len())
			if float64(rand.Intn(RAND)) / float64(RAND) > mutation {
				t := ps[rand.Intn(len(ps))]
				x := p[t].wi.At(i,j)
				nn.wi.Set(i,j,x)
			} else {
				x := float64(rand.Intn(RAND)) / float64(RAND) - 0.5
				nn.wi.Set(i,j,x)
			}
		}
	}
	for i:=0;i<nn.outputNodes;i++{
		for j:=0;j<nn.hiddenNodes;j++{
			// fmt.Println(i,j,"outputNodes")
			// fmt.Println(nn.wo.ColView(0).Len())
			// fmt.Println(nn.wo.RowView(0).Len())
			if float64(rand.Intn(RAND)) / float64(RAND) > mutation {
				t := ps[rand.Intn(len(ps))]
				x := p[t].wo.At(i,j)
				nn.wo.Set(i,j,x)
			} else {
				x := float64(rand.Intn(RAND)) / float64(RAND) - 0.5
				nn.wo.Set(i,j,x)
			}
		}
	}

	return nn
}

func (nn *NeuralNetwork)Query(input *mat.Dense)*mat.Dense{
	A := mat.NewDense(nn.hiddenNodes,1,nil)
	A.Product(nn.wi,input)

	// 要素ごとに適用する関数
	sigmoid := func(i, j int, v float64) float64 {
		return 1 / (1 + math.Exp(-v))
	}
	var B mat.Dense
	B.Apply(sigmoid, A)


	C := mat.NewDense(nn.outputNodes,1,nil)
	C.Product(nn.wo,&B)

	var D mat.Dense
	D.Apply(sigmoid, C)

	return &D
}

