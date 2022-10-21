package gonn

import (
	// "fmt"
	"fmt"
	"math"
	"math/rand"

	json "github.com/takoyaki-3/go-json"
	"gonum.org/v1/gonum/mat"
)

type NeuralNetwork struct {
	inputNodes  int        // 入力ノード数
	hiddenNodes int        // 隠れ層ノード数
	outputNodes int        // 出力ノード数
	wi          *mat.Dense // 入力層と隠れ層間の重み
	wo          *mat.Dense // 隠れ層と出力層間の重み
	Score       float64    // 得点
}

const RAND = 10000

func NewRandom05to05Dense(r, c int) *mat.Dense {
	data := []float64{}
	for i := 0; i < r*c; i++ {
		data = append(data, float64(rand.Intn(RAND))/float64(RAND)-0.5)
	}

	return mat.NewDense(r, c, data)
}

const SinglePointCrossover = 1
const TwoPointCrosser = 2
const UniformCrosser = 3

// 乱数を初期値とした新規ニューラルネットワーク作成
func NewNeuralNetwork(inputNodes, outputNodes, hiddenNodes int) *NeuralNetwork {
	nn := new(NeuralNetwork)
	nn.inputNodes = inputNodes
	nn.hiddenNodes = hiddenNodes
	nn.outputNodes = outputNodes

	nn.wi = NewRandom05to05Dense(hiddenNodes, inputNodes)
	nn.wo = NewRandom05to05Dense(outputNodes, hiddenNodes)

	return nn
}

// 親ニューラルネットワークから子ニューラルネットワークを作成
func Parent2Child(p []*NeuralNetwork, mutation float64) *NeuralNetwork {
	nn := new(NeuralNetwork)
	nn.inputNodes = p[0].inputNodes
	nn.hiddenNodes = p[0].hiddenNodes
	nn.outputNodes = p[0].outputNodes

	nn.wi = mat.NewDense(nn.hiddenNodes, nn.inputNodes, nil)
	nn.wo = mat.NewDense(nn.outputNodes, nn.hiddenNodes, nil)
	// fmt.Println(nn.wo.ColView(0).Len())
	// fmt.Println(nn.wo.RowView(0).Len())

	ps := []int{}
	ps = append(ps, rand.Intn(len(p)))
	ps = append(ps, rand.Intn(len(p)))

	for i := 0; i < nn.hiddenNodes; i++ {
		for j := 0; j < nn.inputNodes; j++ {
			// fmt.Println(i,j,"inputNodes")
			// fmt.Println(nn.wi.ColView(0).Len())
			// fmt.Println(nn.wi.RowView(0).Len())
			if float64(rand.Intn(RAND))/float64(RAND) > mutation {
				t := ps[rand.Intn(len(ps))]
				x := p[t].wi.At(i, j)
				nn.wi.Set(i, j, x)
			} else {
				x := float64(rand.Intn(RAND))/float64(RAND) - 0.5
				nn.wi.Set(i, j, x)
			}
		}
	}
	for i := 0; i < nn.outputNodes; i++ {
		for j := 0; j < nn.hiddenNodes; j++ {
			// fmt.Println(i,j,"outputNodes")
			// fmt.Println(nn.wo.ColView(0).Len())
			// fmt.Println(nn.wo.RowView(0).Len())
			if float64(rand.Intn(RAND))/float64(RAND) > mutation {
				t := ps[rand.Intn(len(ps))]
				x := p[t].wo.At(i, j)
				nn.wo.Set(i, j, x)
			} else {
				x := float64(rand.Intn(RAND))/float64(RAND+1) - 1.0
				nn.wo.Set(i, j, x)
			}
		}
	}

	return nn
}

// ニューラルネットワークを用いて入力ベクトルから出力ベクトルを得る
func (nn *NeuralNetwork) Query(inputVec *[]float64) *[]float64 {

	input := mat.NewDense(len(*inputVec),1,nil)
	for i,p:=range *inputVec{
		input.Set(i,0,p)
	}

	A := mat.NewDense(nn.hiddenNodes, 1, nil)
	A.Product(nn.wi, input)

	// 要素ごとに適用する関数
	sigmoid := func(i, j int, v float64) float64 {
		return 1 / (1 + math.Exp(-v))
	}
	var B mat.Dense
	B.Apply(sigmoid, A)

	C := mat.NewDense(nn.outputNodes, 1, nil)
	C.Product(nn.wo, &B)

	var D mat.Dense
	D.Apply(sigmoid, C)

	output := make([]float64,D.RawMatrix().Rows)

	return &output
}

// ニューラルネットワークの一時保存用JSONデータ形式
type Data struct {
	InputNodes  int         `json:"input_nodes"`
	HiddenNodes int         `json:"hidden_nodes"`
	OutputNodes int         `json:"output_nodes"`
	Wi          [][]float64 `json:"wi"`
	Wo          [][]float64 `json:"wo"`
	Score       float64     `json:"score"`
}

func NeuralNetwork2Data(nn *NeuralNetwork)*Data{
	wi := make([][]float64, nn.hiddenNodes)
	for i := 0; i < nn.hiddenNodes; i++ {
		for j := 0; j < nn.inputNodes; j++ {
			x := nn.wi.At(i, j)
			wi[i] = append(wi[i], x)
		}
	}
	wo := make([][]float64, nn.outputNodes)
	for i := 0; i < nn.outputNodes; i++ {
		for j := 0; j < nn.hiddenNodes; j++ {
			x := nn.wo.At(i, j)
			wo[i] = append(wo[i], x)
		}
	}
	data := new(Data)
	*data = Data{
		InputNodes:  nn.inputNodes,
		HiddenNodes: nn.hiddenNodes,
		OutputNodes: nn.outputNodes,
		Wi:          wi,
		Wo:          wo,
		Score:       nn.Score,
	}
	return data
}

func Data2NeuralNetwork(data *Data,nn *NeuralNetwork)error{
	nn.hiddenNodes = data.HiddenNodes
	nn.inputNodes = data.InputNodes
	nn.outputNodes = data.OutputNodes

	nn.wi = mat.NewDense(nn.hiddenNodes, nn.inputNodes, nil)
	nn.wo = mat.NewDense(nn.outputNodes, nn.hiddenNodes, nil)

	for i := 0; i < nn.hiddenNodes; i++ {
		for j := 0; j < nn.inputNodes; j++ {
			nn.wi.Set(i, j, data.Wi[i][j])
		}
	}
	for i := 0; i < nn.outputNodes; i++ {
		for j := 0; j < nn.hiddenNodes; j++ {
			nn.wo.Set(i, j, data.Wo[i][j])
		}
	}
	return nil
}

// ニューラルネットワークをJSON形式で保存
func (nn *NeuralNetwork) Save(path string) error {
	return json.DumpToFile(NeuralNetwork2Data(nn), path)
}

// ニューラルネットワークのJSONファイルを読み込み
func (nn *NeuralNetwork) Load(path string) error {
	var data Data
	if err := json.LoadFromPath(path, &data); err != nil {
		return err
	}
	return Data2NeuralNetwork(&data, nn)
}

// 親世代が格納された配列から子世代を作り出す
func Parents2Children_(parents []*NeuralNetwork,numChildren int, mutation float64)[]*NeuralNetwork{
	children := []*NeuralNetwork{}

	for i:=0;i<numChildren;i++{
		children = append(children, Parent2Child(parents,mutation))
	}

	return children
}

// 遺伝子の比較を行う
func CompNN(nn1 ,nn2 *NeuralNetwork)float64{
	d1 := NeuralNetwork2Data(nn1)
	d2 := NeuralNetwork2Data(nn2)

	n:=0
	c:=0
	for i,w:=range d1.Wi{
		for j,_:=range w{
			if d1.Wi[i][j] == d2.Wi[i][j]{
				n++
			} else {
				fmt.Println(c,d1.Wi[i][j],d2.Wi[i][j])
			}
			c++
		}
	}
	return float64(n)/float64(c)
}
