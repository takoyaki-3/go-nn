package main

import (
	"fmt"
	gonn "github.com/takoyaki-3/go-nn/v2" //ニューラルネットワークライブラリ
	"math/rand"
	"sort"
	"sync"
)

const PrintBoard = false
const LoadNN = true

const NumTrain = 100000
const NumParent = 100
const NumVS = 20
const NumCore = 8
const NextGen = 10
const RandMode = false
const VS_Human = false

const N = 8

func main() {

	// 学習開始の世代数。既にe世代まで学習済みの場合、64206などの自然数を設定する
	e := -1

	// 乱数の初期値を設定。もし学習済みの特定世代のデータを使用する場合は特定世代データを読み込み
	nns := []*gonn.NeuralNetwork{}
	for i := 0; i < NumParent; i++ {
		nn := gonn.NewNeuralNetwork(N*N+1, N*N, 200, "sigmoid-sigmoid")
		nns = append(nns, nn)
	}

	// 学習ループ
	for {
		e++
		// 試合を繰り広げる
		type Case struct {
			IScore float64
			JScore float64
			I      int
			J      int
		}
		cases := []Case{}
		for i := 0; i < len(nns); i++ {
			for jj := 0; jj < NumVS; jj++ {
				j := rand.Intn(len(nns))
				// i vs j を試合パターンに追加
				cases = append(cases, Case{
					I: i,
					J: j,
				})
			}
		}

		// 実際に試合を行う処理
		Parallel(NumCore, len(cases), func(index, rank int) {
			i := cases[index].I
			j := cases[index].J
			if i != j {
				// i vs j
				a := Game(nns[i], nns[j], PrintBoard, RandMode, VS_Human)
				cpu1score := 0
				cpu2score := 0
				if v, ok := a[1]; ok {
					cpu1score = v
				}
				if v, ok := a[-1]; ok {
					cpu2score = v
				}
				if cpu1score >= cpu2score {
					cases[index].IScore += float64(cpu1score)
					cases[index].JScore -= float64(cpu1score)
				}
				if cpu1score <= cpu2score {
					cases[index].IScore -= float64(cpu2score)
					cases[index].JScore += float64(cpu2score)
				}
			}
		})

		// 試合結果を基に各ニューラルネットワークの成績を加点又は減点する
		for _, c := range cases {
			nns[c.I].Score += c.IScore
			nns[c.J].Score += c.JScore
		}

		// ニューラルネットワークを成績順に並び替える
		sort.Slice(nns, func(i, j int) bool {
			return nns[i].Score > nns[j].Score
		})

		// ニューラルネットワークの重みをファイルに保存
		nns[0].SaveWeights("trained_data.json")

		// 結果を標準出力
		fmt.Print("e:", e, ":")
		for _, n := range nns {
			fmt.Print(n.Score, " ")
		}
		fmt.Println("")

		// 突然変異を起こしつつ、子世代を生成する
		er := 0.002
		cs := gonn.Crossover(nns[:NextGen], NumParent, er)
		nns = cs
	}
}

//ゲームをプレイする関数。cpu1とcpu2はニューラルネットワーク、PrintBoardは盤面を表示するかどうかのフラグ、RandModeはランダムに手を選ぶかどうかのフラグ、VS_Humanは人間と対戦するかどうかのフラグ。
func Game(cpu1, cpu2 *gonn.NeuralNetwork, PrintBoard bool, RandMode bool, VS_Human bool) map[int]int {
	board := make([]int, N*N) //盤面を初期化する。空きは0、黒石は1、白石は-1で表す。
	board[3+3*N] = -1         //初期状態で黒石は中央に配置される。
	board[4+4*N] = -1
	board[4+3*N] = 1 //白石は斜め向かいに配置される。
	board[3+4*N] = 1

	player := 1     //最初は黒石から始める。
	finish := false //終了フラグを初期化する。
	for {
		if PrintBoard {
			printInt(board) //盤面を表示する。
		}
		var nn *gonn.NeuralNetwork
		if player == 1 {
			nn = cpu1 //黒石の手番はcpu1のニューラルネットワークを使用する。
		} else {
			nn = cpu2 //白石の手番はcpu2のニューラルネットワークを使用する。
		}
		n := CPU(&board, player, nn, RandMode, VS_Human) //ニューラルネットワークを使用して手を選ぶ。
		if n >= 0 {
			Set(&board, n, player) //手を打つ。
			finish = false         //終了フラグを初期化する。
		} else {
			if finish {
				// fmt.Println("finish")
				break //終了する。
			}
			finish = true //終了フラグを立てる。
		}

		if player == 1 {
			player = -1 //黒石と白石を交互に打つ。
		} else {
			player = 1
		}
	}

	a := map[int]int{}
	for _, p := range board {
		if _, ok := a[p]; !ok {
			a[p] = 0
		}
		a[p]++ //盤面にある石の数をカウントする。
	}
	// fmt.Println(a)
	return a //各石の数を返す。
}

func Set(board *[]int, pos int, player int) bool {
	// 既に石がある場合はエラーを返す。
	if (*board)[pos] != 0 {
		return false
	}
	(*board)[pos] = player //指定された場所に石を置く。

	// 縦横斜めを自分の石の色に変更できるか判定していく。
	mi := []int{-1, 1, 0, 0, -1, 1, 1, -1} //8方向の移動量。
	mj := []int{0, 0, -1, 1, -1, 1, -1, 1}
	i := pos % N //石が置かれた場所のx座標。
	j := pos / N //石が置かれた場所のy座標。
	for k := 0; k < len(mi); k++ {
		if f := check(board, i, j, mi[k], mj[k], player); len(f) > 0 {
			// 変更できる場合は、その方向にある石を自分の色に変更する。
			for _, p := range f {
				(*board)[p] = player
			}
		}
	}
	return true
}

func isOpponent(board *[]int, i, j, player int) bool {
	if i < 0 || i >= N || j < 0 || j >= N {
		return false
	}
	return (*board)[i+j*N] != 0 && (*board)[i+j*N] != player
}

func printBool(board []bool) {
	for i := 0; i < N; i++ {
		for j := 0; j < N; j++ {
			fmt.Print(board[i+j*N], "	")
		}
		fmt.Println("")
	}
}
func printInt(board []int) {
	fmt.Println("***************************")
	fmt.Println(" 01234567")
	for i := 0; i < N; i++ {
		fmt.Print(i)
		for j := 0; j < N; j++ {
			c := '-'
			if board[i+j*N] > 0 {
				c = 'w'
			} else if board[i+j*N] < 0 {
				c = 'b'
			}
			fmt.Printf("%c", c)
		}
		fmt.Println("")
	}
}

// i,jを中心にベクトル(di,dj)方向に確認し、色を自色に変更できるリストを返す
func check(board *[]int, i, j, di, dj, player int) []int {
	a := []int{} // ひっくり返せる箇所リスト
	// 縦に挟めるかチェック
	if isOpponent(board, i+di, j+dj, player) {
		ii := i + di
		jj := j + dj
		a = append(a, ii+jj*N)
		for 0 <= ii && ii < N && 0 <= jj && jj < N {
			ii += di
			jj += dj
			a = append(a, ii+jj*N)
			if ii < 0 || jj < 0 || ii >= N || jj >= N {
				return []int{}
			}
			if (*board)[ii+N*jj] == player {
				// ひっくり返せる
				return a
			}
			if (*board)[ii+N*jj] == 0 {
				// ひっくり返せない
				return []int{}
			}
		}
	}
	return []int{}
}

// 各マスに石を置けるか返す関数
func Available(board *[]int, player int) []bool {

	a := make([]bool, N*N)

	for i := 0; i < N; i++ {
		for j := 0; j < N; j++ {
			pos := i + N*j
			// 既に他のコマが置いてある
			if (*board)[pos] != 0 {
				a[pos] = false
				continue
			}
			// 挟めるかチェック
			mi := []int{-1, 1, 0, 0, -1, 1, 1, -1}
			mj := []int{0, 0, -1, 1, -1, 1, -1, 1}
			for k := 0; k < len(mi); k++ {
				if f := check(board, i, j, mi[k], mj[k], player); len(f) > 0 {
					// 縦横斜めにlen(f)方向挟める
					a[pos] = true
					break
				}
			}
		}
	}

	return a
}

func CPU(board *[]int, player int, nn *gonn.NeuralNetwork, RandMode bool, VS_Human bool) int {
	nextAble := Available(board, player)
	oklist := []int{}
	for i := 0; i < N*N; i++ {
		if nextAble[i] {
			oklist = append(oklist, i)
		}
	}
	if len(oklist) == 0 {
		// 置ける箇所が1マスもない。
		return -1
	}
	if RandMode && player == -1 {
		// ランダムモードの場合。
		return oklist[rand.Int()%len(oklist)]
	}

	// 行列演算をしておけるところを導く。
	input := make([]float64, len(*board)+1)
	input[0] = float64(player)/2.0 + 0.001
	for i, p := range *board {
		input[i+1] = float64(p)/2.0 + 0.001
	}

	output := nn.Forward(input)

	most := 0
	for j := 1; j < len(oklist); j++ {
		if output[oklist[j]] > output[oklist[most]] {
			most = j
		}
	}
	return oklist[most]
}

// コア数,総ループ数,呼び出し関数(インデックス,スレッド番号)
func Parallel(core int, n int, f func(int, int)) {
	wg := sync.WaitGroup{}
	wg.Add(core)
	for rank := 0; rank < core; rank++ {
		go func(rank int) {
			defer wg.Done()
			for i := rank; i < n; i += core {
				f(i, rank)
			}
		}(rank)
	}
	wg.Wait()
}
