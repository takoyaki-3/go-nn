package main

import (
	"bytes"
	"fmt"
	"io"
	"log"
	"math/rand"
	"net/http"
	"os"

	json "github.com/takoyaki-3/go-json"
	gonn "github.com/takoyaki-3/go-nn/v2" //ニューラルネットワークライブラリ
)

const N = 8 //盤面サイズ

type APIQuery struct {
	Board  [][]int `json:"board"`
	X      int     `json:"x"`
	Y      int     `json:"y"`
	Player int     `json:"player"`
	Status string  `json:"status"`
}

func Query2Board(q APIQuery) []int {
	board := make([]int, 8*8)
	for x := 0; x < N; x++ {
		for y := 0; y < N; y++ {
			board[x+y*N] = q.Board[x][y]
		}
	}
	return board
}
func Board2Query(board []int) [][]int {
	a := [][]int{}
	for x := 0; x < N; x++ {
		line := []int{}
		for y := 0; y < N; y++ {
			line = append(line, board[x+y*N])
		}
		a = append(a, line)
	}
	return a
}

func StreamToString(stream io.Reader) string {
	buf := new(bytes.Buffer)
	buf.ReadFrom(stream)
	return buf.String()
}

func main() {

	nn := &gonn.NeuralNetwork{}

	// 学習済みの重みを読み込み
	err := nn.LoadWeights("./trained_data.json")
	if err != nil {
		log.Fatalln(err)
	}
	nn.SetActivationFunction("sigmoid-sigmoid")
	nn.PrintSize()

	http.HandleFunc("/put", func(w http.ResponseWriter, r *http.Request) {

		// CORS 許可
		w.Header().Set("Access-Control-Allow-Headers", "*")
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")

		// オセロのルール上、おけるか判定する
		// 引数：座標、オセロの盤面　戻り値：おけるか判定、置いた後のオセロの盤面
		var q APIQuery
		st := StreamToString(r.Body)
		json.LoadFromString(st, &q)

		board := Query2Board(q)
		availables := Available(&board, q.Player)

		pos := q.X*N + q.Y
		if availables[pos] {
			Set(&board, pos, q.Player)
			q.Board = Board2Query(board)
			q.Status = "true"
		} else {
			q.Status = "false"
		}

		str, _ := json.DumpToString(q)
		fmt.Fprintf(w, str)
	})
	http.HandleFunc("/cpu", func(w http.ResponseWriter, r *http.Request) {

		// CORS 許可
		w.Header().Set("Access-Control-Allow-Headers", "*")
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")

		// ある地点に置いた場合のAIによる次の手
		// 引数：盤面　戻り値：置いた後のオセロの盤面
		var q APIQuery
		st := StreamToString(r.Body)
		json.LoadFromString(st, &q)

		board := Query2Board(q)
		pos := CPU(&board, q.Player, nn, false, false)
		if pos >= 0 {
			// 置けるところがあった場合
			Set(&board, pos, q.Player)
		}
		q.Board = Board2Query(board)

		str, _ := json.DumpToString(q)
		fmt.Fprintf(w, str)
	})

	// 以下、ファイルを返すWebサーバーとしての挙動
	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path[len(r.URL.Path)-1] == '/' {
			r.URL.Path += "index.html"
		}

		fmt.Println("./public" + r.URL.Path)
		f, _ := os.Open("./public" + r.URL.Path)
		io.Copy(w, f)
	})
	http.ListenAndServe(":8080", nil)
}

//ゲームをプレイする関数。cpu1とcpu2はニューラルネットワーク、PrintBoardは盤面を表示するかどうかのフラグ、RandModeはランダムに手を選ぶかどうかのフラグ、VS_Humanは人間と対戦するかどうかのフラグ。
func Game(cpu1, cpu2 *gonn.NeuralNetwork, PrintBoard bool, RandMode bool, VS_Human bool) map[int]int {
	board := make([]int, N*N) //盤面を初期化する。空きは0、黒石は1、白石は-1で表す。
	board[3+3*N] = -1         //初期状態で黒石は中央に配置される。
	board[4+4*N] = -1
	board[4+3*N] = 1 					//白石は斜め向かいに配置される。
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
