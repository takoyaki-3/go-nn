package main

import (
	"fmt"
	"image"
	"image/color"
	"image/png"
	"log"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"sync"
	gonn "github.com/takoyaki-3/go-nn/v2"
)

type Data struct {
	Wi [][]float64 `json:"wi"`
	Wo [][]float64 `json:"wo"`
}

// 同時ファイル読み込みを最大10件に制限するためのセマフォチャネル
const maxConcurrentReads = 10

type genData struct {
	index  int
	values []float64
}

func loadValues(genDir string, inputOutput string, x, y, index int, wg *sync.WaitGroup, ch chan genData, sem chan struct{}) {
	defer wg.Done()

	// セマフォの獲得
	sem <- struct{}{}
	defer func() { <-sem }() // 処理が終わったらセマフォを解放

	values := make([]float64, 0, 100)
	for i := 0; i < 100; i++ {
		filePath := filepath.Join(genDir, fmt.Sprintf("%d.b", i)) // 拡張子を .b に変更

		nn := &gonn.NeuralNetwork{}
		if err := nn.LoadWeightsBinary(filePath); err != nil {
			fmt.Println("File not found: ", filePath)
			// ファイルが見つからない場合、0を追加して処理を続ける
			values = append(values, 0)
			continue
		}
		value := nn.GetWeight1(x, y)
		values = append(values, value)
	}
	sort.Float64s(values)
	ch <- genData{index: index, values: values}
}

func main() {
	trainDir := "../train-binary"      // 学習データが保存されているディレクトリのパス
	inputOutput := "input"   // 'input' か 'output' を指定
	x, y := 0, 0             // x座標とy座標
	startGen, endGen := 25000, 26000 // 世代の開始と終了
	stepGen := 1

	// 世代リストの作成
	generations := []int{}
	for gen := startGen; gen <= endGen; gen += stepGen {
		generations = append(generations, gen)
	}

	// 画像データの初期化
	imageData := make([][]float64, len(generations))
	genLabels := make([]int, len(generations))

	// 並列処理用のチャンネルとWaitGroup
	ch := make(chan genData)
	var wg sync.WaitGroup

	// 最大同時ファイル読み込み件数を制御するセマフォチャンネル
	sem := make(chan struct{}, maxConcurrentReads)

	for i, gen := range generations {
		fmt.Printf("Processing generation %d...\n", gen)
		genDir := filepath.Join(trainDir, strconv.Itoa(gen))

		wg.Add(1)
		go loadValues(genDir, inputOutput, x, y, i, &wg, ch, sem)

		genLabels[i] = gen
	}

	// チャンネルからデータを受け取り、imageDataに追加
	go func() {
		wg.Wait()
		close(ch)
	}()

	for data := range ch {
		imageData[data.index] = data.values
	}

	// 画像の生成
	imgHeight := len(imageData)
	imgWidth := 100
	img := image.NewRGBA(image.Rect(0, 0, imgWidth, imgHeight))

	// 値をカラーマップに従って色に変換
	for i := 0; i < imgHeight; i++ {
		for j := 0; j < imgWidth; j++ {
			// 例外処理
			if len(imageData[i]) <= j {
				fmt.Printf("Warning: generation %d, index=%d is out of range\n", genLabels[i], j)
				continue
			}
			val := imageData[i][j]
			col := valueToColor(val)
			img.Set(j, i, col)
		}
	}

	// 画像をファイルに保存
	// 保存先のファイルをstart,endから作成
	outputFileName := fmt.Sprintf("output_image_1117_%dto%d.png", startGen, endGen)
	outputFile, err := os.Create(outputFileName)
	if err != nil {
		log.Fatalf("Failed to create file: %v", err)
	}
	defer outputFile.Close()

	if err := png.Encode(outputFile, img); err != nil {
		log.Fatalf("Failed to encode image: %v", err)
	}

	fmt.Println("画像を %s に保存しました。", outputFileName)
}

// 値を色に変換する関数
func valueToColor(value float64) color.Color {
	// 値を [-1, 1] の範囲にクランプ
	if value < -1 {
		value = -1
	} else if value > 1 {
		value = 1
	}

	// 値に応じて青から赤までの色を生成
	r := uint8((value + 1) * 127.5)
	b := uint8((1 - value) * 127.5)
	return color.RGBA{R: r, G: 0, B: b, A: 255}
}
