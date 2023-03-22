// 盤面の一辺のマス数
const size = 8;
// 1マスの大きさ
const grid_size = 64;
// 線の太さ
const wd = 2;
// ゲームの状態を保持するオブジェクト
const gameStatus = {
  board: []
};
// 盤面を作成
for (let x = 0; x < size; x++) {
  const row = [];
  for (let y = 0; y < size; y++) {
    // 各マスに0を格納
    row.push(0);
  }
  // 各行をboardプロパティに格納
  gameStatus.board.push(row);
}
// 中央の4つのマスにそれぞれ1、-1を格納
gameStatus.board[3][3] = 1;
gameStatus.board[4][3] = -1;
gameStatus.board[3][4] = -1;
gameStatus.board[4][4] = 1;
// Canvasに描画する関数
function draw(canvas, gameStatus) {
  // Canvasが利用可能か確認
  if (canvas.getContext) {
    // 2D描画コンテキストを取得
    const ctx = canvas.getContext('2d');
    // 背景色を設定
    ctx.fillStyle = "#000";
    // 背景を描画
    ctx.fillRect(0, 0, grid_size * size, grid_size * size);
    // 各マスを描画
    for (let x = 0; x < size; x++) {
      for (let y = 0; y < size; y++) {
        // マスの色を設定
        ctx.fillStyle = "#080";
        // マスを描画
        ctx.fillRect(x * grid_size + wd, y * grid_size + wd, grid_size - wd * 2, grid_size - wd * 2);
        // マスの状態に応じて、石を描画
        if (gameStatus.board[y][x] == 1) {
          ctx.fillStyle = "#000";
        } else if (gameStatus.board[y][x] == -1) {
          ctx.fillStyle = "#fff";
        } else {
          // マスの状態が0の場合は、描画を終了
          continue;
        }
        // 円を描画
        ctx.beginPath();
        ctx.arc(x * grid_size + grid_size / 2, y * grid_size + grid_size / 2, grid_size / 2 - wd * 2, 0, Math.PI * 2, false);
        ctx.fill();
      }
    }
  } else {
    // Canvasが利用不可能の場合
    console.log("Canvas is not available.");
  }
  // 石の数を更新
  updateStoneCounts(gameStatus);
}
// CPUによる盤面の評価を行う関数
function cpu(gameStatus) {
  // 評価に必要なデータをまとめる
  const query = {
    board: gameStatus.board,
    player: -1
  };
  // APIのURL
  const url = '/cpu';
  // HTTPメソッド
  const method = "POST";
  // リクエストボディ
  const body = JSON.stringify(query);
  // リクエストヘッダー
  const headers = {};
  // APIにリクエストを送信
  fetch(url, {
    method,
    headers,
    body
  }).then(function (data) {
    // レスポンスデータをJSON形式に変換
    return data.json();
  }).then(function (json) {
    // 評価結果をゲーム状態に格納
    gameStatus.board = json.board;
    // Canvasを更新
    draw(canvas, gameStatus);
  });
}
// 石の数をカウントして画面に表示する関数
function updateStoneCounts(gameStatus) {
  // 黒石の数
  let blackCount = 0;
  // 白石の数
  let whiteCount = 0;
  // 各マスの状態を確認
  for (let x = 0; x < size; x++) {
    for (let y = 0; y < size; y++) {
      // マスが黒石の場合
      if (gameStatus.board[y][x] == 1) {
        blackCount++;
      } else if (gameStatus.board[y][x] == -1) {
        // マスが白石の場合
        whiteCount++;
      }
    }
  }
  // 画面に石の数を表示
  document.getElementById('black-count').innerText = blackCount;
  document.getElementById('white-count').innerText = whiteCount;
}