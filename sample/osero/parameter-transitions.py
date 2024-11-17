import json
import os
import numpy as np

# 最終世代と個体ID
final_generation = 64206
target_individual_id = 0

# 分析する世代の範囲と間隔
generations = list(range(1000, final_generation + 1, 100))

# 可視化する重みの数
max_weights_to_plot = 100  # 必要に応じて調整

# 重みの位置をランダムに選択するかどうか
sample_weights = True

# 重みの値を格納する辞書
wi_data = {}
wo_data = {}

# 最終世代のネットワークを読み込み、重みの形状を取得
final_gen_dir = f"./train/{final_generation}"
final_file_path = os.path.join(final_gen_dir, f"{target_individual_id}.json")
with open(final_file_path, "r") as f:
    data = json.load(f)
    wi_shape = (len(data["wi"]), len(data["wi"][0]))
    wo_shape = (len(data["wo"]), len(data["wo"][0]))

# 追跡する重みの位置を選択
if sample_weights:
    num_wi_weights = wi_shape[0] * wi_shape[1]
    num_wo_weights = wo_shape[0] * wo_shape[1]
    wi_indices = np.random.choice(num_wi_weights, max_weights_to_plot, replace=False)
    wo_indices = np.random.choice(num_wo_weights, max_weights_to_plot, replace=False)
else:
    wi_indices = np.arange(max_weights_to_plot)
    wo_indices = np.arange(max_weights_to_plot)

# インデックスを行と列に変換
wi_positions = [(idx // wi_shape[1], idx % wi_shape[1]) for idx in wi_indices]
wo_positions = [(idx // wo_shape[1], idx % wo_shape[1]) for idx in wo_indices]

# 各世代の重みを収集
for generation in generations:
    generation_dir = f"./train/{generation}"
    print(f"Processing generation {generation}...")

    # 対象個体のニューラルネットワークを読み込み
    file_path = os.path.join(generation_dir, f"{target_individual_id}.json")
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            data = json.load(f)

        # 選択した位置の重みの値を記録
        for (row_idx, col_idx) in wi_positions:
            key = (row_idx, col_idx)
            value = data["wi"][row_idx][col_idx]
            if key not in wi_data:
                wi_data[key] = []
            wi_data[key].append((generation, value))

        for (row_idx, col_idx) in wo_positions:
            key = (row_idx, col_idx)
            value = data["wo"][row_idx][col_idx]
            if key not in wo_data:
                wo_data[key] = []
            wo_data[key].append((generation, value))

# Chart.jsで可視化するためのデータを準備
wi_data_json = {
    "labels": generations,
    "datasets": []
}
wo_data_json = {
    "labels": generations,
    "datasets": []
}

def random_color():
    return f"rgba({np.random.randint(0,256)},{np.random.randint(0,256)},{np.random.randint(0,256)},1)"

# wiのデータセットを準備
for key, values in wi_data.items():
    row_idx, col_idx = key
    value_dict = dict(values)  # 世代から値へのマッピング
    data_values = [value_dict.get(gen, None) for gen in generations]
    wi_data_json["datasets"].append({
        "label": f"wi[{row_idx}][{col_idx}]",
        "data": data_values,
        "fill": False,
        "borderColor": random_color(),
    })

# woのデータセットを準備
for key, values in wo_data.items():
    row_idx, col_idx = key
    value_dict = dict(values)
    data_values = [value_dict.get(gen, None) for gen in generations]
    wo_data_json["datasets"].append({
        "label": f"wo[{row_idx}][{col_idx}]",
        "data": data_values,
        "fill": False,
        "borderColor": random_color(),
    })

# JSONファイルに保存
with open("wi_data_b.json", "w") as f:
    json.dump(wi_data_json, f)
with open("wo_data_b.json", "w") as f:
    json.dump(wo_data_json, f)

# HTMLファイルを生成
html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Neural Network Weight Analysis</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
    <div style="width: 80%; margin: 0 auto;">
        <h1>Input-Hidden Weights</h1>
        <canvas id="wiChart"></canvas>
        <h1>Hidden-Output Weights</h1>
        <canvas id="woChart"></canvas>
    </div>

    <script>
        const wiData = {json.dumps(wi_data_json)};
        const woData = {json.dumps(wo_data_json)};

        const wiCtx = document.getElementById('wiChart').getContext('2d');
        new Chart(wiCtx, {{
            type: 'line',
            data: wiData,
            options: {{
                scales: {{
                    y: {{
                        beginAtZero: true
                    }}
                }}
            }}
        }});

        const woCtx = document.getElementById('woChart').getContext('2d');
        new Chart(woCtx, {{
            type: 'line',
            data: woData,
            options: {{
                scales: {{
                    y: {{
                        beginAtZero: true
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>
"""

# HTMLファイルを保存
with open("weight_analysis_b.html", "w") as f:
    f.write(html_content)
