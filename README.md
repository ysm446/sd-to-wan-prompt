# WAN Prompt Generator

Stable Diffusionで生成された画像とそのプロンプトから、Vision-Language Model（VLM）を使ってWAN 2.2用の動画生成プロンプトを作成するアプリケーション

## 主要機能

- PNG画像からメタデータ（プロンプト情報）を自動抽出
- VLMが画像とSDプロンプトを分析し、WAN 2.2用動画プロンプトを生成
- スタイルプリセット（穏やか / ダイナミック / シネマティック / アニメ風）
- 追加指示によるカスタマイズ
- Hugging Faceからのモデル自動ダウンロード
- 複数モデルの切り替え機能

## 技術スタック

- **Python 3.10+**
- **Gradio 6.0+** - ユーザーインターフェース
- **Transformers** - VLMモデル推論
- **Pillow** - 画像処理
- **PyTorch** - 深層学習フレームワーク

## 対応モデル

- **Qwen2.5-VL系** (3B, 7B)
- **Qwen3-VL系** (4B, 8B) - 最新世代
- **Huihui-AI abliterated** (試験用)

## プロジェクト構成

```
sd-to-wan-prompt/
├── data/
│   ├── sd_outputs/          # SD生成画像の保存先
│   ├── database/            # メタデータDB
│   └── downloads/           # ダウンロードした画像等
├── models/                  # ローカルVLMモデル保存先
├── src/
│   ├── core/                # コアモジュール
│   │   ├── image_parser.py
│   │   ├── model_manager.py
│   │   ├── vlm_interface.py
│   │   └── vlm_interface_gguf.py  # GGUF形式対応
│   ├── ui/                  # UIモジュール
│   │   └── gradio_app.py
│   └── utils/               # ユーティリティ
│       └── config_loader.py
├── config/                  # 設定ファイル
│   ├── settings.yaml
│   └── model_presets.yaml
├── scripts/                 # スクリプト
│   ├── setup.py
│   ├── test_model.py
│   ├── check_cuda.py
│   └── check_gpu.py
├── requirements.txt
├── start.bat                # Windows起動スクリプト
└── app.py                   # エントリーポイント
```

## セットアップ方法

### 1. リポジトリのクローン

```bash
git clone <repository-url>
cd sd-to-wan-prompt
```

### 2. Conda環境の作成（推奨）

```bash
# Conda環境を作成
conda create -n sd-to-wan-prompt python=3.10 -y

# 環境をアクティベート
conda activate sd-to-wan-prompt
```

### 3. 依存関係のインストール

```bash
pip install -r requirements.txt
```

### 4. 初期セットアップ

```bash
python scripts/setup.py
```

## 使用方法

### アプリケーションの起動

```bash
# Conda環境をアクティベート
conda activate sd-to-wan-prompt

# アプリケーションを起動
python app.py
```

ブラウザで `http://localhost:7860` を開く

### 基本的な使い方

1. **モデル管理タブ**
   - プリセットから使用したいモデルを選択
   - ダウンロードボタンをクリック
   - ダウンロード完了後、モデルをロード

2. **プロンプト生成タブ**
   - SD画像をアップロード（PNGメタデータが自動抽出される）
   - スタイルプリセットを選択、または「WANプロンプト生成」ボタンをクリック
   - 追加指示がある場合は入力欄に記載
   - 生成されたプロンプトをコピーしてWAN 2.2で使用

3. **設定タブ**
   - Temperature、Max Tokens等のパラメータを調整

## スタイルプリセット

| スタイル | 説明 |
|---------|------|
| **穏やか** | ゆっくりとした動き、スムーズなカメラパン、呼吸や髪の揺れなど繊細な表現 |
| **ダイナミック** | エネルギッシュな動き、トラッキングショット、明確なモーション |
| **シネマティック** | ドラマチックなカメラワーク、感情的な表現、雰囲気のあるライティング |
| **アニメ風** | 誇張された表現、風エフェクト、日本アニメ風のダイナミックなポーズ |

## モデルのダウンロード

### UI上でダウンロード

1. 「モデル管理」タブを開く
2. プリセットを選択（例: qwen3-vl-4b）
3. 「ダウンロード開始」ボタンをクリック

### プリセット一覧

**Qwen2.5-VL シリーズ**
- **qwen2.5-vl-3b**: 軽量版（VRAM ~5GB）- 低メモリ環境向け
- **qwen2.5-vl-7b**: 高品質（VRAM ~8-10GB）

**Qwen3-VL シリーズ（最新世代）**
- **qwen3-vl-4b**: バランス型（VRAM ~6GB）- 推奨
- **qwen3-vl-8b**: 高性能（VRAM ~10-12GB）

**試験用**
- **huihui-qwen3-vl-4b-abliterated**: フィルタ除去版 4B
- **huihui-qwen3-vl-8b-abliterated**: フィルタ除去版 8B

## 設定ファイル

### config/settings.yaml

アプリケーションの基本設定を管理

```yaml
app:
  name: "WAN Prompt Generator"
  version: "0.1.0"

paths:
  image_folder: "./data/sd_outputs"
  models_dir: "./models"

model:
  default: "qwen3-vl-4b"
  device: "cuda"
  dtype: "float16"

inference:
  temperature: 0.7
  max_tokens: 1024
  top_p: 0.9

wan_prompt:
  default_style: "cinematic"
  style_presets:
    calm:
      description: "穏やかな動き"
    dynamic:
      description: "ダイナミック"
    cinematic:
      description: "シネマティック"
    anime:
      description: "アニメ風"

ui:
  theme: "soft"
  share: false
  server_port: 7860
```

### config/model_presets.yaml

モデルプリセットの定義

## トラブルシューティング

### モデルのロードに失敗する

- GPUメモリが不足している可能性があります
- より小さいモデル（3B/4Bモデル）を試してください
- `config/settings.yaml`の`device`を`cpu`に変更してください（遅くなります）

### 画像にメタデータがない

- Stable Diffusionで生成された画像でない可能性があります
- PNG形式で保存されているか確認してください
- A1111 WebUIやComfyUIで生成された画像は対応しています

### ダウンロードが進まない

- インターネット接続を確認してください
- Hugging Faceへのアクセスが制限されていないか確認してください

## システム要件

### 最小要件

- Python 3.10以上
- RAM: 16GB以上
- ストレージ: 20GB以上（モデル保存用）

### 推奨要件

- Python 3.10以上
- GPU: NVIDIA GPU（VRAM 12GB以上）
- RAM: 32GB以上
- ストレージ: 50GB以上

## ライセンス

MIT License

## 貢献

バグ報告や機能提案は、GitHubのIssuesでお願いします。

## サポート

問題が発生した場合は、以下を確認してください：

1. 依存関係が正しくインストールされているか
2. Conda環境が正しくアクティベートされているか
3. モデルファイルが正しくダウンロードされているか

詳細なログを確認する場合は、ターミナルの出力を確認してください。
