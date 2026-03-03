# WAN Prompt Generator

Stable Diffusionで生成された画像とそのプロンプトから、Vision-Language Model（VLM）を使ってWAN 2.2用の動画生成プロンプトを作成するアプリケーション。GradioによるWebUI、FastAPIによるREST API、Electronによるデスクトップアプリの3モードに対応。

## 主要機能

- PNG画像からメタデータ（プロンプト情報）を自動抽出（A1111 WebUI / ComfyUI形式対応）
- SD以外の一般画像にも対応（VLMが画像を直接分析）
- VLMが画像とSDプロンプトを分析し、WAN 2.2用動画プロンプトを生成
- 出力言語選択（English / 日本語）
- スタイルプリセット（なし / 穏やか / ダイナミック / シネマティック / アニメ風）
- 出力項目の個別選択（シーン / アクション / カメラ / スタイル / WANプロンプト）
- 追加指示によるカスタマイズ
- セッション保存・読み込み（JSON形式）
- 生成プロンプトのTXTファイル保存
- 設定の自動保存（言語、スタイル、出力項目、推論設定、使用モデル）
- Hugging Faceからのモデル自動ダウンロード
- 複数モデルの切り替え機能
- FastAPI REST APIモード（Electronデスクトップアプリ連携）

## 技術スタック

- **Python 3.10+**
- **Gradio 6.0+** - ユーザーインターフェース
- **FastAPI / Uvicorn** - REST APIサーバー（APIモード）
- **Transformers** - VLMモデル推論
- **Pillow** - 画像処理
- **PyTorch** - 深層学習フレームワーク
- **Electron** - デスクトップアプリラッパー（オプション）

## 対応モデル

- **Qwen2.5-VL系** (3B, 7B)
- **Qwen3-VL系** (4B, 8B) - 最新世代
- **Huihui-AI abliterated** (試験用、フィルタ除去版)

## プロジェクト構成

```
sd-to-wan-prompt/
├── app.py                      # エントリーポイント（--mode gradio/api/help）
├── start.bat                   # Windows起動スクリプト（Conda環境対応）
├── convert_txt_to_json.bat     # TXT→JSONセッション変換バッチ
├── requirements.txt
├── config/
│   ├── settings.yaml           # アプリケーション設定
│   └── model_presets.yaml      # VLMモデルプリセット定義
├── src/
│   ├── core/
│   │   ├── image_parser.py     # PNGメタデータ抽出
│   │   ├── model_manager.py    # VLMダウンロード・管理
│   │   └── vlm_interface.py    # VLM推論・WAN用プロンプト生成
│   ├── ui/
│   │   └── gradio_app.py       # Gradio UIメインモジュール
│   ├── api/
│   │   ├── server.py           # FastAPI REST APIサーバー
│   │   └── service.py          # ビジネスロジックサービス
│   └── utils/
│       └── config_loader.py    # YAML設定読み込み
├── scripts/
│   ├── setup.py                # 初期セットアップ
│   ├── test_model.py           # モデル検証テスト
│   ├── check_cuda.py           # CUDA環境確認
│   ├── check_gpu.py            # GPU状態確認
│   └── convert_txt_to_json.py  # TXTをセッションJSONに変換
├── data/
│   ├── sd_outputs/             # SD生成画像の保存先
│   ├── database/               # メタデータDB
│   └── downloads/              # ダウンロードファイル
├── models/                     # ローカルVLMモデル保存先
└── desktop/
    └── electron/               # Electronデスクトップアプリ
```

## セットアップ方法

### 1. リポジトリのクローン

```bash
git clone <repository-url>
cd sd-to-wan-prompt
```

### 2. Conda環境の作成（推奨）

```bash
conda create -n wan-prompt python=3.10 -y
conda activate wan-prompt
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
# Gradio WebUI（デフォルト）
conda activate wan-prompt
python app.py

# Windowsの場合はstart.batも使用可能
start.bat gradio
```

ブラウザで `http://localhost:7860` を開く

### 基本的な使い方

1. **モデル管理タブ**
   - プリセットから使用したいモデルを選択
   - ダウンロードボタンをクリック
   - ダウンロード完了後、モデルをロード

2. **プロンプト生成タブ**
   - SD画像をアップロード（PNGメタデータが自動抽出される）
   - ※SD以外の一般画像も使用可能
   - 出力言語を選択（English / 日本語）
   - スタイルプリセットを選択（任意）
   - 出力項目を選択（不要な項目のチェックを外すと出力が短くなる）
   - 追加指示がある場合は入力欄に記載
   - 「WANプロンプト生成」ボタンをクリック
   - 生成されたプロンプトをコピーしてWAN 2.2で使用
   - セッション保存ボタンでJSON形式に保存可能
   - ※設定は自動保存され、次回起動時に復元される

3. **設定タブ**
   - Temperature、Max Tokens等のパラメータを調整

## スタイルプリセット

| スタイル | 説明 |
|---------|------|
| **なし** | スタイル指定なし（VLMが画像から自然に判断） |
| **穏やか** | ゆっくりとした動き、スムーズなカメラパン、呼吸や髪の揺れなど繊細な表現 |
| **ダイナミック** | エネルギッシュな動き、トラッキングショット、明確なモーション |
| **シネマティック** | ドラマチックなカメラワーク、感情的な表現、雰囲気のあるライティング |
| **アニメ風** | 誇張された表現、風エフェクト、日本アニメ風のダイナミックなポーズ |

## 出力項目

| 項目 | 説明 |
|------|------|
| **シーン** | 画像に基づいた視覚的なシーンの説明 |
| **アクション** | 追加する動き・モーションの説明 |
| **カメラ** | カメラの動き（静止、パン、ズーム、ドリー等） |
| **スタイル** | 視覚的なスタイルと雰囲気 |
| **WANプロンプト** | 上記を組み合わせた最終プロンプト |

不要な項目のチェックを外すことで、出力を短くできます。

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

**試験用（フィルタ除去版）**
- **huihui-qwen3-vl-4b-abliterated**: VRAM ~6GB
- **huihui-qwen3-vl-8b-abliterated**: VRAM ~10-12GB

## セッション保存・読み込み

生成したプロンプトをJSONセッションファイルとして保存することで、後から結果を復元できます。

```json
{
  "image_path": "/path/to/image.png",
  "prompt": "...",
  "additional_instruction": "...",
  "wan_prompt": "..."
}
```

### TXTファイルからJSONへの変換

既存のTXTプロンプトファイルをセッションJSON形式に一括変換できます。

```bash
# Pythonスクリプト（直接実行）
python scripts/convert_txt_to_json.py [ターゲットディレクトリ] --recursive --overwrite

# Windowsバッチスクリプト
convert_txt_to_json.bat [ターゲットディレクトリ] [--overwrite]
```

TXTファイルの期待フォーマット:
```
=== Original Prompt ===
[元のSDプロンプト]
=== Additional Instruction ===
[追加指示]
=== Generated WAN Prompt ===
[生成されたWANプロンプト]
```

## 設定ファイル

### config/settings.yaml

```yaml
app:
  name: "WAN Prompt Generator"
  version: "0.1.0"

model:
  default: "qwen3-vl-4b"
  device: "cuda"
  dtype: "float16"    # float16, bfloat16, float32

inference:
  temperature: 0.7
  max_tokens: 1024
  top_p: 0.9

wan_prompt:
  default_style: "cinematic"

ui:
  theme: "soft"
  share: false
  server_port: 7860
```

### config/model_presets.yaml

モデルプリセットの定義（Hugging FaceリポジトリIDとローカル保存名）。新しいモデルを追加する際はこのファイルを編集してください。

## Electronデスクトップモード

`desktop/electron/` 以下にElectronデスクトップアプリが含まれています。

### バックエンドの起動

```bash
python app.py --mode api --host 127.0.0.1 --port 7861
# または
start.bat api
```

### Electronアプリの起動

```bash
cd desktop/electron
npm install
npm start
# または
start.bat electron
```

### 環境変数

| 変数名 | デフォルト | 説明 |
|--------|-----------|------|
| `PYTHON_EXECUTABLE` | （システムPython） | Pythonインタープリタのパス |
| `WAN_API_HOST` | `127.0.0.1` | バックエンドホスト |
| `WAN_API_PORT` | `7861` | バックエンドポート |

> **Note**: TauriをElectronの代わりに使用したい場合は、同じバックエンド（`app.py --mode api`）を再利用し、フロントエンドシェルのみ置き換えてください。

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

### CUDA / GPU環境の確認

```bash
python scripts/check_cuda.py   # CUDA環境確認
python scripts/check_gpu.py    # GPU状態確認
```

## システム要件

### 最小要件

- Python 3.10以上
- RAM: 16GB以上
- ストレージ: 20GB以上（モデル保存用）

### 推奨要件

- GPU: NVIDIA GPU（VRAM 12GB以上）
- RAM: 32GB以上
- ストレージ: 50GB以上

## ライセンス

MIT License
