# WAN Prompt Generator

Stable Diffusionで生成された画像とそのプロンプトから、Vision-Language Model（VLM）を使ってWAN 2.2用の動画生成プロンプトを作成するWebアプリケーション。

## 技術スタック

- **Python 3.10+**
- **Gradio 6.0+** - WebUIフレームワーク
- **PyTorch 2.0+** - 深層学習フレームワーク
- **Transformers 4.40+** - VLMモデル推論
- **Qwen VL** - 主要なVLMモデル（Qwen2-VL, Qwen2.5-VL, Qwen3-VL対応）

## ディレクトリ構造

```
sd-to-wan-prompt/
├── app.py                      # エントリーポイント
├── config/
│   ├── settings.yaml           # アプリケーション設定
│   └── model_presets.yaml      # VLMモデルプリセット定義
├── src/
│   ├── core/
│   │   ├── image_parser.py     # PNGメタデータ抽出
│   │   ├── model_manager.py    # VLMダウンロード・管理
│   │   └── vlm_interface.py    # Transformers形式VLM推論（generate_wan_prompt_stream含む）
│   ├── ui/
│   │   └── gradio_app.py       # Gradio UIメインモジュール
│   └── utils/
│       └── config_loader.py    # YAML設定読み込み
├── data/
│   ├── sd_outputs/             # SD生成画像保存先
│   └── database/               # メタデータDB
├── models/                     # ローカルVLMモデル保存先
└── scripts/                    # セットアップ・テストスクリプト
```

## 主要コンポーネント

| モジュール | 役割 |
|-----------|------|
| `app.py` | ConfigLoaderで設定を読み込み、PromptAnalyzerUIを起動 |
| `image_parser.py` | PNG画像からプロンプト、ネガティブプロンプト、設定パラメータを抽出 |
| `model_manager.py` | Hugging Faceからのモデルダウンロード、ローカルモデル管理 |
| `vlm_interface.py` | Transformers形式VLMのモデルロード・推論、WAN用プロンプト生成 |
| `gradio_app.py` | 3タブ構成のWebUI（プロンプト生成、モデル管理、設定） |

## 主要機能

- **SD画像からメタデータ抽出**: PNG画像からプロンプト情報を自動取得
- **一般画像対応**: SD以外の画像もVLMで分析してプロンプト生成可能
- **WAN 2.2プロンプト生成**: VLMが画像とSDプロンプトを分析し、動画生成用プロンプトを作成
- **出力言語選択**: English / 日本語
- **スタイルプリセット**: なし / 穏やか / ダイナミック / シネマティック / アニメ風
- **出力項目選択**: シーン / アクション / カメラ / スタイル / WANプロンプトを個別に選択可能
- **追加指示対応**: ユーザーの追加指示を反映したプロンプト生成
- **ストリーミング出力**: リアルタイムでプロンプト生成結果を表示
- **設定キャッシュ**: 言語、スタイル、出力項目、推論設定、使用モデルを自動保存

## コーディング規約

- 言語: 日本語コメント可、変数名・関数名は英語
- 型ヒント: 必須
- docstring: Google スタイル
- インデント: スペース4つ
- 最大行長: 100文字

## 起動方法

```bash
python app.py
# ブラウザで http://localhost:7860 にアクセス
```

## 設定ファイル

- `config/settings.yaml`: モデル設定、推論パラメータ、UIテーマ、WANプロンプト設定など
- `config/model_presets.yaml`: VLMモデルのHugging FaceリポジトリIDとローカル保存名

## 開発時の注意

- VLMモデルは大容量（数GB〜数十GB）のため、`models/` ディレクトリはgit管理対象外
- GPU（CUDA）推奨、CPU動作も可能だが低速
- 新しいモデルプリセット追加時は `config/model_presets.yaml` を編集
