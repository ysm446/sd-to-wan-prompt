# WAN Prompt Generator

Stable Diffusionで生成された画像とそのプロンプトから、Vision-Language Model（VLM）を使ってWAN 2.2用の動画生成プロンプトを作成するWebアプリケーション。FastAPIによるREST API、Electronによるデスクトップアプリの2モードに対応。

## 技術スタック

- **Python 3.10+**
- **FastAPI / Uvicorn** - REST APIサーバー（Electronデスクトップ連携用）
- **PyTorch 2.0+** - 深層学習フレームワーク
- **Transformers 4.40+** - VLMモデル推論
- **Qwen VL** - 主要なVLMモデル（Qwen2.5-VL, Qwen3-VL対応）
- **Electron** - デスクトップアプリラッパー（desktop/electron/）

## ディレクトリ構造

```
sd-to-wan-prompt/
├── app.py                      # エントリーポイント（FastAPI APIサーバー起動）
├── start.bat                   # Windows起動スクリプト（Conda環境対応）
├── convert_txt_to_json.bat     # TXT→JSONセッション変換バッチ
├── requirements.txt
├── config/
│   ├── settings.yaml           # アプリケーション設定
│   └── model_presets.yaml      # VLMモデルプリセット定義
├── src/
│   ├── core/
│   │   ├── image_parser.py     # PNGメタデータ抽出（A1111/ComfyUI形式対応）
│   │   ├── model_manager.py    # VLMダウンロード・管理
│   │   └── vlm_interface.py    # Transformers形式VLM推論（generate_wan_prompt_stream含む）
│   ├── api/
│   │   ├── server.py           # FastAPI REST APIサーバー
│   │   └── service.py          # ビジネスロジックサービス（PromptService）
│   └── utils/
│       └── config_loader.py    # YAML設定読み込み
├── scripts/
│   ├── setup.py                # 初期セットアップ（ディレクトリ作成・依存確認）
│   ├── check_cuda.py           # CUDA環境確認
│   ├── check_gpu.py            # GPU状態確認
│   ├── test_model.py           # モデル検証テスト
│   └── convert_txt_to_json.py  # TXTファイルをセッションJSONに変換
├── data/
│   ├── sd_outputs/             # SD生成画像保存先
│   ├── database/               # メタデータDB
│   └── downloads/              # ダウンロードファイル
├── models/                     # ローカルVLMモデル保存先（git管理外）
└── desktop/
    └── electron/               # Electronデスクトップアプリ
```

## 主要コンポーネント

| モジュール | 役割 |
|-----------|------|
| `app.py` | ConfigLoaderで設定を読み込み、FastAPI APIサーバーを起動 |
| `image_parser.py` | PNG画像からプロンプト、ネガティブプロンプト、設定パラメータを抽出（A1111/ComfyUI対応） |
| `model_manager.py` | Hugging Faceからのモデルダウンロード、ローカルモデル管理・検証 |
| `vlm_interface.py` | Transformers形式VLMのモデルロード・推論、WAN用プロンプト生成（ストリーミング対応） |
| `server.py` | FastAPI REST APIサーバー、CORS対応、NDJSONストリーミングエンドポイント |
| `service.py` | PromptServiceクラス、スレッドセーフな状態管理、セッション保存・読み込み |
| `config_loader.py` | YAML設定読み込み・保存 |

## 主要機能

- **SD画像からメタデータ抽出**: PNG画像からプロンプト情報を自動取得（A1111 WebUI / ComfyUI形式対応）
- **一般画像対応**: SD以外の画像もVLMで分析してプロンプト生成可能
- **WAN 2.2プロンプト生成**: VLMが画像とSDプロンプトを分析し、動画生成用プロンプトを作成
- **出力言語選択**: English / 日本語
- **スタイルプリセット**: なし / 穏やか / ダイナミック / シネマティック / アニメ風
- **出力項目選択**: シーン / アクション / カメラ / スタイル / WANプロンプトを個別に選択可能
- **追加指示対応**: ユーザーの追加指示を反映したプロンプト生成
- **ストリーミング出力**: リアルタイムでプロンプト生成結果を表示
- **セッション保存・読み込み**: 生成結果をJSONで保存し、次回復元可能
- **テキスト保存**: 生成プロンプトをTXTファイルに保存
- **TXT→JSON変換**: 既存のTXTファイルをセッションJSON形式に一括変換
- **設定キャッシュ**: 言語、スタイル、出力項目、推論設定、使用モデルを自動保存（settings_cache.json）
- **FastAPI APIモード**: Electronデスクトップアプリとの連携用REST API
- **デバイス選択**: CUDA / CPU、dtype（float16 / bfloat16 / float32）選択可能
- **自動アンロード**: 生成後にモデルをVRAMから解放するオプション

## 起動方法

```bash
# FastAPI APIサーバー（Electronデスクトップアプリ用）
python app.py --host 127.0.0.1 --port 7861

# Windowsバッチスクリプト（Conda環境対応）
start.bat api
start.bat electron
```

## API エンドポイント

| メソッド | パス | 説明 |
|---------|------|------|
| GET | `/health` | ヘルスチェック |
| GET | `/models` | ローカルモデル一覧 |
| POST | `/models/load` | モデルロード |
| POST | `/models/download` | モデルダウンロード |
| POST | `/models/unload` | モデルアンロード |
| POST | `/image/parse` | 画像メタデータ抽出 |
| POST | `/generate` | プロンプト生成（同期） |
| POST | `/generate/stream` | プロンプト生成（NDJSONストリーミング） |
| GET/POST | `/settings` | 設定取得・更新 |
| POST | `/session/save` | セッションJSON保存 |
| POST | `/session/load` | セッションJSON読み込み |

## 設定ファイル

- `config/settings.yaml`: モデル設定、推論パラメータ、WANプロンプト設定など
- `config/model_presets.yaml`: VLMモデルのHugging FaceリポジトリIDとローカル保存名

## コーディング規約

- 言語: 日本語コメント可、変数名・関数名は英語
- 型ヒント: 必須
- docstring: Google スタイル
- インデント: スペース4つ
- 最大行長: 100文字

## 開発時の注意

- VLMモデルは大容量（数GB〜数十GB）のため、`models/` ディレクトリはgit管理対象外
- GPU（CUDA）推奨、CPU動作も可能だが低速
- 新しいモデルプリセット追加時は `config/model_presets.yaml` を編集
- `service.py`の`PromptService`はRLockでスレッドセーフに実装されている
- セッションJSONはUTF-8 / UTF-8 BOM / UTF-16のマルチエンコーディング対応
- Electronフロントエンドはfetch経由でFastAPI（ポート7861）と通信する
