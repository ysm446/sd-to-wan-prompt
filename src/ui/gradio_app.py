"""
Gradio UI実装
"""
import gradio as gr
import json
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from src.core.image_parser import ImageParser
from src.core.model_manager import ModelManager
from src.core.vlm_interface import VLMInterface
from src.utils.config_loader import ConfigLoader


class PromptAnalyzerUI:
    """メインUIクラス"""

    def __init__(self, config: Dict):
        """
        Args:
            config: settings.yamlから読み込んだ設定
        """
        self.config = config
        self.model_manager = ModelManager(config['paths']['models_dir'])
        self.current_vlm: Optional[VLMInterface] = None
        self.current_image_path: Optional[str] = None
        self.current_metadata: Optional[Dict] = None
        self.selected_model_path: Optional[str] = None  # 選択されているモデルのパス
        self.last_model_cache_file = Path(".last_model_cache.json")

        # モデルプリセットを読み込み
        config_loader = ConfigLoader()
        self.model_presets = config_loader.load_model_presets()

    def create_interface(self) -> gr.Blocks:
        """
        Gradio UIを構築

        UI構成:
        - タブ1: 画像分析
        - タブ2: モデル管理
        - タブ3: 設定
        """
        # カスタムCSS（フォント変更）
        custom_css = """
        * {
            font-family: "Segoe UI", "Yu Gothic", "Meiryo", Arial, sans-serif !important;
        }
        """

        # キャッシュから推論設定を読み込み（なければconfigのデフォルト値を使用）
        cached_settings = self.load_inference_settings()
        initial_temperature = cached_settings.get('temperature', self.config['inference']['temperature'])
        initial_max_tokens = cached_settings.get('max_tokens', self.config['inference']['max_tokens'])
        initial_top_p = cached_settings.get('top_p', self.config['inference']['top_p'])

        with gr.Blocks(title="WAN Prompt Generator", css=custom_css) as interface:
            gr.Markdown("# WAN Prompt Generator")
            gr.Markdown("SD画像からWAN 2.2用の動画プロンプトを生成します")

            with gr.Tabs():
                # タブ1: プロンプト生成
                with gr.Tab("プロンプト生成"):
                    with gr.Row():
                        # 左側: 画像表示
                        with gr.Column(scale=1):
                            image_display = gr.Image(
                                label="SD画像をアップロード",
                                type="filepath",
                                sources=["upload"],
                                height=400
                            )

                            # プロンプト情報表示
                            with gr.Accordion("元のSD情報", open=True):
                                prompt_display = gr.Textbox(
                                    label="Prompt",
                                    lines=3,
                                    interactive=False
                                )
                                negative_prompt_display = gr.Textbox(
                                    label="Negative Prompt",
                                    lines=2,
                                    interactive=False
                                )
                                settings_display = gr.Code(
                                    label="Settings",
                                    language="json",
                                    interactive=False,
                                    lines=5
                                )

                        # 右側: プロンプト生成
                        with gr.Column(scale=1):
                            # 生成結果表示エリア
                            output_textbox = gr.Textbox(
                                label="生成されたWANプロンプト",
                                lines=18,
                                max_lines=25,
                                interactive=True  # コピーできるようにinteractiveに
                            )
                            context_info = gr.Markdown(
                                value="<small style='color: gray;'>--</small>",
                                elem_id="context-info"
                            )

                            # 追加指示入力欄
                            additional_input = gr.Textbox(
                                label="追加指示（オプション）",
                                placeholder="例: カメラをズームアウトさせて、髪をなびかせてください",
                                lines=2
                            )

                            # スタイルプリセットボタン
                            gr.Markdown("### スタイルプリセット")
                            with gr.Row():
                                style_calm = gr.Button("穏やか", size="sm")
                                style_dynamic = gr.Button("ダイナミック", size="sm")
                            with gr.Row():
                                style_cinematic = gr.Button("シネマティック", size="sm")
                                style_anime = gr.Button("アニメ風", size="sm")

                            # 現在選択中のスタイル
                            current_style = gr.State(value="cinematic")

                            generate_btn = gr.Button("WANプロンプト生成", variant="primary", size="lg")

                            # モデル選択
                            with gr.Accordion("モデル設定", open=False):
                                model_dropdown = gr.Dropdown(
                                    label="使用するモデル",
                                    choices=[],
                                    value=None,
                                    interactive=True
                                )
                                with gr.Row():
                                    load_model_btn = gr.Button("モデルをロード")
                                    unload_model_btn = gr.Button("モデルをクリア")
                                model_status = gr.Textbox(
                                    label="モデル状態",
                                    value="モデル未ロード",
                                    interactive=False
                                )

                # タブ2: モデル管理
                with gr.Tab("モデル管理"):
                    gr.Markdown("### ローカルモデル")
                    refresh_models_btn = gr.Button("モデル一覧を更新")
                    local_models_display = gr.DataFrame(
                        headers=["モデル名", "パス", "サイズ"],
                        datatype=["str", "str", "str"],
                        label="保存済みモデル"
                    )

                    gr.Markdown("### モデルをダウンロード")
                    with gr.Row():
                        with gr.Column():
                            preset_dropdown = gr.Dropdown(
                                label="プリセット",
                                choices=list(self.model_presets.keys()),
                                value=None
                            )
                            repo_id_input = gr.Textbox(
                                label="Repository ID",
                                placeholder="Qwen/Qwen2-VL-7B-Instruct",
                                value=""
                            )
                            local_name_input = gr.Textbox(
                                label="ローカル保存名",
                                placeholder="qwen2-vl-7b",
                                value=""
                            )
                            download_btn = gr.Button("ダウンロード開始", variant="primary")

                        with gr.Column():
                            preset_info = gr.Markdown("プリセットを選択すると詳細が表示されます")
                            download_status = gr.Textbox(
                                label="ダウンロード状態",
                                value="",
                                interactive=False,
                                lines=5
                            )

                # タブ3: 設定
                with gr.Tab("設定"):
                    with gr.Row():
                        with gr.Column():
                            temperature_slider = gr.Slider(
                                label="Temperature",
                                info="ランダム性を制御（低い値=正確、高い値=創造的）。画像分析では0.1～0.3を推奨",
                                minimum=0.0,
                                maximum=2.0,
                                value=initial_temperature,
                                step=0.1
                            )
                            max_tokens_slider = gr.Slider(
                                label="Max Tokens",
                                info="生成する最大トークン数（文章の長さ）",
                                minimum=64,
                                maximum=2048,
                                value=initial_max_tokens,
                                step=64
                            )
                            top_p_slider = gr.Slider(
                                label="Top P",
                                info="語彙の多様性を制御。0.9前後を推奨",
                                minimum=0.0,
                                maximum=1.0,
                                value=initial_top_p,
                                step=0.05
                            )

            # イベントハンドラー
            # 画像アップロード（changeイベントで処理）
            image_display.change(
                fn=self.on_image_upload,
                inputs=[image_display],
                outputs=[prompt_display, negative_prompt_display, settings_display]
            )

            # WANプロンプト生成
            generate_btn.click(
                fn=self.generate_wan_prompt,
                inputs=[additional_input, current_style, temperature_slider, max_tokens_slider],
                outputs=[output_textbox, context_info, model_status]
            )

            # スタイルプリセットボタン - スタイルを設定して生成
            style_calm.click(
                fn=lambda: "calm",
                outputs=[current_style]
            ).then(
                fn=self.generate_wan_prompt,
                inputs=[additional_input, current_style, temperature_slider, max_tokens_slider],
                outputs=[output_textbox, context_info, model_status]
            )

            style_dynamic.click(
                fn=lambda: "dynamic",
                outputs=[current_style]
            ).then(
                fn=self.generate_wan_prompt,
                inputs=[additional_input, current_style, temperature_slider, max_tokens_slider],
                outputs=[output_textbox, context_info, model_status]
            )

            style_cinematic.click(
                fn=lambda: "cinematic",
                outputs=[current_style]
            ).then(
                fn=self.generate_wan_prompt,
                inputs=[additional_input, current_style, temperature_slider, max_tokens_slider],
                outputs=[output_textbox, context_info, model_status]
            )

            style_anime.click(
                fn=lambda: "anime",
                outputs=[current_style]
            ).then(
                fn=self.generate_wan_prompt,
                inputs=[additional_input, current_style, temperature_slider, max_tokens_slider],
                outputs=[output_textbox, context_info, model_status]
            )

            # モデル管理
            refresh_models_btn.click(
                fn=self.refresh_local_models,
                outputs=[local_models_display, model_dropdown]
            )

            # モデルドロップダウンの変更時に選択を保存
            def save_selected_model(path):
                self.selected_model_path = path
                self.save_last_model_path(path) if path else None

            model_dropdown.change(
                fn=save_selected_model,
                inputs=[model_dropdown],
                outputs=[]
            )

            load_model_btn.click(
                fn=self.load_vlm_model,
                inputs=[model_dropdown],
                outputs=[model_status, context_info]
            )

            unload_model_btn.click(
                fn=self.unload_vlm_model,
                outputs=[model_status, context_info]
            )

            preset_dropdown.change(
                fn=self.update_preset_info,
                inputs=[preset_dropdown],
                outputs=[preset_info, repo_id_input, local_name_input]
            )

            download_btn.click(
                fn=self.download_model,
                inputs=[repo_id_input, local_name_input],
                outputs=[download_status]
            )

            # 推論設定の変更時にキャッシュを更新
            def on_settings_change(temp, tokens, top_p):
                self.save_inference_settings(temp, tokens, top_p)

            temperature_slider.change(
                fn=on_settings_change,
                inputs=[temperature_slider, max_tokens_slider, top_p_slider],
                outputs=[]
            )
            max_tokens_slider.change(
                fn=on_settings_change,
                inputs=[temperature_slider, max_tokens_slider, top_p_slider],
                outputs=[]
            )
            top_p_slider.change(
                fn=on_settings_change,
                inputs=[temperature_slider, max_tokens_slider, top_p_slider],
                outputs=[]
            )

            # 初期ロード
            interface.load(
                fn=self.refresh_local_models,
                outputs=[local_models_display, model_dropdown]
            )

        return interface

    def on_image_upload(self, image_path: str) -> Tuple:
        """画像がアップロードされたときの処理"""
        try:
            # 画像パスがNoneまたは空の場合はクリア
            if not image_path:
                self.current_image_path = None
                self.current_metadata = None
                return "", "", "{}"

            # 画像パスを保存
            self.current_image_path = image_path

            # メタデータを抽出
            self.current_metadata = ImageParser.extract_metadata(image_path)

            # SettingsをJSON文字列に変換
            settings_json = json.dumps(self.current_metadata['settings'], indent=2, ensure_ascii=False)

            return (
                self.current_metadata['prompt'],
                self.current_metadata['negative_prompt'],
                settings_json
            )
        except Exception as e:
            print(f"画像読み込みエラー: {e}")
            import traceback
            traceback.print_exc()
            # エラーが発生した場合も状態をクリア
            self.current_image_path = None
            self.current_metadata = None
            return "画像の読み込みに失敗しました。もう一度ドロップしてください。", "", "{}"

    def _get_model_status(self) -> str:
        """現在のモデル状態を取得"""
        if self.current_vlm is None:
            return "モデル未ロード"
        if self.selected_model_path:
            return f"✓ モデルロード済み: {Path(self.selected_model_path).name}"
        return "モデルロード済み"

    def generate_wan_prompt(
        self,
        additional_instruction: str,
        style_preset: str,
        temperature: float,
        max_tokens: int
    ):
        """WAN 2.1プロンプトを生成（ストリーミング対応）"""
        max_tokens_int = int(max_tokens)

        # モデルが未ロードで、モデルが選択されている場合は自動ロード
        if self.current_vlm is None and self.selected_model_path:
            yield "モデルをロード中...", "<small style='color: gray;'>モデルをロード中...</small>", "モデルをロード中..."

            # モデルをロード
            status, context = self.load_vlm_model(self.selected_model_path)

            if "✓" not in status:
                yield f"エラー: モデルのロードに失敗しました\n{status}", "<small style='color: gray;'>--</small>", status
                return

        if self.current_vlm is None:
            yield "エラー: モデルを選択してロードしてください", "<small style='color: gray;'>--</small>", "モデル未選択"
            return

        if not self.current_image_path or self.current_metadata is None:
            yield "エラー: 画像をアップロードしてください", self._get_context_info_simple(), self._get_model_status()
            return

        prompt_text = self.current_metadata['prompt']

        try:
            # VLMでストリーミング生成
            response = ""
            start_time = time.time()
            for chunk in self.current_vlm.generate_wan_prompt_stream(
                image_path=self.current_image_path,
                sd_prompt=prompt_text,
                additional_instruction=additional_instruction or "",
                style_preset=style_preset or "cinematic",
                temperature=temperature,
                max_tokens=max_tokens_int
            ):
                response += chunk
                yield response, self._get_context_info_simple(), self._get_model_status()

            # 生成時間を表示
            elapsed_time = time.time() - start_time
            context_with_time = f"<small style='color: gray;'>生成完了 ({elapsed_time:.1f}秒)</small>"
            yield response, context_with_time, self._get_model_status()

        except Exception as e:
            yield f"エラー: {str(e)}", self._get_context_info_simple(), self._get_model_status()

    def _get_context_info_simple(self) -> str:
        """シンプルなコンテキスト情報を取得"""
        if self.current_vlm is None:
            return "<small style='color: gray;'>--</small>"

        context_length = self.current_vlm.get_context_length()
        if context_length > 0:
            return f"<small style='color: gray;'>📊 コンテキスト長: {context_length:,}</small>"
        return "<small style='color: gray;'>--</small>"

    def refresh_local_models(self) -> Tuple:
        """ローカルモデル一覧を更新"""
        models = self.model_manager.list_local_models()

        # DataFrameデータを作成
        df_data = [[m['name'], m['path'], m['size']] for m in models]

        # ドロップダウン用の選択肢
        choices = [m['path'] for m in models]

        # 前回使用したモデルを読み込み
        last_model_path = self.load_last_model_path()

        # 前回のモデルがまだ存在する場合は初期値に設定
        if last_model_path and last_model_path in choices:
            self.selected_model_path = last_model_path
            return df_data, gr.Dropdown(choices=choices, value=last_model_path)

        return df_data, gr.Dropdown(choices=choices)

    def load_vlm_model(self, model_path: str) -> Tuple[str, str]:
        """VLMモデルをロード"""
        if not model_path:
            return "エラー: モデルが選択されていません", "<small style='color: gray;'>--</small>"

        # 選択されたモデルパスを保存
        self.selected_model_path = model_path

        try:
            # 既存モデルをアンロード
            if self.current_vlm is not None:
                self.current_vlm.unload_model()

            # モデルをロード
            self.current_vlm = VLMInterface(
                model_path=model_path,
                device=self.config['model']['device'],
                dtype=self.config['model']['dtype']
            )

            # コンテキスト長を取得
            context_length = self.current_vlm.get_context_length()
            if context_length > 0:
                context_info = f"<small style='color: gray;'>📊 CONTEXT: 0 / {context_length:,}</small>"
            else:
                context_info = "<small style='color: gray;'>📊 CONTEXT: 0</small>"

            # 最後に使用したモデルとして保存
            self.save_last_model_path(model_path)

            return f"✓ モデルをロードしました: {Path(model_path).name}", context_info

        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            return f"✗ エラー: {str(e)}\n\n詳細:\n{error_detail}", "<small style='color: gray;'>--</small>"

    def unload_vlm_model(self) -> Tuple[str, str]:
        """VLMモデルをアンロードしてVRAMを解放"""
        if self.current_vlm is None:
            return "モデルは既にアンロードされています", "<small style='color: gray;'>--</small>"

        try:
            self.current_vlm.unload_model()
            self.current_vlm = None
            return "✓ モデルをアンロードしました（VRAMを解放）", "<small style='color: gray;'>--</small>"
        except Exception as e:
            return f"✗ アンロード失敗: {str(e)}", "<small style='color: gray;'>--</small>"

    def update_preset_info(self, preset_name: str) -> Tuple:
        """プリセット情報を表示"""
        if not preset_name or preset_name not in self.model_presets:
            return "プリセットを選択すると詳細が表示されます", "", ""

        preset = self.model_presets[preset_name]

        info_md = f"""
### {preset_name}

**説明**: {preset['description']}
**推奨用途**: {preset['recommended_for']}
**Repository ID**: `{preset['repo_id']}`
"""

        return info_md, preset['repo_id'], preset['local_name']

    def save_last_model_path(self, model_path: str):
        """最後に使用したモデルのパスを保存（settings含む）"""
        try:
            # 既存のデータを読み込み
            data = {}
            if self.last_model_cache_file.exists():
                try:
                    data = json.loads(self.last_model_cache_file.read_text(encoding='utf-8'))
                except:
                    pass

            # モデルパスを更新
            data["last_model"] = model_path

            self.last_model_cache_file.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding='utf-8'
            )
        except Exception as e:
            print(f"警告: モデルパスの保存に失敗しました: {e}")

    def save_inference_settings(self, temperature: float, max_tokens: int, top_p: float):
        """推論設定を保存"""
        try:
            # 既存のデータを読み込み
            data = {}
            if self.last_model_cache_file.exists():
                try:
                    data = json.loads(self.last_model_cache_file.read_text(encoding='utf-8'))
                except:
                    pass

            # 設定を更新
            data["inference_settings"] = {
                "temperature": temperature,
                "max_tokens": int(max_tokens),
                "top_p": top_p
            }

            self.last_model_cache_file.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding='utf-8'
            )
        except Exception as e:
            print(f"警告: 推論設定の保存に失敗しました: {e}")

    def load_last_model_path(self) -> Optional[str]:
        """最後に使用したモデルのパスを読み込み"""
        try:
            if self.last_model_cache_file.exists():
                data = json.loads(self.last_model_cache_file.read_text(encoding='utf-8'))
                return data.get("last_model")
        except Exception as e:
            print(f"警告: モデルパスの読み込みに失敗しました: {e}")
        return None

    def load_inference_settings(self) -> dict:
        """推論設定を読み込み"""
        try:
            if self.last_model_cache_file.exists():
                data = json.loads(self.last_model_cache_file.read_text(encoding='utf-8'))
                return data.get("inference_settings", {})
        except Exception as e:
            print(f"警告: 推論設定の読み込みに失敗しました: {e}")
        return {}

    def download_model(self, repo_id: str, local_name: str) -> str:
        """モデルをダウンロード"""
        if not repo_id:
            return "エラー: Repository IDを入力してください"

        try:
            # ダウンロード実行
            downloaded_path = self.model_manager.download_model(
                repo_id=repo_id,
                local_name=local_name if local_name else None
            )

            return f"✓ ダウンロード完了\n保存先: {downloaded_path}"

        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            return f"✗ ダウンロード失敗\nエラー: {str(e)}\n\n詳細:\n{error_detail}"
