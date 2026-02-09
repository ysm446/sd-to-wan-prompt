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
        self.settings_cache_file = Path("settings_cache.json")

        # モデルプリセットを読み込み
        config_loader = ConfigLoader()
        self.model_presets = config_loader.load_model_presets()

        # デバイス設定をキャッシュから読み込み
        self.device_settings = self.load_device_settings()

    def create_interface(self) -> gr.Blocks:
        """
        Gradio UIを構築

        UI構成:
        - タブ1: 画像分析
        - タブ2: モデル管理
        - タブ3: 設定
        """
        # カスタムCSS（フォント変更）
        # 画像アップロードエリアのフルスクリーン／拡大アイコンを非表示にするルールを追加
        custom_css = """
        * {
            font-family: "Segoe UI", "Yu Gothic", "Meiryo", Arial, sans-serif !important;
        }
        /* Hide Gradio image fullscreen/expand icons by common attributes and keywords (case-insensitive) */
        button[aria-label*="full" i],
        button[title*="full" i],
        button[aria-label*="expand" i],
        button[title*="expand" i],
        button[aria-label*="fullscreen" i],
        button[title*="fullscreen" i] {
            display: none !important;
        }
        /* Additional selectors for possible Gradio class names */
        .gr-image .gr-button, .gr-image .gr-button--icon, .gr-image__open-fullscreen, .gr-image__expand {
            display: none !important;
        }
        """

        # キャッシュから推論設定を読み込み（なければconfigのデフォルト値を使用）
        cached_settings = self.load_inference_settings()
        initial_temperature = cached_settings.get('temperature', self.config['inference']['temperature'])
        initial_max_tokens = cached_settings.get('max_tokens', self.config['inference']['max_tokens'])
        initial_top_p = cached_settings.get('top_p', self.config['inference']['top_p'])

        with gr.Blocks(title="WAN Prompt Generator", css=custom_css) as interface:
            gr.Markdown("# WAN Prompt Generator <small style='font-size:0.4em; color:gray;'>SD画像からWAN 2.2用の動画プロンプトを生成</small>")

            with gr.Tabs():
                # タブ1: プロンプト生成
                with gr.Tab("プロンプト生成"):
                    with gr.Row():
                        # 左側: 画像表示
                        with gr.Column(scale=1):
                            image_display = gr.Image(
                                label="画像をアップロード",
                                type="filepath",
                                sources=["upload"],
                                height=400
                            )
                            image_filename_display = gr.Markdown(
                                value="<small style='color: gray;'>--</small>"
                            )

                            # プロンプト情報表示
                            with gr.Accordion("メタデータ情報", open=True):
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
                            with gr.Row():
                                context_info = gr.Markdown(
                                    value="<small style='color: gray;'>--</small>",
                                    elem_id="context-info"
                                )
                                mini_unload_btn = gr.Button(
                                    value="モデル未ロード",
                                    variant="secondary",
                                    size="sm",
                                    interactive=False,
                                    min_width=60,
                                    scale=0
                                )

                            # 追加指示入力欄
                            additional_input = gr.Textbox(
                                label="追加指示（オプション）",
                                placeholder="例: カメラをズームアウトさせて、髪をなびかせてください",
                                lines=2
                            )

                            generate_btn = gr.Button("WANプロンプト生成", variant="primary", size="lg")

                            # 出力言語・スタイルプリセット選択（横一列）
                            cached_settings = self.load_generation_settings()
                            with gr.Row():
                                language_dropdown = gr.Dropdown(
                                    label="出力言語",
                                    choices=["English", "日本語"],
                                    value=cached_settings.get("language", "English"),
                                    interactive=True,
                                    scale=1
                                )
                                style_dropdown = gr.Dropdown(
                                    label="スタイルプリセット",
                                    choices=[
                                        ("なし", None),
                                        ("穏やか", "calm"),
                                        ("ダイナミック", "dynamic"),
                                        ("シネマティック", "cinematic"),
                                        ("アニメ風", "anime")
                                    ],
                                    value=cached_settings.get("style_preset"),
                                    interactive=True,
                                    scale=1
                                )

                            # 出力項目選択
                            cached_sections = self.load_output_sections()
                            output_sections = gr.CheckboxGroup(
                                label="出力項目",
                                choices=[
                                    ("シーン", "scene"),
                                    ("アクション", "action"),
                                    ("カメラ", "camera"),
                                    ("スタイル", "style"),
                                    ("WANプロンプト", "prompt")
                                ],
                                value=cached_sections,
                                interactive=True
                            )

                            # モデル選択
                            with gr.Accordion("モデル設定", open=False):
                                model_dropdown = gr.Dropdown(
                                    label="使用するモデル",
                                    choices=[],
                                    value=None,
                                    interactive=True
                                )

                                # デバイス設定
                                with gr.Row():
                                    device_dropdown = gr.Dropdown(
                                        label="計算デバイス",
                                        choices=self._get_available_devices(),
                                        value=self.device_settings['device'],
                                        interactive=True,
                                        scale=1,
                                        info="モデルの計算に使用するデバイス"
                                    )
                                    dtype_dropdown = gr.Dropdown(
                                        label="データ型",
                                        choices=[
                                            ("BFloat16（推奨・高速）", "bfloat16"),
                                            ("Float16（GPU高速）", "float16"),
                                            ("Float32（高精度・低速）", "float32")
                                        ],
                                        value=self.device_settings['dtype'],
                                        interactive=True,
                                        scale=1
                                    )

                                device_info = gr.Markdown(value=self._get_device_info())

                                with gr.Row():
                                    load_model_btn = gr.Button("モデルをロード")
                                    unload_model_btn = gr.Button("モデルをクリア")
                                model_status = gr.Textbox(
                                    label="モデル状態",
                                    value="モデル未ロード",
                                    interactive=False
                                )

                                # 自動アンロード設定
                                cached_auto_unload = self.load_auto_unload_setting()
                                auto_unload_checkbox = gr.Checkbox(
                                    label="プロンプト生成後に自動アンロード",
                                    value=cached_auto_unload,
                                    info="WANプロンプト生成完了後、自動的にモデルをアンロードしてVRAMを解放します",
                                    interactive=True
                                )

                            # テキストファイル保存
                            cached_save_dir = self.load_save_directory()
                            with gr.Accordion("テキストファイル保存", open=False):
                                save_dir_input = gr.Textbox(
                                    label="保存先フォルダ",
                                    placeholder="例: D:\\images\\output",
                                    value=cached_save_dir,
                                    lines=1
                                )
                                save_btn = gr.Button(
                                    "📄 テキストファイルに保存", variant="primary"
                                )
                                save_status = gr.Textbox(
                                    label="保存結果",
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
            # 画像アップロード（upload/clearイベントで分離して処理）
            image_display.upload(
                fn=self.on_image_upload,
                inputs=[image_display],
                outputs=[image_filename_display, prompt_display, negative_prompt_display, settings_display]
            )
            image_display.clear(
                fn=self.on_image_clear,
                inputs=[],
                outputs=[image_filename_display, prompt_display, negative_prompt_display, settings_display]
            )

            # WANプロンプト生成
            generate_btn.click(
                fn=self.generate_wan_prompt,
                inputs=[additional_input, style_dropdown, language_dropdown, output_sections, temperature_slider, max_tokens_slider, auto_unload_checkbox],
                outputs=[output_textbox, context_info, model_status, mini_unload_btn]
            )

            # 自動アンロード設定の変更時にキャッシュを保存
            auto_unload_checkbox.change(
                fn=self.save_auto_unload_setting,
                inputs=[auto_unload_checkbox],
                outputs=[]
            )

            # テキストファイル保存
            save_btn.click(
                fn=self.save_prompt_to_file,
                inputs=[save_dir_input, output_textbox, additional_input],
                outputs=[save_status]
            )

            # 保存先フォルダの変更時にキャッシュを保存
            save_dir_input.change(
                fn=self.save_save_directory,
                inputs=[save_dir_input],
                outputs=[]
            )

            # 出力項目の変更時にキャッシュを保存
            output_sections.change(
                fn=self.save_output_sections,
                inputs=[output_sections],
                outputs=[]
            )

            # 言語・スタイルの変更時にキャッシュを保存
            language_dropdown.change(
                fn=lambda lang, style: self.save_generation_settings(lang, style),
                inputs=[language_dropdown, style_dropdown],
                outputs=[]
            )
            style_dropdown.change(
                fn=lambda lang, style: self.save_generation_settings(lang, style),
                inputs=[language_dropdown, style_dropdown],
                outputs=[]
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
                fn=self.load_vlm_model_with_btn,
                inputs=[model_dropdown],
                outputs=[model_status, context_info, mini_unload_btn]
            )

            unload_model_btn.click(
                fn=self.unload_vlm_model_with_btn,
                outputs=[model_status, context_info, mini_unload_btn]
            )

            # デバイス変更時のイベント
            device_dropdown.change(
                fn=self.on_device_change,
                inputs=[device_dropdown, dtype_dropdown],
                outputs=[model_status, context_info, mini_unload_btn, device_info]
            )

            dtype_dropdown.change(
                fn=self.on_device_change,
                inputs=[device_dropdown, dtype_dropdown],
                outputs=[model_status, context_info, mini_unload_btn, device_info]
            )

            mini_unload_btn.click(
                fn=self.unload_vlm_model_with_btn,
                outputs=[model_status, context_info, mini_unload_btn]
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

    def on_image_clear(self) -> Tuple:
        """画像がクリアされたときの処理"""
        self.current_image_path = None
        self.current_metadata = None
        return "<small style='color: gray;'>--</small>", "", "", "{}"

    def on_image_upload(self, image_path: str) -> Tuple:
        """画像がアップロードされたときの処理"""
        try:
            # 画像パスがNoneまたは空の場合はクリア
            if not image_path:
                return self.on_image_clear()

            # 画像パスを保存
            self.current_image_path = image_path

            # メタデータを抽出
            self.current_metadata = ImageParser.extract_metadata(image_path)

            # SettingsをJSON文字列に変換
            settings_json = json.dumps(self.current_metadata['settings'], indent=2, ensure_ascii=False)

            # ファイル名を表示
            filename = Path(image_path).name
            filename_md = f"<small>📁 {filename}</small>"

            return (
                filename_md,
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
            return "<small style='color: gray;'>--</small>", "画像の読み込みに失敗しました。もう一度ドロップしてください。", "", "{}"

    def _get_unload_btn_update(self):
        """ミニアンロードボタンの表示状態を取得"""
        if self.current_vlm is not None:
            return gr.update(
                value="⏏ アンロード", variant="stop", interactive=True
            )
        else:
            return gr.update(
                value="モデル未ロード", variant="secondary", interactive=False
            )

    def load_vlm_model_with_btn(
        self, model_path: str
    ) -> Tuple[str, str, dict]:
        """VLMモデルをロード（ミニボタン更新付き）"""
        status, context = self.load_vlm_model(model_path)
        return status, context, self._get_unload_btn_update()

    def unload_vlm_model_with_btn(self) -> Tuple[str, str, dict]:
        """VLMモデルをアンロード（ミニボタン更新付き）"""
        status, context = self.unload_vlm_model()
        return status, context, self._get_unload_btn_update()

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
        output_language: str,
        output_sections: List[str],
        temperature: float,
        max_tokens: int,
        auto_unload: bool
    ):
        """WAN 2.2プロンプトを生成（ストリーミング対応）"""
        max_tokens_int = int(max_tokens)
        # 出力項目が空の場合はデフォルトを使用
        if not output_sections:
            output_sections = ["scene", "action", "camera", "style", "prompt"]

        # モデルが未ロードで、モデルが選択されている場合は自動ロード
        if self.current_vlm is None and self.selected_model_path:
            yield "モデルをロード中...", "<small style='color: gray;'>モデルをロード中...</small>", "モデルをロード中...", gr.update()

            # モデルをロード
            status, context = self.load_vlm_model(self.selected_model_path)

            if "✓" not in status:
                yield f"エラー: モデルのロードに失敗しました\n{status}", "<small style='color: gray;'>--</small>", status, self._get_unload_btn_update()
                return

        if self.current_vlm is None:
            yield "エラー: モデルを選択してロードしてください", "<small style='color: gray;'>--</small>", "モデル未選択", self._get_unload_btn_update()
            return

        if not self.current_image_path or self.current_metadata is None:
            yield "エラー: 画像をアップロードしてください", self._get_context_info_simple(), self._get_model_status(), self._get_unload_btn_update()
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
                style_preset=style_preset,
                output_language=output_language or "English",
                output_sections=output_sections,
                temperature=temperature,
                max_tokens=max_tokens_int
            ):
                response += chunk
                yield response, self._get_context_info_simple(), self._get_model_status(), self._get_unload_btn_update()

            # 生成時間を表示
            elapsed_time = time.time() - start_time
            context_with_time = f"<small style='color: gray;'>生成完了 ({elapsed_time:.1f}秒)</small>"
            yield response, context_with_time, self._get_model_status(), self._get_unload_btn_update()

            # 自動アンロードが有効な場合はモデルをアンロード
            if auto_unload:
                self.unload_vlm_model()
                yield response, "<small style='color: gray;'>✓ モデルを自動アンロードしました</small>", "✓ モデルをアンロードしました（VRAMを解放）", self._get_unload_btn_update()

        except Exception as e:
            yield f"エラー: {str(e)}", self._get_context_info_simple(), self._get_model_status(), self._get_unload_btn_update()

    def _get_context_info_simple(self) -> str:
        """シンプルなコンテキスト情報を取得"""
        if self.current_vlm is None:
            return "<small style='color: gray;'>--</small>"

        context_length = self.current_vlm.get_context_length()
        if context_length > 0:
            return f"<small style='color: gray;'>📊 コンテキスト長: {context_length:,}</small>"
        return "<small style='color: gray;'>--</small>"

    def _get_device_info(self) -> str:
        """現在のデバイス情報を取得"""
        import torch

        info_lines = []

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            info_lines.append(f"✓ **CUDA利用可能**")
            info_lines.append(f"  - GPU: {gpu_name}")
            info_lines.append(f"  - VRAM: {gpu_memory:.1f} GB")
        else:
            info_lines.append("⚠ **CUDA利用不可** - CPUモードのみ")

        return "<small>" + "<br>".join(info_lines) + "</small>"

    def _get_available_devices(self) -> List[Tuple[str, str]]:
        """利用可能なデバイス選択肢を取得"""
        import torch

        choices = [
            ("自動選択（GPU優先）", "auto"),
            ("CPU", "cpu")
        ]

        if torch.cuda.is_available():
            choices.append(("CUDA（GPU）", "cuda"))

        return choices

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
            # ドロップダウンのvalueを設定してUIに表示
            return df_data, gr.update(choices=choices, value=last_model_path)

        # 前回のモデルがない場合は選択状態をクリア
        self.selected_model_path = None
        return df_data, gr.update(choices=choices, value=None)

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

            # キャッシュからデバイス設定を読み込み
            device_settings = self.load_device_settings()

            # モデルをロード
            self.current_vlm = VLMInterface(
                model_path=model_path,
                device=device_settings['device'],
                dtype=device_settings['dtype']
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

    def on_device_change(
        self,
        device: str,
        dtype: str
    ) -> Tuple[str, str, dict, str]:
        """
        デバイス/データ型変更時の処理

        処理フロー:
        1. 設定を settings_cache.json に保存
        2. モデルがロード済みの場合:
           - 既存モデルをアンロード
           - 新しいデバイス設定でモデルを再ロード
           - UIを更新
        3. モデル未ロードの場合:
           - 設定のみ保存（次回ロード時に反映）

        Args:
            device: 新しいデバイス設定
            dtype: 新しいデータ型設定

        Returns:
            (model_status, context_info, mini_unload_btn_update, device_info)
        """
        # 設定を保存
        self.save_device_settings(device, dtype)

        # デバイス情報を更新
        device_info_text = self._get_device_info()

        # モデルがロード済みの場合は再ロード
        if self.current_vlm is not None and self.selected_model_path:
            try:
                # アンロード
                self.current_vlm.unload_model()
                self.current_vlm = None

                # 新しいデバイスでロード
                self.current_vlm = VLMInterface(
                    model_path=self.selected_model_path,
                    device=device,
                    dtype=dtype
                )

                context_length = self.current_vlm.get_context_length()
                context_info = f"<small style='color: gray;'>📊 CONTEXT: 0 / {context_length:,}</small>"

                device_name = {"auto": "自動選択", "cpu": "CPU", "cuda": "CUDA（GPU）"}.get(device, device)
                return (
                    f"✓ デバイスを変更しました: {device_name}\nモデル: {Path(self.selected_model_path).name}",
                    context_info,
                    self._get_unload_btn_update(),
                    device_info_text
                )
            except Exception as e:
                return (
                    f"✗ デバイス変更エラー: {str(e)}",
                    "<small style='color: gray;'>--</small>",
                    self._get_unload_btn_update(),
                    device_info_text
                )

        # モデル未ロード時
        return (
            "デバイス設定を保存しました（次回ロード時に反映）",
            "<small style='color: gray;'>--</small>",
            self._get_unload_btn_update(),
            device_info_text
        )

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
            if self.settings_cache_file.exists():
                try:
                    data = json.loads(self.settings_cache_file.read_text(encoding='utf-8'))
                except:
                    pass

            # モデルパスを更新
            data["last_model"] = model_path

            self.settings_cache_file.write_text(
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
            if self.settings_cache_file.exists():
                try:
                    data = json.loads(self.settings_cache_file.read_text(encoding='utf-8'))
                except:
                    pass

            # 設定を更新
            data["inference_settings"] = {
                "temperature": temperature,
                "max_tokens": int(max_tokens),
                "top_p": top_p
            }

            self.settings_cache_file.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding='utf-8'
            )
        except Exception as e:
            print(f"警告: 推論設定の保存に失敗しました: {e}")

    def load_last_model_path(self) -> Optional[str]:
        """最後に使用したモデルのパスを読み込み"""
        try:
            if self.settings_cache_file.exists():
                data = json.loads(self.settings_cache_file.read_text(encoding='utf-8'))
                return data.get("last_model")
        except Exception as e:
            print(f"警告: モデルパスの読み込みに失敗しました: {e}")
        return None

    def load_inference_settings(self) -> dict:
        """推論設定を読み込み"""
        try:
            if self.settings_cache_file.exists():
                data = json.loads(self.settings_cache_file.read_text(encoding='utf-8'))
                return data.get("inference_settings", {})
        except Exception as e:
            print(f"警告: 推論設定の読み込みに失敗しました: {e}")
        return {}

    def save_output_sections(self, sections: List[str]):
        """出力項目の選択状態を保存"""
        try:
            data = {}
            if self.settings_cache_file.exists():
                try:
                    data = json.loads(self.settings_cache_file.read_text(encoding='utf-8'))
                except:
                    pass

            data["output_sections"] = sections

            self.settings_cache_file.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding='utf-8'
            )
        except Exception as e:
            print(f"警告: 出力項目の保存に失敗しました: {e}")

    def load_output_sections(self) -> List[str]:
        """出力項目の選択状態を読み込み"""
        default_sections = ["scene", "action", "camera", "style", "prompt"]
        try:
            if self.settings_cache_file.exists():
                data = json.loads(self.settings_cache_file.read_text(encoding='utf-8'))
                return data.get("output_sections", default_sections)
        except Exception as e:
            print(f"警告: 出力項目の読み込みに失敗しました: {e}")
        return default_sections

    def save_generation_settings(self, language: str, style_preset: str):
        """言語・スタイルプリセットの選択状態を保存"""
        try:
            data = {}
            if self.settings_cache_file.exists():
                try:
                    data = json.loads(self.settings_cache_file.read_text(encoding='utf-8'))
                except:
                    pass

            data["generation_settings"] = {
                "language": language,
                "style_preset": style_preset
            }

            self.settings_cache_file.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding='utf-8'
            )
        except Exception as e:
            print(f"警告: 生成設定の保存に失敗しました: {e}")

    def load_generation_settings(self) -> dict:
        """言語・スタイルプリセットの選択状態を読み込み"""
        try:
            if self.settings_cache_file.exists():
                data = json.loads(self.settings_cache_file.read_text(encoding='utf-8'))
                return data.get("generation_settings", {})
        except Exception as e:
            print(f"警告: 生成設定の読み込みに失敗しました: {e}")
        return {}

    def save_save_directory(self, save_dir: str):
        """保存先フォルダをキャッシュに保存"""
        try:
            data = {}
            if self.settings_cache_file.exists():
                try:
                    data = json.loads(self.settings_cache_file.read_text(encoding='utf-8'))
                except:
                    pass

            data["save_directory"] = save_dir

            self.settings_cache_file.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding='utf-8'
            )
        except Exception as e:
            print(f"警告: 保存先フォルダの保存に失敗しました: {e}")

    def load_save_directory(self) -> str:
        """保存先フォルダをキャッシュから読み込み"""
        try:
            if self.settings_cache_file.exists():
                data = json.loads(self.settings_cache_file.read_text(encoding='utf-8'))
                return data.get("save_directory", "")
        except Exception as e:
            print(f"警告: 保存先フォルダの読み込みに失敗しました: {e}")
        return ""

    def save_device_settings(self, device: str, dtype: str):
        """デバイス設定を settings_cache.json に保存"""
        try:
            # 既存データを読み込み
            data = {}
            if self.settings_cache_file.exists():
                try:
                    data = json.loads(self.settings_cache_file.read_text(encoding='utf-8'))
                except:
                    pass

            # デバイス設定を更新
            data["device_settings"] = {
                "device": device,
                "dtype": dtype
            }

            self.settings_cache_file.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding='utf-8'
            )
        except Exception as e:
            print(f"警告: デバイス設定の保存に失敗しました: {e}")

    def load_device_settings(self) -> Dict[str, str]:
        """デバイス設定を settings_cache.json から読み込み"""
        # デフォルト値（settings.yaml から取得）
        default_settings = {
            "device": self.config.get('model', {}).get('device', 'auto'),
            "dtype": self.config.get('model', {}).get('dtype', 'bfloat16')
        }

        try:
            if self.settings_cache_file.exists():
                data = json.loads(self.settings_cache_file.read_text(encoding='utf-8'))
                return data.get("device_settings", default_settings)
        except Exception as e:
            print(f"警告: デバイス設定の読み込みに失敗しました: {e}")

        return default_settings

    def save_auto_unload_setting(self, auto_unload: bool):
        """自動アンロード設定をキャッシュに保存"""
        try:
            data = {}
            if self.settings_cache_file.exists():
                try:
                    data = json.loads(self.settings_cache_file.read_text(encoding='utf-8'))
                except:
                    pass

            data["auto_unload"] = auto_unload

            self.settings_cache_file.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding='utf-8'
            )
        except Exception as e:
            print(f"警告: 自動アンロード設定の保存に失敗しました: {e}")

    def load_auto_unload_setting(self) -> bool:
        """自動アンロード設定をキャッシュから読み込み"""
        try:
            if self.settings_cache_file.exists():
                data = json.loads(self.settings_cache_file.read_text(encoding='utf-8'))
                return data.get("auto_unload", False)
        except Exception as e:
            print(f"警告: 自動アンロード設定の読み込みに失敗しました: {e}")
        return False

    def save_prompt_to_file(
        self,
        save_dir: str,
        output_text: str,
        additional_instruction: str
    ) -> str:
        """元のプロンプト・追加指示・生成プロンプトをテキストファイルに保存"""
        if not self.current_image_path:
            return "エラー: 画像がアップロードされていません"

        if not save_dir or not save_dir.strip():
            return "エラー: 保存先フォルダを指定してください"

        if not output_text or not output_text.strip():
            return "エラー: 生成されたプロンプトがありません"

        # 画像ファイル名から .txt のファイル名を作成
        image_filename = Path(self.current_image_path).stem + ".txt"
        save_path = Path(save_dir.strip()) / image_filename

        # テキスト内容を組み立て
        lines: list[str] = []

        # 元のプロンプト
        original_prompt = ""
        if self.current_metadata:
            original_prompt = self.current_metadata.get('prompt', '')
        if original_prompt:
            lines.append("=== Original Prompt ===")
            lines.append(original_prompt)
            lines.append("")

        # 追加指示
        if additional_instruction and additional_instruction.strip():
            lines.append("=== Additional Instruction ===")
            lines.append(additional_instruction.strip())
            lines.append("")

        # 生成されたWANプロンプト
        lines.append("=== Generated WAN Prompt ===")
        lines.append(output_text.strip())
        lines.append("")

        try:
            save_path.write_text("\n".join(lines), encoding='utf-8')
            return f"✓ 保存しました: {save_path}"
        except Exception as e:
            return f"✗ 保存に失敗しました: {e}"

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
