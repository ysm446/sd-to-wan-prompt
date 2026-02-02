"""
Vision-Language Model推論インターフェース
"""
from pathlib import Path
from typing import Optional, Generator, List
from threading import Thread
import torch
import json
from transformers import AutoProcessor, TextIteratorStreamer
from PIL import Image


class VLMInterface:
    """Vision-Language Model推論インターフェース"""

    def __init__(self, model_path: str, device: str = "auto", dtype: str = "bfloat16"):
        """
        Args:
            model_path: ローカルモデルのパス
            device: デバイス指定 ("auto", "cuda:0", "cpu")
            dtype: データ型 ("bfloat16", "float16", "float32")
        """
        self.model_path = Path(model_path)
        self.device = device
        self.dtype = dtype
        self.model = None
        self.processor = None

        self.load_model(str(model_path))

    def load_model(self, model_path: str):
        """
        モデルをメモリにロード

        実装詳細:
        - Qwen2VLForConditionalGeneration または AutoModel を使用
        - device_map="auto" で自動GPU配置
        - torch.bfloat16 または torch.float16
        """
        print(f"モデルを読み込み中: {model_path}")

        # データ型の設定
        torch_dtype = self._get_torch_dtype()

        try:
            import traceback

            # プロセッサー（トークナイザー + 画像プロセッサー）をロード
            print(f"  プロセッサーを読み込み中...")
            self.processor = AutoProcessor.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            print(f"  ✓ プロセッサーの読み込み完了")

            # モデルをロード
            # デバイスマップの設定（CPUモード時は明示的にCPUを指定）
            if self.device == "cpu":
                device_map = {"": "cpu"}
            else:
                device_map = "auto"

            # config.jsonからアーキテクチャを判定して適切なモデルクラスを選択
            model_class = self._get_model_class(model_path)

            print(f"  モデルクラス: {model_class.__name__}")
            print(f"  デバイスマップ: {device_map}")
            print(f"  モデルを読み込み中...")
            self.model = model_class.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                device_map=device_map,
                trust_remote_code=True
            )

            print(f"✓ モデルの読み込みが完了しました")
            print(f"  デバイスマップ: auto")
            print(f"  データ型: {self.dtype}")

            # デバイス情報を表示
            if hasattr(self.model, 'device'):
                print(f"  モデルデバイス: {self.model.device}")
            elif hasattr(self.model, 'hf_device_map'):
                print(f"  デバイスマップ: {self.model.hf_device_map}")

            # CUDA使用状況
            if torch.cuda.is_available():
                print(f"  ✓ CUDA利用可能")
                print(f"  GPU: {torch.cuda.get_device_name(0)}")
                print(f"  GPU メモリ: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            else:
                print(f"  ⚠ CUDA利用不可 - CPUで動作中（非常に遅くなります）")

        except Exception as e:
            import traceback
            print(f"✗ モデルの読み込みに失敗しました")
            print(f"  エラー: {e}")
            print(f"\n詳細なエラートレース:")
            traceback.print_exc()
            raise

    def analyze_image_with_prompt(
        self,
        image_path: str,
        prompt_text: str,
        question: str,
        temperature: float = 0.7,
        max_tokens: int = 512
    ) -> str:
        """
        画像とプロンプトを分析

        Args:
            image_path: 分析対象の画像パス
            prompt_text: 元のSDプロンプト
            question: ユーザーの質問
            temperature: 生成温度
            max_tokens: 最大トークン数

        Returns:
            VLMの回答テキスト
        """
        if self.model is None or self.processor is None:
            raise RuntimeError("モデルがロードされていません")

        # 画像を読み込み
        image = Image.open(image_path).convert('RGB')

        # システムメッセージとユーザーメッセージを構築
        conversation = [
            {
                "role": "system",
                "content": "あなたは画像分析の専門家です。Stable Diffusionで生成された画像とそのプロンプトを評価してください。"
            },
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"元のプロンプト:\n{prompt_text}\n\n質問: {question}"}
                ]
            }
        ]

        # プロセッサーでテキストと画像を処理
        text_prompt = self.processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.processor(
            text=[text_prompt],
            images=[image],
            return_tensors="pt",
            padding=True
        )

        # GPUに転送（必要な場合）
        if self.device == "cuda" or (self.device == "auto" and torch.cuda.is_available()):
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # 推論
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0
            )

        # デコード
        generated_text = self.processor.batch_decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )[0]

        # プロンプト部分を削除（応答のみを返す）
        response = generated_text.split("assistant\n")[-1].strip()

        return response

    def analyze_image_with_prompt_stream(
        self,
        image_path: str,
        prompt_text: str,
        question: str,
        temperature: float = 0.7,
        max_tokens: int = 512
    ) -> Generator[str, None, None]:
        """
        画像とプロンプトを分析（ストリーミング版）

        Args:
            image_path: 分析対象の画像パス
            prompt_text: 元のSDプロンプト
            question: ユーザーの質問
            temperature: 生成温度
            max_tokens: 最大トークン数

        Yields:
            生成されたテキストの断片
        """
        if self.model is None or self.processor is None:
            raise RuntimeError("モデルがロードされていません")

        # 画像を読み込み
        image = Image.open(image_path).convert('RGB')

        # システムメッセージとユーザーメッセージを構築
        conversation = [
            {
                "role": "system",
                "content": "あなたは画像分析の専門家です。Stable Diffusionで生成された画像とそのプロンプトを評価してください。"
            },
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"元のプロンプト:\n{prompt_text}\n\n質問: {question}"}
                ]
            }
        ]

        # プロセッサーでテキストと画像を処理
        text_prompt = self.processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.processor(
            text=[text_prompt],
            images=[image],
            return_tensors="pt",
            padding=True
        )

        # GPUに転送（必要な場合）
        if self.device == "cuda" or (self.device == "auto" and torch.cuda.is_available()):
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # ストリーマーを作成
        streamer = TextIteratorStreamer(
            self.processor.tokenizer,
            skip_special_tokens=True,
            skip_prompt=True
        )

        # 生成パラメータ
        generation_kwargs = dict(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            streamer=streamer
        )

        # 別スレッドで生成を実行
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        # ストリーマーからトークンを順次yield
        for text in streamer:
            yield text

        thread.join()

    def generate_wan_prompt_stream(
        self,
        image_path: str,
        sd_prompt: str,
        additional_instruction: str = "",
        style_preset: str = "cinematic",
        output_language: str = "English",
        output_sections: List[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024
    ) -> Generator[str, None, None]:
        """
        WAN 2.2用の動画プロンプトをストリーミング生成

        Args:
            image_path: 分析対象の画像パス
            sd_prompt: 元のSDプロンプト
            additional_instruction: ユーザーの追加指示
            style_preset: スタイルプリセット名 (calm, dynamic, cinematic, anime)
            output_language: 出力言語 (English, 日本語)
            output_sections: 出力する項目のリスト (scene, action, camera, style, prompt)
            temperature: 生成温度
            max_tokens: 最大トークン数

        Yields:
            生成されたテキストの断片
        """
        # デフォルトの出力項目
        if output_sections is None:
            output_sections = ["scene", "action", "camera", "style", "prompt"]
        if self.model is None or self.processor is None:
            raise RuntimeError("モデルがロードされていません")

        # 画像を読み込み
        image = Image.open(image_path).convert('RGB')

        # スタイルプリセットに基づくヒント（Noneや空の場合はスタイルヒントなし）
        style_hints = {
            "calm": "Focus on gentle, slow movements. Camera should move slowly with smooth pans. Actions should be subtle like breathing, hair swaying, or soft expressions.",
            "dynamic": "Focus on energetic, fast movements. Camera can use tracking shots or quick cuts. Actions should be dynamic with clear motion.",
            "cinematic": "Focus on dramatic, cinematic quality. Camera should use dolly movements or dramatic angles. Include emotional expressions and atmospheric lighting.",
            "anime": "Focus on anime-style movements. Include exaggerated expressions, wind effects, and dynamic poses typical of Japanese animation."
        }
        style_hint = style_hints.get(style_preset) if style_preset else None

        # 出力項目テンプレート
        section_templates_ja = {
            "scene": "**シーン**: [画像に基づいて視覚的なシーンを詳しく説明]",
            "action": "**アクション**: [追加する動き・モーションを具体的に説明]",
            "camera": "**カメラ**: [カメラの動き: 静止、スローパン、ズームイン/アウト、ドリー、トラッキングなど]",
            "style": "**スタイル**: [視覚的なスタイルと雰囲気を説明]",
            "prompt": "---\n**WAN 2.2用プロンプト**:\n[上記の要素をすべて組み合わせた1つの段落を書いてください。WAN 2.2にそのままコピー＆ペーストできるようにしてください。簡潔かつ描写的に。動きと映画的な品質に焦点を当ててください。]"
        }
        section_templates_en = {
            "scene": "**Scene**: [Describe the visual scene in detail based on the image]",
            "action": "**Action**: [Describe the motion/movement to add - be specific about what moves and how]",
            "camera": "**Camera**: [Describe camera movement: static, slow pan, zoom in/out, dolly, tracking, etc.]",
            "style": "**Style**: [Describe the visual style and mood]",
            "prompt": "---\n**Final Prompt for WAN 2.2**:\n[Write a single paragraph combining all elements. This should be copy-paste ready for WAN 2.2. Write in English, be concise but descriptive. Focus on motion and cinematic qualities.]"
        }

        # 言語に応じたシステムプロンプトを動的構築
        if output_language == "日本語":
            # 選択された項目のみ含める
            format_lines = [section_templates_ja[s] for s in output_sections if s in section_templates_ja]
            format_section = "\n".join(format_lines) if format_lines else "自由な形式で出力してください。"

            system_prompt = f"""あなたはWAN 2.2（テキストから動画を生成するAIモデル）向けの動画生成プロンプトを作成する専門家です。
与えられた画像とStable Diffusionのプロンプトを分析し、動画プロンプトを生成してください。

【重要】以下の指定されたセクションのみを出力してください。指定されていないセクションは出力しないでください。各セクションは簡潔に1-2文で書いてください。

{format_section}"""
        else:
            # 選択された項目のみ含める
            format_lines = [section_templates_en[s] for s in output_sections if s in section_templates_en]
            format_section = "\n".join(format_lines) if format_lines else "Output in a free format."

            system_prompt = f"""You are an expert in creating video generation prompts for WAN 2.2 (a text-to-video AI model).
Your task is to analyze the given image and its Stable Diffusion prompt, then generate a video prompt.

**IMPORTANT:** Output ONLY the sections specified below. Do NOT add any other sections. Keep each section concise (1-2 sentences).

{format_section}"""

        # ユーザーメッセージを構築
        has_sd_prompt = bool(sd_prompt.strip())

        if has_sd_prompt:
            # SDプロンプトがある場合（従来の動作）
            user_message = f"""Original SD Prompt:
{sd_prompt}"""
        else:
            # SDプロンプトがない場合（一般画像）
            user_message = "This is a general image (not from Stable Diffusion)."

        # スタイルヒントがある場合のみ追加
        if style_hint:
            user_message += f"\n\nStyle Direction: {style_hint}"

        # 追加指示の扱い
        if additional_instruction:
            if has_sd_prompt:
                # SDプロンプトがある場合は「追加指示」として追加
                user_message += f"\n\nAdditional Instructions: {additional_instruction}"
            else:
                # SDプロンプトがない場合は「主要な指示」として強調
                user_message += f"\n\nUser Instructions (IMPORTANT - follow these closely): {additional_instruction}"

        if output_language == "日本語":
            user_message += "\n\nこの画像と情報に基づいて、WAN 2.2用の動画プロンプトを日本語で生成してください。"
        else:
            user_message += "\n\nPlease generate a WAN 2.2 video prompt based on this image and information."

        # システムメッセージとユーザーメッセージを構築
        conversation = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_message}
                ]
            }
        ]

        # プロセッサーでテキストと画像を処理
        text_prompt = self.processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.processor(
            text=[text_prompt],
            images=[image],
            return_tensors="pt",
            padding=True
        )

        # GPUに転送（必要な場合）
        if self.device == "cuda" or (self.device == "auto" and torch.cuda.is_available()):
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # ストリーマーを作成
        streamer = TextIteratorStreamer(
            self.processor.tokenizer,
            skip_special_tokens=True,
            skip_prompt=True
        )

        # 生成パラメータ
        generation_kwargs = dict(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            streamer=streamer
        )

        # 別スレッドで生成を実行
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        # ストリーマーからトークンを順次yield
        for text in streamer:
            yield text

        thread.join()

    def chat(
        self,
        message: str,
        image: Optional[Image.Image] = None,
        temperature: float = 0.7,
        max_tokens: int = 512
    ) -> str:
        """
        シンプルなチャットインターフェース

        Args:
            message: ユーザーメッセージ
            image: PIL Image オブジェクト（オプション）
            temperature: 生成温度
            max_tokens: 最大トークン数

        Returns:
            VLMの回答
        """
        if self.model is None or self.processor is None:
            raise RuntimeError("モデルがロードされていません")

        # メッセージを構築
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": message}
                ]
            }
        ]

        # 画像がある場合は追加
        if image is not None:
            conversation[0]["content"].insert(0, {"type": "image"})
            images = [image]
        else:
            images = None

        # テキストプロンプトを作成
        text_prompt = self.processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True
        )

        # 入力を処理
        inputs = self.processor(
            text=[text_prompt],
            images=images,
            return_tensors="pt",
            padding=True
        )

        # GPUに転送
        if self.device == "cuda" or (self.device == "auto" and torch.cuda.is_available()):
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # 推論
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0
            )

        # デコード
        generated_text = self.processor.batch_decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )[0]

        # 応答部分を抽出
        response = generated_text.split("assistant\n")[-1].strip()

        return response

    def get_context_length(self) -> int:
        """モデルの最大コンテキスト長を取得"""
        if self.model is None:
            return 0

        # モデルの設定からコンテキスト長を取得
        if hasattr(self.model.config, 'max_position_embeddings'):
            return self.model.config.max_position_embeddings
        elif hasattr(self.model.config, 'max_seq_len'):
            return self.model.config.max_seq_len
        elif hasattr(self.model.config, 'n_positions'):
            return self.model.config.n_positions
        else:
            return 0  # 不明な場合

    def count_tokens(self, text: str) -> int:
        """テキストのトークン数をカウント"""
        if self.processor is None:
            return 0

        tokens = self.processor.tokenizer.encode(text)
        return len(tokens)

    def unload_model(self):
        """メモリからモデルをアンロード"""
        if self.model is not None:
            del self.model
            self.model = None

        if self.processor is not None:
            del self.processor
            self.processor = None

        # GPUメモリをクリア
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("モデルをアンロードしました")

    def _get_model_class(self, model_path: str):
        """
        config.jsonからアーキテクチャを読み取って適切なモデルクラスを返す

        Args:
            model_path: モデルディレクトリのパス

        Returns:
            モデルクラス
        """
        config_path = Path(model_path) / "config.json"

        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)

                architecture = config.get('architectures', [None])[0]
                print(f"  検出されたアーキテクチャ: {architecture}")

                # アーキテクチャに基づいてモデルクラスを選択
                if architecture == "Qwen2_5_VLForConditionalGeneration":
                    try:
                        from transformers import Qwen2_5_VLForConditionalGeneration
                        return Qwen2_5_VLForConditionalGeneration
                    except ImportError:
                        print(f"  警告: Qwen2_5_VLForConditionalGenerationが見つかりません。AutoModelForVision2Seqを使用します。")
                        from transformers import AutoModelForVision2Seq
                        return AutoModelForVision2Seq

                elif architecture == "Qwen2VLForConditionalGeneration":
                    try:
                        from transformers import Qwen2VLForConditionalGeneration
                        return Qwen2VLForConditionalGeneration
                    except ImportError:
                        print(f"  警告: Qwen2VLForConditionalGenerationが見つかりません。AutoModelForVision2Seqを使用します。")
                        from transformers import AutoModelForVision2Seq
                        return AutoModelForVision2Seq

                elif architecture == "Qwen3VLForConditionalGeneration":
                    try:
                        from transformers import Qwen3VLForConditionalGeneration
                        return Qwen3VLForConditionalGeneration
                    except ImportError:
                        print(f"  警告: Qwen3VLForConditionalGenerationが見つかりません。AutoModelForVision2Seqを使用します。")
                        from transformers import AutoModelForVision2Seq
                        return AutoModelForVision2Seq

                else:
                    # 未知のアーキテクチャでもgenerate()メソッドを持つモデルを使用
                    print(f"  未知のアーキテクチャです。AutoModelForVision2Seqを使用します。")
                    from transformers import AutoModelForVision2Seq
                    return AutoModelForVision2Seq

            except Exception as e:
                print(f"  警告: config.jsonの読み込みに失敗しました: {e}")
                print(f"  AutoModelForVision2Seqを使用します。")
                from transformers import AutoModelForVision2Seq
                return AutoModelForVision2Seq
        else:
            print(f"  警告: config.jsonが見つかりません。AutoModelForVision2Seqを使用します。")
            from transformers import AutoModelForVision2Seq
            return AutoModelForVision2Seq

    def _get_torch_dtype(self):
        """データ型文字列をtorch dtypeに変換"""
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        return dtype_map.get(self.dtype, torch.bfloat16)
