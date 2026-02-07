"""
PNG画像からStable Diffusionのメタデータを抽出
"""
from pathlib import Path
from typing import Dict, Optional
from PIL import Image
import re
import json


class ImageParser:
    """PNG画像のメタデータ抽出クラス"""

    @staticmethod
    def extract_metadata(image_path: str) -> Dict:
        """
        PNGのメタデータを抽出

        Args:
            image_path: 画像ファイルパス

        Returns:
            {
                'path': str,
                'filename': str,
                'size': tuple,
                'prompt': str,
                'negative_prompt': str,
                'settings': dict  # steps, CFG, sampler等
            }
        """
        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(f"画像ファイルが見つかりません: {image_path}")

        with Image.open(image_path) as img:
            # 基本情報
            metadata = {
                'path': str(image_path),
                'filename': image_path.name,
                'size': img.size,
                'prompt': '',
                'negative_prompt': '',
                'settings': {}
            }

            # PNG infoからパラメータを取得
            if hasattr(img, 'info') and img.info:
                # ComfyUI形式の確認
                if 'prompt' in img.info:
                    try:
                        parsed = ImageParser.parse_comfyui_workflow(img.info['prompt'])
                        if parsed['prompt'] or parsed['negative_prompt']:
                            metadata.update(parsed)
                            return metadata
                    except (json.JSONDecodeError, KeyError, TypeError):
                        # ComfyUI形式ではない、次の形式を試す
                        pass

                # 一般的なキー: 'parameters', 'Description', 'UserComment'
                params_text = None

                for key in ['parameters', 'Parameters', 'Description', 'UserComment']:
                    if key in img.info:
                        params_text = img.info[key]
                        break

                if params_text:
                    parsed = ImageParser.parse_parameters(params_text)
                    metadata.update(parsed)

        return metadata

    @staticmethod
    def parse_parameters(params_text: str) -> Dict:
        """
        parametersテキストをパース

        Args:
            params_text: PNG info の 'parameters' フィールド

        Returns:
            {
                'prompt': str,
                'negative_prompt': str,
                'settings': dict
            }

        例:
            入力: "masterpiece, 1girl\nNegative prompt: bad hands\nSteps: 28, Sampler: DPM++ 2M"
            出力: {
                'prompt': 'masterpiece, 1girl',
                'negative_prompt': 'bad hands',
                'settings': {'steps': 28, 'sampler': 'DPM++ 2M'}
            }
        """
        result = {
            'prompt': '',
            'negative_prompt': '',
            'settings': {}
        }

        if not params_text:
            return result

        # 行ごとに分割
        lines = params_text.strip().split('\n')

        # プロンプト（最初の行、Negative promptより前）
        prompt_lines = []
        settings_line = None

        for i, line in enumerate(lines):
            if line.lower().startswith('negative prompt:'):
                # Negative promptが見つかった
                negative_part = line.split(':', 1)[1].strip()
                result['negative_prompt'] = negative_part
            elif re.match(r'^\s*(Steps|Seed|Size|Model|CFG scale|Sampler)', line, re.IGNORECASE):
                # 設定行
                settings_line = line
                break
            else:
                # プロンプト行
                prompt_lines.append(line)

        result['prompt'] = '\n'.join(prompt_lines).strip()

        # 設定のパース
        if settings_line:
            result['settings'] = ImageParser._parse_settings_line(settings_line)

        return result

    @staticmethod
    def _parse_settings_line(settings_line: str) -> Dict:
        """
        設定行をパース

        Args:
            settings_line: "Steps: 28, Sampler: DPM++ 2M, CFG scale: 7"

        Returns:
            {'steps': 28, 'sampler': 'DPM++ 2M', 'cfg_scale': 7}
        """
        settings = {}

        # カンマで分割
        parts = settings_line.split(',')

        for part in parts:
            part = part.strip()
            if ':' not in part:
                continue

            key, value = part.split(':', 1)
            key = key.strip().lower().replace(' ', '_')
            value = value.strip()

            # 数値に変換を試みる
            try:
                if '.' in value:
                    value = float(value)
                else:
                    value = int(value)
            except ValueError:
                pass  # 文字列のまま

            settings[key] = value

        return settings

    @staticmethod
    def parse_comfyui_workflow(workflow_json: str) -> Dict:
        """
        ComfyUIのワークフローJSONからプロンプトを抽出

        Args:
            workflow_json: PNG info の 'prompt' フィールド（JSON文字列）

        Returns:
            {
                'prompt': str,
                'negative_prompt': str,
                'settings': dict
            }
        """
        result = {
            'prompt': '',
            'negative_prompt': '',
            'settings': {}
        }

        # JSONをパース
        workflow = json.loads(workflow_json)

        # CLIPTextEncodeノードを検索
        text_nodes = {}
        sampler_node = None

        for node_id, node_data in workflow.items():
            class_type = node_data.get('class_type', '')

            # CLIPTextEncodeノードを保存
            if 'CLIPTextEncode' in class_type:
                inputs = node_data.get('inputs', {})
                if 'text' in inputs:
                    text_nodes[node_id] = inputs['text']

            # KSamplerノードを検索（正/負のプロンプトを判別するため）
            elif class_type in ['KSampler', 'KSamplerAdvanced', 'SamplerCustom']:
                sampler_node = node_data

        # KSamplerノードから正/負のプロンプトを判別
        if sampler_node and text_nodes:
            inputs = sampler_node.get('inputs', {})

            # positive/negative入力を確認
            positive_input = inputs.get('positive', [])
            negative_input = inputs.get('negative', [])

            # 入力形式: ["node_id", output_index]
            if isinstance(positive_input, list) and len(positive_input) >= 1:
                pos_node_id = str(positive_input[0])
                if pos_node_id in text_nodes:
                    result['prompt'] = text_nodes[pos_node_id]

            if isinstance(negative_input, list) and len(negative_input) >= 1:
                neg_node_id = str(negative_input[0])
                if neg_node_id in text_nodes:
                    result['negative_prompt'] = text_nodes[neg_node_id]

        # KSamplerが見つからない場合、すべてのCLIPTextEncodeノードを結合
        elif text_nodes:
            all_texts = list(text_nodes.values())
            if len(all_texts) >= 2:
                # 最初をpositive、2番目をnegativeと仮定
                result['prompt'] = all_texts[0]
                result['negative_prompt'] = all_texts[1]
            elif len(all_texts) == 1:
                result['prompt'] = all_texts[0]

        # サンプラー設定を抽出
        if sampler_node:
            inputs = sampler_node.get('inputs', {})
            result['settings'] = {
                'steps': inputs.get('steps'),
                'cfg_scale': inputs.get('cfg'),
                'sampler': inputs.get('sampler_name'),
                'scheduler': inputs.get('scheduler'),
                'seed': inputs.get('seed'),
                'denoise': inputs.get('denoise'),
            }
            # None値を削除
            result['settings'] = {k: v for k, v in result['settings'].items() if v is not None}

        return result
