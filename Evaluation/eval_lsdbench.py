#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LSDBench eval (Qwen3-VL + vLLM, multi-choice accuracy)."""

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path

from vllm import LLM, SamplingParams
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from tqdm import tqdm


# ==================== Answer Judging ====================

def judge_multi_choice(predict_str: str, ground_truth: str) -> float:
    """Match ground-truth letter against <answer>, \\boxed{}, "X." pattern, or last A-D."""
    if ground_truth.endswith('.'):
        ground_truth = ground_truth[:-1]
    ground_truth = ground_truth.strip()

    if "<answer>" in predict_str:
        extracted = predict_str.split("<answer>")[-1].split("</answer>")[0].strip()
        if extracted:
            predict_str = extracted

    if predict_str.strip() == ground_truth:
        return 1.0

    boxed_match = re.search(r'\\boxed\{([^}]*)\}', predict_str)
    if boxed_match and boxed_match.group(1).strip() == ground_truth:
        return 1.0

    matches = re.findall(r'([A-D])\.', predict_str)
    if matches and matches[-1] == ground_truth:
        return 1.0

    matches = re.findall(r'\b([A-D])\b', predict_str)
    if matches and matches[-1] == ground_truth:
        return 1.0

    return 0.0


# ==================== Data Loading ====================

def load_lsdbench_data(data_path, video_dir):
    video_dir = Path(video_dir)
    with open(data_path, 'r') as f:
        raw_data = json.load(f)

    qa_list = []
    skipped = 0
    for idx, item in enumerate(raw_data):
        question = item["question"]
        options = item["options"]
        options_str = ""
        for opt_key, opt_text in options.items():
            options_str += f"{opt_key}. {opt_text}\n"
        full_question = (
            question + "\n" + options_str + "\n"
            'Provide your final answer as a single letter (A, B, C, or D) within <answer> </answer> tags. '
            'Your output format should be "<answer>...</answer>".'
        )

        video_id = item['video_id']

        video_path = None
        for ext in ['.mp4', '.MP4', '.mkv']:
            p = video_dir / f"{video_id}{ext}"
            if p.exists():
                video_path = str(p)
                break

        if video_path is None:
            skipped += 1
            continue

        qa_list.append({
            'id': str(idx),
            'question': full_question,
            'video_id': video_id,
            'video_path': video_path,
            'correct_answer': item['correct_answer'],
            'time_range': item.get('time_range', {}),
            'segment': item.get('segment', ''),
        })

    if skipped > 0:
        print(f"[WARN] skipped {skipped} samples (missing videos)")
    print(f"Loaded {len(qa_list)} / {len(raw_data)} samples")
    return qa_list


# ==================== Video Processing ====================

DEFAULT_FRAME_ARGS = {
    'min_pixels': 16 * 28 * 28,
    'total_pixels': 3584 * 28 * 28,
}


def build_video_input(video_path, fps=2.0, max_frames=0):
    vid_input = {'type': 'video', 'video': video_path, 'fps': fps}
    for k, v in DEFAULT_FRAME_ARGS.items():
        vid_input[k] = v
    if max_frames > 0:
        vid_input['max_frames'] = max_frames
    return vid_input


def load_video_frames(video_path, processor, fps=2.0, max_frames=0):
    vid_input = build_video_input(video_path, fps, max_frames)
    _, video_inputs, video_kwargs = process_vision_info(
        [{'content': [vid_input]}],
        image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True,
        return_video_metadata=True,
    )
    frames = video_inputs[0]
    return frames, video_kwargs


# ==================== Main ====================

def main():
    parser = argparse.ArgumentParser(description='LSDBench Evaluation (vLLM)')
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--data_path', type=str, default=None, help='test.json (or set $DATA_LSDBENCH_JSON)')
    parser.add_argument('--video_dir', type=str, default=None, help='video dir (or set $DATA_LSDBENCH_VIDEO_DIR)')
    parser.add_argument('--output_path', type=str, default='lsdbench_results.json')
    parser.add_argument('--fps', type=float, default=2.0)
    parser.add_argument('--max_frames', type=int, default=32, help='frame count cap')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_tokens', type=int, default=8192)
    parser.add_argument('--max_model_len', type=int, default=81920)
    parser.add_argument('--tensor_parallel_size', type=int, default=1)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--top_k', type=int, default=-1)
    args = parser.parse_args()

    # Defaults from paths.sh env vars; if absent, must be set via CLI.
    if args.data_path is None:
        args.data_path = os.environ.get('DATA_LSDBENCH_JSON')
    if args.video_dir is None:
        args.video_dir = os.environ.get('DATA_LSDBENCH_VIDEO_DIR')
    if args.data_path is None or args.video_dir is None:
        raise ValueError(
            "Please specify --data_path and --video_dir, "
            "or export DATA_LSDBENCH_JSON / DATA_LSDBENCH_VIDEO_DIR "
            "(see Evaluation/paths.sh)."
        )

    print(f"Loading data: {args.data_path}")
    qa_list = load_lsdbench_data(args.data_path, args.video_dir)

    # Resume from existing output
    existing_results = {}
    if os.path.exists(args.output_path):
        with open(args.output_path, 'r') as f:
            prev = json.load(f)
            for r in prev.get('results', []):
                existing_results[r['id']] = r
        print(f"Resuming: {len(existing_results)} samples already done")

    todo_list = [qa for qa in qa_list if qa['id'] not in existing_results]
    print(f"To evaluate: {len(todo_list)}")

    if len(todo_list) == 0:
        print("All done.")
        all_results = list(existing_results.values())
    else:
        print(f"Loading model: {args.ckpt_path}")
        model = LLM(
            model=args.ckpt_path,
            max_model_len=args.max_model_len,
            max_num_seqs=args.batch_size,
            tensor_parallel_size=args.tensor_parallel_size,
            dtype='bfloat16',
        )
        processor = AutoProcessor.from_pretrained(args.ckpt_path)
        sampling_params = SamplingParams(
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            top_k=args.top_k,
            stop_token_ids=[],
        )

        # Group by video so we load each clip's frames only once
        video_groups = defaultdict(list)
        for qa in todo_list:
            video_groups[qa['video_path']].append(qa)

        print(f"Processing {len(video_groups)} unique videos...")

        new_results = []
        for video_path, qa_group in tqdm(video_groups.items(), desc="Videos"):
            try:
                frames, video_kwargs = load_video_frames(
                    video_path, processor, args.fps, args.max_frames
                )
            except Exception as e:
                print(f"[ERROR] failed to load video {video_path}: {e}")
                for qa in qa_group:
                    new_results.append({
                        'id': qa['id'],
                        'video_id': qa['video_id'],
                        'question': qa['question'],
                        'prediction': '',
                        'ground_truth': qa['correct_answer'],
                        'correct': 0.0,
                        'response': f'ERROR: {e}',
                        'error': True,
                    })
                continue

            batch_inputs = []
            for qa in qa_group:
                conversation = [{
                    "role": "user",
                    "content": [
                        {"type": "video", "video": frames},
                        {"type": "text", "text": qa['question']},
                    ]
                }]

                text = processor.apply_chat_template(
                    conversation, tokenize=False, add_generation_prompt=True
                )
                batch_inputs.append({
                    'prompt': text,
                    'multi_modal_data': {'video': frames},
                    'mm_processor_kwargs': video_kwargs,
                })

            outputs = model.generate(batch_inputs, sampling_params=sampling_params)

            for qa, output in zip(qa_group, outputs):
                response = output.outputs[0].text
                correct = judge_multi_choice(response, qa['correct_answer'])

                new_results.append({
                    'id': qa['id'],
                    'video_id': qa['video_id'],
                    'question': qa['question'],
                    'prediction': response,
                    'ground_truth': qa['correct_answer'],
                    'correct': correct,
                })

        all_results = list(existing_results.values()) + new_results

    # ==================== Aggregate ====================
    total = len(all_results)
    correct = sum(1 for r in all_results if r.get('correct', 0.0) == 1.0)
    accuracy = correct / total * 100 if total > 0 else 0

    print(f"\n{'='*60}")
    print(f"LSDBench results")
    print(f"{'='*60}")
    print(f"Model: {args.ckpt_path}")
    print(f"Total: {total}, correct: {correct}, accuracy: {accuracy:.2f}%")
    print(f"{'='*60}")

    output = {
        'model': args.ckpt_path,
        'metrics': {
            'accuracy': accuracy,
            'correct': correct,
            'processed': total,
            'total': len(qa_list),
        },
        'results': all_results,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    with open(args.output_path, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {args.output_path}")


if __name__ == '__main__':
    main()
