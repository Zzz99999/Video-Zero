#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""VNBench (VideoNIAH) eval — 4-try strict score across 9 sub-tasks (ret / ord / cnt)."""

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


# ==================== VNBench Metrics ====================

def compute_vnbench_metrics(results):
    """Official VNBench 4-try logic: a question counts as correct iff all 4 tries hit."""
    group_correct = defaultdict(int)
    group_type = {}

    for r in results:
        vp = r['video_path']
        group_correct[vp] += int(r.get('correct', 0.0) == 1.0)
        group_type[vp] = r['type']

    type_correct = defaultdict(int)
    type_total = defaultdict(int)

    for vp, cnt in group_correct.items():
        t = group_type[vp]
        type_total[t] += 1
        if cnt == 4:
            type_correct[t] += 1

    metrics = {}
    for t in sorted(type_total.keys()):
        acc = type_correct[t] / type_total[t] if type_total[t] > 0 else 0.0
        metrics[t] = acc

    cat_scores = {'ret': 0.0, 'ord': 0.0, 'cnt': 0.0}
    cat_count = {'ret': 0, 'ord': 0, 'cnt': 0}

    for t, acc in metrics.items():
        for cat in ['ret', 'ord', 'cnt']:
            if cat in t:
                cat_scores[cat] += acc
                cat_count[cat] += 1
                break

    for cat in cat_scores:
        if cat_count[cat] > 0:
            cat_scores[cat] /= cat_count[cat]

    metrics.update(cat_scores)
    metrics['Overall'] = (cat_scores['ret'] + cat_scores['ord'] + cat_scores['cnt']) / 3

    return metrics


# ==================== Data Loading ====================

def load_vnbench_data(annotation_path, video_dir):
    video_dir = Path(video_dir)
    with open(annotation_path, 'r') as f:
        raw_data = json.load(f)

    qa_list = []
    skipped = 0
    for idx, item in enumerate(raw_data):
        question = item['question']
        options = item['options']
        full_question = (
            f"{question}\n"
            f"A. {options[0]}\n"
            f"B. {options[1]}\n"
            f"C. {options[2]}\n"
            f"D. {options[3]}\n\n"
            f"Provide your final answer as a single letter (A, B, C, or D) within <answer> </answer> tags. "
            f'Your output format should be "<answer>...</answer>".'
        )

        gt_option = item.get('gt_option', '')
        if not gt_option:
            gt_text = item['gt']
            try:
                gt_idx = options.index(gt_text)
                gt_option = chr(ord('A') + gt_idx)
            except ValueError:
                gt_option = 'A'

        video_basename = os.path.basename(item['video'])
        video_path = video_dir / video_basename

        if not video_path.exists():
            skipped += 1
            continue

        video_stem = video_basename.replace('.mp4', '')
        question_id = f"{video_stem}_{item['try']}"

        qa_list.append({
            'id': str(idx),
            'question_id': question_id,
            'question': full_question,
            'video_path': str(video_path),
            'video_rel': item['video'],
            'gt_option': gt_option,
            'gt_text': item.get('gt', ''),
            'type': item['type'],
            'try': item['try'],
            'length': item.get('length', 0),
            'needle_time': item.get('needle_time', []),
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


def build_video_input(video_path, nframes=0, fps=2.0, max_frames=0):
    vid_input = {'type': 'video', 'video': video_path}
    for k, v in DEFAULT_FRAME_ARGS.items():
        vid_input[k] = v

    if nframes > 0:
        vid_input['nframes'] = nframes
    else:
        vid_input['fps'] = fps
        if max_frames > 0:
            vid_input['max_frames'] = max_frames

    return vid_input


def load_video_frames(video_path, processor, nframes=0, fps=2.0, max_frames=0):
    vid_input = build_video_input(video_path, nframes, fps, max_frames)
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
    parser = argparse.ArgumentParser(description='VNBench Evaluation (vLLM)')
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--annotation_path', type=str, default=None,
                        help='VNBench-main-4try.json (or set $DATA_VNBENCH_ANNO)')
    parser.add_argument('--video_dir', type=str, default=None,
                        help='video dir (or set $DATA_VNBENCH_VIDEO_DIR)')
    parser.add_argument('--output_path', type=str, default='vnbench_results.json')
    parser.add_argument('--nframes', type=int, default=0, help='fixed frames; 0 = use fps mode')
    parser.add_argument('--fps', type=float, default=2.0)
    parser.add_argument('--max_frames', type=int, default=32, help='cap when nframes=0')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_tokens', type=int, default=8192)
    parser.add_argument('--max_model_len', type=int, default=81920)
    parser.add_argument('--tensor_parallel_size', type=int, default=1)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--top_k', type=int, default=-1)
    args = parser.parse_args()

    # Defaults from paths.sh env vars; if absent, must be set via CLI.
    if args.annotation_path is None:
        args.annotation_path = os.environ.get('DATA_VNBENCH_ANNO')
    if args.video_dir is None:
        args.video_dir = os.environ.get('DATA_VNBENCH_VIDEO_DIR')
    if args.annotation_path is None or args.video_dir is None:
        raise ValueError(
            "Please specify --annotation_path and --video_dir, "
            "or export DATA_VNBENCH_ANNO / DATA_VNBENCH_VIDEO_DIR "
            "(see Evaluation/paths.sh)."
        )

    print(f"Loading data: {args.annotation_path}")
    qa_list = load_vnbench_data(args.annotation_path, args.video_dir)

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

        # Group by video so we load each clip's frames only once (4 tries share)
        video_groups = defaultdict(list)
        for qa in todo_list:
            video_groups[qa['video_path']].append(qa)

        print(f"Processing {len(video_groups)} unique videos...")

        new_results = []
        for video_path, qa_group in tqdm(video_groups.items(), desc="Videos"):
            try:
                frames, video_kwargs = load_video_frames(
                    video_path, processor, args.nframes, args.fps, args.max_frames
                )
            except Exception as e:
                print(f"[ERROR] failed to load video {video_path}: {e}")
                for qa in qa_group:
                    new_results.append({
                        'id': qa['id'],
                        'question_id': qa['question_id'],
                        'video_path': qa['video_rel'],
                        'question': qa['question'],
                        'prediction': '',
                        'gt_option': qa['gt_option'],
                        'correct': 0.0,
                        'type': qa['type'],
                        'try': qa['try'],
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
                correct = judge_multi_choice(response, qa['gt_option'])

                new_results.append({
                    'id': qa['id'],
                    'question_id': qa['question_id'],
                    'video_path': qa['video_rel'],
                    'question': qa['question'],
                    'prediction': response,
                    'gt_option': qa['gt_option'],
                    'correct': correct,
                    'type': qa['type'],
                    'try': qa['try'],
                })

        all_results = list(existing_results.values()) + new_results

    # ==================== Aggregate ====================
    metrics = compute_vnbench_metrics(all_results)

    print(f"\n{'='*60}")
    print(f"VNBench results (4-try strict)")
    print(f"{'='*60}")
    print(f"Model: {args.ckpt_path}")

    subtasks = [k for k in sorted(metrics.keys()) if k not in ('ret', 'ord', 'cnt', 'Overall')]
    for t in subtasks:
        print(f"  {t:20s}: {metrics[t]*100:6.2f}%")

    print(f"{'-'*60}")
    for cat in ['ret', 'ord', 'cnt']:
        if cat in metrics:
            print(f"  {cat:20s}: {metrics[cat]*100:6.2f}%")

    print(f"{'-'*60}")
    print(f"  {'Overall':20s}: {metrics.get('Overall', 0)*100:6.2f}%")
    print(f"{'='*60}")

    output = {
        'model': args.ckpt_path,
        'metrics': {k: v * 100 for k, v in metrics.items()},
        'processed': len(all_results),
        'total': len(qa_list),
        'results': all_results,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    with open(args.output_path, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {args.output_path}")

    # Also dump JSONL (compatible with the official eval.py input format).
    jsonl_path = args.output_path.replace('.json', '.jsonl')
    with open(jsonl_path, 'w') as f:
        for r in all_results:
            line = {
                'question_id': r['question_id'],
                'prediction': r['prediction'],
                'gt': r['gt_option'],
                'correct': r['correct'],
                'type': r['type'],
                'try': r['try'],
                'video_path': r['video_path'],
            }
            f.write(json.dumps(line, ensure_ascii=False) + '\n')
    print(f"JSONL saved: {jsonl_path}")


if __name__ == '__main__':
    main()
