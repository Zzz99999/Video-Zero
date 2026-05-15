#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""NExT-GQA eval — VideoQA accuracy + temporal grounding (mIoU / R@IoU / GQA@IoU)."""

import argparse
import csv
import json
import os
import re
from collections import defaultdict

from vllm import LLM, SamplingParams
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from tqdm import tqdm


# ==================== Answer Judging ====================

OPTION_LABELS = ['A', 'B', 'C', 'D', 'E']

def judge_multi_choice(predict_str: str, gt_label: str) -> float:
    """Match ground-truth letter (A-E) against <answer>, \\boxed{}, "X." pattern, or last A-E."""
    if gt_label.endswith('.'):
        gt_label = gt_label[:-1]
    gt_label = gt_label.strip().upper()

    if "<answer>" in predict_str:
        extracted = predict_str.split("<answer>")[-1].split("</answer>")[0].strip()
        if extracted:
            predict_str = extracted

    if predict_str.strip().upper() == gt_label:
        return 1.0

    boxed_match = re.search(r'\\boxed\{([^}]*)\}', predict_str)
    if boxed_match and boxed_match.group(1).strip().upper() == gt_label:
        return 1.0

    matches = re.findall(r'([A-E])\.', predict_str.upper())
    if matches and matches[-1] == gt_label:
        return 1.0

    matches = re.findall(r'\b([A-E])\b', predict_str.upper())
    if matches and matches[-1] == gt_label:
        return 1.0

    return 0.0


def compute_iou_1d(pred_span, gt_span):
    inter_start = max(pred_span[0], gt_span[0])
    inter_end = min(pred_span[1], gt_span[1])
    inter = max(0, inter_end - inter_start)
    union = (pred_span[1] - pred_span[0]) + (gt_span[1] - gt_span[0]) - inter
    return inter / union if union > 0 else 0.0


def parse_time_spans(text):
    """Parse [start,end] / "start-end" / "from start to end (s)" patterns from model output."""
    if "<answer>" in text:
        extracted = text.split("<answer>")[-1].split("</answer>")[0].strip()
        if extracted:
            text = extracted

    spans = []
    bracket_matches = re.findall(r'\[?\s*(\d+\.?\d*)\s*[,\-~]\s*(\d+\.?\d*)\s*\]?', text)
    if bracket_matches:
        for s, e in bracket_matches:
            start, end = float(s), float(e)
            if end > start:
                spans.append([start, end])

    if not spans:
        to_matches = re.findall(r'(\d+\.?\d*)\s*(?:s|seconds?)?\s*(?:to|-)\s*(\d+\.?\d*)\s*(?:s|seconds?)?', text)
        for s, e in to_matches:
            start, end = float(s), float(e)
            if end > start:
                spans.append([start, end])

    return spans


def compute_grounding_metrics(pred_spans, gt_spans, iou_thresholds=[0.3, 0.5]):
    """mIoU averaged over GT spans (best-match per GT); R@t = max-IoU >= threshold."""
    if not pred_spans or not gt_spans:
        return {'mIoU': 0.0, **{f'R@{t}': 0.0 for t in iou_thresholds}}

    best_ious = []
    for gt_span in gt_spans:
        best_iou = max(compute_iou_1d(pred, gt_span) for pred in pred_spans)
        best_ious.append(best_iou)

    mean_iou = sum(best_ious) / len(best_ious)
    max_iou = max(best_ious)

    result = {'mIoU': mean_iou}
    for t in iou_thresholds:
        result[f'R@{t}'] = 1.0 if max_iou >= t else 0.0
    return result


# ==================== Data Loading ====================

def load_nextgqa_data(csv_path, gsub_path, map_path, video_root):
    """Load NExT-GQA QAs (CSV + grounding annotations + video_id->path map)."""
    with open(map_path, 'r') as f:
        vid_map = json.load(f)
    with open(gsub_path, 'r') as f:
        gsub = json.load(f)

    qa_list = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            video_id = row['video_id']
            qid = row['qid']
            question = row['question']
            answer_text = row['answer']
            q_type = row['type']

            options = [row[f'a{i}'] for i in range(5)]

            gt_label_idx = None
            for i, opt in enumerate(options):
                if opt.strip().lower() == answer_text.strip().lower():
                    gt_label_idx = i
                    break
            gt_label = OPTION_LABELS[gt_label_idx] if gt_label_idx is not None else '?'

            subpath = vid_map.get(video_id, None)
            if subpath is None:
                continue
            video_path = os.path.join(video_root, subpath + '.mp4')
            if not os.path.exists(video_path):
                # Fallback: try basename in case the dataset is laid out flat
                alt_path = os.path.join(video_root, subpath.split('/')[-1] + '.mp4')
                if os.path.exists(alt_path):
                    video_path = alt_path
                else:
                    continue

            gt_spans = []
            duration = 0.0
            if video_id in gsub:
                vid_info = gsub[video_id]
                duration = vid_info.get('duration', 0)
                loc = vid_info.get('location', {})
                if qid in loc:
                    gt_spans = loc[qid]

            uid = f"{video_id}_{qid}"
            qa_list.append({
                'uid': uid,
                'video_id': video_id,
                'video_path': video_path,
                'qid': qid,
                'question': question,
                'gt_answer_text': answer_text,
                'gt_label': gt_label,
                'question_type': q_type,
                'options': options,
                'gt_spans': gt_spans,
                'duration': duration,
            })

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


# ==================== Prompt Templates ====================

QA_TEMPLATE = (
    'Watch the video and answer the following question.\n\n'
    'Question: {question}\n\n'
    'Options:\n'
    'A. {a0}\n'
    'B. {a1}\n'
    'C. {a2}\n'
    'D. {a3}\n'
    'E. {a4}\n\n'
    'Provide your final answer as a single letter (A, B, C, D, or E) within <answer> </answer> tags. '
    'Your output format should be "<answer>...</answer>".'
)

GROUNDING_TEMPLATE = (
    'Watch the video and answer the following question.\n\n'
    'Question: {question}\n\n'
    'Options:\n'
    'A. {a0}\n'
    'B. {a1}\n'
    'C. {a2}\n'
    'D. {a3}\n'
    'E. {a4}\n\n'
    'Provide your final answer as a single letter (A, B, C, D, or E) within <answer> </answer> tags. '
    'Then, on a new line, provide the relevant time span in seconds within <grounding> </grounding> tags.\n\n'
    'Your output format should be:\n'
    '<answer>A</answer>\n'
    '<grounding>[start, end]</grounding>'
)


# ==================== Main ====================

def main():
    parser = argparse.ArgumentParser(description='NExT-GQA Evaluation')
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True, help='datasets/nextgqa dir')
    parser.add_argument('--video_dir', type=str, required=True, help='NExTVideo/ root')
    parser.add_argument('--output_path', type=str, default='nextgqa_results.json')
    parser.add_argument('--split', type=str, default='test', choices=['test', 'val'])
    parser.add_argument('--mode', type=str, default='qa', choices=['qa', 'grounding', 'both'],
                        help='qa = QA only, grounding = QA+temporal, both = same prompt as grounding')
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

    csv_path = os.path.join(args.data_dir, f'{args.split}.csv')
    gsub_path = os.path.join(args.data_dir, f'gsub_{args.split}.json')
    map_path = os.path.join(args.data_dir, 'map_vid_vidorID.json')

    print(f"Loading data: {csv_path}")
    qa_list = load_nextgqa_data(csv_path, gsub_path, map_path, args.video_dir)
    print(f"Loaded {len(qa_list)} QAs ({sum(1 for q in qa_list if q['gt_spans'])} with grounding annotations)")

    # Resume from existing output
    existing_results = {}
    if os.path.exists(args.output_path):
        with open(args.output_path, 'r') as f:
            prev = json.load(f)
            for r in prev.get('results', []):
                existing_results[r['uid']] = r
        print(f"Resuming: {len(existing_results)} samples already done")

    todo_list = [qa for qa in qa_list if qa['uid'] not in existing_results]
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

        # Group by video so each clip is loaded only once
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
                        'uid': qa['uid'],
                        'video_id': qa['video_id'],
                        'qid': qa['qid'],
                        'question': qa['question'],
                        'gt_label': qa['gt_label'],
                        'gt_answer_text': qa['gt_answer_text'],
                        'question_type': qa['question_type'],
                        'prediction': '',
                        'pred_label': '',
                        'qa_correct': 0.0,
                        'gt_spans': qa['gt_spans'],
                        'pred_spans': [],
                        'grounding_mIoU': 0.0,
                        'grounding_R@0.3': 0.0,
                        'grounding_R@0.5': 0.0,
                        'error': str(e),
                    })
                continue

            batch_inputs = []
            for qa in qa_group:
                if args.mode == 'grounding' or args.mode == 'both':
                    prompt_text = GROUNDING_TEMPLATE.format(
                        question=qa['question'],
                        a0=qa['options'][0], a1=qa['options'][1],
                        a2=qa['options'][2], a3=qa['options'][3],
                        a4=qa['options'][4],
                    )
                else:
                    prompt_text = QA_TEMPLATE.format(
                        question=qa['question'],
                        a0=qa['options'][0], a1=qa['options'][1],
                        a2=qa['options'][2], a3=qa['options'][3],
                        a4=qa['options'][4],
                    )

                conversation = [{
                    "role": "user",
                    "content": [
                        {"type": "video", "video": frames},
                        {"type": "text", "text": prompt_text},
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
                pred_text = output.outputs[0].text

                # QA judging
                pred_label = ''
                if "<answer>" in pred_text:
                    extracted = pred_text.split("<answer>")[-1].split("</answer>")[0].strip()
                    pred_label = extracted.upper()[:1] if extracted else ''

                qa_correct = judge_multi_choice(pred_text, qa['gt_label'])

                # Grounding judging
                pred_spans = []
                grounding_metrics = {'mIoU': 0.0, 'R@0.3': 0.0, 'R@0.5': 0.0}

                if args.mode in ['grounding', 'both']:
                    grounding_text = pred_text
                    if "<grounding>" in pred_text:
                        grounding_text = pred_text.split("<grounding>")[-1].split("</grounding>")[0]

                    pred_spans = parse_time_spans(grounding_text)

                    if qa['gt_spans'] and pred_spans:
                        grounding_metrics = compute_grounding_metrics(pred_spans, qa['gt_spans'])

                result = {
                    'uid': qa['uid'],
                    'video_id': qa['video_id'],
                    'qid': qa['qid'],
                    'question': qa['question'],
                    'gt_label': qa['gt_label'],
                    'gt_answer_text': qa['gt_answer_text'],
                    'question_type': qa['question_type'],
                    'prediction': pred_text,
                    'pred_label': pred_label,
                    'qa_correct': qa_correct,
                    'gt_spans': qa['gt_spans'],
                    'pred_spans': pred_spans,
                    'grounding_mIoU': grounding_metrics['mIoU'],
                    'grounding_R@0.3': grounding_metrics['R@0.3'],
                    'grounding_R@0.5': grounding_metrics['R@0.5'],
                }
                new_results.append(result)

        all_results = list(existing_results.values()) + new_results

    # ==================== Aggregate ====================
    total = len(all_results)

    # --- QA Accuracy ---
    qa_correct_count = sum(1 for r in all_results if r['qa_correct'] == 1.0)
    qa_accuracy = qa_correct_count / total * 100 if total > 0 else 0

    type_stats = defaultdict(lambda: {'total': 0, 'correct': 0})
    for r in all_results:
        qt = r['question_type']
        type_stats[qt]['total'] += 1
        type_stats[qt]['correct'] += int(r['qa_correct'] == 1.0)

    # --- Grounding (over samples with gt_spans) ---
    grounded = [r for r in all_results if r.get('gt_spans')]
    g_total = len(grounded)
    g_miou = sum(r['grounding_mIoU'] for r in grounded) / g_total if g_total > 0 else 0
    g_r03 = sum(r['grounding_R@0.3'] for r in grounded) / g_total * 100 if g_total > 0 else 0
    g_r05 = sum(r['grounding_R@0.5'] for r in grounded) / g_total * 100 if g_total > 0 else 0

    # --- GQA: QA correct AND grounding R@t ---
    gqa_r03 = sum(1 for r in grounded if r['qa_correct'] == 1.0 and r['grounding_R@0.3'] == 1.0) / g_total * 100 if g_total > 0 else 0
    gqa_r05 = sum(1 for r in grounded if r['qa_correct'] == 1.0 and r['grounding_R@0.5'] == 1.0) / g_total * 100 if g_total > 0 else 0

    type_grounding = defaultdict(lambda: {'total': 0, 'mIoU_sum': 0.0, 'R@0.3': 0, 'R@0.5': 0,
                                           'GQA@0.3': 0, 'GQA@0.5': 0})
    for r in grounded:
        qt = r['question_type']
        type_grounding[qt]['total'] += 1
        type_grounding[qt]['mIoU_sum'] += r['grounding_mIoU']
        type_grounding[qt]['R@0.3'] += int(r['grounding_R@0.3'] == 1.0)
        type_grounding[qt]['R@0.5'] += int(r['grounding_R@0.5'] == 1.0)
        type_grounding[qt]['GQA@0.3'] += int(r['qa_correct'] == 1.0 and r['grounding_R@0.3'] == 1.0)
        type_grounding[qt]['GQA@0.5'] += int(r['qa_correct'] == 1.0 and r['grounding_R@0.5'] == 1.0)

    print(f"\n{'='*70}")
    print(f"NExT-GQA results ({args.split} set)")
    print(f"{'='*70}")
    print(f"Model: {args.ckpt_path}")
    print(f"Mode:  {args.mode}")

    print(f"\n--- VideoQA Accuracy ---")
    print(f"Total: {total}, correct: {qa_correct_count}, acc: {qa_accuracy:.2f}%")
    print(f"\n  By question type:")
    for qt in sorted(type_stats.keys()):
        s = type_stats[qt]
        acc = s['correct'] / s['total'] * 100 if s['total'] > 0 else 0
        print(f"    {qt:6s}: {s['correct']:4d}/{s['total']:4d} = {acc:.2f}%")

    if args.mode in ['grounding', 'both']:
        print(f"\n--- Temporal Grounding ---")
        print(f"Samples with grounding annotations: {g_total}")
        print(f"  mIoU:    {g_miou:.4f}")
        print(f"  R@0.3:   {g_r03:.2f}%")
        print(f"  R@0.5:   {g_r05:.2f}%")

        print(f"\n--- Grounded QA (QA correct + Grounding) ---")
        print(f"  GQA@0.3: {gqa_r03:.2f}%")
        print(f"  GQA@0.5: {gqa_r05:.2f}%")

        print(f"\n  By question type (grounding):")
        for qt in sorted(type_grounding.keys()):
            s = type_grounding[qt]
            miou = s['mIoU_sum'] / s['total'] if s['total'] > 0 else 0
            r03 = s['R@0.3'] / s['total'] * 100 if s['total'] > 0 else 0
            r05 = s['R@0.5'] / s['total'] * 100 if s['total'] > 0 else 0
            gqa03 = s['GQA@0.3'] / s['total'] * 100 if s['total'] > 0 else 0
            gqa05 = s['GQA@0.5'] / s['total'] * 100 if s['total'] > 0 else 0
            print(f"    {qt:6s}: mIoU={miou:.4f}  R@0.3={r03:.1f}%  R@0.5={r05:.1f}%  "
                  f"GQA@0.3={gqa03:.1f}%  GQA@0.5={gqa05:.1f}%  (n={s['total']})")

    output = {
        'model': args.ckpt_path,
        'split': args.split,
        'mode': args.mode,
        'total': total,
        'qa_accuracy': qa_accuracy,
        'qa_correct': qa_correct_count,
        'per_question_type': {
            qt: {'total': s['total'], 'correct': s['correct'],
                 'accuracy': s['correct'] / s['total'] * 100 if s['total'] > 0 else 0}
            for qt, s in type_stats.items()
        },
    }

    if args.mode in ['grounding', 'both']:
        output['grounding'] = {
            'total_grounded': g_total,
            'mIoU': g_miou,
            'R@0.3': g_r03,
            'R@0.5': g_r05,
            'GQA@0.3': gqa_r03,
            'GQA@0.5': gqa_r05,
        }
        output['per_type_grounding'] = {
            qt: {
                'total': s['total'],
                'mIoU': s['mIoU_sum'] / s['total'] if s['total'] > 0 else 0,
                'R@0.3': s['R@0.3'] / s['total'] * 100 if s['total'] > 0 else 0,
                'R@0.5': s['R@0.5'] / s['total'] * 100 if s['total'] > 0 else 0,
                'GQA@0.3': s['GQA@0.3'] / s['total'] * 100 if s['total'] > 0 else 0,
                'GQA@0.5': s['GQA@0.5'] / s['total'] * 100 if s['total'] > 0 else 0,
            }
            for qt, s in type_grounding.items()
        }

    output['results'] = all_results

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    with open(args.output_path, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {args.output_path}")


if __name__ == '__main__':
    main()
