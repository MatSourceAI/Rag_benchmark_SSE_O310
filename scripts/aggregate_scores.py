"""Aggregate original per-question metric CSVs; does NOT perform Ragas scoring.

Input: phase,configuration,run_id,question_id,faithfulness,answer_relevance,
context_relevance,context_precision,context_recall.
Strictly rejects missing/non-finite scores and duplicate question identifiers
within a phase/configuration/run. Runs are averaged equally; no silent drops.
"""
import argparse
import csv
import json
import math
import random
import statistics
from collections import defaultdict

METRICS = ['faithfulness','answer_relevance','context_relevance',
           'context_precision','context_recall']

def aggregate(rows, expected_questions=30):
    groups = defaultdict(list)
    seen = set()
    question_sets = {}
    for row in rows:
        key = tuple(row[k] for k in ['phase','configuration','run_id'])
        unique = key + (row['question_id'],)
        if unique in seen:
            raise ValueError(f'Duplicate score record: {unique}')
        seen.add(unique)
        vals = [float(row[m]) for m in METRICS]
        if not all(math.isfinite(x) and 0 <= x <= 1 for x in vals):
            raise ValueError(f'Invalid metric value: {unique}')
        groups[key].append((row['question_id'], vals))
    per_config = defaultdict(list)
    for key, entries in sorted(groups.items()):
        ids = frozenset(q for q,v in entries)
        if len(ids) != expected_questions:
            raise ValueError(f'{key}: expected {expected_questions} questions, got {len(ids)}')
        phase = key[0]
        if phase in question_sets and ids != question_sets[phase]:
            raise ValueError(f'{phase}: question IDs differ across runs/configurations')
        question_sets[phase] = ids
        means = [statistics.mean(v[i] for q,v in entries) for i in range(5)]
        per_config[key[:2]].append(dict(zip(METRICS,means),run_id=key[2],
                                             score=statistics.mean(means)))
    return [{'phase':key[0],'configuration':key[1], 'n_questions_per_run':expected_questions,
             'n_runs':len(runs), 'run_means':runs,
             'metric_means':{m:statistics.mean(r[m] for r in runs) for m in METRICS},
             'score':statistics.mean(r['score'] for r in runs)}
            for key,runs in sorted(per_config.items())]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_csv')
    parser.add_argument('output_json')
    parser.add_argument('--expected-questions',type=int,default=30)
    args=parser.parse_args()
    if args.expected_questions < 1:
        parser.error('expected-questions must be positive')
    with open(args.input_csv,newline='',encoding='utf-8') as f:
        result=aggregate(list(csv.DictReader(f)),args.expected_questions)
    with open(args.output_json,'w',encoding='utf-8') as f:
        json.dump(result,f,indent=2)
if __name__=='__main__':
    main()
