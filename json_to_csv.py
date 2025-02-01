import json
import csv
import statistics
import argparse
import os
from pathlib import Path

def process_results(json_file):
    # Read and parse JSON
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    results = data['results']
    
    # Extract MTBench Average
    mtbench_avg = results['MTBench']['Average']
    
    # Extract Alpaca Eval Length Controlled Winrate
    alpaca_winrate = results['alpaca_eval']['length_controlled_winrate']

    # Extract MMLU Pro
    mmlu_pro = results['leaderboard_mmlu_pro']['acc,none']

    # Extract MixEval
    mixeval = results['MixEval']['gpt-4o-mini-2024-07-18']['metrics']["overall"]

    # Extract MBPP
    mbpp = results['MBPP']['pass@1']
    
    # Calculate average BBH score
    bbh_scores = []
    for key, value in results.items():
        if key.startswith('leaderboard_bbh_') and isinstance(value, dict) and 'acc_norm,none' in value:
            bbh_scores.append(value['acc_norm,none'])
    bbh_avg = statistics.mean(bbh_scores)
    
    # Calculate average GPQA score
    gpqa_scores = []
    for key in ['leaderboard_gpqa_diamond', 'leaderboard_gpqa_extended', 'leaderboard_gpqa_main']:
        if key in results:
            gpqa_scores.append(results[key]['acc_norm,none'])
    gpqa_avg = statistics.mean(gpqa_scores)
    
    # Calculate average MATH score
    math_scores = []
    for key, value in results.items():
        if key.startswith('leaderboard_math_') and isinstance(value, dict) and 'exact_match,none' in value:
            math_scores.append(value['exact_match,none'])
    math_avg = statistics.mean(math_scores)
    
    # Calculate average MUSR score
    musr_scores = []
    for key, value in results.items():
        if key.startswith('leaderboard_musr_') and isinstance(value, dict) and 'acc_norm,none' in value:
            musr_scores.append(value['acc_norm,none'])
    musr_avg = statistics.mean(musr_scores)
    
    # Extract IFEval average (using prompt-level strict accuracy)
    ifeval_score = results['leaderboard_ifeval']['prompt_level_strict_acc,none']
    ifeval_second_score = results['leaderboard_ifeval']['inst_level_loose_acc,none']
    
    # Create output dictionary
    output = {
        'MTBench': mtbench_avg,
        'Alpaca Eval (LC)': alpaca_winrate,
        'BBH': bbh_avg,
        'GPQA': gpqa_avg,
        'MATH': math_avg,
        'MUSR': musr_avg,
        'IFEval (Prompt Level, Strict)': ifeval_score,
        'IFEval (Instance Level, Loose)': ifeval_second_score,
        'MMLU Pro': mmlu_pro,
        'MixEval': mixeval,
        'MBPP': mbpp,
    }
    
    return output

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process evaluation results JSON to CSV')
    parser.add_argument('--json_path', help='Path to the JSON file to process')
    args = parser.parse_args()
    
    # Convert path to Path object and resolve it
    json_path = Path(args.json_path).resolve()
    
    # Ensure the JSON file exists
    if not json_path.exists():
        print(f"Error: File not found: {json_path}")
        return
    
    try:
        # Process the results
        results = process_results(json_path)
        
        # Create output path with same name but .csv extension
        csv_path = json_path.with_suffix('.csv')
        
        # Write to CSV
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Metric', 'Value'])
            for metric, value in results.items():
                writer.writerow([metric, round(value, 4)])
        
        print(f"\nResults have been saved to: {csv_path}")
        print("\nSummary of results:")
        for metric, value in results.items():
            print(f"{metric}: {round(value, 4)}")
            
    except Exception as e:
        print(f"Error processing file: {e}")

if __name__ == '__main__':
    main()