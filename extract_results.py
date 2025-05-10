import re
import json
import os
from collections import defaultdict
import pandas as pd

def parse_log_file(log_filepath):
    """
    Parses the experiment log file to extract results.

    Args:
        log_filepath (str): Path to the log file.

    Returns:
        list: A list of dictionaries, where each dictionary represents
              the results of a single experiment run. Returns None if
              the file cannot be read.
    """
    if not os.path.exists(log_filepath):
        print(f"Error: Log file not found at {log_filepath}")
        return None

    try:
        with open(log_filepath, 'r') as f:
            log_content = f.read()
    except IOError as e:
        print(f"Error reading log file {log_filepath}: {e}")
        return None

    experiments = []
    # --- Find start markers for each experiment ---
    start_marker_pattern = re.compile(r'^-{20,}\n\n-{20,}\nRunning experiment:', re.MULTILINE)
    start_indices = [match.start() for match in start_marker_pattern.finditer(log_content)]
    first_experiment_match = re.search(r'^-{20,}\nRunning experiment:', log_content, re.MULTILINE)

    if first_experiment_match and (not start_indices or first_experiment_match.start() < start_indices[0]):
        start_indices.insert(0, first_experiment_match.start())

    if not start_indices:
        print("Error: Could not find any experiment start markers in the log file.")
        if first_experiment_match:
             print("Found the initial start marker, processing as a single block.")
             start_indices = [first_experiment_match.start()]
        else:
             return None

    end_indices = start_indices[1:] + [len(log_content)]

    # --- Process each block ---
    for block_idx, (start, end) in enumerate(zip(start_indices, end_indices)):
        block = log_content[start:end]
        header_line_match = re.search(r'Running experiment: (.*)', block)
        if not header_line_match:
             print(f"Skipping block {block_idx+1} - Could not find 'Running experiment:' line within block.")
             continue
        header_content = header_line_match.group(1)
        block_header = f"Running experiment: {header_content}"

        if "Experiment completed successfully." not in block:
            print(f"Skipping incomplete block {block_idx+1}: {block_header}")
            continue

        data = {}
        is_seq_adaptive = False

        # --- Parse the header_content ---
        match_start = re.search(
            r'Difficulty=(\w+), Sampler=(\w+), TestConfig=(\S+), (?:uncertainty_power=([\d.]+)|top_k=(\d+)), MaxSteps=(\d+)',
            header_content
        )

        if match_start:
            data['difficulty'] = match_start.group(1)
            data['sampler_base'] = match_start.group(2)
            data['test_config'] = match_start.group(3)
            uncertainty_power = match_start.group(4)
            top_k = match_start.group(5)
            data['max_steps'] = int(match_start.group(6))

            if uncertainty_power is not None:
                data['uncertainty_power'] = float(uncertainty_power)
                data['config_param'] = f"u_pow={data['uncertainty_power']:.1f}"
            elif top_k is not None:
                data['top_k'] = int(top_k)
                data['config_param'] = f"top_k={data['top_k']}"
            else:
                 data['config_param'] = "N/A"
        else:
             match_start_seq = re.search(
                r'Difficulty=(\w+), Sampler=seq_adaptive, TestConfig=(\S+), top_k=(\d+), MaxSteps=(\d+)',
                 header_content
            )
             if match_start_seq:
                 is_seq_adaptive = True
                 data['difficulty'] = match_start_seq.group(1)
                 data['sampler_base'] = 'seq_adaptive'
                 data['test_config'] = match_start_seq.group(2)
                 data['top_k'] = int(match_start_seq.group(3))
                 data['max_steps'] = int(match_start_seq.group(4))
                 data['config_param'] = f"top_k={data['top_k']}"
             else:
                print(f"Skipping block {block_idx+1} - couldn't parse header content: {header_content}")
                continue

        # --- Determine sampler_full name ---
        if is_seq_adaptive and data['test_config'].startswith('ms_') and data['test_config'].split('_')[-1].startswith('seq_adaptive'):
             data['sampler_full'] = data['test_config'].split('_')[-1]
        else:
             data['sampler_full'] = data['sampler_base']

        # --- Extract metrics ---
        metric_regex_str = rf"test/{re.escape(data['difficulty'])}/{re.escape(data['sampler_full'])}/accuracy\s+([\d.]+)"
        accuracy_match = re.search(metric_regex_str, block)

        if accuracy_match:
            metric_accuracy = float(accuracy_match.group(1))
            data['accuracy'] = metric_accuracy
            experiments.append(data)
        else:
            print(f"Skipping block {block_idx+1} - couldn't find accuracy metric for {data.get('sampler_full', 'Unknown')} in block: {block_header}")
            continue

    return experiments

def save_to_json(data, json_filepath):
    """Saves the parsed data to a JSON file."""
    try:
        with open(json_filepath, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Successfully saved parsed data to {json_filepath}")
    except IOError as e:
        print(f"Error writing JSON file {json_filepath}: {e}")
    except TypeError as e:
        print(f"Error serializing data to JSON: {e}")

def generate_latex_table(parsed_data, output_filepath):
    """
    Generates a LaTeX table from the parsed experiment data.

    Args:
        parsed_data (list): List of dictionaries from parse_log_file.
        output_filepath (str): Path to save the generated LaTeX file.
    """
    if not parsed_data:
        print("No data to generate table from.")
        return

    # --- Aggregate data for the table ---
    table_data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    for exp in parsed_data:
        sampler = exp.get('sampler_full')
        config = exp.get('config_param')
        max_steps = exp.get('max_steps')
        difficulty = exp.get('difficulty')
        accuracy = exp.get('accuracy')

        if None in [sampler, config, max_steps, difficulty, accuracy]:
            print(f"Skipping experiment with missing key data during table aggregation: {exp}")
            continue
        table_data[sampler][config][max_steps][difficulty] = accuracy

    # --- Define Sampler Name Mapping and Order ---
    sampler_map = {
        'sim_weighted_adaptive': 'SimWeightedAdaptive',
        'sim_adaptive': 'SimAdaptive',
        'seq_adaptive000': 'SeqAdaptive'
    }
    sampler_order = sorted(list(table_data.keys()))

    # --- Build LaTeX Table String ---
    latex_string = r"""\documentclass{article}
\usepackage{booktabs} % For better table lines (\toprule, \midrule, \bottomrule)
\usepackage{multirow} % For multi-row cells
\usepackage{geometry} % Adjust page margins if needed
\geometry{a4paper, margin=1in} % Or try margin=0.8in if wide

\begin{document}

\begin{table}[htbp]
\centering
\caption{Experiment Accuracy Results}
\label{tab:results}
% Adjust column spacing if needed
\setlength{\tabcolsep}{6pt} % Reduced from 8pt
{\small % Reduce font size for the table
\begin{tabular}{l l c c c c}
\toprule
\multirow{2}{*}{Sampler} & \multirow{2}{*}{Config} & \multirow{2}{*}{Max Steps} & \multicolumn{3}{c}{Accuracy ($\uparrow$)} \\
\cmidrule(lr){4-6} % Partial rule under Accuracy columns
 & & & Easy & Medium & Hard \\
\midrule
"""

    first_sampler_block = True
    for sampler_key in sampler_order:
        if not first_sampler_block:
             latex_string += "\\midrule\n"
        first_sampler_block = False

        display_name = sampler_map.get(sampler_key, sampler_key)
        configs = table_data[sampler_key]

        def sort_key(item):
            k, v = item
            if k.startswith('top_k='):
                try: return (0, int(k.split('=')[1]))
                except ValueError: return (2, 0)
            elif k.startswith('u_pow='):
                 try: return (1, float(k.split('=')[1]))
                 except ValueError: return (2, 1)
            return (2, 2)

        sorted_configs = sorted(configs.items(), key=sort_key)

        first_config_for_sampler = True
        sampler_row_count = sum(len(steps) for steps in configs.values())

        for config_param, steps_data in sorted_configs:
            sorted_steps = sorted(steps_data.items())
            first_step_for_config = True
            config_row_count = len(sorted_steps)

            for max_steps, diff_data in sorted_steps:
                easy_acc = diff_data.get('easy', '-')
                medium_acc = diff_data.get('medium', '-')
                hard_acc = diff_data.get('hard', '-')

                easy_str = f"{easy_acc:.3f}" if isinstance(easy_acc, float) else easy_acc
                medium_str = f"{medium_acc:.3f}" if isinstance(medium_acc, float) else medium_acc
                hard_str = f"{hard_acc:.3f}" if isinstance(hard_acc, float) else hard_acc

                sampler_cell = ""
                if first_config_for_sampler:
                    # Corrected: Use standard string formatting for clarity
                    sampler_cell = "\\multirow{{{}}}{{*}}{{{}}}".format(sampler_row_count, display_name)
                    first_config_for_sampler = False

                config_cell = ""
                latex_config_param = config_param.replace('_', r'\_')
                if first_step_for_config:
                     # Corrected: Use standard string formatting for clarity
                    config_cell = "\\multirow{{{}}}{{*}}{{{}}}".format(config_row_count, latex_config_param)
                    first_step_for_config = False

                latex_string += f"{sampler_cell} & {config_cell} & {max_steps} & {easy_str} & {medium_str} & {hard_str} \\\\\n"

    latex_string += r"""\bottomrule
\end{tabular}
} % End \small
\end{table}

\end{document}
"""

    # --- Save LaTeX Table ---
    try:
        with open(output_filepath, 'w') as f:
            f.write(latex_string)
        print(f"Successfully generated LaTeX table at {output_filepath}")
    except IOError as e:
        print(f"Error writing LaTeX file {output_filepath}: {e}")

# --- Main Execution ---
if __name__ == "__main__":
    log_filename = "final_experiments_2025-05-09-18-21-50.log"
    json_filename = "parsed_results.json"
    latex_filename = "results_table.tex"

    parsed_results = parse_log_file(log_filename)

    if parsed_results:
        print(f"\nTotal experiments parsed: {len(parsed_results)}")
        save_to_json(parsed_results, json_filename)
        generate_latex_table(parsed_results, latex_filename)

        print("\n--- Data Summary (Pandas DataFrame) ---")
        try:
            df = pd.DataFrame(parsed_results)
            print(f"DataFrame shape before pivot: {df.shape}")

            required_cols = ['sampler_full', 'config_param', 'max_steps', 'difficulty', 'accuracy']
            if all(c in df.columns for c in required_cols):
                 print("\nAttempting pivot table...")
                 pivot_df = df.pivot_table(
                     index=['sampler_full', 'config_param', 'max_steps'],
                     columns='difficulty',
                     values='accuracy',
                     dropna=False
                 ).reset_index()
                 print(f"Pivot DataFrame shape: {pivot_df.shape}")
                 difficulty_order = [d for d in ['easy', 'medium', 'hard'] if d in pivot_df.columns]
                 final_cols = ['sampler_full', 'config_param', 'max_steps'] + difficulty_order
                 pivot_df = pivot_df[final_cols]
                 print("\nPivoted DataFrame (full, rounded):")
                 print(pivot_df.round(3).fillna('-').to_string(index=False))
            else:
                 print("\nNot enough columns parsed correctly to display pivot summary.")
                 print(f"Available columns: {df.columns.tolist()}")
                 cols_to_print = [c for c in required_cols if c in df.columns]
                 print("\nRaw DataFrame (relevant columns, rounded):")
                 print(df[cols_to_print].round(3).to_string(index=False))

        except Exception as e:
            print(f"Could not generate pandas summary: {e}")
            if 'df' in locals():
                print("Sample of parsed data (first 5 rows):")
                print(df.head().to_string())
    else:
        print("Parsing failed, no data to process.")