"""
Script to extract surprising sentences above a threshold with their 4 neighboring sentences.

This script finds sentences with surprisingBin scores above a configurable threshold (default 0.4),
then for each one collects 4 neighboring sentences (2 before + 2 after when possible).
Results are saved as JSON with sentences in chronological order within each group.
"""

import os
import json
import pandas as pd
from dataclasses import dataclass

from src.configs import *


@dataclass
class SurprisingConfig:
    surprisal_thres: float
    neighbors_count: int

    env: EnvironmentConfig
    data: DatasetConfig
    llm: LLMConfig


def load_hippocorpus_raw_data():
    """Load and properly aggregate the hippocorpus CSV data."""
    csv_path = os.path.join("artifacts", "text_inputs", "hippocorpus-u20220112", "hcV3-eventAnnotsAggOverWorkers.csv")
    df = pd.read_csv(csv_path)

    print(f"Raw data shape: {df.shape}")
    print(f"Unique sentences: {len(df[['storyIx', 'sentIx']].drop_duplicates())}")

    # Aggregate by taking mean of scores for each unique sentence
    label_columns = ['majorBin', 'minorBin', 'expectedBin', 'surprisingBin', 'eventOrNot']

    # Group by sentence identifier and aggregate
    agg_dict = {
        'sent': 'first',  # Take first occurrence of sentence text
        'memType': 'first',  # Take first occurrence of memory type
        **{col: 'mean' for col in label_columns}  # Average the label scores
    }

    df_agg = df.groupby(['storyIx', 'sentIx']).agg(agg_dict).reset_index()

    print(f"Aggregated data shape: {df_agg.shape}")
    print(f"Each sentence now has unique (storyIx, sentIx): {len(df_agg) == len(df_agg[['storyIx', 'sentIx']].drop_duplicates())}")

    return df_agg


def get_neighbors(df, target_row, neighbors_count=4):
    """
    Get neighboring sentences for a target sentence.

    Args:
        df: DataFrame with all sentences (should be pre-aggregated with unique sentences)
        target_row: The target sentence row
        neighbors_count: Total number of neighbors to collect (default 4)

    Returns:
        List of sentence dictionaries in chronological order
    """
    story_idx = target_row['storyIx']
    sent_idx = target_row['sentIx']

    # Get all sentences from the same story, sorted by sentIx
    story_sentences = df[df['storyIx'] == story_idx].sort_values('sentIx')

    # Find the position of target sentence in the story
    target_pos = None
    for i, (_, row) in enumerate(story_sentences.iterrows()):
        if row['sentIx'] == sent_idx:
            target_pos = i
            break

    if target_pos is None:
        print(f"ERROR: Could not find target sentence with sentIx={sent_idx} in story {story_idx}")
        return []

    # Calculate how many sentences to take before and after
    total_sentences = len(story_sentences)

    # Ideally 2 before and 2 after, but adjust based on story boundaries
    before_count = min(2, target_pos)
    after_count = min(2, total_sentences - target_pos - 1)

    # If we can't get enough from one side, take more from the other
    remaining = neighbors_count - before_count - after_count
    if remaining > 0:
        if before_count < 2 and target_pos > before_count:
            # Can take more from before
            additional_before = min(remaining, target_pos - before_count)
            before_count += additional_before
            remaining -= additional_before

        if remaining > 0 and after_count < 2 and (total_sentences - target_pos - 1) > after_count:
            # Can take more from after
            additional_after = min(remaining, total_sentences - target_pos - 1 - after_count)
            after_count += additional_after

    # Collect the sentences (including the target sentence)
    start_idx = target_pos - before_count
    end_idx = target_pos + after_count + 1  # +1 to include target

    selected_rows = story_sentences.iloc[start_idx:end_idx]

    # Verify target sentence is included
    target_sentIx = target_row['sentIx']
    selected_sentIx_list = selected_rows['sentIx'].tolist()
    if target_sentIx not in selected_sentIx_list:
        print(f"ERROR: Target sentence {target_sentIx} not found in selected sentences {selected_sentIx_list}")
        print(f"Target pos: {target_pos}, start_idx: {start_idx}, end_idx: {end_idx}")
        return []

    # Convert to list of dictionaries
    sentences = []
    label_columns = ['majorBin', 'minorBin', 'expectedBin', 'surprisingBin', 'eventOrNot']

    for _, row in selected_rows.iterrows():
        sentence_dict = {
            'text': row['sent'],
            'story_idx': int(row['storyIx']),
            'sentence_idx': int(row['sentIx']),
            'memType': row['memType']
        }

        # Add all label scores
        for label in label_columns:
            sentence_dict[label] = float(row[label])

        sentences.append(sentence_dict)

    return sentences


def extract_top_surprising_with_neighbors(df, surprisal_thres=0.4, neighbors_count=4):
    """
    Extract sentences with surprisingBin score > surprisal_thres and their neighbors.

    Returns:
        List of sentence groups, each containing sentences in chronological order
    """
    # Filter by surprisingBin score threshold
    df_sorted = df.sort_values('surprisingBin', ascending=False)
    high_surprising = df_sorted[df_sorted['surprisingBin'] > surprisal_thres]

    if len(high_surprising) == 0:
        print(f"No sentences found with surprisingBin score > {surprisal_thres}")
        print(f"Max score in dataset: {df_sorted['surprisingBin'].max():.3f}")
        return []

    print(f"Found {len(high_surprising)} sentences with surprisingBin scores > {surprisal_thres}")
    print(f"Score range: {high_surprising['surprisingBin'].max():.3f} to {high_surprising['surprisingBin'].min():.3f}")

    # Collect neighbors for each high-scoring sentence
    sentence_groups = []
    max_scores_per_group = []

    for idx, (_, row) in enumerate(high_surprising.iterrows()):
        target_score = row['surprisingBin']
        print(f"Processing sentence {idx + 1}/{len(high_surprising)} (target score: {target_score:.3f})")

        neighbors = get_neighbors(df, row, neighbors_count)
        if neighbors:
            # Calculate max surprising score in this group
            group_scores = [s['surprisingBin'] for s in neighbors]
            max_group_score = max(group_scores)
            max_scores_per_group.append(max_group_score)

            # Find which sentence has the max score
            max_sentence_idx = group_scores.index(max_group_score)

            print(f"  Target sentence surprisingBin: {target_score:.3f}")
            print(f"  Max surprisingBin in group: {max_group_score:.3f} (sentence {max_sentence_idx})")
            print(f"  All scores in group: {[f'{s:.3f}' for s in group_scores]}")

            if abs(max_group_score - target_score) > 0.001:  # Allow small floating point differences
                print(f"  WARNING: Max group score ({max_group_score:.3f}) != target score ({target_score:.3f})")

            sentence_groups.append(neighbors)

    # Debug: plot distribution of max scores per group
    print(f"\n=== DEBUG: Max surprising scores per group ===")
    print(f"Expected high scores: {high_surprising['surprisingBin'].values[:10]}")
    print(f"Actual max scores per group (first 10): {max_scores_per_group[:10]}")

    # Check if max scores match the expected scores
    expected_scores = sorted(high_surprising['surprisingBin'].values, reverse=True)
    actual_max_scores = sorted(max_scores_per_group, reverse=True)

    print(f"Do max group scores match expected? {expected_scores[:10] == actual_max_scores[:10]}")

    return sentence_groups


def save_results(sentence_groups, output_path):
    """Save the sentence groups to JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sentence_groups, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(sentence_groups)} sentence groups to {output_path}")


def main():
    """Main function to extract surprising sentences with neighbors."""
    cfg = SurprisingConfig(
        surprisal_thres=0.3,  # Only collect sentences with surprisingBin > 0.4
        neighbors_count=4,

        env=ENV_CFG,
        data=DatasetConfig(
            name="Hippocorpus",
            hf_name="hcV3-eventAnnotsAggOverWorkers.csv",
            num_sequences=1000,
            context_length=None,
        ),
        llm=LLMConfig(
            name="Gemma-2-2B",
            hf_name="google/gemma-2-2b",
            revision=None,
            layer_idx=12,
            hidden_dim=2304,
            batch_size=50,
        ),
    )

    print("Loading hippocorpus raw data...")
    df = load_hippocorpus_raw_data()

    print(f"Total sentences in dataset: {len(df)}")
    print(f"surprisingBin score range: {df['surprisingBin'].min():.3f} to {df['surprisingBin'].max():.3f}")
    print(f"Mean surprisingBin score: {df['surprisingBin'].mean():.3f}")

    # Extract surprising sentences with neighbors
    sentence_groups = extract_top_surprising_with_neighbors(
        df,
        surprisal_thres=cfg.surprisal_thres,
        neighbors_count=cfg.neighbors_count
    )

    # Save results
    output_path = os.path.join("artifacts", "text_inputs", f"surprising_sentences_above_{cfg.surprisal_thres}_with_neighbors.json")
    save_results(sentence_groups, output_path)

    print(f"\nCompleted! Extracted {len(sentence_groups)} sentence groups.")
    print(f"Each group contains up to {cfg.neighbors_count} sentences in chronological order.")


if __name__ == "__main__":
    main()