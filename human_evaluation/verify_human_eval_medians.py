#!/usr/bin/env python3
"""
Human Evaluation Median Verification Script
Extracts median ratings (not means) for expert evaluation
"""

import pandas as pd
import sys
from pathlib import Path

def analyze_human_evaluation_medians(forward_csv, inverse_csv):
    """
    Analyze human evaluation data using MEDIANS for both directions
    """
    
    forward_df = pd.read_csv(forward_csv)
    inverse_df = pd.read_csv(inverse_csv)
    
    print("="*180)
    print("HUMAN EVALUATION ANALYSIS - CLAUDE SONNET 4.5 (USING MEDIANS)")
    print("="*180)
    
    # Forward (C→I) Analysis
    print("\n\n" + "="*180)
    print("FORWARD (C→I) - Collectivistic to Individualistic")
    print("="*180)
    print("Rating Scale: 1=Most Collectivistic, 5=Most Individualistic")
    print(f"Sample Size: {len(forward_df)} stories")
    
    print("\n\nIndividual Story Ratings:")
    print(f"{'Story':<10} {'Original':<15} {'Baseline':<15} {'Abduction':<15} {'Change (B→A)':<15}")
    print("-"*180)
    
    for _, row in forward_df.iterrows():
        story_id = row['Story_id']
        original = row['original']
        baseline = row['baseline']
        abduction = row['abduction']
        change = abduction - baseline
        
        print(f"{story_id:<10} {original:<15.1f} {baseline:<15.1f} {abduction:<15.1f} {change:+.1f}")
    
    # Calculate medians
    forward_orig_median = forward_df['original'].median()
    forward_baseline_median = forward_df['baseline'].median()
    forward_abduction_median = forward_df['abduction'].median()
    
    forward_median_improvement = forward_abduction_median - forward_baseline_median
    forward_abduction_vs_original = forward_abduction_median - forward_orig_median
    
    print("\n\n" + "="*180)
    print("FORWARD (C→I) - MEDIAN STATISTICS")
    print("="*180)
    print(f"\nFormula: Median = middle value when all values are sorted")
    print(f"\nAll baseline scores (sorted): {sorted(forward_df['baseline'].values)}")
    print(f"Median baseline = {forward_baseline_median:.4f}")
    
    print(f"\nAll abduction scores (sorted): {sorted(forward_df['abduction'].values)}")
    print(f"Median abduction = {forward_abduction_median:.4f}")
    
    print(f"\nAll original scores (sorted): {sorted(forward_df['original'].values)}")
    print(f"Median original = {forward_orig_median:.4f}")
    
    print(f"\n\nMedian Statistics:")
    print(f"  Original Median:           {forward_orig_median:.4f}")
    print(f"  Baseline Median:           {forward_baseline_median:.4f}")
    print(f"  Abduction-Guided Median:   {forward_abduction_median:.4f}")
    print(f"\n  Baseline → Abduction improvement: {forward_median_improvement:+.4f} points")
    print(f"  Abduction vs Original shift:      {forward_abduction_vs_original:+.4f} points")
    
    print(f"\nQuartile Information:")
    print(f"  Baseline Q1 (25%):   {forward_df['baseline'].quantile(0.25):.4f}")
    print(f"  Baseline Q3 (75%):   {forward_df['baseline'].quantile(0.75):.4f}")
    print(f"  Abduction Q1 (25%):  {forward_df['abduction'].quantile(0.25):.4f}")
    print(f"  Abduction Q3 (75%):  {forward_df['abduction'].quantile(0.75):.4f}")
    
    # Inverse (I→C) Analysis
    print("\n\n" + "="*180)
    print("INVERSE (I→C) - Individualistic to Collectivistic")
    print("="*180)
    print("Rating Scale: 1=Most Individualistic, 5=Most Collectivistic")
    print(f"Sample Size: {len(inverse_df)} stories")
    
    print("\n\nIndividual Story Ratings:")
    print(f"{'Story':<10} {'Original':<15} {'Baseline':<15} {'Abduction':<15} {'Change (B→A)':<15}")
    print("-"*180)
    
    for _, row in inverse_df.iterrows():
        story_id = row['Story_id']
        original = row['original']
        baseline = row['baseline']
        abduction = row['abduction']
        change = abduction - baseline
        
        print(f"{story_id:<10} {original:<15.1f} {baseline:<15.1f} {abduction:<15.1f} {change:+.1f}")
    
    # Calculate medians
    inverse_orig_median = inverse_df['original'].median()
    inverse_baseline_median = inverse_df['baseline'].median()
    inverse_abduction_median = inverse_df['abduction'].median()
    
    inverse_median_improvement = inverse_abduction_median - inverse_baseline_median
    inverse_abduction_vs_original = inverse_abduction_median - inverse_orig_median
    
    print("\n\n" + "="*180)
    print("INVERSE (I→C) - MEDIAN STATISTICS")
    print("="*180)
    print(f"\nFormula: Median = middle value when all values are sorted")
    print(f"\nAll baseline scores (sorted): {sorted(inverse_df['baseline'].values)}")
    print(f"Median baseline = {inverse_baseline_median:.4f}")
    
    print(f"\nAll abduction scores (sorted): {sorted(inverse_df['abduction'].values)}")
    print(f"Median abduction = {inverse_abduction_median:.4f}")
    
    print(f"\nAll original scores (sorted): {sorted(inverse_df['original'].values)}")
    print(f"Median original = {inverse_orig_median:.4f}")
    
    print(f"\n\nMedian Statistics:")
    print(f"  Original Median:           {inverse_orig_median:.4f}")
    print(f"  Baseline Median:           {inverse_baseline_median:.4f}")
    print(f"  Abduction-Guided Median:   {inverse_abduction_median:.4f}")
    print(f"\n  Baseline → Abduction improvement: {inverse_median_improvement:+.4f} points")
    print(f"  Abduction vs Original shift:      {inverse_abduction_vs_original:+.4f} points")
    
    print(f"\nQuartile Information:")
    print(f"  Baseline Q1 (25%):   {inverse_df['baseline'].quantile(0.25):.4f}")
    print(f"  Baseline Q3 (75%):   {inverse_df['baseline'].quantile(0.75):.4f}")
    print(f"  Abduction Q1 (25%):  {inverse_df['abduction'].quantile(0.25):.4f}")
    print(f"  Abduction Q3 (75%):  {inverse_df['abduction'].quantile(0.75):.4f}")
    
    # Comparison
    print("\n\n" + "="*180)
    print("COMPARISON - MEDIANS ACROSS DIRECTIONS")
    print("="*180)
    print(f"\nAbduction-guided improvements (using MEDIANS):")
    print(f"  Forward (C→I):  {forward_median_improvement:+.4f} points (from {forward_baseline_median:.4f} to {forward_abduction_median:.4f})")
    print(f"  Inverse (I→C):  {inverse_median_improvement:+.4f} points (from {inverse_baseline_median:.4f} to {inverse_abduction_median:.4f})")
    
    print(f"\nAbduction effectiveness (shift toward target, using MEDIANS):")
    print(f"  Forward (C→I):  Moved {forward_abduction_vs_original:.4f} points toward individualistic (original median: {forward_orig_median:.4f})")
    print(f"  Inverse (I→C):  Moved {inverse_abduction_vs_original:.4f} points toward collectivistic (original median: {inverse_orig_median:.4f})")
    
    # Count improvements
    print("\n\n" + "="*180)
    print("STORY-BY-STORY IMPROVEMENTS")
    print("="*180)
    
    forward_improved = (forward_df['abduction'] > forward_df['baseline']).sum()
    forward_flat = (forward_df['abduction'] == forward_df['baseline']).sum()
    forward_degraded = (forward_df['abduction'] < forward_df['baseline']).sum()
    
    inverse_improved = (inverse_df['abduction'] > inverse_df['baseline']).sum()
    inverse_flat = (inverse_df['abduction'] == inverse_df['baseline']).sum()
    inverse_degraded = (inverse_df['abduction'] < inverse_df['baseline']).sum()
    
    print(f"\nForward (C→I):")
    print(f"  Stories improved: {forward_improved}/12 ({forward_improved/12*100:.1f}%)")
    print(f"  Stories flat:     {forward_flat}/12 ({forward_flat/12*100:.1f}%)")
    print(f"  Stories degraded: {forward_degraded}/12 ({forward_degraded/12*100:.1f}%)")
    
    print(f"\nInverse (I→C):")
    print(f"  Stories improved: {inverse_improved}/12 ({inverse_improved/12*100:.1f}%)")
    print(f"  Stories flat:     {inverse_flat}/12 ({inverse_flat/12*100:.1f}%)")
    print(f"  Stories degraded: {inverse_degraded}/12 ({inverse_degraded/12*100:.1f}%)")
    
    # Summary for paper
    print("\n\n" + "="*180)
    print("NUMBERS FOR PAPER (USING MEDIANS - NO HALLUCINATION)")
    print("="*180)
    print(f"\nForward (C→I):")
    print(f"  Original median: {forward_orig_median}")
    print(f"  Baseline median: {forward_baseline_median}")
    print(f"  Abduction median: {forward_abduction_median}")
    print(f"  Improvement: {forward_median_improvement:+.2f} points")
    print(f"  Stories improved: {forward_improved}/12")
    
    print(f"\nInverse (I→C):")
    print(f"  Original median: {inverse_orig_median}")
    print(f"  Baseline median: {inverse_baseline_median}")
    print(f"  Abduction median: {inverse_abduction_median}")
    print(f"  Improvement: {inverse_median_improvement:+.2f} points")
    print(f"  Stories improved: {inverse_improved}/12")

def main():
    if len(sys.argv) < 3:
        print("Usage: python verify_human_eval_medians.py <forward_csv> <inverse_csv>")
        print("\nExample:")
        print("  python verify_human_eval_medians.py human_evaluation_forward.csv human_evaluation_inverse.csv")
        sys.exit(1)
    
    forward_csv = sys.argv[1]
    inverse_csv = sys.argv[2]
    
    if not Path(forward_csv).exists():
        print(f"Error: File not found: {forward_csv}")
        sys.exit(1)
    
    if not Path(inverse_csv).exists():
        print(f"Error: File not found: {inverse_csv}")
        sys.exit(1)
    
    analyze_human_evaluation_medians(forward_csv, inverse_csv)

if __name__ == "__main__":
    main()
