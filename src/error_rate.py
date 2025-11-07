import numpy as np
from typing import List, Union

def calculate_wer(reference: List[Union[str, int]], hypothesis: List[Union[str, int]]) -> (int, int, int, int, float):
    """
    Calculate Word Error Rate (WER) and the number of substitutions, deletions, and insertions.

    Args:
        reference (List): The reference sequence.
        hypothesis (List): The hypothesis sequence.

    Returns:
        A tuple containing:
        - substitutions (int): The number of substitutions.
        - deletions (int): The number of deletions.
        - insertions (int): The number of insertions.
        - num_words (int): The number of words in the reference sequence.
        - wer (float): The Word Error Rate.
    """
    # Add BOS and EOS symbols
    ref = ["<bos>"] + reference + ["<eos>"]
    hyp = ["<bos>"] + hypothesis + ["<eos>"]
    
    n = len(ref)
    m = len(hyp)
    
    # Initialize DP table
    dp = np.zeros((n + 1, m + 1))
    
    for i in range(n + 1):
        dp[i, 0] = i
    for j in range(m + 1):
        dp[0, j] = j
        
    # Fill DP table
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i-1] == hyp[j-1] else 1
            dp[i, j] = min(dp[i-1, j] + 1,       # Deletion
                           dp[i, j-1] + 1,       # Insertion
                           dp[i-1, j-1] + cost)  # Substitution or Correct
    
    # Backtrack to find the number of errors
    substitutions = 0
    deletions = 0
    insertions = 0
    
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i-1] == hyp[j-1]:
            i -= 1
            j -= 1
            continue

        if j > 0 and i > 0 and dp[i, j] == dp[i-1, j-1] + 1:
            substitutions += 1
            i -= 1
            j -= 1
        elif j > 0 and dp[i, j] == dp[i, j-1] + 1:
            insertions += 1
            j -= 1
        elif i > 0 and dp[i, j] == dp[i-1, j] + 1:
            deletions += 1
            i -= 1
        else: # Should not happen in correct DP table
            if i > 0:
                i -= 1
            if j > 0:
                j -= 1

    num_words = len(ref)
    wer = (substitutions + deletions + insertions) / num_words if num_words > 0 else 0.0
    
    return substitutions, deletions, insertions, num_words, wer
