"""
Word Error rate and character Error Rate(CER) Calculation.

Run Directly to see self-test verification against hand- computed examples.

"""

def _edit_distance(ref_tokens, hyp_tokens):
    """Clasic Levenshtein edit distance via dynamic programming 
         Builds a grid where cell [i][j] = minimum edits needed to turn the
    first i tokens of ref into the first j tokens of hyp. Each cell can
    come from: a deletion (from above), an insertion (from the left), or
    a match/substitution (from the diagonal).
 
    Returns (distance, substitutions, deletions, insertions) so we can
    report the breakdown, not just the final number.
    """

    n,m = len(ref_tokens), len(hyp_tokens)

    dp = [[0] * (m+1) for _ in range(n+1)]


    for i in range(n+1):
        dp[i][0] = i
    for j in range(m+1):
        dp[0][j] = j


    for i in range(1,n+1):
        for j in range(1, m+1):
            if ref_tokens[i -1] == hyp_tokens[j -1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1+min(
                    dp[i-1][j],
                    dp[i][j-1],
                    dp[i-1][j-1],)


    subs = dels = ins = 0
    i, j = n,m
    while i>0 or j>0:
        if i > 0 and j> 0 and ref_tokens[i-1] == hyp_tokens[j-1]:
            i,j = i-1, j-1 
        elif i>0 and j>0 and dp[i][j] == dp[i-1][j-1] + 1:
            subs +=1
            i, j = i-1, j-1
        elif i>0 and dp[i][j] == dp[i-1][j] +1:
            dels += 1
            i -= 1

        else:
            ins += 1
            j -= 1

    return dp[n][m], subs, dels, ins

def compute_wer(reference, hypothesis):
    """Word Error Rate: Split on whitespace, compare wprd by word."""

    ref_words = reference.strip().split()
    hyp_words = hypothesis.strip().split()

    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else float('inf')


    distance, subs, dels, ins = _edit_distance(ref_words, hyp_words)
    wer = distance / len(ref_words)

    return {
        "wer" : wer, 
        "substitutions" : subs,
        "deletions": dels, 
        "insertions": ins,
        "ref_word_count": len(ref_words),
    }

def compute_cer(reference, hypothesis):
    """ Character Error Rate: compare character-by-character instead of
    word-by-word. More robust for languages/scripts where word boundaries
    in casual transcription are inconsistent."""

    ref_chars = list(reference.strip())
    hyp_chars = list(hypothesis.strip())

    if len(ref_chars) ==0:
        return 0.0 if len(hyp_chars) == 0 else float("inf")

    distance, subs, dels, ins = _edit_distance(ref_chars, hyp_chars)
    cer = distance / len(ref_chars)


    return {
        "cer": cer,
        "substitutions": subs,
        "deletions": dels,
        "insertions":ins,
        "ref_char_count": len(ref_chars),
    }


def _self_test():
    """
    Verify our implementation against examples we can chect by hand,
    Before trusting this code on real eveluation data.
    
    
    """

    print("Running self-test...")

    # Test 1: identical strings -> WER should be exactly 0
    result = compute_wer("i love machine learning", "i love machine learning")
    assert result["wer"] == 0.0, f"Expected 0.0, got {result['wer']}"
    print(f"  Test 1 (identical): WER = {result['wer']:.3f} (expected 0.000) OK")
 
    # Test 2: one substitution out of 4 words -> WER should be exactly 0.25
    # "machine" -> "deep" is 1 substitution, everything else matches.
    result = compute_wer("i love machine learning", "i love deep learning")
    assert result["wer"] == 0.25, f"Expected 0.25, got {result['wer']}"
    assert result["substitutions"] == 1
    print(f"  Test 2 (1 substitution / 4 words): WER = {result['wer']:.3f} "
          f"(expected 0.250) OK")
 
    # Test 3: one word deleted out of 4 -> WER should be exactly 0.25
    result = compute_wer("i love machine learning", "i love learning")
    assert result["wer"] == 0.25, f"Expected 0.25, got {result['wer']}"
    assert result["deletions"] == 1
    print(f"  Test 3 (1 deletion / 4 words): WER = {result['wer']:.3f} "
          f"(expected 0.250) OK")
 
    # Test 4: one extra word inserted -> WER should be 0.25 (1 insertion / 4 ref words)
    result = compute_wer("i love machine learning", "i really love machine learning")
    assert result["wer"] == 0.25, f"Expected 0.25, got {result['wer']}"
    assert result["insertions"] == 1
    print(f"  Test 4 (1 insertion / 4 ref words): WER = {result['wer']:.3f} "
          f"(expected 0.250) OK")
 
    # Test 5: CER on a simple example
    result = compute_cer("cat", "cot")
    assert result["cer"] == 1 / 3, f"Expected 0.333, got {result['cer']}"
    print(f"  Test 5 (CER, 1 char sub / 3 chars): CER = {result['cer']:.3f} "
          f"(expected 0.333) OK")
 
    print("\nAll self-tests passed. The WER/CER implementation is verified correct.")
 
 
if __name__ == "__main__":
    _self_test()
 