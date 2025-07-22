import spacy
from typing import List

# Load the English language model
try:
    nlp = spacy.load("en_core_web_sm")
except:
    print("We have to download the model first, via `python -m spacy download en_core_web_sm`")

    
def calculate_syntactic_complexity(sentence: str) -> int:
    """
    Calculate syntactic complexity as the sum of all dependency lengths.

    Dependency length is the distance between a dependent token and its head token
    in terms of their positions in the sentence.

    Args:
        sentence: Input sentence to analyze

    Returns:
        Total syntactic complexity (sum of all dependency lengths)
    """
    # Parse the sentence
    doc = nlp(sentence)

    total_complexity = 0

    # For each token, calculate distance to its head
    for token in doc:
        # Skip the root token (which has itself as head)
        if token.head != token:
            # Calculate dependency length as absolute distance
            dependency_length = abs(token.i - token.head.i)
            total_complexity += dependency_length

            # Debug output to show dependencies
            print(
                f"'{token.text}' -> '{token.head.text}' (distance: {dependency_length})"
            )

    return total_complexity


def analyze_sentence_dependencies(sentence: str):
    """
    Provide detailed analysis of dependencies in a sentence.
    """
    doc = nlp(sentence)

    print(f"\nAnalyzing: '{sentence}'")
    print("Token dependencies:")

    for token in doc:
        if token.head != token:
            print(
                f"  {token.i}: '{token.text}' --{token.dep_}--> {token.head.i}: '{token.head.text}'"
            )
        else:
            print(f"  {token.i}: '{token.text}' (ROOT)")

    complexity = calculate_syntactic_complexity(sentence)
    print(f"Total syntactic complexity: {complexity}")
    return complexity


example_sentences = [
    "The cat sat on the mat.",
    "The mat the cat sat on.",
]

if __name__ == "__main__":
    print("Syntactic Complexity Analysis")
    print("=" * 40)

    for sentence in example_sentences:
        complexity = analyze_sentence_dependencies(sentence)
        print()
