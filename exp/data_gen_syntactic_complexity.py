#!/usr/bin/env python3
"""
Script to generate a fluent English sentence using Anthropic's Claude Haiku model.
Combines defined subject, object, time, and location into a natural sentence.
"""

import anthropic
import os
import json
import random
from typing import Dict, List
from src.project_config import INPUTS_DIR

def _query_claude(prompt: str, max_tokens: int = 100) -> str:
    """
    Child function to handle Claude API queries.
    
    Args:
        prompt: The prompt to send to Claude
        max_tokens: Maximum tokens for the response
    
    Returns:
        The response text from Claude
    """
    client = anthropic.Anthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY")
    )
    
    message = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=max_tokens,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    
    return message.content[0].text.strip()

def generate_sentence(subject: str, object: str, time: str, location: str) -> str:
    """
    Generate a fluent English sentence combining subject, object, time, and location.
    
    Args:
        subject: The subject of the sentence
        object: The object in the sentence
        time: The time reference
        location: The location reference
    
    Returns:
        A fluent English sentence
    """
    prompt = f"""Create a single, fluent English sentence that naturally incorporates all of these elements:
- Subject: {subject}
- Object: {object}
- Time: {time}
- Location: {location}
Use a phrasal verb.
Return only the sentence, nothing else."""
    
    return _query_claude(prompt, max_tokens=100)

def rearrange_for_complexity_gramatical_forms(sentence: str) -> Dict[str, str]:
    """
    Rearrange the words of a sentence to increase syntactic complexity
    while maintaining the same meaning and exact word set.
    
    Args:
        sentence: The original sentence to rearrange
    
    Returns:
        A dictionary mapping grammatical forms to their corresponding generations
    """
    prompt = f"""Adapt this sentence to each of the following gramatical forms, to create a more syntactically complex version while keeping the exact same meaning:

    Gramatical forms:
    - word order
    - subordination
    - separation of phrasal verb
    - passive voice
    - cleft construction
    - relative clause
    - particle fronting
    - existential there insertion

"{sentence}"

Rules:
- Maintain the SAME MEANING
- Use the same words unless necessary for new gramatical structure (no adding, removing, or changing words)
- Do not introduce puntctuation, eg no commata.
- Maximize syntactic complexity through word order and separation
- maintain gramatical correctness

Return the result in JSON format with the following structure:
{{
    "original": "{sentence}",
    "word_order": "generated sentence",
    "subordination": "generated sentence",
    "separation_of_phrasal_verb": "generated sentence",
    "passive_voice": "generated sentence",
    "cleft_construction": "generated sentence",
    "relative_clause": "generated sentence",
    "particle_fronting": "generated sentence",
    "existential_there_insertion": "generated sentence"
}}

Return only the JSON object, nothing else."""
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = _query_claude(prompt, max_tokens=300)
            result = json.loads(response)
            return result
        except json.JSONDecodeError as e:
            if attempt == max_retries - 1:
                raise ValueError(f"Failed to parse JSON after {max_retries} attempts. Last error: {e}")
            continue
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            continue

def phrasal_verb_separation_variations(sentence: str) -> List[str]:
    """
    Generate all possible variations of phrasal verb separation for the original sentence.
    
    Args:
        sentence: The original sentence containing a phrasal verb
    
    Returns:
        A list of all possible phrasal verb separation variations
    """
    prompt = f"""Given this sentence with a phrasal verb, generate ALL possible variations of phrasal verb separation while maintaining the exact same meaning:

"{sentence}"

Rules:
- Maintain the SAME MEANING
- Keep all the same words (no adding, removing, or changing words)
- Only vary the separation/position of the phrasal verb particle
- Do not introduce punctuation
- Include both separated and unseparated forms where grammatically correct
- Generate ALL valid variations, not just one or two

Return the result as a JSON list of strings:
["variation 1", "variation 2", "variation 3", ...]

Return only the JSON array, nothing else."""
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = _query_claude(prompt, max_tokens=200)
            result = json.loads(response)
            return result
        except json.JSONDecodeError as e:
            if attempt == max_retries - 1:
                raise ValueError(f"Failed to parse JSON after {max_retries} attempts. Last error: {e}")
            continue
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            continue

def main():
    # Define 10 diverse words for each category
    subjects = ["John", "Sarah", "the teacher", "my neighbor", "the detective", "the artist", "her brother", "the scientist", "the chef", "the musician"]
    objects = ["trash", "books", "flowers", "tools", "letters", "dishes", "paintings", "equipment", "groceries", "instruments"]
    times = ["yesterday", "last week", "this morning", "earlier today", "last month", "two days ago", "last evening", "recently", "last Friday", "an hour ago"]
    locations = ["backyard", "kitchen", "office", "garden", "garage", "studio", "laboratory", "restaurant", "library", "park"]
    
    # Generate 50 random combinations
    all_results = []
    
    try:
        for i in range(50):
            subject = random.choice(subjects)
            obj = random.choice(objects)
            time = random.choice(times)
            location = random.choice(locations)
            
            print(f"Generating combination {i+1}/50: {subject}, {obj}, {time}, {location}")
            
            # Generate original sentence
            sentence = generate_sentence(subject, obj, time, location)
            
            # Generate phrasal verb separation variations
            variations = phrasal_verb_separation_variations(sentence)
            
            # Store results
            result = {
                "combination_id": i + 1,
                "components": {
                    "subject": subject,
                    "object": obj,
                    "time": time,
                    "location": location
                },
                "original_sentence": sentence,
                "phrasal_verb_variations": variations
            }
            all_results.append(result)
        
        # Write results to INPUTS_DIR as JSON
        output_file = os.path.join(INPUTS_DIR, "syntactic_complexity_phrasal_verbs.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        print(f"Generated {len(all_results)} combinations with phrasal verb variations")
        print(f"Results saved to: {output_file}")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())