"""
Preprocess CaSiNo dataset for semantic analysis.

This script extracts individual participant contributions from CaSiNo dialogues,
along with personality metadata (SVO and Big Five Agreeableness) for stratification.

Output: results/CASINO_PREPROCESSED.jsonl
Each line contains one participant's messages from one dialogue, with:
- dialogue_id: unique identifier
- participant_id: mturk_agent_1 or mturk_agent_2
- messages: all messages from this participant in the dialogue
- full_dialogue: complete dialogue for context
- svo: Social Value Orientation (proself, prosocial, etc.)
- agreeableness: Big Five agreeableness score (1.0-7.0)
- outcomes: points_scored, satisfaction, opponent_likeness
"""

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

def extract_participant_data(dialogue: dict) -> list[dict]:
    """Extract data for each participant in a dialogue."""
    results = []
    
    dialogue_id = dialogue['dialogue_id']
    chat_logs = dialogue['chat_logs']
    participant_info = dialogue['participant_info']
    
    # Build full dialogue text
    full_dialogue_lines = []
    for turn in chat_logs:
        speaker = turn['id']
        text = turn['text']
        full_dialogue_lines.append(f"{speaker}: {text}")
    
    full_dialogue = "\n".join(full_dialogue_lines)
    
    # Extract each participant's data
    for participant_id in ['mturk_agent_1', 'mturk_agent_2']:
        pinfo = participant_info.get(participant_id, {})
        personality = pinfo.get('personality', {})
        
        # Get this participant's messages
        participant_messages = []
        for turn in chat_logs:
            if turn['id'] == participant_id:
                participant_messages.append(turn['text'])
        
        # Skip if no messages (shouldn't happen, but safety check)
        if not participant_messages:
            continue
        
        # Extract SVO and agreeableness
        svo = personality.get('svo', 'unknown')
        big_five = personality.get('big-five', {})
        agreeableness = big_five.get('agreeableness', None)
        
        # Extract outcomes
        outcomes = pinfo.get('outcomes', {})
        
        results.append({
            'id': f"{dialogue_id}_{participant_id}",
            'dialogue_id': dialogue_id,
            'participant_id': participant_id,
            'messages': participant_messages,
            'messages_combined': "\n\n".join(participant_messages),
            'full_dialogue': full_dialogue,
            'svo': svo,
            'agreeableness': agreeableness,
            'points_scored': outcomes.get('points_scored'),
            'satisfaction': outcomes.get('satisfaction'),
            'opponent_likeness': outcomes.get('opponent_likeness'),
            'word_count': sum(len(m.split()) for m in participant_messages)
        })
    
    return results


def main():
    print("Loading CaSiNo dataset...")
    with open('casino.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Found {len(data)} dialogues")
    
    # Process all dialogues
    all_participants = []
    svo_counts = {}
    agreeableness_buckets = {'low': 0, 'medium': 0, 'high': 0}
    
    for dialogue in data:
        participants = extract_participant_data(dialogue)
        all_participants.extend(participants)
        
        for p in participants:
            # Count SVO
            svo = p['svo']
            svo_counts[svo] = svo_counts.get(svo, 0) + 1
            
            # Bucket agreeableness (1-7 scale)
            # Low: 1-3, Medium: 3-5, High: 5-7
            agree = p['agreeableness']
            if agree is not None:
                if agree <= 3.5:
                    agreeableness_buckets['low'] += 1
                elif agree <= 5.5:
                    agreeableness_buckets['medium'] += 1
                else:
                    agreeableness_buckets['high'] += 1
    
    print(f"\nExtracted {len(all_participants)} participant entries")
    print(f"\nSVO distribution:")
    for svo, count in sorted(svo_counts.items(), key=lambda x: -x[1]):
        print(f"  {svo}: {count} ({100*count/len(all_participants):.1f}%)")
    
    print(f"\nAgreeableness distribution (1-7 scale):")
    for bucket, count in agreeableness_buckets.items():
        print(f"  {bucket}: {count} ({100*count/len(all_participants):.1f}%)")
    
    # Word count stats
    word_counts = [p['word_count'] for p in all_participants]
    print(f"\nWord count statistics:")
    print(f"  Min: {min(word_counts)}")
    print(f"  Max: {max(word_counts)}")
    print(f"  Mean: {sum(word_counts)/len(word_counts):.1f}")
    
    # Save to JSONL
    output_path = Path('results/CASINO_PREPROCESSED.jsonl')
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in all_participants:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"\nSaved to {output_path}")
    
    # Show sample entry
    print("\n--- Sample entry ---")
    sample = all_participants[0]
    print(f"ID: {sample['id']}")
    print(f"SVO: {sample['svo']}")
    print(f"Agreeableness: {sample['agreeableness']}")
    print(f"Points scored: {sample['points_scored']}")
    print(f"Word count: {sample['word_count']}")
    print(f"Messages (first 200 chars): {sample['messages_combined'][:200]}...")


if __name__ == '__main__':
    main()

