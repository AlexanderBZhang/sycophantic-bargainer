"""Quick exploration of CaSiNo dataset structure."""
import json
import sys
sys.stdout.reconfigure(encoding='utf-8')

with open('casino.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"Total dialogues: {len(data)}")
print()

# Look at first dialogue
d = data[0]
print(f"Top-level keys: {list(d.keys())}")
print()

if 'participant_info' in d:
    print(f"participant_info keys: {list(d['participant_info'].keys())}")
    for pkey in d['participant_info']:
        print(f"  {pkey} keys: {list(d['participant_info'][pkey].keys())}")
print()

if 'chat_logs' in d:
    print(f"chat_logs has {len(d['chat_logs'])} turns")
    print(f"First turn: {d['chat_logs'][0]}")
print()

if 'annotations' in d:
    print(f"annotations type: {type(d['annotations'])}")
    if isinstance(d['annotations'], list) and len(d['annotations']) > 0:
        print(f"annotations[0]: {d['annotations'][0]}")
print()

# Check for personality data
for pkey in d.get('participant_info', {}):
    pinfo = d['participant_info'][pkey]
    print(f"{pkey}:")
    print(f"  personality: {pinfo.get('personality', 'N/A')}")
    print(f"  demographics: {pinfo.get('demographics', 'N/A')}")
    print(f"  outcomes: {pinfo.get('outcomes', 'N/A')}")
    print()

