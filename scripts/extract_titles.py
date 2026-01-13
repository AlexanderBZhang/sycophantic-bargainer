#!/usr/bin/env python3
"""Extract employee titles from Enron email signatures."""

import json
import re
from collections import defaultdict, Counter

person_titles = defaultdict(Counter)

# Words that are NOT names
not_names = ['was', 'the', 'and', 'for', 'that', 'with', 'this', 'from', 'have', 
             'been', 'will', 'would', 'could', 'should', 'their', 'there', 'were',
             'vice', 'president', 'director', 'manager', 'board', 'access', 'without',
             'become', 'below', 'above', 'after', 'before', 'during', 'through',
             'between', 'within', 'nominations', 'brazos', 'executive', 'senior',
             'says', 'california', 'gas', 'transmission', 'operations', 'account',
             'client', 'privilege', 'attorney', 'message', 'technology', 'officer',
             'energy', 'supply', 'north', 'america', 'bar', 'regarding', 'texas',
             'licensed', 'learning', 'technologies', 'erisa', 'join', 'name', 'eight',
             'associate', 'tenaska', 'plant', 'confidential', 'communication', 'key', 'be']

def is_valid_name(name):
    parts = name.lower().split()
    if len(parts) != 2:
        return False
    for part in parts:
        if part in not_names:
            return False
        if len(part) < 2:
            return False
        # Must start with letter
        if not part[0].isalpha():
            return False
    return True

print('Analyzing emails for titles...')

with open('results/ENRON_EMAILS_CLEANED.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        body = data.get('body', '')
        
        sig = body[-1000:] if len(body) > 1000 else body
        
        sig_patterns = [
            r'([A-Z][a-z]+ [A-Z][a-z]+)\n((?:Senior |Executive )?Vice President[^\n]{0,50})',
            r'([A-Z][a-z]+ [A-Z][a-z]+)\n((?:Managing )?Director[^\n]{0,50})',
            r'([A-Z][a-z]+ [A-Z][a-z]+)\n(President[^\n]{0,30})',
            r'([A-Z][a-z]+ [A-Z][a-z]+)\n(General Counsel[^\n]{0,30})',
            r'([A-Z][a-z]+ [A-Z][a-z]+)\n(Chief [A-Z][a-z]+ Officer)',
            r'([A-Z][a-z]+ [A-Z][a-z]+)\n(CEO|CFO|COO|CTO|CIO)',
            r'([A-Z][a-z]+ [A-Z][a-z]+)\n(Manager[^\n]{0,40})',
        ]
        
        for pattern in sig_patterns:
            matches = re.findall(pattern, sig, re.IGNORECASE)
            for name, title in matches:
                name = name.title()
                if is_valid_name(name):
                    title = title.strip()[:60]
                    person_titles[name][title] += 1

# Write clean results
with open('ENRON_EMPLOYEES_WITH_TITLES.txt', 'w', encoding='utf-8') as f:
    f.write('ENRON EMPLOYEES WITH IDENTIFIED TITLES\n')
    f.write('=' * 70 + '\n')
    f.write('Extracted from 156,109 emails via signature block analysis\n')
    f.write('=' * 70 + '\n\n')
    
    sorted_people = sorted(person_titles.items(), key=lambda x: -sum(x[1].values()))
    
    count = 0
    for name, titles in sorted_people:
        if sum(titles.values()) >= 2:
            top_title = titles.most_common(1)[0][0]
            f.write(f'{name}: {top_title}\n')
            count += 1

print(f'Found {count} employees with verified titles')
print('Saved to ENRON_EMPLOYEES_WITH_TITLES.txt')

