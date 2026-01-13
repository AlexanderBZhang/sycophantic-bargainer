#!/usr/bin/env python3
"""
Enron Email Preprocessing

Reads all emails from maildir/, decodes quoted-printable encoding,
extracts metadata, and saves to a consolidated JSONL file.

Output: results/ENRON_EMAILS_CLEANED.jsonl
Errors: enron_parsing_errors.log

Usage:
    python preprocess_enron.py
    python preprocess_enron.py --limit 1000  # For testing
    python preprocess_enron.py --sample 10   # Show 10 sample emails
"""

import os
import re
import json
import quopri
import argparse
import email.utils
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
import time


# Configuration
MAILDIR = Path("maildir")
OUTPUT_FILE = Path("results/ENRON_EMAILS_CLEANED.jsonl")
ERROR_LOG = Path("enron_parsing_errors.log")


def get_all_email_paths(maildir: Path) -> List[str]:
    """Get all email file paths from the maildir structure.
    
    Returns string paths instead of Path objects because Windows
    has issues with filenames ending in dots (like '1.', '2.').
    We use the \\?\ extended path prefix to handle this.
    """
    paths = []
    maildir_abs = maildir.resolve()
    
    for root, dirs, files in os.walk(maildir_abs):
        for f in files:
            # Skip hidden files and non-email files
            if not f.startswith('.') and not f.endswith(('.db', '.pst')):
                # Use extended path syntax for Windows compatibility
                full_path = os.path.join(root, f)
                if os.name == 'nt':  # Windows
                    full_path = '\\\\?\\' + full_path
                paths.append(full_path)
    return paths


def path_to_relative_id(path: str, maildir: Path) -> str:
    """Convert a path to a relative ID for the email."""
    # Remove the extended path prefix if present
    if path.startswith('\\\\?\\'):
        path = path[4:]
    
    # Get relative path
    maildir_str = str(maildir.resolve())
    if path.startswith(maildir_str):
        rel = path[len(maildir_str):].lstrip('\\/')
    else:
        rel = path
    
    # Normalize to forward slashes
    return rel.replace('\\', '/')


def decode_quoted_printable(text: str) -> str:
    """Decode quoted-printable encoding to clean text."""
    try:
        # Handle soft line breaks (=\n or =\r\n)
        text = re.sub(r'=\r?\n', '', text)
        
        # Decode the rest using quopri
        decoded = quopri.decodestring(text.encode('latin-1', errors='replace'))
        
        # Try to decode as UTF-8, fall back to latin-1
        try:
            return decoded.decode('utf-8')
        except UnicodeDecodeError:
            return decoded.decode('latin-1', errors='replace')
    except Exception as e:
        return text  # Return original on failure


def parse_email_headers(content: str) -> Tuple[Dict[str, str], str]:
    """Parse email headers and return (headers_dict, body)."""
    headers = {}
    
    # Split into header and body sections
    parts = content.split('\n\n', 1)
    header_section = parts[0]
    body = parts[1] if len(parts) > 1 else ""
    
    # Parse headers (handle multi-line with tab continuation)
    current_header = None
    current_value = []
    
    for line in header_section.split('\n'):
        if line.startswith((' ', '\t')) and current_header:
            # Continuation of previous header
            current_value.append(line.strip())
        elif ':' in line:
            # Save previous header
            if current_header:
                headers[current_header] = ' '.join(current_value)
            
            # Start new header
            key, value = line.split(':', 1)
            current_header = key.strip()
            current_value = [value.strip()]
        else:
            # End of headers
            break
    
    # Save last header
    if current_header:
        headers[current_header] = ' '.join(current_value)
    
    return headers, body


def extract_name_from_header(header_value: str) -> Tuple[str, str]:
    """Extract email and name from a header value like 'Name <email@domain.com>'."""
    if not header_value:
        return "", ""
    
    # Try to parse with email.utils
    name, email_addr = email.utils.parseaddr(header_value)
    
    # Clean up
    name = name.strip()
    email_addr = email_addr.strip().lower()
    
    return email_addr, name


def extract_multiple_recipients(header_value: str) -> Tuple[List[str], List[str]]:
    """Extract multiple email addresses and names from a header."""
    if not header_value:
        return [], []
    
    emails = []
    names = []
    
    # Split by comma (but be careful with names containing commas)
    # Use email.utils.getaddresses for robust parsing
    addresses = email.utils.getaddresses([header_value])
    
    for name, addr in addresses:
        if addr:
            emails.append(addr.lower().strip())
            names.append(name.strip() if name else "")
    
    return emails, names


def parse_date(date_str: str) -> str:
    """Parse email date to ISO format."""
    if not date_str:
        return ""
    
    try:
        # Parse the date using email.utils
        parsed = email.utils.parsedate_to_datetime(date_str)
        return parsed.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return date_str  # Return original if parsing fails


def process_email(path: str, maildir: Path) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Process a single email file and return (data_dict, error_message)."""
    try:
        # Read file
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        
        # Parse headers and body
        headers, body = parse_email_headers(content)
        
        # Determine encoding
        transfer_encoding = headers.get('Content-Transfer-Encoding', '').lower()
        
        # Decode body if needed
        if 'quoted-printable' in transfer_encoding:
            body = decode_quoted_printable(body)
        
        # Extract metadata
        from_email, from_name = extract_name_from_header(headers.get('From', ''))
        
        # Prefer X-From for names (usually more complete)
        if headers.get('X-From'):
            x_from_email, x_from_name = extract_name_from_header(headers.get('X-From', ''))
            if x_from_name:
                from_name = x_from_name
        
        # Get recipients
        to_emails, to_names = extract_multiple_recipients(headers.get('To', ''))
        
        # Prefer X-To for names
        if headers.get('X-To'):
            _, x_to_names = extract_multiple_recipients(headers.get('X-To', ''))
            if x_to_names and any(x_to_names):
                to_names = x_to_names
        
        # Create relative ID from path
        email_id = path_to_relative_id(path, maildir)
        
        # Build result
        result = {
            "id": email_id,
            "from_email": from_email,
            "from_name": from_name,
            "to_emails": to_emails,
            "to_names": to_names,
            "subject": headers.get('Subject', ''),
            "date": parse_date(headers.get('Date', '')),
            "body": body.strip(),
            "word_count": len(body.split())
        }
        
        return result, None
        
    except Exception as e:
        return None, f"{path}: {type(e).__name__}: {str(e)}"


def show_samples(paths: List[str], n: int = 10):
    """Show sample processed emails."""
    import random
    random.seed(42)
    sample_paths = random.sample(paths, min(n, len(paths)))
    
    print(f"\n{'=' * 70}")
    print(f"SAMPLE EMAILS ({n} random samples)")
    print('=' * 70)
    
    for i, path in enumerate(sample_paths, 1):
        result, error = process_email(path, MAILDIR)
        rel_id = path_to_relative_id(path, MAILDIR)
        
        print(f"\n--- Sample {i}: {rel_id} ---")
        if error:
            print(f"ERROR: {error}")
        else:
            print(f"From: {result['from_name']} <{result['from_email']}>")
            print(f"To: {', '.join(result['to_names'])} ({len(result['to_emails'])} recipients)")
            print(f"Subject: {result['subject'][:60]}...")
            print(f"Date: {result['date']}")
            print(f"Words: {result['word_count']}")
            print(f"Body preview: {result['body'][:200]}...")


def main():
    parser = argparse.ArgumentParser(description="Preprocess Enron emails")
    parser.add_argument('--limit', type=int, help="Limit number of emails to process")
    parser.add_argument('--sample', type=int, help="Show N sample emails and exit")
    args = parser.parse_args()
    
    print("=" * 70)
    print("ENRON EMAIL PREPROCESSING")
    print("=" * 70)
    
    # Check if maildir exists
    if not MAILDIR.exists():
        print(f"ERROR: {MAILDIR} not found!")
        print("Please extract the Enron email archive first.")
        return
    
    # Get all email paths
    print(f"\nScanning {MAILDIR}...")
    paths = get_all_email_paths(MAILDIR)
    print(f"Found {len(paths):,} email files")
    
    # Sample mode
    if args.sample:
        show_samples(paths, args.sample)
        return
    
    # Apply limit
    if args.limit:
        paths = paths[:args.limit]
        print(f"Processing first {len(paths):,} emails (--limit)")
    
    # Process emails
    print(f"\nProcessing {len(paths):,} emails...")
    start_time = time.time()
    
    results = []
    errors = []
    
    for i, path in enumerate(paths):
        if (i + 1) % 10000 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            remaining = (len(paths) - i - 1) / rate
            print(f"  {i+1:,}/{len(paths):,} ({rate:.0f}/sec, ~{remaining:.0f}s remaining)")
        
        result, error = process_email(path, MAILDIR)
        
        if result:
            results.append(result)
        if error:
            errors.append(error)
    
    elapsed = time.time() - start_time
    print(f"Done in {elapsed:.1f}s ({len(paths)/elapsed:.0f} emails/sec)")
    
    # Save results
    OUTPUT_FILE.parent.mkdir(exist_ok=True)
    
    print(f"\nSaving to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    file_size = OUTPUT_FILE.stat().st_size / (1024 * 1024)
    print(f"Saved {len(results):,} emails ({file_size:.1f} MB)")
    
    # Save errors
    if errors:
        print(f"\nSaving {len(errors):,} errors to {ERROR_LOG}...")
        with open(ERROR_LOG, 'w', encoding='utf-8') as f:
            for error in errors:
                f.write(error + '\n')
    
    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print("=" * 70)
    print(f"Total files scanned: {len(paths):,}")
    print(f"Successfully processed: {len(results):,}")
    print(f"Errors: {len(errors):,}")
    print(f"Success rate: {100*len(results)/len(paths):.1f}%")
    
    # Sample stats
    if results:
        total_words = sum(r['word_count'] for r in results)
        avg_words = total_words / len(results)
        print(f"\nTotal words: {total_words:,}")
        print(f"Average words per email: {avg_words:.0f}")


if __name__ == "__main__":
    main()

