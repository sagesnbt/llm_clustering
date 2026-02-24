import csv
import json
import os
from pathlib import Path

def read_gestures(csv_path):
    gestures = []
    encodings = ['utf-8', 'utf-8-sig', 'latin1']
    
    for encoding in encodings:
        try:
            with open(csv_path, 'r', encoding=encoding) as f:
                content = f.read()
                # Remove any BOM if present
                if content.startswith('\ufeff'):
                    content = content[1:]
                
                # Create a list of lines and pass it to DictReader
                lines = content.splitlines()
                reader = csv.DictReader(lines, delimiter=';')
                
                for row in reader:
                    gestures.append({
                        'id': row['gesture_name'].lower().replace(' ', '_'),
                        'content': f"{row['gesture_name']}: {row['gesture_definition']}"
                    })
                break  # If successful, break the loop
        except (UnicodeDecodeError, KeyError):
            if encoding == encodings[-1]:  # If this was the last encoding to try
                raise  # Re-raise the exception
            continue  # Try the next encoding
    return gestures

def save_json(data, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

if __name__ == '__main__':
    # Read the CSV file
    csv_path = Path(__file__).parent.parent.parent / 'data' / 'full_list.csv'
    output_path = Path(__file__).parent.parent.parent / 'data' / 'gestures.json'
    
    gestures = read_gestures(csv_path)
    save_json(gestures, output_path)
    print(f"Processed {len(gestures)} gestures. Saved to {output_path}")