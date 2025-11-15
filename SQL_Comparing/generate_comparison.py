#!/usr/bin/env python3
"""
SQL Query Comparison Tool
Automatically reads from:
- model_queries.txt.rtf
- correct_queries.txt.rtf
"""

import difflib
import sys
import os
import re
from html import escape

# Hardcoded file paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_QUERIES_FILE = os.path.join(SCRIPT_DIR, 'model_queries.txt.rtf')
CORRECT_QUERIES_FILE = os.path.join(SCRIPT_DIR, 'correct_queries.txt.rtf')

def extract_text_from_rtf(rtf_content):
    """Extract plain text from RTF content by removing RTF codes"""
    lines = []
    
    # Split by lines
    for line in rtf_content.split('\n'):
        # Only process lines that contain SQL keywords
        if not any(keyword in line for keyword in ['SELECT', 'FROM', 'WHERE', 'INSERT', 'UPDATE', 'DELETE']):
            continue
        
        # Extract SQL from RTF line by removing RTF formatting codes
        # Strategy: Remove all RTF control codes, keep the actual SQL text
        
        # Step 1: Remove RTF color and formatting codes (like \cf2, \strokec2, \cb3, etc.)
        cleaned = re.sub(r'\\[a-z]+\d+\s*', ' ', line)  # Remove \cf2, \strokec2, etc.
        cleaned = re.sub(r'\\[a-z]+\s+', ' ', cleaned)  # Remove \f0, \fs24, etc.
        
        # Step 2: Remove RTF groups (text in braces that are formatting)
        # But be careful - we want to keep SQL text, not formatting
        # Remove simple RTF groups first
        cleaned = re.sub(r'\\[a-z]+\d*\s*', ' ', cleaned)
        
        # Step 3: Remove braces (but preserve content)
        # Replace { and } with spaces, but keep the text inside
        cleaned = cleaned.replace('{', ' ').replace('}', ' ')
        
        # Step 4: Remove trailing backslashes and clean up
        cleaned = cleaned.rstrip('\\').strip()
        
        # Step 5: Clean up multiple spaces
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        # Step 6: Verify it's actually a SQL query
        if cleaned and cleaned.startswith('SELECT') and len(cleaned) > 20:
            lines.append(cleaned)
    
    # If we still don't have enough lines, try more aggressive extraction
    if len(lines) < 10:
        # More aggressive: remove everything that looks like RTF codes
        text = rtf_content
        # Remove all RTF control sequences
        text = re.sub(r'\\[a-z]+\d*', '', text)
        # Remove braces
        text = text.replace('{', ' ').replace('}', ' ')
        # Split and find SQL lines
        for line in text.split('\n'):
            cleaned = re.sub(r'\s+', ' ', line.strip())
            if cleaned.startswith('SELECT') and len(cleaned) > 20:
                if cleaned not in lines:  # Avoid duplicates
                    lines.append(cleaned)
    
    return lines

def read_queries_from_file(filename):
    """Read SQL queries from a file, handling both plain text and RTF formats"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if it's RTF format
        if content.strip().startswith('{\\rtf'):
            # Extract text from RTF
            queries = extract_text_from_rtf(content)
        else:
            # Plain text file
            queries = [line.strip() for line in content.split('\n') if line.strip()]
        
        return queries
    except UnicodeDecodeError:
        # Try with different encoding
        with open(filename, 'r', encoding='latin-1') as f:
            content = f.read()
        if content.strip().startswith('{\\rtf'):
            queries = extract_text_from_rtf(content)
        else:
            queries = [line.strip() for line in content.split('\n') if line.strip()]
        return queries

def highlight_differences(text1, text2):
    """Create HTML with highlighted differences - both sides show red for any difference"""
    s = difflib.SequenceMatcher(None, text1, text2)
    output1 = []
    output2 = []
    
    for tag, i1, i2, j1, j2 in s.get_opcodes():
        if tag == 'equal':
            output1.append(escape(text1[i1:i2]))
            output2.append(escape(text2[j1:j2]))
        elif tag == 'delete':
            output1.append(f'<span class="different">{escape(text1[i1:i2])}</span>')
        elif tag == 'insert':
            output2.append(f'<span class="different">{escape(text2[j1:j2])}</span>')
        elif tag == 'replace':
            output1.append(f'<span class="different">{escape(text1[i1:i2])}</span>')
            output2.append(f'<span class="different">{escape(text2[j1:j2])}</span>')
    
    return ''.join(output1), ''.join(output2)

def generate_html_comparison(model_queries, correct_queries, output_file='sql_comparison.html'):
    """Generate HTML comparison file"""
    
    # Calculate statistics
    model_avg_length = sum(len(q) for q in model_queries) / len(model_queries)
    correct_avg_length = sum(len(q) for q in correct_queries) / len(correct_queries)
    
    # Generate HTML
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SQL Query Comparison</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            color: #333;
        }}
        
        .container {{
            max-width: 1800px;
            margin: 0 auto;
        }}
        
        .header {{
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            margin-bottom: 30px;
        }}
        
        .header h1 {{
            color: #667eea;
            margin-bottom: 20px;
            font-size: 2.5em;
            text-align: center;
        }}
        
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}
        
        .stat-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }}
        
        .stat-card h3 {{
            font-size: 1em;
            margin-bottom: 10px;
            opacity: 0.9;
        }}
        
        .stat-card .value {{
            font-size: 2em;
            font-weight: bold;
        }}
        
        .stat-card .subtext {{
            font-size: 0.9em;
            margin-top: 5px;
            opacity: 0.8;
        }}
        
        .query-comparison {{
            background: white;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            margin-bottom: 30px;
            overflow: hidden;
        }}
        
        .query-header {{
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px 25px;
            font-size: 1.2em;
            font-weight: bold;
        }}
        
        .query-content {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 2px;
            background: #e0e0e0;
        }}
        
        .query-side {{
            background: white;
            padding: 25px;
        }}
        
        .query-side h3 {{
            color: #667eea;
            margin-bottom: 15px;
            font-size: 1.1em;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }}
        
        .query-side.model h3 {{
            color: #e74c3c;
        }}
        
        .query-side.model h3::before {{
            content: "🤖 ";
        }}
        
        .query-side.correct h3 {{
            color: #27ae60;
        }}
        
        .query-side.correct h3::before {{
            content: "✓ ";
        }}
        
        .sql-code {{
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
            line-height: 1.6;
            white-space: pre-wrap;
            word-break: break-all;
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            border: 1px solid #dee2e6;
        }}
        
        .different {{
            background-color: #ffcdd2;
            color: #d32f2f;
            font-weight: bold;
            padding: 2px 4px;
            border-radius: 3px;
        }}
        
        .legend {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 30px;
            display: flex;
            justify-content: center;
            gap: 30px;
            flex-wrap: wrap;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }}
        
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        
        .legend-color {{
            width: 30px;
            height: 20px;
            border-radius: 4px;
            border: 1px solid #ddd;
        }}
        
        .match-indicator {{
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: bold;
            margin-left: 10px;
        }}
        
        .match {{
            background: #e8f5e9;
            color: #2e7d32;
        }}
        
        .differ {{
            background: #ffebee;
            color: #c62828;
        }}
        
        @media (max-width: 1200px) {{
            .query-content {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 SQL Query Comparison Analysis</h1>
            <div class="stats">
                <div class="stat-card">
                    <h3>Total Queries</h3>
                    <div class="value">{len(model_queries)}</div>
                </div>
                <div class="stat-card">
                    <h3>Model Average Length</h3>
                    <div class="value">{model_avg_length:.1f}</div>
                    <div class="subtext">characters</div>
                </div>
                <div class="stat-card">
                    <h3>Correct Average Length</h3>
                    <div class="value">{correct_avg_length:.1f}</div>
                    <div class="subtext">characters</div>
                </div>
                <div class="stat-card">
                    <h3>Difference</h3>
                    <div class="value">{model_avg_length - correct_avg_length:.1f}</div>
                    <div class="subtext">{((model_avg_length / correct_avg_length - 1) * 100):.1f}% shorter</div>
                </div>
            </div>
        </div>
        
        <div class="legend">
            <div class="legend-item">
                <div class="legend-color" style="background-color: #ffcdd2; border-color: #d32f2f;"></div>
                <span><strong>Red Highlight:</strong> Differences between model and correct query</span>
            </div>
        </div>
"""

    # Add each query comparison
    for i in range(min(len(model_queries), len(correct_queries))):
        model_q = model_queries[i]
        correct_q = correct_queries[i]
        
        model_highlighted, correct_highlighted = highlight_differences(model_q, correct_q)
        
        is_match = model_q == correct_q
        match_class = "match" if is_match else "differ"
        match_text = "✓ Exact Match" if is_match else "✗ Differences Found"
        
        html_content += f"""
        <div class="query-comparison">
            <div class="query-header">
                Query #{i+1}
                <span class="match-indicator {match_class}">{match_text}</span>
            </div>
            <div class="query-content">
                <div class="query-side model">
                    <h3>Model Generated ({len(model_q)} chars)</h3>
                    <div class="sql-code">{model_highlighted}</div>
                </div>
                <div class="query-side correct">
                    <h3>Correct Answer ({len(correct_q)} chars)</h3>
                    <div class="sql-code">{correct_highlighted}</div>
                </div>
            </div>
        </div>
"""

    html_content += """
    </div>
</body>
</html>
"""

    # Write to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return output_file

def main():
    # Automatically use hardcoded file paths
    model_file = MODEL_QUERIES_FILE
    correct_file = CORRECT_QUERIES_FILE
    
    try:
        print(f"Reading model queries from: {model_file}")
        if not os.path.exists(model_file):
            print(f"❌ Error: File not found - {model_file}")
            sys.exit(1)
        model_queries = read_queries_from_file(model_file)
        print(f"  Loaded {len(model_queries)} queries")
        
        print(f"Reading correct queries from: {correct_file}")
        if not os.path.exists(correct_file):
            print(f"❌ Error: File not found - {correct_file}")
            sys.exit(1)
        correct_queries = read_queries_from_file(correct_file)
        print(f"  Loaded {len(correct_queries)} queries")
        
        if len(model_queries) == 0:
            print("❌ Error: No queries found in model_queries.txt.rtf")
            sys.exit(1)
        if len(correct_queries) == 0:
            print("❌ Error: No queries found in correct_queries.txt.rtf")
            sys.exit(1)
        
        print("\nGenerating HTML comparison...")
        output_file = generate_html_comparison(model_queries, correct_queries)
        
        print(f"\n✅ Success! HTML comparison generated: {output_file}")
        print(f"   Total queries compared: {min(len(model_queries), len(correct_queries))}")
        print(f"\nOpen the file in your browser to view the comparison.")
        
    except FileNotFoundError as e:
        print(f"❌ Error: File not found - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
