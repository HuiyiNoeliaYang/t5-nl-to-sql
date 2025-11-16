#!/usr/bin/env python3
"""
Script to fix a missing record in a pickle file by executing the corresponding SQL query.

Usage:
    python fix_missing_record.py \
        --sql_file results/t5_ft_t5_ft_with8batch_test.sql \
        --pkl_file records/t5_ft_t5_ft_with8batch_test.pkl \
        --query_index 431
"""

import argparse
import pickle
import sqlite3
import sys

DB_PATH = 'data/flight_database.db'

def compute_record(query):
    """Execute a single SQL query and return results."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        cursor.execute(query)
        rec = cursor.fetchall()
        error_msg = ""
    except Exception as e:
        rec = []
        error_msg = f"{type(e).__name__}: {e}"
    finally:
        conn.close()
    
    return rec, error_msg

def fix_missing_record(sql_file, pkl_file, query_index=None):
    """
    Fix missing record(s) in pickle file by executing SQL queries.
    
    If query_index is None, finds all empty records and fixes them.
    If query_index is specified, fixes only that index.
    """
    # Load SQL queries
    print(f"Loading SQL queries from {sql_file}...")
    with open(sql_file, 'r') as f:
        sql_queries = [q.strip() for q in f.readlines()]
    
    # Load pickle file
    print(f"Loading records from {pkl_file}...")
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    
    # Handle different pickle formats
    if isinstance(data, tuple) and len(data) == 2:
        records, error_msgs = data
    elif isinstance(data, list):
        records = data
        error_msgs = [""] * len(data)
    else:
        print(f"Error: Unexpected pickle format. Expected tuple (records, error_msgs) or list of records.")
        sys.exit(1)
    
    print(f"Found {len(records)} records and {len(sql_queries)} SQL queries")
    
    # Find indices with missing records (empty lists or "Query timed out" errors)
    missing_indices = []
    if query_index is not None:
        if query_index < 0 or query_index >= len(records):
            print(f"Error: query_index {query_index} is out of range (0-{len(records)-1})")
            sys.exit(1)
        missing_indices = [query_index]
    else:
        # Find all empty records or timeout errors
        for i, (rec, error_msg) in enumerate(zip(records, error_msgs)):
            if not rec or (error_msg and "timeout" in error_msg.lower()):
                missing_indices.append(i)
    
    if not missing_indices:
        print("✓ No missing records found! All records are complete.")
        return
    
    print(f"\nFound {len(missing_indices)} missing record(s) at indices: {missing_indices}")
    
    # Fix each missing record
    fixed_count = 0
    for idx in missing_indices:
        if idx >= len(sql_queries):
            print(f"⚠️  Warning: Index {idx} is out of range for SQL queries. Skipping.")
            continue
        
        sql_query = sql_queries[idx]
        print(f"\nFixing index {idx}...")
        print(f"SQL: {sql_query[:100]}..." if len(sql_query) > 100 else f"SQL: {sql_query}")
        
        rec, error_msg = compute_record(sql_query)
        
        if error_msg:
            print(f"  ⚠️  Error executing query: {error_msg}")
            records[idx] = rec  # Still save empty result
            error_msgs[idx] = error_msg
        else:
            print(f"  ✓ Success! Retrieved {len(rec)} record(s)")
            records[idx] = rec
            error_msgs[idx] = ""
            fixed_count += 1
    
    # Save updated pickle file
    print(f"\nSaving updated records to {pkl_file}...")
    with open(pkl_file, 'wb') as f:
        if isinstance(data, tuple):
            pickle.dump((records, error_msgs), f)
        else:
            pickle.dump(records, f)
    
    print(f"✓ Fixed {fixed_count}/{len(missing_indices)} missing record(s)")
    print(f"✓ Updated pickle file saved: {pkl_file}")

def main():
    parser = argparse.ArgumentParser(description='Fix missing records in a pickle file')
    parser.add_argument('--sql_file', type=str, required=True,
                        help='Path to SQL file containing queries')
    parser.add_argument('--pkl_file', type=str, required=True,
                        help='Path to pickle file containing records')
    parser.add_argument('--query_index', type=int, default=None,
                        help='Specific query index to fix (0-indexed). If not specified, fixes all missing records.')
    
    args = parser.parse_args()
    
    fix_missing_record(args.sql_file, args.pkl_file, args.query_index)

if __name__ == "__main__":
    main()

