import sqlite3
import requests
import json
import re
import random


def get_schema(db_path):
    """Get schema from SQLite database"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='payroll'")
    schema = cursor.fetchone()[0]
    conn.close()
    return schema


def get_random_rows(db_path, num_rows=3):
    """Get random rows from the payroll table and return as JSON"""
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row  # This enables column access by name
        cursor = conn.cursor()

        # First get the total row count
        cursor.execute("SELECT COUNT(*) FROM payroll")
        total_rows = cursor.fetchone()[0]
        
        if total_rows == 0:
            conn.close()
            return json.dumps([], indent=2)

        # Select random rows
        random_indices = random.sample(range(1, total_rows + 1), min(num_rows, total_rows))
        results = []

        for idx in random_indices:
            cursor.execute("SELECT * FROM payroll LIMIT 1 OFFSET ?", (idx - 1,))
            row = cursor.fetchone()
            if row:
                results.append(dict(row))

        conn.close()
        return json.dumps(results, indent=2)
    except sqlite3.Error as e:
        raise sqlite3.Error(f"Error accessing sample data: {e}")


def query_ollama(schema, sample_data=None):
    """Query Ollama API for SQL generation"""
    # Build the prompt with schema and sample data if available
    prompt = (f"""From the following table description in SQLite please write a query to list the
              average hourly rate for people working in the LAPD for each year\n\n{schema}""")

    if sample_data:
        prompt += f"\n\nHere are some sample rows from the table for reference:\n{sample_data}"

    response = requests.post('http://localhost:11434/api/generate',
                             json={
                                 "model": "qwen3:latest",
                                 "prompt": prompt,
                                 "stream": False,
                                "options": {
                                    "num_ctx": 32768,
                                    "temperature": 0.7,
                                }

                             })

    if response.status_code == 200:
        return response.json()['response']
    else:
        return f"Error: {response.status_code}"


def execute_query(query, db_path):
    """Execute SQL query and return results"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(query)
    results = cursor.fetchall()
    conn.close()
    return results


def main():
    db_path = 'city_payroll.db'

    # Get schema
    schema = get_schema(db_path)
    print("\nSchema:")
    print(schema)

    # Get sample data
    sample_data = get_random_rows(db_path)
    print("\nSample Data:")
    print(sample_data)

    # Get query from Ollama
    generated_query = query_ollama(schema, sample_data)
    print("\nGenerated Query:")
    print(generated_query)

    # Extract the SQL for the query with proper error checking
    sql_pattern = r"```sql\s*(.*?)\s*```"
    matches = re.findall(sql_pattern, generated_query, re.DOTALL)

    if not matches:
        # If no SQL extras blocks found, try to extract SQL directly
        # Look for SELECT statements
        select_pattern = r"(SELECT.*?;?)"
        select_matches = re.findall(select_pattern, generated_query, re.DOTALL | re.IGNORECASE)
        
        if select_matches:
            sql_query = select_matches[0].strip()
            # Remove trailing semicolon if present
            if sql_query.endswith(';'):
                sql_query = sql_query[:-1]
        else:
            print("\nError: No SQL query found in the generated response.")
            print("Raw response:")
            print(generated_query)
            return
    else:
        sql_query = matches[0].strip()

    print(f"\nExtracted SQL Query: {sql_query}")

    # Execute the query
    try:
        results = execute_query(sql_query, db_path)

        # Format results as markdown table
        markdown_output = "| Year | Average |\n|-----------------|-------|\n"
        for year, average in results:
            markdown_output += f"| {year} | {average} |\n"

        print("\nResults:")
        print(markdown_output)

    except sqlite3.Error as e:
        print(f"Error executing query: {e}")


if __name__ == "__main__":
    main()