import sqlite3
import requests
import json
import re
import random


def get_schema(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='payroll'")
    schema = cursor.fetchone()[0]
    conn.close()
    return schema


def get_random_rows(db_path, num_rows=3):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM payroll")
    total_rows = cursor.fetchone()[0]

    if total_rows == 0:
        conn.close()
        return json.dumps([], indent=2)

    random_indices = random.sample(range(1, total_rows + 1), min(num_rows, total_rows))
    results = []

    for idx in random_indices:
        cursor.execute("SELECT * FROM payroll LIMIT 1 OFFSET ?", (idx - 1,))
        row = cursor.fetchone()
        if row:
            results.append(dict(row))

    conn.close()
    return json.dumps(results, indent=2)


def query_ollama(schema, sample_data=None):
    prompt = f"""From the following table description in SQLite please write a query to list the
average hourly rate for people working in the LAPD for each year\n\n{schema}"""

    if sample_data:
        prompt += f"\n\nSample rows:\n{sample_data}"

    response = requests.post(
        'http://localhost:11434/api/generate',
        json={
            "model": "qwen3-vl:4b-instruct",
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_ctx": 32768,
                "temperature": 0.7,
            }
        }
    )

    if response.status_code == 200:
        return response.json()['response']
    else:
        return f"Error: {response.status_code}"


def execute_query(query, db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(query)
    results = cursor.fetchall()
    conn.close()
    return results


def extract_sql(generated_query):
    sql_pattern = r"```sql\s*(.*?)\s*```"
    matches = re.findall(sql_pattern, generated_query, re.DOTALL)

    if not matches:
        select_pattern = r"(SELECT.*?;?)"
        select_matches = re.findall(select_pattern, generated_query, re.DOTALL | re.IGNORECASE)

        if select_matches:
            sql_query = select_matches[0].strip()
            if sql_query.endswith(';'):
                sql_query = sql_query[:-1]
            return sql_query
        else:
            return None
    else:
        return matches[0].strip()


def main():
    db_path = 'city_payroll.db'

    schema = get_schema(db_path)
    print("\nSchema:")
    print(schema)

    sample_data = get_random_rows(db_path)
    print("\nSample Data:")
    print(sample_data)

    generated_query = query_ollama(schema, sample_data)
    print("\nGenerated Query:")
    print(generated_query)

    sql_query = extract_sql(generated_query)

    if not sql_query:
        print("\nError: No SQL query found")
        print("Raw response:")
        print(generated_query)
        return

    print(f"\nExtracted SQL: {sql_query}")

    try:
        results = execute_query(sql_query, db_path)

        print("\nResults:")
        print("| Year | Average |")
        print("|------|---------|")
        for year, average in results:
            print(f"| {year} | {average} |")

    except sqlite3.Error as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()