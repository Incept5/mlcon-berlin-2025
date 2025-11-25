"""LA City Payroll SQL Query Generator using Groq LLM

This script demonstrates text-to-SQL capabilities using Groq's cloud API.
It retrieves the database schema and sample data, sends them to a Groq LLM to generate
a SQL query based on a natural language prompt, then executes the generated query
and displays the results.

Workflow:
1. Extract database schema from SQLite
2. Get random sample rows for context
3. Send schema + samples to Groq with natural language prompt
4. Parse generated SQL from LLM response
5. Execute SQL query and display results

Requirements:
- SQLite database created by payroll.py (city_payroll.db)
- groq package installed (pip install groq)
- GROQ_API_KEY environment variable set

Example Query: "List average hourly rate for LAPD employees by year"
"""

import sqlite3
import json
import re
import random
import os
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def get_schema(db_path):
    """Retrieve the SQL schema definition for the payroll table.
    
    Queries the SQLite system table (sqlite_master) to get the CREATE TABLE
    statement that defines the payroll table structure. This schema will be
    sent to the LLM to help it understand what columns are available and
    their data types.
    
    Parameters:
    -----------
    db_path : str
        Path to the SQLite database file
    
    Returns:
    --------
    str
        The SQL CREATE TABLE statement for the payroll table
    
    Example Output:
    ---------------
    CREATE TABLE payroll (
        Year INTEGER,
        Department_Title TEXT,
        Job_Class_Title TEXT,
        Base_Pay REAL,
        ...
    )
    """
    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Query sqlite_master (system table) for the CREATE TABLE statement
    # sqlite_master stores metadata about all tables, indices, etc.
    cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='payroll'")
    schema = cursor.fetchone()[0]
    
    # Clean up database connection
    conn.close()
    return schema


def get_random_rows(db_path, num_rows=3):
    """Extract random sample rows from the payroll table.
    
    Provides the LLM with example data to help it understand:
    - Actual data formats and values
    - Column naming conventions
    - Data types in practice
    - Typical value ranges
    
    This context improves the quality of generated SQL queries by showing
    the LLM what the actual data looks like, not just the schema.
    
    Parameters:
    -----------
    db_path : str
        Path to the SQLite database file
    num_rows : int, optional
        Number of random rows to retrieve (default: 3)
    
    Returns:
    --------
    str
        JSON-formatted string containing random sample rows
        Returns empty array JSON if no data exists
    
    Implementation Note:
    --------------------
    Uses LIMIT/OFFSET for random sampling rather than ORDER BY RANDOM()
    for better performance on large tables.
    """
    # Connect to database with Row factory for dict-like row access
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row  # Enables column name access
    cursor = conn.cursor()

    # Get total number of rows to determine valid sampling range
    cursor.execute("SELECT COUNT(*) FROM payroll")
    total_rows = cursor.fetchone()[0]

    # Handle empty table case
    if total_rows == 0:
        conn.close()
        return json.dumps([], indent=2)

    # Generate random indices within the valid range
    # Use min() to handle cases where num_rows > total_rows
    random_indices = random.sample(range(1, total_rows + 1), min(num_rows, total_rows))
    results = []

    # Fetch each random row using LIMIT/OFFSET
    # This is more efficient than ORDER BY RANDOM() on large tables
    for idx in random_indices:
        cursor.execute("SELECT * FROM payroll LIMIT 1 OFFSET ?", (idx - 1,))
        row = cursor.fetchone()
        if row:
            # Convert Row object to dictionary for JSON serialization
            results.append(dict(row))

    # Clean up and return formatted JSON
    conn.close()
    return json.dumps(results, indent=2)


def query_groq(schema, sample_data=None):
    """Send database schema and natural language query to Groq LLM for SQL generation.
    
    Constructs a prompt containing:
    1. The database schema (table structure)
    2. Sample rows (optional, for better context)
    3. A natural language query description
    
    The LLM analyzes this information and generates an appropriate SQL query.
    
    Parameters:
    -----------
    schema : str
        The SQL CREATE TABLE statement defining the table structure
    sample_data : str, optional
        JSON-formatted sample rows to provide additional context
    
    Returns:
    --------
    str
        The LLM's response, which should contain a SQL query
        Returns error message if request fails
    
    API Configuration:
    ------------------
    - Model: llama-3.3-70b-versatile (fast, high-quality language model)
    - Temperature: 0.7 (balanced between creativity and consistency)
    - Max tokens: 1024 (sufficient for SQL query generation)
    - Top_p: 0.95 (nucleus sampling for quality outputs)
    
    Example Prompt:
    ---------------
    "From the following table description in SQLite please write a query to list the
    average hourly rate for people working in the LAPD for each year
    
    [schema here]
    
    Sample rows:
    [sample data here]"
    """
    try:
        # Get API key from environment
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        
        # Initialize Groq client
        client = Groq(api_key=api_key)
        
        # Construct the prompt with task description and schema
        # This specific prompt asks for average hourly rates by year for LAPD
        prompt = f"""From the following table description in SQLite please write a query to list the
average hourly rate for people working in the LAPD for each year\n\n{schema}"""

        # Append sample data if provided for better context
        # Sample data helps the LLM understand actual column values and formats
        if sample_data:
            prompt += f"\n\nSample rows:\n{sample_data}"

        # Prepare messages with system prompt for SQL generation
        messages = [
            {
                "role": "system",
                "content": "You are an expert SQL query generator. Generate only valid SQL queries without explanations unless asked. Format SQL queries in code blocks."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        # Make API request to Groq
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",  # Fast, high-quality model
            messages=messages,
            temperature=0.7,    # Balance between deterministic and creative
            max_tokens=1024,    # Sufficient for SQL queries
            top_p=0.95,         # Nucleus sampling for quality
            stream=False        # Get complete response at once
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"Error: {e}"


def execute_query(query, db_path):
    """Execute a SQL query against the payroll database.
    
    Takes a SQL query string (typically generated by the LLM) and executes it
    against the SQLite database, returning all result rows.
    
    Parameters:
    -----------
    query : str
        SQL query to execute (e.g., "SELECT Year, AVG(Hourly_Rate) FROM payroll ...")
    db_path : str
        Path to the SQLite database file
    
    Returns:
    --------
    list of tuples
        Query results, where each tuple is a row
        Empty list if query returns no rows
    
    Raises:
    -------
    sqlite3.Error
        If the SQL query is invalid or execution fails
    
    Example:
    --------
    >>> execute_query("SELECT Year, AVG(Base_Pay) FROM payroll GROUP BY Year", "city_payroll.db")
    [(2020, 75000.50), (2021, 77500.25), (2022, 80000.75)]
    """
    # Connect to the database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Execute the query (may raise sqlite3.Error if invalid)
    cursor.execute(query)
    
    # Fetch all results as list of tuples
    results = cursor.fetchall()
    
    # Clean up connection
    conn.close()
    return results


def extract_sql(generated_query):
    """Extract SQL query from LLM-generated text.
    
    LLMs often wrap code in markdown code blocks or include explanatory text.
    This function uses regex patterns to extract just the SQL query portion.
    
    Extraction Strategy:
    1. First, look for SQL in markdown code blocks (```sql ... ```)
    2. If not found, search for SELECT statements in the raw text
    3. Clean up extracted SQL (remove trailing semicolons)
    
    Parameters:
    -----------
    generated_query : str
        The complete response from the LLM, which may contain:
        - Markdown code blocks
        - Explanatory text
        - Multiple SQL statements
        - Comments
    
    Returns:
    --------
    str or None
        Extracted SQL query string, or None if no valid SQL found
    
    Examples:
    ---------
    Input: "Here's your query:\n```sql\nSELECT * FROM payroll;\n```"
    Output: "SELECT * FROM payroll"
    
    Input: "You can use: SELECT Year, AVG(Base_Pay) FROM payroll GROUP BY Year;"
    Output: "SELECT Year, AVG(Base_Pay) FROM payroll GROUP BY Year"
    """
    # Pattern 1: Look for SQL within markdown code blocks
    # Matches: ```sql ... ``` or ```SQL ... ```
    sql_pattern = r"```sql\s*(.*?)\s*```"
    matches = re.findall(sql_pattern, generated_query, re.DOTALL | re.IGNORECASE)

    # If no markdown code blocks found, try Pattern 2
    if not matches:
        # Pattern 2: Look for raw SELECT statements in the text
        # Matches: SELECT ... (up to semicolon or end of string)
        # re.IGNORECASE handles "SELECT", "select", "Select", etc.
        select_pattern = r"(SELECT.*?;?)"
        select_matches = re.findall(select_pattern, generated_query, re.DOTALL | re.IGNORECASE)

        if select_matches:
            # Clean up the extracted SQL
            sql_query = select_matches[0].strip()
            
            # Remove trailing semicolon (SQLite doesn't require it and it can cause issues)
            if sql_query.endswith(';'):
                sql_query = sql_query[:-1]
            return sql_query
        else:
            # No SQL found using either pattern
            return None
    else:
        # Return the first match from markdown code block, cleaned up
        return matches[0].strip()


def main():
    """Main execution function demonstrating text-to-SQL with Groq.
    
    Workflow:
    1. Retrieve database schema
    2. Get random sample rows for context
    3. Send schema + samples to LLM with natural language query
    4. Extract SQL from LLM response
    5. Execute generated SQL
    6. Display results in formatted table
    
    Error Handling:
    - Validates SQL extraction before execution
    - Catches and reports SQL execution errors
    - Shows raw LLM response if SQL extraction fails
    """
    # Path to the SQLite database created by payroll.py
    db_path = 'city_payroll.db'

    print("=" * 70)
    print("LA City Payroll SQL Query Generator using Groq")
    print("=" * 70)
    print("\nRequirements:")
    print("  - GROQ_API_KEY environment variable set")
    print("  - groq library installed (pip install groq)")
    print("  - SQLite database (city_payroll.db)")
    print()

    # Step 1: Get the database schema (table structure)
    # This tells the LLM what columns exist and their types
    schema = get_schema(db_path)
    print("Schema:")
    print(schema)

    # Step 2: Get random sample rows for additional context
    # This shows the LLM actual data formats and typical values
    sample_data = get_random_rows(db_path)
    print("\nSample Data:")
    print(sample_data)

    # Step 3: Send schema and samples to Groq for SQL generation
    # The LLM will generate a SQL query based on the natural language prompt
    print("\n" + "=" * 70)
    print("Generating SQL query with Groq...")
    print("=" * 70)
    generated_query = query_groq(schema, sample_data)
    print("\nGenerated Response:")
    print(generated_query)

    # Step 4: Extract clean SQL from the LLM's response
    # The response may contain markdown, explanations, or other text
    sql_query = extract_sql(generated_query)

    # Validate that we successfully extracted SQL
    if not sql_query:
        print("\nError: No SQL query found")
        print("Raw response:")
        print(generated_query)
        return  # Exit if we couldn't find valid SQL

    print(f"\nExtracted SQL: {sql_query}")

    # Step 5: Execute the generated SQL query
    try:
        print("\n" + "=" * 70)
        print("Executing query...")
        print("=" * 70)
        results = execute_query(sql_query, db_path)

        # Step 6: Display results in formatted markdown-style table
        print("\nResults:")
        print("| Year | Average |")
        print("|------|---------|")
        for year, average in results:
            print(f"| {year} | {average:.2f} |")

    except sqlite3.Error as e:
        # Handle SQL execution errors (invalid query, missing columns, etc.)
        print(f"Error executing SQL: {e}")


# Entry point: only execute if script is run directly (not imported)
if __name__ == "__main__":
    # Execute the main workflow
    main()
