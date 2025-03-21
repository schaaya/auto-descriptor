import os
from dotenv import load_dotenv
from openai import AsyncAzureOpenAI
import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
import asyncio
import time

load_dotenv()

model="gpt-4o"
client = AsyncAzureOpenAI(  
    api_key = os.getenv("AZURE_OPENAI_KEY"),  
    api_version = "2023-03-15-preview",
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
)
    
TEST_DB = os.getenv("TEST_DB")
engine = create_engine(TEST_DB)

st.set_page_config(layout="wide")


async def chat_completion_request(user_id=None, messages=None, model="gpt-4o", tools=None, response_format=None, stop=None, max_tokens=None, temperature=None):
    
    if tools is not None:
        tool_choice = "auto"
    else:
        tool_choice = None
    
    if response_format is True:
        response_format = {"type": "json_object"}
    else:
        response_format = None
        
    if model == "gpt-35-turbo-16k":
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
    else:
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
            stop=stop,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
    completion_tokens = response.usage.completion_tokens
    prompt_tokens = response.usage.prompt_tokens
    total_tokens = response.usage.total_tokens
    return response
# -------------------------------------------------

# Query the list of databases using sys.databases
@st.cache_data
def get_db():
    query = "SELECT name as database_name FROM sys.databases ORDER BY name"
    try:
        return pd.read_sql(query, engine)
    except Exception as e:
        st.error(f"Error retrieving databases: {e}")
        return pd.DataFrame(columns=["database_name"])

# Query all the tables in the selected database that have at least 10 rows.
@st.cache_data
def get_schema_tables(db_name):
    query = f"""
    SELECT s.name as table_schema, t.name as table_name
    FROM [{db_name}].sys.tables t
    JOIN [{db_name}].sys.schemas s ON t.schema_id = s.schema_id
    JOIN [{db_name}].sys.dm_db_partition_stats p ON t.object_id = p.object_id
    WHERE p.index_id IN (0,1)
    GROUP BY s.name, t.name
    HAVING SUM(p.row_count) >= 10
    ORDER BY s.name, t.name
    """
    try:
        return pd.read_sql(query, engine)
    except Exception as e:
        st.error(f"Error retrieving tables: {e}")
        return pd.DataFrame(columns=["table_schema", "table_name"])

# Generate a basic DDL for a given table by querying INFORMATION_SCHEMA
def get_table_ddl(fully_qualified_table_name):
    parts = fully_qualified_table_name.split('.')
    if len(parts) == 3:
        db, schema, table = parts
    else:
        db = "your_database"  # change as needed
        schema = "dbo"
        table = parts[0]
    query = f"""
    SELECT COLUMN_NAME, DATA_TYPE, CHARACTER_MAXIMUM_LENGTH, NUMERIC_PRECISION, NUMERIC_SCALE, IS_NULLABLE, ORDINAL_POSITION
    FROM [{db}].INFORMATION_SCHEMA.COLUMNS
    WHERE TABLE_SCHEMA = '{schema}' AND TABLE_NAME = '{table}'
    ORDER BY ORDINAL_POSITION
    """
    df = pd.read_sql(query, engine)
    ddl = f"CREATE TABLE {schema}.{table} (\n"
    column_definitions = []
    for _, row in df.iterrows():
        col_def = f"  {row['COLUMN_NAME']} {row['DATA_TYPE']}"
        # Add length/precision details for certain data types
        if row['DATA_TYPE'] in ('varchar', 'char', 'nvarchar', 'nchar'):
            if row['CHARACTER_MAXIMUM_LENGTH'] == -1:
                col_def += "(max)"
            elif row['CHARACTER_MAXIMUM_LENGTH'] is not None:
                col_def += f"({row['CHARACTER_MAXIMUM_LENGTH']})"
        elif row['DATA_TYPE'] in ('decimal', 'numeric'):
            col_def += f"({row['NUMERIC_PRECISION']},{row['NUMERIC_SCALE']})"
        col_def += " NOT NULL" if row['IS_NULLABLE'] == 'NO' else " NULL"
        column_definitions.append(col_def)
    ddl += ",\n".join(column_definitions)
    ddl += "\n);"
    return ddl

st.title("Auto Data Description Generator")
st.subheader("Choose a specific table to generate table and columns descriptions.")

# Allow the user to choose a database (if available)
a, b, c = st.columns(3)
with a:
    db_df = get_db()
    if not db_df.empty:
        db_option = st.selectbox('Choose a database', db_df['database_name'])
    else:
        # Fallback: use the connected database from the connection string
        db_option = 'cerebra-gpt-test-db'

# Retrieve tables from the selected database
df = get_schema_tables(db_option)
if not df.empty:
    with b:
        list_of_schemas = sorted(df['table_schema'].unique())
        schema_option = st.selectbox('Choose a schema', list_of_schemas)
    with c:
        list_of_tables = sorted(df[df["table_schema"] == schema_option]["table_name"].unique())
        table_option = st.selectbox('Choose a table', list_of_tables)

    fully_qualified_table_name = f"{db_option}.{schema_option}.{table_option}"
    
    # Display the first 20 rows of the table
    query_table = f"SELECT TOP 20 * FROM [{db_option}].[{schema_option}].[{table_option}]"
    try:
        df_table = pd.read_sql(query_table, engine)
        st.dataframe(df_table, use_container_width=True)
    except Exception as e:
        st.error(f"Error retrieving table data: {e}")
    
    # Get a sample extract of 20 random rows
    query_sample = f"SELECT TOP 20 * FROM [{db_option}].[{schema_option}].[{table_option}] ORDER BY NEWID()"
    try:
        df_sample = pd.read_sql(query_sample, engine)
        extract = df_sample.to_csv(index=False)
    except Exception as e:
        st.error(f"Error retrieving sample data: {e}")
        extract = ""
    
    # Option to upload a DDL file for the table
    uploaded_ddl = st.file_uploader("Upload DDL file for the table (optional)", type=["sql", "txt"])
    if uploaded_ddl is not None:
        ddl_text = uploaded_ddl.read().decode("utf-8")
    else:
        ddl_text = get_table_ddl(fully_qualified_table_name)
    
    with st.expander("Table DDL"):
        st.code(ddl_text)
    
    human_information = st.text_area("Additional context about the table that SQL Server should consider:", "Not Applicable")
    
    llm_name = st.selectbox("Large Language Model", [
        'mixtral-8x7b', 'snowflake-arctic', 'mistral-large', 'reka-flash',
        'reka-core', 'llama2-70b-chat', 'llama3-8b', 'llama3-70b', 'mistral-7b', 'gemma-7b', 'gpt-4o'
    ])
    
    if st.button('Ask Cortex to generate descriptions'):
        first_ddl_prompt = f"""
        You are a tool with one goal: Generate descriptions and a primary key for a specific SQL Server table and its related columns. 
        You will only respond with markdown text. 
        Do not provide explanations.
        Include a description of the table, outlining its overall purpose and contents.
        Include a description for every column of what it represents, including how it relates to other columns.
        Descriptions can be up to 30 words. 
        In the markdown, include a dbt_constraints.primary_key test with the most likely primary key columns for the table.
        The primary key may have only one column or may need multiple columns to be unique.
        Only specify the dbt_constraints.primary_key test in a "tests:" block.
        To help you, I will give you context about the table, the table ddl and a sample CSV extract of the data.
        I need you to return markdown text exactly like : 
        ###
        Alter Table DDL:
        ---------------
        ```sql
        ALTER TABLE [{db_option}].[{schema_option}].[{table_option}] SET COMMENT = '{{ table description }}';
        ALTER TABLE [{db_option}].[{schema_option}].[{table_option}] ADD CONSTRAINT {table_option}_PK PRIMARY KEY ("{{ PRIMARY_KEY_COLUMN_1 }}", "{{ PRIMARY_KEY_COLUMN_2 }}");
        ALTER TABLE [{db_option}].[{schema_option}].[{table_option}] ALTER COLUMN "{{ COLUMN_NAME_1 }}" COMMENT '{{ column description }}';
        ALTER TABLE [{db_option}].[{schema_option}].[{table_option}] ALTER COLUMN "{{ COLUMN_NAME_2 }}" COMMENT '{{ column description }}';
        ```
        DBT YAML:
        ---------
        ```yaml
        version: 2
        models:
        - name: "{table_option}"
            description: "{{ table description }}"
            columns:
            - name: "{{ COLUMN_NAME_1 }}"
                description: "{{ column description }}"
            - name: "{{ COLUMN_NAME_2 }}"
                description: "{{ column description }}"
            tests:
            - dbt_constraints.primary_key:
                column_names:
                    - "{{ PRIMARY_KEY_COLUMN_1 }}"
                    - "{{ PRIMARY_KEY_COLUMN_2 }}"
        sources:
        - name: "{db_option}_{schema_option}"
            database: "{db_option}"
            schema: "{schema_option}"
            tables:
            - name: "{table_option}"
                description: "{{ table description }}"
                columns:
                - name: "{{ COLUMN_NAME_1 }}"
                    description: "{{ column description }}"
                - name: "{{ COLUMN_NAME_2 }}"
                    description: "{{ column description }}"
                tests:
                - dbt_constraints.primary_key:
                    column_names:
                        - "{{ PRIMARY_KEY_COLUMN_1 }}"
                        - "{{ PRIMARY_KEY_COLUMN_2 }}"
        ```
        ###
        context:
        {human_information}
        ###
        ddl:
        {ddl_text}
        ###
        data:
        {extract}
        """
        # Prepare the message for the chat completion
        messages = [{"role": "user", "content": first_ddl_prompt}]
        
        with st.spinner("Running SQL Server Cortex queries"):
            st.code(first_ddl_prompt)
            try:
                # Call the asynchronous LLM function using asyncio.run
                response = asyncio.run(chat_completion_request(messages=messages, model=llm_name))
                # Extract the text response (adjust this if your response structure differs)
                llm_response = response.choices[0].message.content
                # Remove any escape characters if needed
                llm_response = llm_response.replace("\\_", "_")
                st.subheader("Response")
                st.markdown(llm_response)
            except Exception as e:
                st.error(f"Error in LLM call: {e}")
else:
    st.write("No tables with sufficient data found in the selected database.")
