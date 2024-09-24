import requests
import psycopg2
import json
import sys
import time


# PostgreSQL connection parameters
db_params = {
    'dbname': 'postgres',
    'user': 'exrec',
    'password': 'wollstonecraft',
    'host': '34.34.106.184',
    'port': '5432'
}

conn = psycopg2.connect(**db_params)
cur = conn.cursor()

# Create table for a_names
cur.execute("""
    CREATE TABLE IF NOT EXISTS a_names (
        id VARCHAR PRIMARY KEY,
        orcid VARCHAR,
        name VARCHAR
    )
""")

# Create table for a_aff
cur.execute("""
    CREATE TABLE IF NOT EXISTS a_aff (
        id VARCHAR,
        aff_id VARCHAR,
        aff_name VARCHAR,    
        year INT,
        CONSTRAINT unique_aff UNIQUE (id, aff_id, year)
    )
""")

# Create table for a_topics
cur.execute("""
    CREATE TABLE IF NOT EXISTS a_topics (
        id VARCHAR,
        topic_id VARCHAR,
        topic_name VARCHAR,
        topic_count INT,
        CONSTRAINT unique_topic UNIQUE (id, topic_id)
    )
""")

# Create table for a_stats
cur.execute("""
    CREATE TABLE IF NOT EXISTS a_stats (
        id VARCHAR PRIMARY KEY,
        works_count INT,
        cited_count INT,
        h_index FLOAT,
        mean_citedness FLOAT
    )
""")

conn.commit()

# Define batch size
batch_size = 50
offset = 0
api_url = "https://api.openalex.org/authors"

while True:
    # Fetch a batch of author IDs from w_abs_sn
    cur.execute(f"SELECT auth_id FROM w_auth_sn LIMIT {batch_size} OFFSET %s", (offset,))
    author_ids = cur.fetchall()

    if not author_ids:
        print("All authors processed.")
        break

    # Convert list of tuples to a pipe-separated string
    author_ids_str = '|'.join(str(id[0]) for id in author_ids)
    print(author_ids_str)
    # Check if any author IDs are already in a_names
    cur.execute(f"SELECT id FROM a_names WHERE id = ANY(%s)", (author_ids,))
    existing_ids = set(row[0] for row in cur.fetchall())
    print(existing_ids)

    # Filter out existing author_ids
    author_ids_to_fetch = [id[0] for id in author_ids if id[0] not in existing_ids]
    print(author_ids_to_fetch)
    
    if not author_ids_to_fetch:
        print(f"No new authors in batch {offset // batch_size + 1}. Skipping API call.")
        offset += batch_size
        continue

    # Set up parameters for the OpenAlex API request
    params = {
        'filter': f'ids.openalex:{ "|".join(author_ids_to_fetch) }',
        'per_page': 100,
        'select': 'id,orcid,display_name,works_count,cited_by_count,summary_stats,affiliations,topics',
        'mailto': 'a@journaltierlist.com'
    }

    # Try querying the API
    try:
        response = requests.get(api_url, params=params)
        response.raise_for_status()
        data = response.json()

        # Process the data
        for author in data['results']:
            id = author.get('id')
            orcid = author.get('orcid')
            display_name = author.get('display_name')
            works_count = author.get('works_count')
            cited_by_count = author.get('cited_by_count')
            
            # Extract summary_stats
            summary_stats = author.get('summary_stats', {})
            mean_citedness = summary_stats.get('2yr_mean_citedness')
            h_index = summary_stats.get('h_index')

            # Extract affiliations
            affiliations_data = []
            for affiliation in author.get('affiliations', []):
                institution = affiliation.get('institution', {})
                affiliations_data.append({
                    'institution_id': institution.get('id'),
                    'institution_display_name': institution.get('display_name'),
                    'years': affiliation.get('years', [])
                })

            # Extract topics
            topics_data = []
            for topic in author.get('topics', []):
                topics_data.append({
                    'topic_id': topic.get('id'),
                    'topic_display_name': topic.get('display_name'),
                    'topic_count': topic.get('count')
                })

            # Insert into a_names
            cur.execute("""
                INSERT INTO a_names (id, orcid, name)
                VALUES (%s, %s, %s)
                ON CONFLICT (id) DO NOTHING
            """, (id, orcid, display_name))

            # Insert into a_stats
            cur.execute("""
                INSERT INTO a_stats (id, works_count, cited_count, h_index, mean_citedness)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (id) DO NOTHING
            """, (id, works_count, cited_by_count, h_index, mean_citedness))

            # Insert affiliations into a_aff
            for affiliation in affiliations_data:
                for year in affiliation['years']:
                    cur.execute("""
                        INSERT INTO a_aff (id, aff_id, aff_name, year)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT DO NOTHING
                    """, (id, affiliation['institution_id'], affiliation['institution_display_name'], year))

            # Insert topics into a_topics
            for topic in topics_data:
                cur.execute("""
                    INSERT INTO a_topics (id, topic_id, topic_name, topic_count)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT DO NOTHING
                """, (id, topic['topic_id'], topic['topic_display_name'], topic['topic_count']))

        conn.commit()

    except requests.exceptions.RequestException as e:
        print(f"API request failed for batch {offset // batch_size + 1}: {e}")
        time.sleep(5)  # Delay before retrying the next batch

    # Move to the next batch
    offset += batch_size
    time.sleep(0.2)  # Small delay between batches

print("Processing complete.")