from flask import Flask, request, jsonify, render_template
from sentence_transformers import SentenceTransformer
import torch
import psycopg2
from psycopg2 import pool
import os
app = Flask(__name__)

# Load the e5-multilingual-large model using SentenceTransformer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
model = SentenceTransformer('intfloat/e5-large-v2', device=device)

# Set up the connection pool

if os.getenv('ENV') == 'production':
    # GCP Production using Unix socket
    pg_pool = psycopg2.pool.SimpleConnectionPool(
        minconn=1,
        maxconn=10,
        dbname='postgres',
        user='exrec',
        password='wollstonecraft',
        host='/cloudsql/expertrecommender:europe-west4:exrec',  # GCP Unix socket path
        port='5432'
    )
else:
    # Local development using TCP
    pg_pool = psycopg2.pool.SimpleConnectionPool(
        minconn=1,
        maxconn=10,
        dbname='postgres',
        user='exrec',
        password='wollstonecraft',
        host='34.34.106.184',  # Local PostgreSQL host
        port='5432'
    )


# Function to calculate the embedding for a new abstract using e5-multilingual-large
def get_embeddings(text):
    query = "query: " + text  # Prefix the text with "query:"
    embeddings = model.encode(query, convert_to_numpy=True)
    return embeddings

# Route for the home page (index)
@app.route('/')
def index():
    return render_template('index.html')

# /query route to accept the 'q' parameter and return results
@app.route('/query', methods=['GET', 'POST'])

# A function to query the database using the connection pool1
def query():
    if request.method == 'POST':
        data = request.form['q']

        if not data:
            return jsonify({"error": "Missing query text"}), 400

        print(" 1 getting embeddings from query")
        new_embedding = get_embeddings(data).tolist()

        try:
            # Step 2: Get a connection from the pool
            print(" 2 Getting connection from the pool")
            conn = pg_pool.getconn()
            if conn:
                cur = conn.cursor()

                # Step 3: Execute the SQL query using the new embedding
                print(' 3 Executing sql query calculating distances')
                # sql_query = """
                #     SELECT ev.id, ev.embedding <=> %s::vector AS distance, 
                #         COALESCE(wt.title, etw.title) AS title,
                #         COALESCE(was.auth_id, wae.auth_id) AS auids

                #     FROM e5_vectors ev
                #     LEFT JOIN w_titles_sn wt ON ev.id = wt.id
                #     LEFT JOIN w_titles_els etw ON ev.id = etw.id
                #     LEFT JOIN w_auth_sn was ON ev.id = was.id
                #     LEFT JOIN w_auth_els wae ON ev.id = wae.id
                #     ORDER BY distance
                #     LIMIT 10;
                # """

                sql_query='''
                        WITH top_works AS (
                            SELECT ev.id, 
                                ev.embedding <=> %s::vector AS distance, 
                                COALESCE(wt.title, etw.title) AS title,
                                COALESCE(was.auth_id, wae.auth_id) AS auids
                            FROM e5_vectors ev
                            LEFT JOIN w_titles_sn wt ON ev.id = wt.id
                            LEFT JOIN w_titles_els etw ON ev.id = etw.id
                            LEFT JOIN w_auth_sn was ON ev.id = was.id
                            LEFT JOIN w_auth_els wae ON ev.id = wae.id
                            ORDER BY distance
                            LIMIT 100
                        )

                        SELECT tw.id, 
                            tw.distance, 
                            tw.title, 
                            STRING_AGG(tw.auids, ', ') AS aggregated_auids
                        FROM top_works tw
                        GROUP BY tw.id, tw.distance, tw.title
                        ORDER BY tw.distance;'''

        


                cur.execute(sql_query, (new_embedding,))

                results = cur.fetchall()
                print(" 4 ok")

                # Step 4: Convert results to a list of dictionaries for the template
                results_list = [
                    {"id": row[0], "distance": row[1], "title": row[2], "auid": row[3]} 
                    for row in results
                ]

                # Step 5: Release the connection back to the pool
                pg_pool.putconn(conn)

                # Step 6: Return the results in HTML format
                return render_template('query.html', results=results_list)

        except Exception as e:
            return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)