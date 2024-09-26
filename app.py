from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

from sentence_transformers import SentenceTransformer
import torch
import psycopg2
from psycopg2 import pool
import os
from datetime import datetime
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://127.0.0.1:8080"}})



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
    query = "passage: " + text  # Prefix the text with "query:"
    embeddings = model.encode(query, convert_to_numpy=True)
    return embeddings

# Route for the home page (index)
@app.route('/')
def index():
    return render_template('index.html')

# /query route to accept the 'q' parameter and return results
@app.route('/query', methods=['GET'])

# A function to query the database using the connection pool1
def query():
    if request.method == 'GET':
        data = request.args.get('q')

        if not data:
            return jsonify({"error": "Missing query text"}), 400

        print(" 1 getting embeddings from query ", datetime.now().strftime("%H:%M:%S"))
        new_embedding = get_embeddings(data).tolist()

        try:
            # Step 2: Get a connection from the pool
            print(" 2 Getting connection from the pool ", datetime.now().strftime("%H:%M:%S"))
            conn = pg_pool.getconn()
            if conn:
                cur = conn.cursor()

                # Step 3: Execute the SQL query using the new embedding
                print(' 3 Executing sql query calculating distances', datetime.now().strftime("%H:%M:%S"))

                # sql_query='''
                #         SET LOCAL hnsw.ef_search = 40;
                #         WITH top_works AS (
                #             SELECT ev.id, 
                #                 ev.embedding <=> %s::vector AS distance, 
                #                 COALESCE(wt.title, etw.title) AS title
                #             FROM e5_vectors ev
                #             LEFT JOIN w_titles_sn wt ON ev.id = wt.id
                #             LEFT JOIN w_titles_els etw ON ev.id = etw.id
                #             ORDER BY distance
                #             LIMIT 10
                #         )

                #         SELECT tw.id, 
                #             tw.distance, 
                #             tw.title, 
                #             STRING_AGG(COALESCE(was.auth_id, wae.auth_id), ', ') AS aggregated_auids
                #         FROM top_works tw
                #         LEFT JOIN w_auth_sn was ON tw.id = was.id
                #         LEFT JOIN w_auth_els wae ON tw.id = wae.id
                #         GROUP BY tw.id, tw.distance, tw.title
                #         ORDER BY tw.distance;'''

                sql_query='''
                        SET LOCAL hnsw.ef_search = 40;
                        WITH top_works AS (
                            SELECT ev.id, 
                                ev.embedding <=> %s::vector AS distance, 
                                wt.title AS title
                            FROM e5_vectors ev
                            LEFT JOIN w_titles wt ON ev.id = wt.id
                            ORDER BY distance
                            LIMIT 10
                        ),
                        awd AS (
                        SELECT tw.id, 
                            tw.distance, 
                            tw.title, 
                            wa.auth_id auid
                        FROM top_works tw
                        LEFT JOIN w_auth wa ON tw.id = wa.id
                        )
                        SELECT awd.id, awd.title, awd.auid, awd.distance, a_names.orcid, a_names.name
                        FROM awd 
                        LEFT JOIN a_names ON awd.auid = a_names.id
                        ORDER BY awd.distance
                        '''
                


                cur.execute(sql_query, (new_embedding,))

                results = cur.fetchall()
                print(" 4 ok ", datetime.now().strftime("%H:%M:%S"))

                # Step 4: Convert results to a list of dictionaries for the template
                results_list = [
                    {"id": row[0], "title": row[1], "auid": row[2], "distance": row[3], "orcid": row[4], "name": row[5]} 
                    for row in results
                ]

                # Step 5: Release the connection back to the pool
                pg_pool.putconn(conn)

                # Step 6: Return the results in HTML format
                return jsonify(results_list)

        except Exception as e:
            return jsonify({"error": str(e)}), 500
    return jsonify({"error": str(e)}), 500
if __name__ == "__main__":
    app.run(debug=True)