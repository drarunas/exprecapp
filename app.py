from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
import torch
import psycopg2
from psycopg2 import pool
import os
from datetime import datetime
import pandas as pd
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["http://127.0.0.1:8080", "https://exrecapp.web.app"]}})

# Load the e5-multilingual-large model using SentenceTransformer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
model = SentenceTransformer('intfloat/e5-large-v2', device=device)

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
    return jsonify({"error": "This is not a valid route"}), 400

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

                sql_query='''
                        SET LOCAL hnsw.ef_search = 20;
                        WITH top_works AS (
                            SELECT ev.id, 
                                ev.embedding <=> %s::vector AS distance, 
                                wt.title AS title
                            FROM e5_vectors ev
                            LEFT JOIN w_titles wt ON ev.id = wt.id
                            ORDER BY distance
                            LIMIT 20
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

                results_list = [
                    {"id": row[0], "title": row[1], "auid": row[2], "distance": row[3], "orcid": row[4], "name": row[5]} 
                    for row in results
                ]
               
                # Step 5: Release the connection back to the pool
                pg_pool.putconn(conn)

                return jsonify(results_list)

        except Exception as e:
            return jsonify({"error": str(e)}), 500
    return jsonify({"error": str(e)}), 500

@app.route('/queryauthors', methods=['GET'])
# A function to query the database using the connection pool1
def queryauthors():
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
                
                sql_query='''
                    SET LOCAL hnsw.ef_search = 20;

                    WITH work_distances AS (
                        SELECT awd.auid, 
                            awd.work_id, 
                            MIN(awd.distance) AS distance, 
                            wt.title
                        FROM (
                            SELECT ev.id, ev.embedding <=> %s::vector AS distance
                            FROM e5_vectors ev
                            ORDER BY distance
                            LIMIT 20
                        ) AS top_works
                        JOIN w_auth wa ON top_works.id = wa.id
                        JOIN LATERAL (
                            SELECT wa.auth_id AS auid, ev.id AS work_id, ev.embedding <=> %s::vector AS distance
                            FROM e5_vectors ev
                            JOIN w_auth wa2 ON ev.id = wa2.id
                            WHERE wa.auth_id = wa2.auth_id
                            ORDER BY ev.embedding <=> %s::vector
                            LIMIT 3
                        ) AS awd ON awd.auid = wa.auth_id
                        JOIN w_titles wt ON awd.work_id = wt.id
                        GROUP BY awd.auid, awd.work_id, wt.title
                    )
                    SELECT auid, work_id, distance,
                        AVG(distance) OVER (PARTITION BY auid) AS avg_distance_per_auid
                    FROM work_distances
                    ORDER BY avg_distance_per_auid, distance
                    '''

                cur.execute(sql_query, (new_embedding,new_embedding,new_embedding,))
                # List of tuples
                results = cur.fetchall()
                print(" 4 ok ", datetime.now().strftime("%H:%M:%S"))
                # Return the connection to the pool
                pg_pool.putconn(conn)

                # Convert list of tuples to DataFrame
                df = pd.DataFrame(results, columns=['author', 'work', 'work_dist', 'avg_dist'])
                grouped_data = (
                    df.groupby(['author', 'avg_dist'])
                    .apply(lambda x: x[['work', 'work_dist']].to_dict(orient='records'))
                    .reset_index(name='works')
                    .sort_values(by='avg_dist', ascending=True) 
                ).to_dict(orient='records')
                # Serialize your sorted_results
                return jsonify(grouped_data)

        except Exception as e:
            return jsonify({"error": str(e)}), 500
    return jsonify({"error": str(e)}), 500










# /query route to accept the 'q' parameter and return results
@app.route('/queryresearch', methods=['GET'])
# A function to query the database using the connection pool1
def queryresearch():
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

                sql_query='''
                        SET LOCAL hnsw.ef_search = 20;
                        WITH top_works AS (
                            SELECT ev.id, 
                                ev.embedding <=> %s::vector AS distance, 
                                wt.title AS title
                            FROM e5_vectors ev
                            LEFT JOIN w_titles wt ON ev.id = wt.id
                            ORDER BY distance
                            LIMIT 20
                        ),
                        awd AS (
                        SELECT tw.id, 
                            tw.distance, 
                            tw.title, 
                            wa.auth_id auid
                        FROM top_works tw
                        LEFT JOIN w_auth wa ON tw.id = wa.id
                        )
                        SELECT awd.id, awd.title, array_agg(awd.auid) as auid_array, awd.distance, array_agg(a_names.orcid) AS orcid_array, array_agg(a_names.name) AS name_array
                        FROM awd 
                        LEFT JOIN a_names ON awd.auid = a_names.id
                        GROUP BY awd.id, awd.title, awd.distance
                        ORDER BY awd.distance
                        '''
                
                cur.execute(sql_query, (new_embedding,))

                results = cur.fetchall()
                print(" 4 ok ", datetime.now().strftime("%H:%M:%S"))

                results_list = [
                    {"id": row[0], "title": row[1], "auid": row[2], "distance": row[3], "orcid": row[4], "name": row[5]} 
                    for row in results
                ]
               
                # Step 5: Release the connection back to the pool
                pg_pool.putconn(conn)

                return jsonify(results_list)

        except Exception as e:
            return jsonify({"error": str(e)}), 500
    return jsonify({"error": str(e)}), 500






@app.route('/querytopics', methods=['GET'])
def querytopics():
    if request.method == 'GET':
        data = request.args.get('q')

        if not data:
            return jsonify({"error": "Missing query text"}), 400

        print(" 1 getting embeddings from query ", datetime.now().strftime("%H:%M:%S"))
        new_embedding = get_embeddings(data).tolist()

        try:
            print(" 2 Getting connection from the pool ", datetime.now().strftime("%H:%M:%S"))
            conn = pg_pool.getconn()
            if conn:
                cur = conn.cursor()
                print(' 3 Executing sql query calculating distances', datetime.now().strftime("%H:%M:%S"))
                sql_query='''
                        SET LOCAL hnsw.ef_search = 20;
                        SELECT 
                            top.topic_id_c, 
                            top.topic_name,
                            top.summary,
                            ev.embedding <=> %s::vector AS distance
                        FROM topic_e5_vectors ev
                        LEFT JOIN topics top  ON ev.topic_id = top.topic_id
                        ORDER BY distance
                        LIMIT 10
                        '''
                
                cur.execute(sql_query, (new_embedding,))

                results = cur.fetchall()
                print(" 4 ok ", datetime.now().strftime("%H:%M:%S"))

                results_list = [
                    {"topic_id": row[0], "topic_name": row[1], "summary": row[2], "distance": row[3]} 
                    for row in results
                ]
               
                # Step 5: Release the connection back to the pool
                pg_pool.putconn(conn)

                return jsonify(results_list)

        except Exception as e:
            return jsonify({"error": str(e)}), 500
    return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)