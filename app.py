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
CORS(app, resources={r"/*": {"origins": ["http://127.0.0.1:8080", "https://exrecapp.web.app", "https://exprlabs.com"]}})

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
        page = int(request.args.get('page', 1))
        limit = int(request.args.get('limit', 10))  # Default limit
        with_emails_only = request.args.get('with_emails_only', 'false').lower() == 'true'
        min_h_index = int(request.args.get('minh', 0))
        check_for_cois = request.args.get('coi-check', 'false').lower() == 'true'
        authors = request.args.get('authors', [])
        if len(authors.strip())<2:
            check_for_cois = False
        authors = [author.strip() for author in request.args.get('authors', '').split(',')]
        print(check_for_cois)
        print(authors)


        offset = (page - 1) * limit

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
                    SET LOCAL hnsw.ef_search = 1000;
                    WITH
                    AUS AS (
                    SELECT ap.auth_id, an.name, ap.embedding <=> %s::vector AS distance,
                    ast.works_count, ast.h_index
                    FROM author_profiles ap
                    LEFT JOIN a_names an ON ap.auth_id = an.id
                    LEFT JOIN a_stats ast ON ap.auth_id = ast.id
                    LEFT JOIN a_emails_agg ae ON ap.auth_id = ae.auth_id
                    WHERE ((%s = false OR ae.emails IS NOT NULL)
                            AND ast.h_index > %s)
                    ORDER BY distance
                    LIMIT %s OFFSET %s
                    ),
                    EMAILS AS
                    (
                    SELECT AUS.auth_id, ARRAY_AGG(AE.email) as emails
                    FROM AUS
                    LEFT JOIN a_emails AE ON AUS.auth_id = AE.AUTH_ID
                    GROUP BY AUS.auth_id
                    ),
                    AFFS AS 
                    (
                    SELECT AUS.auth_id, af.aff_name, af.year
                    FROM AUS
                    LEFT JOIN a_aff af ON AUS.auth_id = af.id
                    ),
                    AGGAFFS AS
                    (
                    SELECT auth_id,  ARRAY_AGG(aff_name) as aff_names, ARRAY_AGG(year) as aff_years
                    FROM AFFS
                    GROUP BY auth_id
                    ),
                    TOPS AS 
                    (
                    SELECT AUS.auth_id, atop.topic_name, atop.topic_count
                    FROM AUS
                    LEFT JOIN a_topics atop ON AUS.auth_id = atop.id
                    ),
                    AGGTOPS AS
                    (
                    SELECT auth_id, ARRAY_AGG(topic_name) as topic_names, ARRAY_AGG(topic_count) as topic_counts
                    FROM TOPS
                    GROUP BY auth_id
                    )
                    SELECT AUS.auth_id, name, distance, works_count, h_index, aff_names, aff_years, topic_names, topic_counts, EMAILS.emails
                    FROM AUS
                    LEFT JOIN AGGAFFS ON AUS.auth_id = AGGAFFS.auth_id
                    LEFT JOIN AGGTOPS ON AUS.auth_id = AGGTOPS.auth_id
                    LEFT JOIN EMAILS ON AUS.auth_id = EMAILS.auth_id
                    ORDER BY distance
                    '''

                cur.execute(sql_query, (new_embedding, with_emails_only, min_h_index, limit, offset))
                results = cur.fetchall()
                print(" 4 ok ", datetime.now().strftime("%H:%M:%S"))
                
                
                pg_pool.putconn(conn)

               
                results_list = []
                for row in results:
                    
                    # Zip the affiliations with their years
                    affs_with_years = list(zip(row[5], row[6]))  # row[5] = affs, row[6] = aff_years
                    topics_with_counts = list(zip(row[7], row[8]))
                    sorted_topics_with_counts = sorted(topics_with_counts, key=lambda x: x[1], reverse=True)

                    
                    # Find the latest year
                    latest_year = max(row[6])
                    
                    # Filter affiliations that correspond to the latest year
                    latest_affs = [aff for aff, year in affs_with_years if year == latest_year]

                    result= {
                        "auth_id": row[0],
                        "name": row[1],
                        "distance": row[2],
                        "works_count": row[3],
                        "h_index": row[4],
                        "affs": latest_affs,  # Only affiliations from the latest year
                        "aff_years": [latest_year],  # The latest year
                        "topics": sorted_topics_with_counts[:5],
                        "emails": row[9],
                        "conflicts": 0
                    }

                    if check_for_cois:
                        conflicts = check_for_coi_coauthors(row[0], authors)
                        result["cois"]=conflicts
                    results_list.append(result)
                        

               

                return jsonify(results_list)
                

        except Exception as e:
            print(str(e))
            return jsonify({"error": str(e)}), 500
    print(str(e))
    return jsonify({"error": str(e)}), 500



@app.route('/coi-coauthors', methods=['GET'])
def coi_coauthors():
    # Get expert id and list of author ids from request arguments
    expert_id = request.args.get('expert')
    authors_param = request.args.get('authors')  # Comma-separated string
    
    # Split authors into a list
    authors = authors_param.split(',') if authors_param else []    
    # Check if both expert and authors are provided
    if not expert_id or not authors:
        return jsonify({'error': 'Expert and a list of authors are required'}), 400
    
    # Query to get the co-authored works between expert and each author
    query = """
    SELECT w1.id, wt.title, w2.auth_id as coauthor_id, an.name as coauthor_name
    FROM w_auth w1
    JOIN w_auth w2 ON w1.id = w2.id
    LEFT JOIN w_titles wt ON w1.id = wt.id
    LEFT JOIN a_names an ON w2.auth_id = an.id
    WHERE w1.auth_id = %s
      AND w2.auth_id = %s;
    """
    
    coauthorships = []

    # Connect to the database and execute the query for each author
    conn = pg_pool.getconn()
    if conn:
        try:
            cur = conn.cursor()
            
            for author_id in authors:
                print(author_id)
                cur.execute(query, (expert_id, author_id))
                results = cur.fetchall()
                
                # Add the coauthored work results to the list
                for row in results:
                    coauthorships.append({
                        'work_id': row[0],
                        'title': row[1],
                        'coauthor_id': row[2],
                        'coauthor_name': row[3]
                    })
        finally:
            pg_pool.putconn(conn)

    # If no co-authorships found, return a message
    if not coauthorships:
        return jsonify({'message': 'No co-authored works found'}), 404

    # Return the list of co-authored works with the expert and authors
    return jsonify({'coauthorships': coauthorships}), 200







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
                        SET LOCAL hnsw.ef_search = 40;
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





def check_for_coi_coauthors(expert, authors):
    expert_id = expert
    # Query to get the co-authored works between expert and each author
    query = """
    SELECT w1.id, wt.title, w2.auth_id as coauthor_id, an.name as coauthor_name
    FROM w_auth w1
    JOIN w_auth w2 ON w1.id = w2.id
    LEFT JOIN w_titles wt ON w1.id = wt.id
    LEFT JOIN a_names an ON w2.auth_id = an.id
    WHERE w1.auth_id = %s
      AND w2.auth_id = %s;
    """
    
    coauthorships = []

    # Connect to the database and execute the query for each author
    conn = pg_pool.getconn()
    if conn:
        try:
            cur = conn.cursor()
            
            for author_id in authors:
                cur.execute(query, (expert_id, author_id))
                results = cur.fetchall()
                
                # Add the coauthored work results to the list
                for row in results:
                    coauthorships.append({
                        'work_id': row[0],
                        'title': row[1],
                        'coauthor_id': row[2],
                        'coauthor_name': row[3]
                    })
        finally:
            pg_pool.putconn(conn)

    if not coauthorships:
        return 0

    return [expert_id, coauthorships]



if __name__ == "__main__":
    app.run(debug=True)