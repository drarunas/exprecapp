from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
import torch
import psycopg2
from psycopg2 import pool
import os
from datetime import datetime
import pandas as pd
import ast
import json
import google.generativeai as ai
import typing_extensions as typing
import enum
import requests
import asyncio
import aiohttp


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["http://127.0.0.1:8080", "https://exrecapp.web.app", "https://exprlabs.com"]}})

ai.configure(api_key=os.environ["GEMINI_API_KEY"])

# Load the e5-multilingual-large model using SentenceTransformer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
model = SentenceTransformer('intfloat/e5-large-v2', device=device)

if os.getenv('ENV') == 'production':
    # GCP Production using Unix socket
    pg_pool = psycopg2.pool.SimpleConnectionPool(
        minconn=1,
        maxconn=100,
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
    # query = "query: " + text  # Prefix the text with "query:"
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
        if len(authors)<2:
            check_for_cois = False
        authors = [author.strip() for author in request.args.get('authors', '').split(',')]
        


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
                    SET LOCAL hnsw.ef_search = 200;
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
                    INNER JOIN a_aff_history af ON AUS.auth_id = af.id
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
                    --LEFT JOIN a_topics atop ON AUS.auth_id = atop.id
                    LEFT JOIN a_recent_topics atop ON AUS.auth_id = atop.author_id
                    ),
                    AGGTOPS AS
                    (
                    SELECT auth_id, ARRAY_AGG(topic_name) as topic_names, ARRAY_AGG(topic_count) as topic_counts
                    FROM TOPS
                    GROUP BY auth_id
                    )
                    SELECT AUS.auth_id, name, distance, works_count, h_index, aff_names, aff_years, topic_names, topic_counts, EMAILS.emails
                    FROM AUS
                    INNER JOIN AGGAFFS ON AUS.auth_id = AGGAFFS.auth_id
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
                        "cois": 0
                    }

                    if check_for_cois:
                        conflicts = check_for_coi_coauthors(row[0], authors)
                        result["cois"]=conflicts
                    results_list.append(result)

                #return jsonify(results_list)
                return jsonify({"results": results_list, "vector": new_embedding})

                

        except Exception as e:
            print(str(e))
            return jsonify({"error": str(e)}), 500
    print(str(e))
    return jsonify({"error": str(e)}), 500

@app.route('/match_works', methods=['POST'])
def match_works():
    if request.method == 'POST':
        data = request.json
        author_id = data.get('author_id')
        vector = data.get('vector')
        
        # embedding = json.loads(vector)
        embedding = vector

        if not author_id or not vector:
            return jsonify({"error": "Missing author_id or abstract"}), 400

        try:
            # # Step 1: Get the embedding for the abstract
            # print("Calculating embedding for abstract...", datetime.now().strftime("%H:%M:%S"))
            # embedding = get_embeddings(abstract).tolist()

            # Step 2: Get a connection from the pool
            print("Getting connection from the pool...", datetime.now().strftime("%H:%M:%S"))
            conn = pg_pool.getconn()
            if conn:
                cur = conn.cursor()

                # Step 3: Execute the SQL query to find 3 closest works based on the embedding
                print('Executing SQL query to find closest works...', datetime.now().strftime("%H:%M:%S"))
                sql_query = '''
                    SET LOCAL hnsw.ef_search = 200;
                    WITH FILTERED_WORKS AS (
                        SELECT ev.id, ev.embedding
                        FROM e5_vectors ev
                        INNER JOIN w_auth wa ON ev.id = wa.id
                        WHERE wa.auth_id = %s
                    ),
                    UNIQUEWORKS AS (
                    SELECT MIN(fw.id) AS w_id, MIN(fw.embedding <=> %s::vector) AS distance
                    FROM FILTERED_WORKS fw
                    INNER JOIN w_auth wa ON fw.id = wa.id
                    GROUP BY wa.id
                    ORDER BY distance
                    LIMIT 3
                    )
                    SELECT w_id, distance, wt.title, COALESCE(was.doi, wae.doi) as doi
                    FROM UNIQUEWORKS uw
                    LEFT JOIN w_titles wt ON uw.w_id = wt.id
                    LEFT JOIN w_abs_sn was ON uw.w_id = was.id
                    LEFT JOIN w_abs_els wae ON uw.w_id = wae.id

                '''

                cur.execute(sql_query, (author_id, embedding,))
                results = cur.fetchall()
                print('OK', datetime.now().strftime("%H:%M:%S"))

                # Step 4: Return the results
                pg_pool.putconn(conn)
                
                results_list = []
                for row in results:
                    result = {
                        "work_id": row[0],
                        "distance": row[1],
                        "title": row[2],
                        "doi": row[3]
                    }
                    results_list.append(result)

                return jsonify(results_list)

        except Exception as e:
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

                # sql_query='''
                #         SET LOCAL hnsw.ef_search = 200;
                #         WITH top_works AS (
                #             SELECT ev.id, 
                #                 ev.embedding <=> %s::vector AS distance, 
                #                 wt.title AS title,
                #                 COALESCE(was.inv_abstract, wae.inv_abstract) AS abs,
                #                 COALESCE(was.doi, wae.doi) AS doi,
                #                 oaw.publication_year as year,
                #                 oaw.cited_by_count as citations

                                
                #             FROM e5_vectors ev
                #             LEFT JOIN w_titles wt ON ev.id = wt.id
                #             LEFT JOIN w_abs_sn was ON ev.id = was.id
                #             LEFT JOIN w_abs_els wae  ON ev.id = wae.id
                #             LEFT JOIN oa_works oaw ON ev.id = oaw.id
                            

                #             WHERE (SELECT MAX(extracted_number) 
                #             FROM (SELECT unnest(regexp_matches(COALESCE(was.inv_abstract, wae.inv_abstract), '(?<!")\d+(?!")', 'g'))::bigint AS extracted_number) AS subquery) BETWEEN 50 AND 1000 
                #             AND oaw.cited_by_count > 10
                #             ORDER BY distance
                #             LIMIT 20
                #         ),
                #         awd AS (
                #         SELECT tw.id, 
                #             tw.distance, 
                #             tw.title, 
                #             wa.auth_id auid,
                #             tw.abs abs,
                #             tw.doi,
                #             tw.year,
                #             tw.citations
                #         FROM top_works tw
                #         LEFT JOIN w_auth wa ON tw.id = wa.id
                #         )
                #         SELECT awd.id, awd.title, array_agg(awd.auid) as auid_array, awd.distance, array_agg(a_names.orcid) AS orcid_array, array_agg(a_names.name) AS name_array, awd.abs as abs, awd.doi as doi, awd.year, awd.citations
                #         FROM awd 
                #         LEFT JOIN a_names ON awd.auid = a_names.id
                #         GROUP BY awd.id, awd.title, awd.distance, awd.abs, awd.doi, awd.year, awd.citations
                #         ORDER BY awd.distance
                #         '''
                sql_query='''
                        SET LOCAL hnsw.ef_search = 200;
                        WITH top_works AS (
                            SELECT ev.id, 
                                ev.embedding <=> %s::vector AS distance, 
                                oaw.title AS title,
                                oaw.abstract_inverted_index AS abs,
                                oaw.doi AS doi,
                                oaw.publication_year as year,
                                oaw.cited_by_count as citations

                                
                            FROM e5_vectors ev
                            LEFT JOIN oa_works oaw ON ev.id = oaw.id
                            

                            WHERE (SELECT MAX(extracted_number) 
                            FROM (SELECT unnest(regexp_matches(oaw.abstract_inverted_index, '(?<!")\d+(?!")', 'g'))::bigint AS extracted_number) AS subquery) BETWEEN 50 AND 1000 
                            AND oaw.cited_by_count > 1
                            ORDER BY distance
                            LIMIT 50
                        ),
                        awd AS (
                        SELECT tw.id, 
                            tw.distance, 
                            tw.title, 
                            wa.auth_id auid,
                            tw.abs abs,
                            tw.doi,
                            tw.year,
                            tw.citations
                        FROM top_works tw
                        LEFT JOIN w_auth wa ON tw.id = wa.id
                        )
                        SELECT awd.id, awd.title, array_agg(awd.auid) as auid_array, awd.distance, array_agg(a_names.orcid) AS orcid_array, array_agg(a_names.name) AS name_array, awd.abs as abs, awd.doi as doi, awd.year, awd.citations
                        FROM awd 
                        LEFT JOIN a_names ON awd.auid = a_names.id
                        GROUP BY awd.id, awd.title, awd.distance, awd.abs, awd.doi, awd.year, awd.citations
                        ORDER BY awd.distance
                        '''
                try:
                    cur.execute(sql_query, (new_embedding,))

                    results = cur.fetchall()
                    
                    print(" 4 ok ", datetime.now().strftime("%H:%M:%S"))
                  
                    
                    results_list = [
                        {"id": row[0], "title": row[1], "auid": row[2], "distance": row[3], "orcid": row[4], "name": row[5], "abstract": convert_inverted(ast.literal_eval(row[6])), "doi":row[7], "year":row[8], "citations":row[9]} 
                        for row in results
                    ]


                    #abstracts = [convert_inverted(ast.literal_eval(row[6])) for row in results]
                    
                    
                    # Step 5: Release the connection back to the pool
                    pg_pool.putconn(conn)

                    return jsonify({"results": results_list})
                except Exception as e:
                    print(e)
                    return jsonify({"error": str(e)}), 500

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


@app.route('/pre_review', methods=['POST'])
def pre_review():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and file.content_type == 'application/pdf':
        try:
            # Save the uploaded file temporarily
            filepath = os.path.join('/tmp', file.filename)  # Use a temporary directory
            file.save(filepath)
            ai.configure(api_key="AIzaSyAfO1dlSduFQHwQ7tidvUngiiFK1PJBb7I")

            class HumanStudy(enum.Enum):
                human = "human"
                non_human = "non_human"
                human_s_data = "secondary human data"

            class ArticleType(enum.Enum):
                primary = "primary research"
                lit_review = "narrative literature review"
                sys_review = "systematic review / meta-analysis"

            class Review (typing.TypedDict):
                title: str
                research_question: str
                key_takeaway: str
                authors: list[str]
                summary: str
                methods: list[str]
                fields: list[str]
                ethics: str
                human_study: HumanStudy
                data: str
                article_type: ArticleType
                sample_size: int
                known: str
                new_advance: str
                
            # Create the model
            generation_config = ai.GenerationConfig(
                temperature=0.,
                top_p=0.95,
                top_k=64,
                max_output_tokens=8192,
                response_mime_type="application/json",
                response_schema=Review  # Not list[Review]
            )

            model = ai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config=generation_config)
            
            doc = ai.upload_file(filepath)
            

            query = '''Generate a  summary of the attached research paper in JSON format, strictly adhering to the structure defined by the 'Review' type. 
            Include:
            the title, 
            main research question in one sentence and inquestion form,
            one sentence key takeaway,
            author list,
            a longer summary (what they did, how, what htey found, one paragraph at most),
            the scientific method names employed (including statistical techniques),
            science fields that paper belongs to,
            any ethics or IRB approval/waiver that they mention,
            whether this is a human subjects study with human participants,
            do they share their research data (this is usually in a section on data sharing),
            and article type,
            sample size (for example, how many participants, or None),
            what they say has been known before,
            what they say is new in what's discovered here over what's been known before (describe and include verbatim quotes),
            If you cannot find data in the document, output None.'''
            response = model.generate_content( [query, doc])

            parsed = json.loads(response.text)
            print(json.dumps(parsed, indent=4))

            

            # Remove the temporary file
            os.remove(filepath)

            # Return the parsed JSON response
            return jsonify(parsed)

        except Exception as e:
            return jsonify({'error': str(e)}),



def convert_inverted(inverted_abstract):
    # Find the maximum index to determine the size of the output list
    max_index = max(max(indices) for indices in inverted_abstract.values())
    
    # Create an empty list of the size of the maximum index + 1 (for zero-based index)
    abstract_list = [''] * (max_index + 1)
    
    # Iterate through the dictionary and place the words at their respective positions
    for word, positions in inverted_abstract.items():
        for position in positions:
            abstract_list[position] = word
    
    # Join the list to form the regular abstract, separating words by a space
    return ' '.join(abstract_list)

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

def summarize_abs(data, abstracts):
    ai.configure(api_key=os.environ["GEMINI_API_KEY"])


    # Create the model
    generation_config = {
    "temperature": 0.5,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
    }

    model = ai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
    )

    chat_session = model.start_chat(
    history=[]
    )
    query='User query:"' + data +'". Given this user query, summarize the main scientific result from the following abstracts in 1 sentence. Be concise, and direct, as if you were a scientist stating facts. Summary should be based on all abstracts, not just one. If the user query is a question, answer it only on the follwoing abstracts. If the question is yes/no question, include "Yes" or "No" or "Uncertain" based on abstracts at the beginning of your answer. If user query is not a question (but a passage or abstract), ignore it, and just summarize the following (but not the user query itself). Base your answers only on the abstracts below and nothing else. Abstract array:' + '\n' + str(abstracts)
    response = chat_session.send_message(query)

    return response.text


@app.route('/summarize_study', methods=['POST'])
def summarize_study():
    data = request.get_json()
    if 'abstract' not in data:
        return jsonify({'error': 'No abstract'}), 400
    abstract = data['abstract']

    ai.configure(api_key=os.environ["GEMINI_API_KEY"])
    # Create the model
    generation_config = {
    "temperature": 0.2,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
    }

    model = ai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
    )

    chat_session = model.start_chat(
    history=[]
    )
    query='Provide a one sentence summary of the following research results. Be concise, but specific, mentioning facts. Answer based only on the following research summary. Study:' + '\n' + str(abstract)
    try:
        response = chat_session.send_message(query)
    except ai.types.generation_types.StopCandidateException as e:
        # Log the exception or return a safe message
        response = {"error": "The response was flagged for safety by the model. You can still read the full summary below."}
        return jsonify({"summary": response["error"]})

    return jsonify({"summary": response.text})



def fetch_passage_summary(passage, index, user_query):
    url = "https://generativelanguage.googleapis.com/v1beta/models/aqa:generateAnswer"
    headers = {"Content-Type": "application/json"}
    data = {
        "contents": [{"parts": [{"text": user_query}]}],
        "answer_style": "ABSTRACTIVE",
        "inline_passages": {"passages": [passage]}
    }
    
    response = requests.post(url, headers=headers, params={"key": os.environ["GEMINI_API_KEY"]}, data=json.dumps(data))
    response_data = response.json()
    print(response_data["answer"]["content"])
    print("...")
    
    text_content = response_data["answer"]["content"]["parts"][0]["text"]
    passage_ids = [
        attr["sourceId"]["groundingPassage"]["passageId"]
        for attr in response_data["answer"]["groundingAttributions"]
    ]
    
    return {"summary": text_content, "passageIds": passage_ids, "index": index}

def run_queries_sequentially(passages, user_query):
    results = []
    for index, passage in enumerate(passages):
        result = fetch_passage_summary(passage, index, user_query)
        results.append(result)
    return results

@app.route('/summarize_research_results', methods=['POST'])
def summarize_research_results():
    data = request.get_json()
    
    if 'abstracts' not in data:
        return jsonify({'error': 'No abstracts'}), 400
    if 'query' not in data:
        return jsonify({'error': 'No user query'}), 400

    abstracts = data['abstracts']
    user_query = data['query']
    passages = [{"id": str(index + 1), "content": {"parts": [{"text": item['abstract']}]}} for index, item in enumerate(abstracts)]

    class QueryType(enum.Enum):
        question = "question"
        research_summary = "research_summary"

    class QueryDesc(typing.TypedDict):
        type: QueryType
                


    ai.configure(api_key=os.environ["GEMINI_API_KEY"])
    generation_config = ai.GenerationConfig(
        temperature=0.9,
        top_p=0.95,
        top_k=64,
        max_output_tokens=8192,
        response_mime_type="application/json",
        response_schema=QueryDesc
    )
    model = ai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
    )

    query='User query:  ' + user_query + '. What is the type of this query -- a question of any kind, or a research summary?'
    try:
        response = model.generate_content(query)
    except ai.types.generation_types.StopCandidateException as e:
        # Log the exception or return a safe message
        response = {"error": "The response was flagged for safety by the model. You can still read the full summary below."}
        return jsonify({"summary": response["error"]})
    generation_config = ai.GenerationConfig(
        temperature=0.5,
        top_p=0.95,
        top_k=64,
        max_output_tokens=8192,
        response_mime_type="text/plain",
    )

    model = ai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
    )

        
    if json.loads(response.text)["type"] == "question":
        

        url = "https://generativelanguage.googleapis.com/v1beta/models/aqa:generateAnswer"
        headers = {"Content-Type": "application/json"}
        data = {
            "contents": [{
                "parts": [{"text": user_query}]
            }],
            "answer_style": "ABSTRACTIVE",
            "inline_passages": {
                "passages": passages
            }
        }

        # aqa_response = requests.post(url, headers=headers, params={"key": "AIzaSyAfO1dlSduFQHwQ7tidvUngiiFK1PJBb7I"}, data=json.dumps(data))
        # print(json.dumps(aqa_response.json(), indent=4))
        query='User question: ' + user_query + '. Given only research studies below, answer this question in one sentence. Base your answer only on the text below and nothing else.  If it is a yes/no question, start by syaing yes or no. Research studies:' + '\n' + str(abstracts)
        # text_content = aqa_response.json()["answer"]["content"]["parts"][0]["text"]
        # passage_ids = [attr["sourceId"]["groundingPassage"]["passageId"] for attr in aqa_response.json()["answer"]["groundingAttributions"]]
        # print (text_content)
        # print(passage_ids)
        # return jsonify({"summary": text_content, "passageIds": passage_ids})





    else:        
        query=' Prior studies: ' + str(abstracts) + '. In 2 sentences, summarize Prior Studies. use the following format: "Prior studies have shown that... .". Describing prior studies, provide a reference to the work where the specific fact comes from in [Name et al YEAR] format.'

    try:
        response = model.generate_content(query)
        
    except ai.types.generation_types.StopCandidateException as e:
        # Log the exception or return a safe message
        response = {"error": "The response was flagged for safety by the model."}
        return jsonify({"summary": response["error"]})


    return jsonify({"summary": response.text})

if __name__ == "__main__":
    app.run(debug=True)
