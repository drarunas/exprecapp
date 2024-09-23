from flask import Flask, request, jsonify, render_template
from sentence_transformers import SentenceTransformer
import torch
import psycopg2
import os
# Initialize the Flask app
app = Flask(__name__)

# Load the e5-multilingual-large model using SentenceTransformer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
model = SentenceTransformer('intfloat/e5-large-v2', device=device)

# PostgreSQL connection parameters
db_params = {
    'dbname': 'postgres',
    'user': 'exrec',
    'password': 'wollstonecraft',
    'host': '34.34.106.184',
    'port': '5432'
}

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
def query():
    if request.method == 'POST':
        # Get the query from the form input
        data = request.form['q']
   
        
        if not data:
            return jsonify({"error": "Missing query text"}), 400

        # Step 1: Calculate the embedding for the new abstract
        print('Calculating embeddings')
        new_embedding = get_embeddings(data).tolist()


        try:
            # Step 2: Connect to the PostgreSQL database
            conn = psycopg2.connect(**db_params)
            cur = conn.cursor()
            print('Finding similar')
            # Step 3: Execute the SQL query using the new embedding
            sql_query = """
                SELECT ev.id, ev.embedding <=> %s::vector AS distance, wt.title
                FROM e5_vectors ev
                LEFT JOIN w_titles_sn wt ON ev.id = wt.id
                ORDER BY distance
                LIMIT 100;
            """
            cur.execute(sql_query, (new_embedding,))
            results = cur.fetchall()
            cur.close()
            conn.close()

            # Step 4: Convert results to a list of dictionaries for the template
            results_list = [
                {"id": row[0], "distance": row[1], "title": row[2]} 
                for row in results
            ]

            # Step 5: Return the results in HTML format
            return render_template('query.html', results=results_list)

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return render_template('query.html')

if __name__ == "__main__":
    app.run(debug=True)