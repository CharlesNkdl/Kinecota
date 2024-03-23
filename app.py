from flask import Flask, render_template, request
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import torch

app = Flask(__name__)

model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
ref = pd.read_csv('data/kineacte.csv', encoding='utf-8', delimiter=';')
reftext = ref["texte"].tolist()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/similarity', methods=['POST'])
def calculate_similarity():
    query = request.form['query']
    query_embedding = model.encode(query)
    sim = []
    for i in range(len(ref)):
           sim.append((cosine_similarity(query_embedding.reshape(1, -1), model.encode(ref["texte"][i]).reshape(1, -1)).item() , i))
    sim.sort(reverse=True)
    top5 = []
    for i in range(10):
        top5.append((sim[i][0], ref["texte"][sim[i][1]], ref["cotation"][sim[i][1]]))
    return render_template('index.html', query=query, ref=top5)

if __name__ == '__main__':
	app.run(debug=True)
