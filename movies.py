from flask import Flask
from recommender import recommend
from models import nmf
from flask import render_template
from flask import request

app=Flask(__name__)

@app.route('/')
def home():
    return render_template('main.html')

@app.route('/recommender')
def recommender():
    query = request.args
    recommendation, similarity = recommend(query)
    return render_template('recommender.html',
                            recommendation,
                            similarity)

# if __name__=='__main__':
#     app.run(debug=True)
