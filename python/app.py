import os
import json
from flask import Flask, request, send_file

from recs import Recsystem

DIR = os.path.abspath('..')
WWW_BASE = DIR + '/www'
QS_BASE = DIR + '/data/Qs'

# maybe we could pass these as request parameters eventually
username = 'ttrojan77'

rc = Recsystem(DIR)
rc.init_user(username)

app = Flask(__name__)


@app.route('/')
def show_index():
    return send_file(WWW_BASE + '/index.html')


@app.route('/<path:path>')
def show_file(path):
    return send_file(WWW_BASE + '/' + path)

@app.route('/api/tests', methods=["GET"])
def get_tests():
    return json.dumps({
        "tests":rc.all_test_types
    })

@app.route('/api/keywords', methods=["GET"])
def get_keywords():
    exam = request.args.get("exam")
    grade = request.args.get("grade")
    keywords = rc.get_keywords(exam ,grade)
    found_answers = "yes"
    if len(keywords) == 0:
        found_answers = "no"
    return json.dumps({
        "keywords" : keywords,
        "found_answers" : found_answers
    })

@app.route('/api/list', methods=["POST"])
def get_questions_by_keywords():

    data = request.data
    dataDict = json.loads(data)

    keywords = dataDict['keywords']
    exam = dataDict['exam']
    grade = dataDict["grade"]
    return json.dumps({"questions" : 
    rc.get_question_by_keywords_grade(grade, keywords, exam)})


@app.route('/api/question', methods=['GET'])
def get_question():
    # retrieve user from query string, TODO migrate to using authentication
    user_id = request.args.get('user')
    q, a, b, c, d = rc.send_question()

    print(user_id)

    return json.dumps({
        'question': q,
        'answers': [{
            'id': 'A',
            'text': a
        }, {
            'id': 'B',
            'text': b
        }, {
            'id': 'C',
            'text': c
        }, {
            'id': 'D',
            'text': d
        }]
    })


@app.route('/api/answer', methods=['POST'])
def post_answer():
    user_id = request.args.get('user')

    # letter corresponding to selected answer (e.g. A, B, C)
    selected = request.args.get('selected')
    print(selected)
    rc.prep_next_q(selected)

    return json.dumps({
        'correct': selected
    })

# TODO
# GET -> Types of Exams
# POST -> exam type, Grade Level, Keywords (or load from firebase created)
# GET -> Correct answer
# Add ID to each of the questions


if __name__ == '__main__':
    app.run()