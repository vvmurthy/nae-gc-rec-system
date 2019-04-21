import os
import json
from flask import Flask, request, send_file
import sqlite3

import logging
logging.basicConfig(level=logging.INFO)

from recs import Recsystem

DIR = os.path.abspath('..')
WWW_BASE = DIR + '/www'
QS_BASE = DIR + '/data/Qs'

rc = Recsystem(DIR)

app = Flask(__name__)

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


@app.route('/api/question', methods=['POST'])
def get_question():
    
    # TODO: Use actual authentication
    data = request.data
    dataDict = json.loads(data)

    keywords = dataDict['keywords']
    exam = dataDict['exam']
    username = dataDict['username']
    grade = dataDict["grade"]
    qs_answered, percentage, _, _, question_id = rc.init_user_vars(
        username, grade, exam, keywords
    )
    questions = []
    conn = sqlite3.connect(rc.DB_NAME)
    rc.get_question_by_id(question_id, questions, conn)
    conn.close()
    logging.info(questions)
    return json.dumps({
        'question': questions,
        "qs_answered" : qs_answered,
        "percentage" : percentage
    })

@app.route('/api/answer', methods=['POST'])
def post_answer():
    data = request.data
    dataDict = json.loads(data)

    keywords = dataDict['keywords']
    exam = dataDict['exam']
    username = dataDict['username']
    grade = dataDict["grade"]
    answer = dataDict["answer"]
    question_id = dataDict["question_id"]
    logging.info("Received Question ID: " + question_id)
    response = rc.prep_next_q(answer, username, grade, exam, keywords, question_id)
    
    return json.dumps(response)

# TODO
# GET -> Types of Exams
# POST -> exam type, Grade Level, Keywords (or load from firebase created)
# GET -> Correct answer
# Add ID to each of the questions


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)