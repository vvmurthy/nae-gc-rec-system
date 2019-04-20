import pandas as pd
import numpy as np
import sqlite3
import spacy

import logging
logging.basicConfig(level=logging.INFO)


class Recsystem:

    """
    Recsystem(BASE, username)

    Recommendation system (draft) class. Contains methods needed to
    preprocess data and process responses to user input.

    Parameters
    ----------
    BASE : string
        The absolute directory where the Allen AI CSV files are stored.

    test_type : string
        The username of the user accessing the api

    Attributes
    ----------
    base : string
        contains BASE as a class variable

    df : pandas Dataframe
        contains preprocessed CSV data of questions, responses, and auxilary data about each question

    grade_range : int
        contains the number of grades the data spans over. In this dataset, `grade_range` is 7.

    grade : int
        contains user-entered grade as a class variable

    last_percentage : float
        Once a user has responded to a question, `last_percentage` contains the percentage of students
        who answered correctly to the question the user just responded to.

    qs_answered : int
        The total number of questions the user has answered during a session.

    percentage : float
        The percentage of answered questions the user has answered correctly

    last_question : string
        Once a user has responded to a question, `last_question` contains the exact question the
        user has just responded to.

    A : string
        The answer corresponding to option A (or 1 on select tests)

    B : string
        The answer corresponding to option B (or 2 on select tests)

    C : string
        The answer corresponding to option C (or 3 on select tests)

    D : string
        The answer corresponding to option D (or 4 on select tests)

    test_type : string
        Contains user-entered test_type as a class variable.

    answered_questions : List of strings
        contains the questionIDs of the questions the user has answered previously.

    index : int
        The index in the dataframe of the questoin the user has just responded to. Once updated
        through the `updates()` function, it will contain the index of the question the user will
        respond to.

    Methods
    -------
    prep_next_q(answer)
        Ranks all questions by a similarity function then determines which question is most similar that the
        user has not answered. Then picks the index of the most similar question, and updates class variables
        accordingly.

    send_question()
        Sends the question and four answers in a format printable by the user. Currently a stub method- will
        probably have to convert Q + answer into JSON in this function.

    send_user_stats()
        Sends the variable user stats- such as percentage correct and number of questions answered

    send_q_stats()
        Sends statistics about the question, including what percent of students
        got it right in the user's grade category

    update_sql()
        Sends user information that has been updated after each question to the database

    Reads
    -----
    questions : sql table
        Contains `self.df` in SQL table form. Used to store preprocessed data.

    users : sql table
        Contains user variable info for each user. Some sections are mutable, unlike
        `questions` which is generally immutable
    
    keywords : sql table
        Links questions against specific "keywords" for faster indexing. 

    """

    def __init__(self, BASE):

        # Define dataset of questions
        self.base = BASE

        # Define word tokenizer
        self.nlp = spacy.load('en_core_web_lg')

        # Define static values of dataset
        self.grade_range = 7  # grades 3 - 9 inclusive
        self.min_grade = 3
        self.max_grade = 9
        self.all_test_types = ['ACTAAP', 'AIMS', 'Louisiana Educational Assessment Program',
                               'MCAS', 'MEA', 'MSA', 'TIMSS', 'WASL',
                               'Alaska Department of Education and Early Development',
                               'California Standards Test', 'FCAT', 'Maryland School Assessment',
                               'MEAP', 'NAEP', 'North Carolina READY End-of-Grade Assessment',
                               'NYSEDREGENTS', 'Ohio Achievement Tests', 'TAKS',
                               'Virginia Standards of Learning - Science', 'AMP']

        # Preprocess dataset and read into sql if not done already
        self.conn = sqlite3.connect('recs.db')
        self.c = self.conn.cursor()
        self.c.execute("SELECT count(*) FROM sqlite_master WHERE type='table'")
        tables = self.c.fetchall()[0][0]

        # Add questions to database
        logging.info("Number of tables: " + str(tables))
        if tables < 1:
            try:
                logging.info("Creating Users Table")
                create_table_sql = "CREATE TABLE IF NOT EXISTS users ( \
                                    id integer PRIMARY KEY, \
                                    qs_answered int not null, \
                                    username text not null, \
                                    percentage long not null, \
                                    answered_questions text not null, \
                                    grade integer not null, \
                                    test_type text not null \
                                );"
                self.c.execute(create_table_sql)
                create_user_sql = "INSERT into users (username, grade \
                    , percentage, answered_questions, qs_answered, test_type \
                        ) values (\"ttrojan77\", 3, 0, \"[]\", 0, \"MCAS\");"
                self.c.execute(create_user_sql)
            except Exception as e:
                logging.warn(e)
                logging.warn("Could not add users table to sql")

        if tables < 2:
            self._preprocess_data()
        self.df = pd.read_sql('SELECT * FROM questions;', self.conn)
        
        # Add keywords to database
        
        if tables < 3:
            try:
                logging.info("Creating Keywords Table")
                create_table_sql = "CREATE TABLE IF NOT EXISTS keywords ( \
                                    id integer PRIMARY KEY, \
                                    examName text NOT NULL, \
                                    questionId integer NOT NULL, \
                                    grade integer NOT NULL, question text NOT NULL, keyword text NOT NULL \
                                );"
                self.c.execute(create_table_sql)
                self.df.apply(lambda row: self._init_keywords(row), axis=1)
            except Exception as e:
                logging.warn(e)
                logging.warn("Could not add keywords table to sql")
        
        if tables < 4:
            try:
                logging.info("Creating user_similarity Table")
                create_table_sql = "CREATE TABLE IF NOT EXISTS user_similarity ( \
                                    id integer PRIMARY KEY, \
                                    username text NOT NULL, \
                                    questionId integer NOT NULL, \
                                    similarity long NOT NULL \
                                );"
                self.c.execute(create_table_sql)
                self.conn.commit()
            except Exception as e:
                logging.warn(e)
                logging.warn("Could not add keywords table to sql")

        # Clear the dataframe for future operations
        self.df = None

        # Define temporary variable index- which question currently being answered
        # index is initially a random question that the user has not answered
        self.index = None
        self.conn.commit()
        self.conn.close()
    
    def get_question_by_keywords_grade(self, grade, keywords, exam):
        conn = sqlite3.connect('recs.db')
        questions = []
        question_ids = set()
        sql = "SELECT questionID from keywords where examName=(?) and grade=(?) and keyword=(?)"
        c = conn.cursor()
        for keyword in keywords:
            c.execute(sql, (exam, grade, keyword))
            response = c.fetchall()
            for re in response:
                question_id = str(re[0])
                if "\\" in question_id or len(question_id) == 1:
                    continue            
                question_ids.add(question_id)
        conn.close()
        conn = sqlite3.connect('recs.db')
        
        for question_id in question_ids:
            self.get_question_by_id(question_id, questions, conn)
        conn.close()
        return questions
    
    def get_question_by_id(self, question_id, questions, conn):
        c = conn.cursor()
        sql = "SELECT questionID, originalQuestionID, AnswerKey, examName, year, question, A, B, C, D from questions where questionID=(?)"
        c.execute(sql, (question_id,))
        response = c.fetchall()
        for re in response:
            question_id = re[0]
            if "\\" in question_id or len(question_id) == 1:
                return  
            original_id = re[1]
            answer_key = re[2].lower()
            exam_name = re[3]
            year = re[4]
            question = re[5]
            a = re[6]
            b = re[7]
            c = re[8]
            d = re[9]

            question_dict = {
                "originalID" : original_id,
                "answerKey" : answer_key,
                "questionID" : question_id,
                "examName" : exam_name,
                "A" : a,
                "B" : b,
                "C" : c,
                "D" : d,
                "question" : question,
                "year" : year
            }
            questions.append(question_dict)


    def init_user_vars(self, username, grade, exam, keywords):
        conn = sqlite3.connect('recs.db')
        c = conn.cursor()
        try:
            user = pd.read_sql('SELECT * FROM users WHERE username=' + '"' + username + '";', conn)
            qs_answered = int(user['qs_answered'][0])
        except IndexError:
            create_user_sql = "INSERT into users (username, grade \
                    , percentage, answered_questions, qs_answered, test_type \
                        ) values (\"" + username  + "\", " + grade + \
                            ", 0, \"[]\", 0, \"" + exam + "\");"
            c.execute(create_user_sql)
            conn.commit()
            user = pd.read_sql('SELECT * FROM users WHERE username=' + '"' + username + '";', conn)

        # Define user variables
        qs_answered = int(user['qs_answered'][0])
        percentage = float(user['percentage'][0])
        test_type = user['test_type'][0]
        
        if user['answered_questions'][0] == '[]':
            answered_questions = []
        else:
            answered_questions = user['answered_questions'][0].split('____')
        
        # Seed the last question if the user hasn't answered any
        question_id = None
        if len(answered_questions) == 0:
            for keyword in keywords:
                find_questions_sql = "SELECT questionID from keywords where keyword=(?) and grade=(?)"
                c.execute(find_questions_sql, (keyword, grade))
                response = c.fetchall()
                for re in response:
                    question_id = str(re[0])
                    break
            
            # Send the similarity to db
            find_all_questions = "SELECT questionId from questions"
            c.execute(find_all_questions)
            response = c.fetchall()
            for re in response:
                question_id_query = re[0]
                similarity = 0.0
                if question_id_query == question_id:
                    similarity = 90.0
                insert = "INSERT into user_similarity (username, questionId, similarity) VALUES (?, ?, ?)"
                c.execute(insert, (username, question_id_query, similarity))
                conn.commit()

        else:
            find_questions_sql = "SELECT similarity, questionID from user_similarity where username=(?)"
            c.execute(find_questions_sql, (username,))
            response = c.fetchall()
            similarity = 0
            for re in response:
                similarity_query = float(re[0])
                question_id_query = str(re[1])
                if similarity_query > similarity and (question_id_query not in answered_questions):
                    question_id = question_id_query
                break
        conn.close()
        return qs_answered, percentage, test_type, answered_questions, question_id
    
    def _init_keywords(self, df_row):
        exam = df_row["examName"]
        grade = df_row["schoolGrade"]
        question = df_row["question"]
        question_id = df_row["questionID"]

        sql = "INSERT INTO keywords (examName, grade, question, keyword, questionID) \
              VALUES(?,?,?, ?, ?)"

        q1 = self.nlp(question)
        words = set()
        for word in q1:
            if word.pos_ == 'NOUN':
                words.add(word)

        for word in words:
            self.c.execute(sql, (exam, grade, question, word.text, question_id))
        
    
    def get_keywords(self, exam, grade):
        sql = "SELECT keyword from keywords where examName=(?) and grade=(?)"
        self.conn = sqlite3.connect('recs.db')
        self.c = self.conn.cursor()
        self.c.execute(sql, (exam, grade))
        response = self.c.fetchall()
        keywords = set()
        for re in response:
            keyword = re[0].lower()
            if "\\" in keyword or len(keyword) == 1:
                continue            
            keywords.add(keyword)
        keywords = list(keywords)
        return keywords

    def _preprocess_data(self):

        # Read CSVs from dataset
        df1 = pd.read_csv(self.base + '/data/Qs/ElementarySchool/Elementary-NDMC-Train.csv')
        df2 = pd.read_csv(self.base + '/data/Qs/ElementarySchool/Elementary-NDMC-Test.csv')
        df3 = pd.read_csv(self.base + '/data/Qs/ElementarySchool/Elementary-NDMC-Dev.csv')

        df4 = pd.read_csv(self.base + '/data/Qs/MiddleSchool/Middle-NDMC-Train.csv')
        df5 = pd.read_csv(self.base + '/data/Qs/MiddleSchool/Middle-NDMC-Test.csv')
        df6 = pd.read_csv(self.base + '/data/Qs/MiddleSchool/Middle-NDMC-Dev.csv')

        df = pd.concat([df1, df2, df3, df4, df5, df6])
        del df['subject']
        del df['category']

        # Remove all diagram questions and free responses
        df = df[df.isMultipleChoiceQuestion == 1]
        df = df[df.includesDiagram == 0]

        # 1)) Data preprocessing- we must standardize some of the test names
        df.loc[df.examName == 'California Standards Test - Science', 'examName'] = 'California Standards Test'
        df.loc[df.examName == 'Maryland School Assessment - Science', 'examName'] = 'Maryland School Assessment'
        df.loc[df.examName == 'Alaska Department of Education & Early Development',
               'examName'] = 'Alaska Department of Education and Early Development'
        df.loc[df.examName == 'Alaska Dept. of Education & Early Development',
               'examName'] = 'Alaska Department of Education and Early Development'

        # 2)) we split each question to question and answers
        df7 = df.apply(lambda row: self._answers(row['question'], row['questionID']), axis=1)
        del df['question']
        df = df.merge(df7)

        # 3)) We create a toy distribution for each question based on the grade the question was assigned
        dist_df = df.apply(lambda row: self._correct(row['schoolGrade'], row['questionID']), axis=1)
        df = df.merge(dist_df)
        df['questionID'] = df['questionID'].astype(str)
        df['originalQuestionID'] = df['originalQuestionID'].astype(str)
        df['AnswerKey'] = df['AnswerKey'].astype(str)
        df['examName'] = df['examName'].astype(str)
        df['year'] = df['year'].astype(str)
        df['question'] = df['question'].astype(str)
        df['A'] = df['A'].astype(str)
        df['B'] = df['B'].astype(str)
        df['C'] = df['C'].astype(str)
        df['D'] = df['D'].astype(str)

        df.head()

        df.to_sql("questions", self.conn, if_exists="replace")

    def _answers(self, qs, id):

        def split_answer(qs, letter):
            response = []
            split_letter = '(' + letter + ')'
            splits = qs.split(split_letter)
            if len(splits) > 1:
                response = splits[1]
                qs = splits[0]
            return response, qs

        D, qs = split_answer(qs, 'D')
        C, qs = split_answer(qs, 'C')
        B, qs = split_answer(qs, 'B')
        A, qs = split_answer(qs, 'A')

        if len(A) == 0:
            D, qs = split_answer(qs, '4')
            C, qs = split_answer(qs, '3')
            B, qs = split_answer(qs, '2')
            A, qs = split_answer(qs, '1')

        A = "A. " + A.strip()
        B = "B. " + B.strip()
        C = "C. " + C.strip()
        if isinstance(D, str):
            D = "D. " + D.strip()

        return pd.Series({"questionID": id, "question":qs, "A": A, "B": B, "C":C, "D":D})
        
    # 3)) We create toy distributions of student responses by grade
    def _correct(self, q_grade, id):
        grade = int(q_grade)
        qs_answered = np.random.randint(1, 501, self.grade_range) # students in each grade who answered question

        # First we get a random mean between [10, 95]
        mu = 75*np.random.rand() + 20

        # Then we get a random standard deviation between [5, 20]
        sigma = 15*np.random.rand() + 5

        # Then we create a normal distribution, and take 7 random numbers, sort them
        # and return them as probabilities
        probs = np.random.normal(mu, sigma, 1000)
        np.random.shuffle(probs)
        distribution = probs[0:self.grade_range]
        distribution = np.sort(distribution)

        # Then we generate random numbers for each grade- mu is the grade of q
        series = {}

        for n in range(0, self.grade_range):
            series["Distribution" + str(n + self.min_grade)] = distribution[n]
            series["Distribution" + str(n + self.min_grade) + '_users'] = qs_answered[n]
        series['questionID'] = id

        return pd.Series(series)

    def _similarity_of_id(self, question_id):
        conn = sqlite3.connect('recs.db')
        c = conn.cursor()
        select = "SELECT question from questions where questionID=(?)"
        c.execute(select, (question_id,))
        response = c.fetchall()
        for re in response:
            last_question = re[0]
            break
        conn.close()
        
        q1 = self.nlp(last_question)
        word_list_1 = []
        for word in q1:
                if word.pos_ == 'NOUN':
                    word_list_1.append(word.text)
        tokenized_q1 = " ".join(sorted(word_list_1))
        if len(word_list_1) == 0:
            return None
        tk_q1 = self.nlp(tokenized_q1)
        return tk_q1

    def _match_profile(self, row, username, exam, grade, last_percentage, 
        percentage, qs_answered, tk_q1):

        grade = int(grade)

        # we scale each individual score to be out of 100
        difference_score = 0.0

        # Compare grade
        grade_weight = 1
        q_grade = int(row['schoolGrade'])
        diff = q_grade - grade
        difference_score += (100 * (diff) / float(6)) * grade_weight

        # Compare percentage
        percentage_weight = 1
        q_percent = row["Distribution" + str(grade)]
        q_answers = row["Distribution" + str(grade) + "_users"]
        difference_last = abs(q_percent - last_percentage) * percentage_weight
        difference_percent = abs(q_percent - percentage) * percentage_weight * qs_answered 
        difference_score = difference_last + difference_percent + difference_score

        # compare last question by token sort ratio
        # For (small amounts of) optimization
        # We arbitrarily fail 75% of the questions
        # This should make it more random as well
        if np.random.randint(0, 4) == 0:
            q2 = self.nlp(row['question'])
            word_list_2 = []
            for word in q2:
                if word.pos_ == 'NOUN':
                    word_list_2.append(word.text)

            if tk_q1 is None or len(word_list_2) == 0:
                sim = 0
            else:
                tokenized_q2 = " ".join(sorted(word_list_2))
                # TypeError occurs if sentence has no nouns
                try:
                    sim = tk_q1.similarity(self.nlp(tokenized_q2))
                except TypeError:
                    sim = 0
        else:
            sim = 0

        difference_q = 1 - sim
        question_weight = 100
        difference_score += question_weight * (100 * difference_q)

        # Compare test type
        difference_test = 100
        if exam == row['examName']:
            difference_test = 0
        test_weight = 1
        difference_score += difference_test * test_weight

        total_weight = test_weight + question_weight + grade_weight + 2 * percentage_weight
        similarity = 100 - difference_score / float(total_weight)

        assert similarity < 100 and similarity > 0

        # Update similarity
        conn = sqlite3.connect('recs.db')
        c = conn.cursor()
        update = "UPDATE user_similarity set similarity=(?) where username=(?) and questionId=(?)"
        c.execute(update, (similarity, username, row['questionID']))
        conn.commit()
        conn.close()

    # 5)) We implement a function to check if an answer is correct
    def _check_answer(self, answer, question_id):
        sql = "SELECT AnswerKey from questions where questionID=(?)"
        conn = sqlite3.connect('recs.db')
        c = conn.cursor()
        c.execute(sql, (question_id,))
        response = c.fetchall()
        correct = False
        count = 0
        for re in response:
            count += 1
            answer_key = re[0]
            if answer.lower().strip() == answer_key.lower().strip():
                correct = True
            break
        
        assert count == 1

        conn.close()
        return correct

    # 6)) We implement a function to find the minimum dissimilar that the user has not already answered
    def prep_next_q(self, answer, username, grade, exam, keywords, question_id):

        correct = self._check_answer(answer, question_id)
        last_percentage = 100
        if correct:
            last_percentage = 0 
        qs_answered, percentage, _, answered_questions, q_id = self.init_user_vars(username, grade, exam, keywords)
        
        assert q_id.lower() == question_id.lower()
        
        self._updates(correct, qs_answered, percentage, exam, answered_questions,
            q_id, grade, username)
        qs_answered, percentage, _, answered_questions, _ = self.init_user_vars(username, grade, exam, keywords)
        logging.info("Finish updates")

        tk_q1 = self._similarity_of_id(q_id)

        # Get similarity
        conn = sqlite3.connect('recs.db')
        df = pd.read_sql('SELECT * FROM questions;', conn)
        df.apply(lambda row: self._match_profile(row, username, exam, grade, last_percentage, 
        percentage, qs_answered, tk_q1), axis=1)
        conn.close()
        logging.info("Finished Similarity + Updates")
        return {
            "correct" : ("correct" if correct else "incorrect")
        }

    # 7)) We update student  / db statistics based on answered question
    def _updates(self, correct, qs_answered, percentage, test_type, answered_questions,
        question_id, grade, username):

        # Update percentage
        qs_right = int(percentage * qs_answered)
        if correct:
            qs_right += 1

        # Update qs answered
        qs_answered += 1
        percentage = qs_right / float(qs_answered)

        # Update answered questions
        answered_questions.append(question_id)
        assert len(answered_questions) == qs_answered
        str_questions = "____".join(answered_questions)

        # Update in DB
        conn = sqlite3.connect('recs.db')
        c = conn.cursor()
        user_update = "UPDATE users set answered_questions=(?) , qs_answered=(?) , percentage=(?) WHERE username=(?) and grade=(?)"
        c.execute(user_update, (str_questions, qs_answered, percentage, username, grade))
        conn.commit()
        conn.close()

        # Update db number in grade who answered + percentage correct
        conn = sqlite3.connect('recs.db')
        get_users = "SELECT Distribution" + str(grade) + "_users" +  ", Distribution" + str(grade) +  " from questions WHERE questionID=(?)"
        c = conn.cursor()
        c.execute(get_users, (question_id,))
        response = c.fetchall()
        for re in response:
            users = int(re[0])
            percent = int(re[1])
            prev_correct = int(users * percent)
            if correct:
                prev_correct += 1
            percent = prev_correct / float(users)
            
            sql_update = "UPDATE questions set Distribution" + str(grade) + "_users" + "=(?), Distribution" + str(grade) + "=(?) where questionID=(?)"
            c = conn.cursor()
            c.execute(sql_update, (users, percent, question_id))
        conn.commit()
        conn.close()
        return qs_answered, percentage, test_type, answered_questions

    # Ends sql connection
    def _end_session(self):
        self.conn.close()

