import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import pickle
import logging
from flask import Flask, request, jsonify
logging.basicConfig(filename= 'logging.log', level= logging.INFO , message= '%(levelname)s, %(asctime)s, %(message)s')

app = Flask(__name__)


@app.route('/mass_class', methods = ['GET', 'POST'])
def fetching_data():
    try:
        user = request.json['user']
        host = request.json['host']
        database = request.json['database']
        password = request.json['password']
        import mysql.connector as conn
        mydb = conn.connect(user = user, host = host, database = database, passwd = password)
        logging.info('connection is successfull')
    except Exception as e:
        logging.exception(str(e)+ 'error occured during connection to database')
    try:
        cursor = mydb.cursor()
        cursor.execute('select * from class_algeria')
        fetch_data = [list(i) for i in cursor.fetchall()]
        down_data = pd.DataFrame(fetch_data, columns=('Day',  'Month',  'Humidity',  'Wind',  'Rain',  'DMC',    'DC',  'Spread',  'Region',  'Classes'))    
        X_inserted = down_data.drop('Classes', axis =1)
    except Exception as e:
        logging.exception(str(e)+ 'sth happened while fetching data from database')
    try:
        loaded_class_model = pickle.load(open('class_model.pkl', 'rb'))
        y_pred = loaded_class_model.predict(X_inserted)
        y_modified = ['Fire' if i ==1  else 'Not Fire' for i in y_pred]
        y_modified_s = str(y_modified)
        return jsonify(y_modified_s)
    except Exception as e:
        logging.exception(str(e)+ 'sth happened while predicting the data with model')




@app.route('/mass_regress', methods = ['GET', 'POST'])
def fetchingreg_data():
    try:
        user = request.json['user']
        host = request.json['host']
        database = request.json['database']
        password = request.json['password']
        import mysql.connector as conn
        mydb = conn.connect(user = user, host = host, database = database, passwd = password)
        logging.info('connection is successfull')
    except Exception as e:
        logging.exception(str(e)+ 'error occured during connection to database')
    try:
        cursor = mydb.cursor()
        cursor.execute('select * from regression_algeria')
        fetch_data = [list(i) for i in cursor.fetchall()]
        down_data = pd.DataFrame(fetch_data, columns=('Day',  'Month',  'Humidity',  'Wind',  'Rain',  'DMC',    'DC',  'Spread',  'Region',  'Classes'))    
        X_inserted = down_data.drop('Rain', axis =1)
    except Exception as e:
        logging.exception(str(e)+ 'sth happened while fetching data from database')
    try:
        loaded_class_model = pickle.load(open('regression_model.pkl', 'rb'))
        y_pred = loaded_class_model.predict(X_inserted)
        #Mim_max_X.inverse_transform(y_pred.reshape(-1, 1))
        y_modified_s = str(np.round(y_pred,2))
        return jsonify(y_modified_s)
    except Exception as e:
        logging.exception(str(e)+ 'sth happened while predicting the data with model')




@app.route('/single_regress', methods = ['POST'])
def single_regression():
    try:
        data=request.json['data']
        print(data)
        new_data=[list(data.values())]
    
    except Exception as e:
        logging.exception(str(e)+ 'error occured whıle fetchıng data from postman')

    try:
        loaded_class_model = pickle.load(open('regression_model.pkl', 'rb'))
        y_pred = loaded_class_model.predict(new_data)[0]
        y_modified_s = str(y_pred)
        return jsonify(y_modified_s)
    except Exception as e:
        logging.exception(str(e)+ 'sth happened while predicting the data with model')



@app.route('/single_class', methods = ['POST'])
def single_classif():
    try:
        data=request.json['data']
        print(data)
        new_data=[list(data.values())]
    
    except Exception as e:
        logging.exception(str(e)+ 'error occured whıle fetchıng data from postman')

    try:
        loaded_class_model = pickle.load(open('class_model.pkl', 'rb'))
        y_pred = loaded_class_model.predict(new_data)[0]
        y_str = str(y_pred)
        return jsonify(y_str)
    except Exception as e:
        logging.exception(str(e)+ 'sth happened while predicting the data with model')


@app.route('/single_class_html', methods=['POST'])
def single_class_html():
    try:
        data = [float(x) for x in request.form.values()]
        new_data = [np.array(data)]
        print(new_data)
    except Exception as e:
        logging.exception(str(e) + 'error occured while fetching data from html page')

    try:
        loaded_class_model = pickle.load(open('class_model.pkl', 'rb'))
        y_pred = loaded_class_model.predict(new_data)[0]

        return render_template('home.html', prediction_text="Airflow pressure is {}".format(y_pred))
    except Exception as e:
        logging.exception(str(e) + 'sth happened while predicting the data with model')


if __name__ == '__main__':
    app.run(port = 8000)







