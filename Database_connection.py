import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import pickle
logging.basicConfig(filename='logging.log' , level= logging.INFO, format = '%(levelname)s, %(asctime)s %(message)s'  )



def class_table():
    #make connection to the database
    try:
        import mysql.connector as conn
        mydb = conn.connect(user = 'root',host = 'localhost', database = 'test5', passwd= '431170')
        logging.info('Successfully connected the database')
    except Exception as e:
        logging.exception(str(e)+ 'connection error')
    #make cursor and table in database
    try:
        cursor = mydb.cursor()
        cursor.execute('create table test5.class_algeria(day float, month float, humidity float , wind float, rain float, dmc float, dc float, spread float, region float, classes float)')
        mydb.commit()
        logging.info('table created')
        return 'table created in test5 database and class_algerian is formed'
    except Exception as e:
        logging.exception(str(e )+ 'error occured while new table form')


def scale_insert_database():
    try:
        data = pd.read_csv('df_final_with10.csv', index_col=0)
        sample_data = data.sample(20)
        #scaling data
        X_columns  = ['Humidity', 'Wind', 'Rain', 'DMC', 'DC', 'Spread']
        X_scale = sample_data[X_columns]
        from sklearn.preprocessing import MinMaxScaler
        Mim_max_X = MinMaxScaler()
        my_data = sample_data.copy()
        X_part = pd.DataFrame(Mim_max_X.fit_transform(X_scale), columns = X_columns, index=sample_data.index)
        my_data.drop(X_columns,axis = 1, inplace = True)
        df_new = my_data.join(X_part)
        col_order = ['Day',  'Month',  'Humidity',  'Wind',  'Rain',  'DMC',    'DC',  'Spread',  'Region',  'Classes']
        df_new = df_new[col_order]
        cursor = mydb.cursor()
        cursor.execute('delete from test5.class_algeria')
        mydb.commit()
        for i, rows in df_new.iterrows():
            row_in = tuple(rows)
            cursor.execute('insert into test5.class_algeria values(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s )', row_in)
        mydb.commit()
        logging.info('insertion completed successfully')
        return 'scaling and inserting into test5.class_algeria is done'

    except Exception as e:
        logging.exception(str(e), 'error occured while insertion')
        mydb.rollback()

def table_regression():
    try:
        import mysql.connector as conn
        mydb = conn.connect(user = 'root',host = 'localhost', database = 'test5', passwd= '431170')
        logging.info('Successfully connected the database')
    except Exception as e:
        logging.exception(str(e)+ 'connection error')
    #make cursor and table in database
    try:
        cursor = mydb.cursor()
        cursor.execute('create table test5.regression_algeria(day float, month float, humidity float , wind float, rain float, dmc float, dc float, spread float, region float, classes float)')
        mydb.commit()
        logging.info('table created')
        return 'table for regression is formed'
    except Exception as e:
        logging.exception(str(e )+ 'error occured while new table form')



def scale_table_regress():
    try:
        data = pd.read_csv('df_final_with10.csv', index_col=0)
        sample_data = data.sample(20)
        #scaling data
        X_columns  = ['Humidity', 'Wind',  'DMC', 'DC', 'Spread']
        X_scale = sample_data[X_columns]
        from sklearn.preprocessing import MinMaxScaler
        Mim_max_X = MinMaxScaler()
        my_data = sample_data.copy()
        X_part = pd.DataFrame(Mim_max_X.fit_transform(X_scale), columns = X_columns, index=sample_data.index)
        my_data.drop(X_columns,axis = 1, inplace = True)
        df_new = my_data.join(X_part)
        col_order = ['Day',  'Month',  'Humidity',  'Wind',  'Rain',  'DMC',    'DC',  'Spread',  'Region',  'Classes']
        df_new = df_new[col_order]
        cursor = mydb.cursor()
        cursor.execute('delete from test5.regression_algeria')
        mydb.commit()
        for i, rows in df_new.iterrows():
            row_in = tuple(rows)
            cursor.execute('insert into test5.regression_algeria values(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s )', row_in)
        mydb.commit()
        logging.info('insertion completed successfully')

        return 'regression table insertion and scaling is done'

    except Exception as e:
        logging.exception(str(e), 'error occured while insertion')
        mydb.rollback()



if __name__ == '__main__':
    scale_insert_database()