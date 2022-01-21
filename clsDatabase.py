#DB
import pymssql as sql
import datetime

server = '192.168.0.76:2433'
user = 'sample'
password = 'qwe123!@#'
db = 'TEST'

conn = sql.connect(server=server, user=user, password=password, database=db, charset='utf8')

def SelectQuery(query) : 
    dataList = []

    cursor = conn.cursor(as_dict=True)
    cursor.execute(query)
    row = cursor.fetchone()

    while row :
        for colCount in row :
            if(type(row[colCount]) == type(str)) :
                row[colCount] = row[colCount].encode('ISO-8859-1').decode('euc-kr')
                #데이터가 string이면 인코딩 디코딩하여 한글깨짐 방지

        dataList.append(row)
        row = cursor.fetchone()

    cursor.close()

    return dataList


def InsertQuery(query) :
    cursor = conn.cursor()

    cursor.execute(query)

    cursor.close()
    conn.commit()


def ConnectionClose() : 
    conn.close()
