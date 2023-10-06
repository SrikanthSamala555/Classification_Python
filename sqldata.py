import pyodbc

class sqlconnection:
    def __init__(self):
            self.SERVER = ''
            self.DATABASE = ''
            self.USERNAME = ''
            self.PASSWORD = ''
    def create_connection(self):
        connectionString = f'DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={self.SERVER};DATABASE={self.DATABASE};UID={self.USERNAME};PWD={self.PASSWORD}'
        conn = pyodbc.connect(connectionString)
        return conn
    
    def fetch_cpt_desc(self,input:str):
        print("----input")
        print(input)
        conn = self.create_connection()
        cursor = conn.cursor()
        sqlquery = "SELECT distinct ProcedureCode AS CPTCodes, Description FROM dbo.ProcedureCodes WHERE ProcedureCode IN ("+input+")"
        data= cursor.execute(sqlquery).fetchall()
        print(sqlquery)
        desc=""
        for row in data:
            desc=desc+ str(row)
        return desc
        
    
    def fetch_pos_desc(self,input:str):
        conn = self.create_connection()
        cursor = conn.cursor()
        sqlquery = "SELECT distinct Code AS POSCode, Name as Description FROM dbo.POSCodes WHERE Code IN ("+input+")"
        data= cursor.execute(sqlquery).fetchall()
        desc=""
        for row in data:
            desc=desc+ str(row)
        return desc
     
    def fetch_rev_desc(self,input:str):
        conn = self.create_connection()
        cursor = conn.cursor()
        sqlquery = "SELECT distinct  RevenueCode as Code, Description as Description FROM dbo.RevCodes WHERE RevenueCode IN ("+input+")"
        data= cursor.execute(sqlquery).fetchall()
        desc=""
        for row in data:
            desc=desc+ str(row)
        return desc
    def dispose(self):
        self.cursor.close()
        self.conn.close()
        

