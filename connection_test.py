import pyodbc

conn = pyodbc.connect(
   "Driver={ODBC Driver 18 for SQL Server};"
   "Server=device-data.c7y0avuyjpgj.eu-west-2.rds.amazonaws.com,1433;"
   "Database=DeviceManagement;"
   "UID=darshtp;"
   "PWD=darshtp_boja;"
   "Encrypt=yes;"
   "TrustServerCertificate=yes;"
)
print("Connected!")
