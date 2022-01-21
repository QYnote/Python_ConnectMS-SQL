import clsDatabase as db
import pandas as pd

query = 'SELECT * FROM TradingTest'

data = db.SelectQuery(query)

db.ConnectionClose()
