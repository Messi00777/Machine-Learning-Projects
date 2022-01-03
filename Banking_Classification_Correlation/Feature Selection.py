from Preprocessing import *

features = list(data.select_dtypes(include='object').columns)
print(features)

