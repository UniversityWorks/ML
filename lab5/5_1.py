import pandas as pd
 
K = 5
 
languages = ["Python", "JavaScript", "Java", "C++", "Ruby"][:K]
series = pd.Series(languages, index=range(1, K + 1))
 
print(f"Series з {K} мовами програмування:")
print(series)
 
