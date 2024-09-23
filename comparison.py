import pandas as pd
import os
from helpers import extract_domain_name

file_path = os.path.join(os.getcwd(), 'comparison/pool_corporation.xlsx')
df = pd.read_excel(file_path)

column1 = df['GTD']
column2 = df['AgentsOutput']

column1 = set([extract_domain_name(value) for value in column1])

common_values = set(column1).intersection(set(column2))

unique_gtd_domains = column1.difference(set(column2))

print("Common values:", common_values)
print("Other GTD domains:", unique_gtd_domains)