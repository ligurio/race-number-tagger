# -*- coding: utf-8 -*-

import csv
import json
import os

#RESULTS = 'Batch_3180686_batch_results-1-100.csv'
RESULTS = 'Batch_3183602_batch_results-2-1000.csv'

with open('rejected-2.txt', 'r') as reject:
    regected_images = reject.read().splitlines()
    
rejected_hits = []
with open(RESULTS, 'r') as fp:
    reader = csv.DictReader(fp)
    header = next(reader, None)
    for row in reader:
        filename = row['Input.objects_to_find']
        if filename not in regected_images:
            row['Approve'] = 'x'
        else:
            row['Reject'] = 'incomplete label'
        rejected_hits.append(row)
        
csv_table = open('Batch_3183602_batch_results-2-1000-rejected.csv', 'w')
with csv_table:
    writer = csv.DictWriter(csv_table, quotechar='"', fieldnames=header)
    writer.writeheader()
    for hit in rejected_hits:
        writer.writerow(hit)
