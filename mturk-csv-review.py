# -*- coding: utf-8 -*-

"""
Принимает два параметра - файл с результатами в csv и путь к изображениям.
Показывает аннотированные фото и ждет ответа.
Потом результаты записывает в отдельную CSV.

- Можно сохранять аннотированные фото на диск.
- Можно загружать фото из S3 для ревью, если их нет локально.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
import csv
import json
import urllib
import os
import sys

def check_annotation(boxes):

    for box in boxes:
        if not box['label'].isdigit():
            raise ValueError('Wrong label %s' % box['label'])


def draw_box(filename, boxes, image_path, save = False):

    """
    Draw annotated box on an image
    """
    im = np.array(Image.open(os.path.join(image_path, filename)), dtype=np.uint8a)
    fig, ax = plt.subplots(1)
    ax.imshow(im)

    for box in boxes:
        print "\t", box
        rect = patches.Rectangle((box["left"],box["top"]),box["width"],box["height"],
                                linewidth=1,edgecolor='r',facecolor='none')
        ax.add_patch(rect)
        ax.annotate(box['label'], (box["left"] + 25, box["top"] - 25), color='red', weight='bold',
                    fontsize=15, ha='center', va='center')
    plt.show()

    if save:
        name = filename.split('.')[0]
        annotated_image = name + '-annotate.png'
        print "Saved to %s" % annotated_image
        fig.savefig(annotated_image)
    plt.close(fig)
    plt.close('all')


def write_csv(filename, hits, header):

    csv_table = open(filename, 'w')
    with csv_table:
        writer = csv.DictWriter(csv_table, quotechar='"', fieldnames=header)
        writer.writeheader()
        for hit in hits:
            writer.writerow(hit)


def read_csv(filename):

    with open(filename, 'r') as fp:
        reader = csv.DictReader(fp)
        header = next(reader, None)
        hits = list(reader)

    return hits


def download(url, filename):

    urllib.urlretrieve (url, filename)


def lookup_hit(hits, hit_id):

    for line in hits:
        if line['HITId'] == hit_id and line['AssignmentStatus'] != 'Rejected':
            return line


def main():

    DOWNLOAD = False
    REJECTED_ONLY = True
    results = sys.argv[1]
    image_path = sys.argv[2]
    hits = read_csv(results)

    updated_hits = []
    for line in hits:

        if REJECTED_ONLY:
            if line['AssignmentStatus'] == 'Rejected':
                line = lookup_hit(hits, line['HITId'])
            else:
                updated_hits.append(line)
                continue
        
        boxes = json.loads(line['Answer.annotation_data'])
        filename = line['Input.objects_to_find']
        url = line['Input.image_url']
        status = line['AssignmentStatus']
        print filename, url, status

        try:
            check_annotation(boxes)
            draw_box(filename, boxes, image_path)
        except Exception as e:
            print filename, e

        approve = raw_input("Please enter Y or reason: ")
        if approve == "y":
            line['Approve'] = 'x'
        else:
            line['Reject'] = approve
        updated_hits.append(line)

        if not os.path.exists(filename) and DOWNLOAD:
            download(url, filename)
            print "Downloading %s" % filename

    updated_csv = os.path.basename(results).split('.')[0] + '-reviewed.csv'
    write_csv(updated_csv, updated_hits, header)
    print "++++++ Updated file with results: ", reviewed_csv


if __name__ == '__main__':
    main()
