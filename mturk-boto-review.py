# https://blog.mturk.com/tutorial-retrieving-bounding-box-image-annotations-from-mturk-253b86cb7502

import boto
from boto.mturk.connection import MTurkConnection
from boto.mturk.question import HTMLQuestion
from boto.mturk.layoutparam import LayoutParameter
from boto.mturk.layoutparam import LayoutParameters
import json

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import requests
from io import BytesIO
from PIL import Image

# Create your connection to MTurk
mtc = MTurkConnection(aws_access_key_id='AKIAJRNJOVSR66N43YIQ',
aws_secret_access_key='5ZaO6VMWHUsKDQT2ncLO8jRGv3z0QUKynLqDl4FC',
host="mechanicalturk.sandbox.amazonaws.com",
#host='mechanicalturk.amazonaws.com',
proxy="proxy.avp.ru",
proxy_port="8080",
proxy_user="svc_linuxrepoproxy",
proxy_pass="linux#Repo8proX$y")

# This is the value you received when you created the HIT
# You can also retrieve HIT IDs by calling GetReviewableHITs
# and SearchHITs. See the links to read more about these APIs.
hit_id = "31S7M7DAGFPYLGQYRI5SD078ZHUTLH"
result = mtc.get_assignments(hit_id)
assignment = result[0]
worker_id = assignment.WorkerId
for answer in assignment.answers[0]:
  if answer.qid == 'annotation_data':
    worker_answer = json.loads(answer.fields[0])

# Load the image from the HIT
response = requests.get('http://turk.s3.amazonaws.com/stop_sign_picture.jpg')
img = Image.open(BytesIO(response.content))
im = np.array(img, dtype=np.uint8)

# Create figure, axes, and display the image
fig,ax = plt.subplots(1)
ax.imshow(im)

# Draw the bounding box
for answer in worker_answer:
    rect = patches.Rectangle((answer['left'],answer['top']),answer['width'],answer['height'],linewidth=1,edgecolor='#32cd32',facecolor='none')
    ax.add_patch(rect)

# Show the bounding box
plt.show()
