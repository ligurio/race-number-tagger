import boto
from boto.mturk.connection import MTurkConnection
from boto.mturk.question import HTMLQuestion
from boto.mturk.layoutparam import LayoutParameter
from boto.mturk.layoutparam import LayoutParameters

mtc = MTurkConnection(aws_access_key_id='AKIAJRNJOVSR66N43YIQ',
aws_secret_access_key='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',
#host="mechanicalturk.sandbox.amazonaws.com",
host='mechanicalturk.amazonaws.com',
proxy="proxy.avp.ru",
proxy_port="8080",
proxy_user="svc_linuxrepoproxy",
proxy_pass="linux#Repo8proX$y")

image_url = LayoutParameter('image_url', 'http://turk.s3.amazonaws.com/stop_sign_picture.jpg')
obj_to_find = LayoutParameter('objects_to_find','stop sign')

params   = LayoutParameters([ image_url, obj_to_find ])

response = mtc.create_hit(
  hit_layout    = "3R7WSKJ9DZ47TWI1O5J3D5ERRVGPTM",
  layout_params = params, 
  hit_type      = "3WDXCWYEASQ8X3ABSVLUQIFM6QNJY5"
)
                          
# The response included several fields that will be helpful later
hit_type_id = response[0].HITTypeId
hit_id = response[0].HITId
print("Your HIT has been created. You can see it at this link:")
print("https://www.mturk.com/mturk/preview?groupId={}".format(hit_type_id))
print("Your HIT ID is: {}".format(hit_id))
