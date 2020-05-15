# this lambda has layers mailparser
import os
import io
import boto3
import json
import csv
import logging
import email 
import sys
from hashlib import md5
sys.path.insert(1, '/opt')
import mailparser
import string
from botocore.exceptions import ClientError

# grab environment variables
ENDPOINT_NAME = ''
vocabulary_length = 9013
runtime= boto3.client('runtime.sagemaker')
s3 = boto3.client("s3")

if sys.version_info < (3,):
    maketrans = string.maketrans
else:
    maketrans = str.maketrans

def vectorize_sequences(sequences, vocabulary_length):
    seq = [0]*vocabulary_length
    results = []
    for i in range(len(sequences)):
        results.append(seq)
    for i, sequence in enumerate(sequences):
      for j in sequence:
        results[i][j] = 1. 
    return results

def one_hot_encode(messages, vocabulary_length):
    data = []
    for msg in messages:
        temp = one_hot(msg, vocabulary_length)
        data.append(temp)
    return data

def one_hot(text, n,
            filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
            lower=True,
            split=' '):
      return hashing_trick(text, n,
                         hash_function='md5',
                         filters=filters,
                         lower=lower,
                         split=split)

def hashing_trick(text, n,
                  hash_function=None,
                  filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                  lower=True,
                  split=' '):
    if hash_function is None:
        hash_function = hash
    elif hash_function == 'md5':
        hash_function = lambda w: int(md5(w.encode()).hexdigest(), 16)

    seq = text_to_word_sequence(text,
                                filters=filters,
                                lower=lower,
                                split=split)
    return [int(hash_function(w) % (n - 1) + 1) for w in seq]

def text_to_word_sequence(text,
                          filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                          lower=True, split=" "):

    if lower:
        text = text.lower()

    if sys.version_info < (3,):
        if isinstance(text, unicode):
            translate_map = dict((ord(c), unicode(split)) for c in filters)
            text = text.translate(translate_map)
        elif len(split) == 1:
            translate_map = maketrans(filters, split * len(filters))
            text = text.translate(translate_map)
        else:
            for c in filters:
                text = text.replace(c, split)
    else:
        translate_dict = dict((c, split) for c in filters)
        translate_map = maketrans(translate_dict)
        text = text.translate(translate_map)

    seq = text.split(split)
    return [i for i in seq if i]


def lambda_handler(event, context):
    print("Received event: " , event)
    
    file_obj = event["Records"][0]
    filename = str(file_obj["s3"]['object']['key'])
    bucket_name = str(file_obj['s3']['bucket']['name'])
    print('bucket-name is:', bucket_name)
    fileObj = s3.get_object(Bucket = bucket_name, Key=filename)
    read = fileObj['Body'].read()
    print('read bytes are :',read )
    
    mail = mailparser.parse_from_bytes(read)
    
    body = mail.body.partition('--- mail_boundary ---')[0]
    body = body.replace('\r','')
    body = body.replace('\n',' ')
    print('mail body is:', body)
    
    
    #get the email from s3
    EMAIL_RECEIVE_DATE = mail.headers['Date'].partition(' -')[0]
    EMAIL_SUBJECT = mail.headers['Subject']
    email_address = mail.headers['From']
    EMAIL_BODY = body
    print('data, subject, from are :',EMAIL_RECEIVE_DATE,EMAIL_SUBJECT,email_address )
    

    # process the texts for prediction
    test_messages = [body]
    one_hot_test_messages = one_hot_encode(test_messages, vocabulary_length)
    encoded_test_messages = vectorize_sequences(one_hot_test_messages, vocabulary_length)
    msg = json.dumps(encoded_test_messages)
   
    
    response = runtime.invoke_endpoint(EndpointName=ENDPOINT_NAME,
                                       ContentType='application/json',
                                       Body=msg)
    
    result = json.loads(response['Body'].read().decode())
    print("result: ", result)
    predicted_label = result["predicted_label"][0]
    if predicted_label[0] < 1:
        predicted_label = 'NOT Spam'
    else:
        predicted_label = 'Spam'
    predicted_probability = result["predicted_probability"][0][0]
    
    # resend the email to remind the sender
    response_text = "We received your email from \"{}\" sent at \"{}\" with the subject \"{}\".\n \nHere is a 240 character sample of the email body:\n\n\"{}\".\n \nThis email was categorized as \"{}\" with a {:2.4}% confidence.".format(str(email_address), str(EMAIL_RECEIVE_DATE), str(EMAIL_SUBJECT),  \
    str(EMAIL_BODY), str(predicted_label), str(predicted_probability * 100))
    
    
    
    #send a receipt email back using SES

    SENDER = "" # the domain set email
    RECIPIENT = "" 
    AWS_REGION = "us-east-1"
    SUBJECT = "The classification result of your email sent to us"
    BODY_TEXT = (response_text)
    CHARSET = "UTF-8"
    
    client = boto3.client('ses',region_name=AWS_REGION)
    #Provide the contents of the email.
    email_response = client.send_email(
        Destination={
            'ToAddresses': [
                RECIPIENT,
            ],
        },
        Message={
            'Body': {
                'Text': {
                    'Charset': CHARSET,
                    'Data': BODY_TEXT,
                },
            },
            'Subject': {
                'Charset': CHARSET,
                'Data': SUBJECT,
            },
        },
        Source=SENDER)  
    return {
        'statusCode': 200,
        'body': "The receipt email has been sent."
    }
    

    
    