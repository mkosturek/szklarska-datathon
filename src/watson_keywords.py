import os
import sys
import pandas as pd
from watson_developer_cloud import NaturalLanguageUnderstandingV1, WatsonApiException
from watson_developer_cloud.natural_language_understanding_v1 import Features, KeywordsOptions

watson_client = NaturalLanguageUnderstandingV1(
  username=os.environ['API_USERNAME'],
  password=os.environ['API_PASSWORD'],
  version='2018-03-16'
)


def analyze_text(api_client, text, limit=5):
    return api_client.analyze(
        text=text,
        features=Features(
            keywords=KeywordsOptions(
                emotion=True,
                sentiment=True,
                limit=limit
            )
        ),
        language='en'
    )


def write_header(file, number=5):
    header = ['_cached_page_id']
    fields_name = [
        'keyword',
        'relevance',
        'sentiment_score',
        'sentiment_label',
        'emotion_sadness',
        'emotion_joy',
        'emotion_fear',
        'emotion_disgust',
        'emotion_anger',
    ]
    header.extend('{}_{}'.format(f, i) for i in range(1, number) for f in fields_name)
    file.write(','.join(header) + '\n')


empty_emotion = dict(
    sadness=-1,
    joy=-1,
    fear=-1,
    disgust=-1,
    anger=-1,
)


def write_data(file, page_id, response):
    file.write(page_id)
    for keyword in response['keywords']:
        fields = [
            '"{}"'.format(keyword['text']),
            keyword['relevance'],
            keyword['sentiment']['score'],
            keyword['sentiment']['label'],
            keyword.get('emotion', empty_emotion)['sadness'],
            keyword.get('emotion', empty_emotion)['joy'],
            keyword.get('emotion', empty_emotion)['fear'],
            keyword.get('emotion', empty_emotion)['disgust'],
            keyword.get('emotion', empty_emotion)['anger'],
        ]
        file.write(',' + ','.join(map(str, fields)))
    file.write('\n')


def scrap(df, start_row):
    with open(export_filename, 'a') as f:
        for i, row in df[start_row:].iterrows():
            print(i, '/', len(df))
            try:
                keywords = analyze_text(watson_client, row['text'])
            except WatsonApiException:
                return i
            print(keywords)
            write_data(f, row['_cached_page_id'], keywords)


if __name__ == '__main__':
    df = pd.read_csv('../data/processed_erc.csv')
    export_filename = '../data/erc_watson_keywords.csv'
    try:
        last_row = int(sys.argv[-1])
    except ValueError:
        last_row = 0

    while last_row is not None:
        last_row = scrap(df, last_row)
