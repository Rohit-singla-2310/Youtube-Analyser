import re
import google
import pickle
import string
from urllib.parse import urlparse, parse_qs
from googleapiclient.discovery import build
from flask import Flask, request, render_template
from tensorflow.python.keras.models import load_model
from keras_preprocessing.sequence import pad_sequences

app = Flask(__name__)


POSITIVE = "POSITIVE"
NEGATIVE = "NEGATIVE"
NEUTRAL = "NEUTRAL"
SENTIMENT_THRESHOLDS = (0.4, 0.7)
SEQUENCE_LENGTH = 300


tokenizer = pickle.load(open('tokenizer.pkl', 'rb'))
model_new = load_model("model.h5")


def get_id(url):

    # print(type(url))
    u_pars = urlparse(url)
    quer_v = parse_qs(u_pars.query).get('v')
    if quer_v:
        return quer_v[0]
    pth = u_pars.path.split('/')
    if pth:
        return pth[-1]


# url="https://www.youtube.com/watch?v=LgKyRBHVSTU"

def remove_emoji(text):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', str(text))


def get_comments(youtube, video_id, comments=[], token=''):

    video_response = youtube.commentThreads().list(part='snippet', videoId=video_id, pageToken=token).execute()
    for item in video_response['items']:
        comment = item['snippet']['topLevelComment']
        text = comment['snippet']['textDisplay']
        comments.append(text)
    if "nextPageToken" in video_response:
        return get_comments(youtube, video_id, comments, video_response['nextPageToken'])
    else:
        return comments


def remove_url(text):
    url_pattern = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    return url_pattern.sub(r'', text)


def clean_text(text):
    delete_dict = {sp_character: '' for sp_character in string.punctuation}
    delete_dict[' '] = ' '
    table = str.maketrans(delete_dict)
    text1 = str(text).translate(table)
    # print('cleaned:'+text1)
    textArr = str(text1).split()
    text2 = ' '.join([w for w in textArr if (not w.isdigit() and (not w.isdigit() and len(w) > 3))])

    return text2.lower()


def decode_sentiment(score, include_neutral=True):
    if include_neutral:
        label = NEUTRAL
        if score <= SENTIMENT_THRESHOLDS[0]:
            label = NEGATIVE
        elif score >= SENTIMENT_THRESHOLDS[1]:
            label = POSITIVE

        return label
    else:
        return NEGATIVE if score < 0.5 else POSITIVE


def predicti(text, include_neutral=True):
    # Tokenize text
    x_test = pad_sequences(tokenizer.texts_to_sequences([text]), maxlen=SEQUENCE_LENGTH)
    # Predict
    score = model_new.predict([x_test])[0]
    # Decode sentiment
    label = decode_sentiment(score, include_neutral=include_neutral)

    return {"label": label, "score": float(score)}


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    url = [str(x) for x in request.form.values()]#Convert string inputs to float.
    print(url)
    # features = [np.array(int_features)]  #Convert to the form [[a, b]] for input to the model
    youtube = build('youtube', 'v3',
                    developerKey="AIzaSyC4Id7WXn1DjZesAxkkc2N2-oPKGGk2tMU", cache_discovery=False)

    vID = get_id(url[0])

    response = youtube.commentThreads().list(
        part='snippet',
        maxResults=100,
        textFormat='plainText',
        order='time',
        videoId=vID
    ).execute()

    comments = []
    comments = get_comments(youtube, vID, comments)

    pre_comments = []
    for i in range(len(comments)):
        # preprocessing
        text = comments[i]
        text = remove_url(text)
        text = remove_emoji(text)
        text = clean_text(text)
        pre_comments.append(text)

    n = len(pre_comments)
    negative = 0
    positive = 0
    neutral = 0
    for i in pre_comments:
        k = predicti(i)
        if k['label'] == 'NEUTRAL':
            neutral = neutral + 1
        elif k['label'] == 'POSITIVE':
            positive = positive + 1
        else:
            negative = negative + 1

    print(n)
    print(positive)
    print(negative)
    print(neutral)

    return render_template('index.html', prediction_text='Percent Positive Comments {:.2%}'.format(positive/n), negative='Percent Negative Comments {:.2%}'.format(negative/n), neutral ='Percent Neutral Comments {:.2%}'.format(neutral/n))


if __name__ == "__main__":
    app.run()
