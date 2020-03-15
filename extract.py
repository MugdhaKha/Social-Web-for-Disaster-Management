import tweepy
import sys

ACCESS_TOKEN = "1039752572268105728-dwVVD6csJLwsHHQgb4MRZK4Iw994ES"
ACCESS_TOKEN_SECRET = "FJjVvqAxVJwsy1KinygDojlu9EqGIGXkR9nNktu7CRD3n"
CONSUMER_KEY = "UVzzciwoMsjUAFbnMwFjU2MKf"
CONSUMER_SECRET = "1RavkcGGRy7FfvuLLs7UwSQOuACxjKZDF1iE9alRcscQ9QlC1V"

auth = tweepy.OAuthHandler(CONSUMER_KEY , CONSUMER_SECRET)
auth.set_access_token(ACCESS_TOKEN , ACCESS_TOKEN_SECRET)

api = tweepy.API(auth)

arr = [592632778451132417,
592683588258308098,
592549938073980928,
592279426030641153,
592012112500666370,
592542239563378688
]

i=1
non_bmp_map = dict.fromkeys(range(0x10000, sys.maxunicode + 1), 0xfffd)

for id_of_tweet in arr:
    try:
        tweet = api.get_status(id=id_of_tweet)
    except ConnectionError as c:
        print(c)
        exit
    except tweepy.error.TweepError as e:
        continue
    print (i,':- ',tweet.text.translate(non_bmp_map),'\n')
    i+=1
