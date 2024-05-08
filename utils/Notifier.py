import requests

from utils import credentials


def notify(platform, symbol, *text):
    try:
        message = credentials.TELEGRAM_BOT_URL + platform + " "
        if text:
            message += " " + text[0] + " "
        message += '`' + symbol + '`' + '&parse_mode=MarkDown'

        requests.get(message)

    except Exception as watch_exc:
        print("NOTIFIER", watch_exc)
