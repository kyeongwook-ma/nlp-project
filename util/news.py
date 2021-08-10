import json
import re

import requests
from bs4 import BeautifulSoup
from selenium import webdriver

NAVER_CLIENT_ID = 'PP9m_gkbOYtwfpKjuvrk'
NAVER_CLIENT_SECRET = 'dZEnzKBqIz'

search_query = '올림'
encode_type = 'json'

# 출력할
max_display = 100

# 정렬 기준 date (시간순), sim(관련도 순)
sort = 'date'

headhers = {
    'X-Naver-Client-Id': NAVER_CLIENT_ID,
    'X-Naver-Client-Secret': NAVER_CLIENT_SECRET
}

driver = webdriver.Chrome('./chromedriver')
driver.implicitly_wait(1)


def parse_contents(link):
    driver.get(link)
    soup = BeautifulSoup(driver.page_source, 'html.parser')

    print(soup)


def crawl_news(query, start=1, offset=1000):
    def remove_html(content):
        cleaner = re.compile(r"<.*?>")
        cleaned_text = re.sub(cleaner, '', content)
        cleaned_text = cleaned_text.replace('&quot;', '"')
        cleaned_text = cleaned_text.replace('&amp;', '&')
        cleaned_text = cleaned_text.replace('&#039;', "'")
        cleaned_text = cleaned_text.replace('&lt;', '<')
        cleaned_text = cleaned_text.replace('&gt;', '>')
        return cleaned_text

    news = []

    for idx in range(start, offset):
        url = 'https://openapi.naver.com/v1/search/news.{}?' \
              'query={}&' \
              'display={}&' \
              'start={}&' \
              'sort={}'.format(encode_type, query, max_display, start, sort)

        resp = requests.get(url, headers=headhers)
        raw_news = json.loads(resp.text)

        # title
        # link
        # description
        # originallink
        for n in raw_news['items']:
            contents = parse_contents(n['link'])

            news.append({
                'link': n['link'],
                'contents': contents,
                'title': remove_html(n['title']),
                'description': remove_html(n['description'])
            })

    return news
