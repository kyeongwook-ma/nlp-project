import json
import re

import requests
import torch
from bs4 import BeautifulSoup
from selenium import webdriver
from transformers import BartForConditionalGeneration
from transformers import PreTrainedTokenizerFast

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
    def clean_sent(sent):
        special_symbol = re.compile(r"[\{\}\[\]\/?,;:|\)*~`!^\-_+<>@\#$&▲▶◆◀■【】\\\=\(\'\"]")
        content_pattern = re.compile(
            r"본문 내용|TV플레이어| 동영상 뉴스|flash 오류를 우회하기 위한 함수 추가function  flash removeCallback|tt|앵커 멘트|xa0")
        nl_symbol_removed_txt = sent.replace('\\n', '') \
            .replace('\\t', '') \
            .replace('\\r', '')
        special_symbol_removed_txt = re.sub(special_symbol, ' ', nl_symbol_removed_txt)
        end_phrase_removed_content = re.sub(content_pattern, '', special_symbol_removed_txt)
        blank_removed_content = re.sub(' +', ' ', end_phrase_removed_content)
        reversed_content = ''.join(reversed(blank_removed_content))

        content = ''

        for i in range(0, len(blank_removed_content)):
            # ".다"
            if reversed_content[i: i + 2] == '.다':
                content = ''.join(reversed(reversed_content[i:]))
                break

        return content

    def is_valid_str(sent):
        return sent is not None

    contents = ''

    driver.get(link)
    soup = BeautifulSoup(driver.page_source, 'html.parser')

    # div id articleBodyContents
    body_contents = soup.find('div', {'id': 'articleBodyContents'})

    if body_contents is None:
        return ''

    # body_contents 자식들을 childs
    childs = body_contents.children

    for c in childs:
        content_str = c.string
        if is_valid_str(content_str):
            contents += clean_sent(content_str)

    return contents


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
            link = n['link']

            if 'https://news.naver.com' not in link:
                continue

            contents = parse_contents(link)

            news.append({
                'link': link,
                'contents': contents,
                'title': remove_html(n['title']),
                'description': remove_html(n['description'])
            })

    return news


tokenizer = PreTrainedTokenizerFast.from_pretrained(
    'gogamza/kobart-summarization',
    bos_token='<s>', eos_token='</s>', unk_token='<unk>',
    pad_token='<pad>', mask_token='<mask>'
)

model = BartForConditionalGeneration.from_pretrained(
    'gogamza/kobart-summarization'
)


def summarize(sent):
    raw_input_ids = tokenizer.encode(sent)

    input_ids = [tokenizer.bos_token_id] + raw_input_ids + [tokenizer.eos_token_id]

    summary_ids = model.generate(
        torch.tensor([input_ids]),
        max_length=120,
        num_beams=5,
        repetition_penalty=1.0,
        eos_token_id=tokenizer.eos_token_id
    )

    summarized = tokenizer.batch_decode(summary_ids.tolist(),
                                        skip_special_tokens=True
                                        )[0]

    return summarized
