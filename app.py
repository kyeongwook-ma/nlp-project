from flask import Flask
from flask import request
from flask_sqlalchemy import SQLAlchemy

from util.email_sender import send_email
from util.news import crawl_news, summarize

app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///news.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)


class News(db.Model):
    __tablename__ = 'news'

    id = db.Column('id', db.Integer, primary_key=True, autoincrement=True)
    title = db.Column(db.Text, nullable=False)
    link = db.Column(db.Text, nullable=False, unique=True)
    description = db.Column(db.Text, nullable=False)
    contents = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())
    updated_at = db.Column(db.DateTime, default=db.func.current_timestamp(),
                           onupdate=db.func.current_timestamp())

    def __init__(self, title, link, description, contents):
        self.title = title
        self.link = link
        self.contents = contents
        self.description = description


@app.route('/', methods=['POST'])
def crawl_news_save():
    query = request.form['query']

    news = crawl_news(query, start=1, offset=2)

    news_records = []

    for n in news:
        news_records.append(News(
            title=n['title'],
            contents=n['contents'],
            link=n['link'],
            description=n['description']
        ))

    db.session.add_all(news_records)
    db.session.commit()

    return 'ok'


@app.route('/subscribe', methods=['GET'])
def subscribe():
    q = request.args.get('query')

    news = crawl_news(q, start=1, offset=2)

    top5_latest_news = news[:5]

    for n in top5_latest_news:
        contents = n['contents']
        summarized_contents = summarize(contents)

        send_email(
            subject=n['title'],
            from_email='kyeongwook.ma@gmail.com',
            to_email='kyeongwook.ma@gmail.com',
            basic_text=summarized_contents
        )




if __name__ == '__main__':
    db.create_all()
    app.run(debug=True)
