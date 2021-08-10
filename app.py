from flask import Flask
from flask import request
from flask_sqlalchemy import SQLAlchemy

from util.news import crawl_news

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
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())
    updated_at = db.Column(db.DateTime, default=db.func.current_timestamp(),
                           onupdate=db.func.current_timestamp())

    def __init__(self, title, link, description):
        self.title = title
        self.link = link
        self.description = description


@app.route('/', methods=['POST'])
def crawl_news_save():
    query = request.form['query']

    news = crawl_news(query, start=1, offset=2)

    news_records = []

    for n in news:
        news_records.append(News(
            title=n['title'],
            link=n['link'],
            description=n['description']
        ))

    db.session.add_all(news_records)
    db.session.commit()

    return 'ok'


if __name__ == '__main__':
    db.create_all()
    app.run(debug=True)
