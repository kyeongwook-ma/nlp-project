from flask import Flask
from flask import request

from util import correct_sent

app = Flask(__name__)


@app.route('/', methods=['POST'])
def input():
    input = request.form['input']
    return correct_sent(input) + 'ddddd'


app.run(debug=True)
