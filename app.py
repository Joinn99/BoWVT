from datetime import timedelta
import os
import argparse
import time
from flask import render_template, redirect, url_for, request
from flask import Flask
import bowvt
import utils

app = Flask(__name__)
app.send_file_max_age_default = timedelta(seconds=1)


@app.route("/")
def index():
    # pkl-list get
    pkl_list = sorted([pkl for pkl in os.listdir(
        'model') if pkl.endswith('.pkl')])
    pkl_select = ['Depth {:s} Branch {:s}'.format(a.replace('.pkl', '').split(
        '-')[2], a.replace('.pkl', '').split('-')[3]) for a in pkl_list]
    return render_template('search.html', bg_path=BG_PATH, logo_path=LOGO_PATH, pkl_list=pkl_select, tar_path=utils.random_image('static/data'))


@app.route("/result/", methods=['GET', 'POST'])
def result():
    time.sleep(1)
    select = request.form.get('pkl-select')
    target = request.form.get('target-img')
    VT.load('model/bow-0.50-' + select.replace('Depth ',
                                               '').replace(' Branch ', '-') + '.pkl')
    result_list = VT.search('static/' + target)
    return render_template('result.html', bg_path=BG_PATH, logo_path=LOGO_PATH, tar_path=target, res_list=result_list)


@app.route("/random/", methods=['GET'])
def random():
    return redirect(url_for('index'))


@app.route("/update/", methods=['GET', 'POST'])
def update():
    target = request.form.get('target-img')
    check_result = request.form.copy().to_dict()
    del check_result['target-img']
    result_list = VT.optimize('static/' + target, check_result)
    return render_template('result.html', bg_path=BG_PATH, logo_path=LOGO_PATH, tar_path=target, res_list=result_list)


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument("-d", "--max_depth",
                        help="Max depth of vocab tree", default=4, type=int)
    PARSER.add_argument(
        "-b", "--branch", help="Branches of each node", default=12, type=int)
    PARSER.add_argument(
        "-s", "--scale", help="Scale image", default=0.5, type=float)
    PARSER.add_argument(
        "-p", "--pre_allocate", help="Pre-Allocate storage for features", default=True, type=bool)

    ARGS = PARSER.parse_args()

    BG_PATH = 'res/background.jpg'
    LOGO_PATH = 'res/Logo.png'
    VT = bowvt.VocabularyTree(ARGS)
    app.run()
