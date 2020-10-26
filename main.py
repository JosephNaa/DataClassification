# -*- coding: utf-8 -*-

import argparse
import json
from crawler import InstaCrawler
from download import ClassifyGender


def get_posts(tag, number):
    insta_crawler = InstaCrawler()
    return insta_crawler.get_posts_tag(tag, number)


def output(data, filepath):
    out = json.dumps(data, ensure_ascii=False)
    if filepath:
        with open(filepath, "w", encoding="utf8") as f:
            f.write(out)
    else:
        print(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-n", "--number", type=int)
    parser.add_argument("-t", "--tag")
    parser.add_argument("-o", "--output")

    args = parser.parse_args()

    output(get_posts(args.tag, args.number or 10000), args.output)
"""
    with open('C:/Users/goQba-D01/PycharmProjects/DataClassification/' + args.tag + '.json', 'rt', encoding='UTF-8') as data_file:
        data = json.load(data_file)

    for i in range(0, len(data)):
        instagramURL = data[i]['img_url']
        #print(instagramURL)
        ClassifyGender(instagramURL, args.tag, i)

    print(args.tag + ' done')
"""


##############################################
# python crawler.py hashtag -t selfie -o ./output.json -n 10000
##############################################