# !/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
"""
import logging
import re

import grequests
import requests

import util.terminal

__author__ = "freemso"

VALID_CATEGORY = [
    "组织机构",
    "学校",
    "行政区划",
    "地点",
    "中国地名",
    "公司",
    "大学",
    "教育机构",
    "科研机构",
]

REGEX_VALID_KEY = re.compile(
    "[成创][立办始](?:时间|人|于)|"
    "地址|始建于|位于|地理位置|占地面积|地点|地区|"
    "(?:学校|机构|公司|组织)(?:名称|属性|占地|地址|类型|简介|机构|隶属)|"
    "内设(?:机构|部门|组织)"
)

REGEX_VALID_ENT = re.compile(
    ".+(?:公司|学校|局|中学|所|会|馆|中心|院|站|部|队|"
    "城|会议|法|条例|学院|系|专业|集团|酒店|军|系统|展|"
    "战争|公园|节|小学|园|区|镇|乡|县|市|省|村|网|大学|"
    "银行)$"
)


if __name__ == '__main__':
    util.terminal.setup_logging_config()

    valid_list = []
    exclude_list = []

    lines = [line.strip().split("\t") for line in open("data/abb2ent.txt")]

    logging.info("Total {} samples.".format(len(lines)))

    logging.info("Filter by matching entity...")
    for line in lines:
        if REGEX_VALID_ENT.match(line[1]):
            valid_list.append(line)
        else:
            exclude_list.append(line)

    logging.info("Include: {}, exclude: {}".format(
        len(valid_list), len(exclude_list)
    ))

    # lines = exclude_list
    # exclude_list = []
    #
    # logging.info("Filter by info box...")
    #
    # iter_times = 5
    # old_len = len(lines)
    # while len(lines) > 0 and iter_times > 0:
    #     bar = util.terminal.ProgressBar(total=len(lines))
    #     err_list = []
    #     iter_times -= 1
    #     max_conn_size = 256
    #     for x in range(0, len(lines), max_conn_size):
    #         request_list = (grequests.get("http://knowledgeworks.cn:20313/cndbpedia/api/entityAVP",
    #                                       params={"entity": ent})
    #                         for abb, ent in lines[x:x + max_conn_size])
    #         response_list = grequests.map(request_list)
    #         for idx, response in enumerate(response_list):
    #             bar.log()
    #             if response:
    #                 if response.status_code == requests.codes.ok:
    #                     try:
    #                         entity_json = response.json()
    #                         include = False
    #
    #                         key_list = []
    #                         for key, value in entity_json["av pair"]:
    #                             key_list.append(key)
    #                             if REGEX_VALID_KEY.match(key):
    #                                 include = True
    #                         # category_list = []
    #                         # for attribute in entity_json["av pair"]:
    #                         #     if attribute[0] == "CATEGORY_ZH":
    #                         #         category_list.append(attribute[1])
    #                         #         if attribute[1] in VALID_CATEGORY:
    #                         #             include = True
    #                         if include:
    #                             valid_list.append((lines[x + idx]))
    #                         else:
    #                             exclude_list.append((lines[x + idx], key_list))
    #                     except Exception:
    #                         logging.info("Error! Entity: {}".format(lines[x+idx][1]))
    #                 else:
    #                     err_list.append(lines[x+idx])
    #                 response.close()
    #             else:
    #                 err_list.append(lines[x+idx])
    #     if len(err_list) > 0:
    #         logging.info("Still {} lines not process. Go another round.".format(len(err_list)))
    #
    #     lines = err_list
    #
    #     if len(lines) == old_len:
    #         break
    #     else:
    #         old_len = len(lines)
    #
    # logging.info("Include: {}, exclude: {}, error: {}, total: {}".format(
    #     len(valid_list), len(exclude_list), len(lines),
    #     len(valid_list) + len(exclude_list) + len(lines),
    # ))

    with open("data/abb2ent_valid.txt", "w") as out_file:
        for line in valid_list:
            out_file.write(line[0]+"\t"+line[1]+"\n")

    with open("data/abb2ent_exclude.txt", "w") as out_file:
        for line in exclude_list:
            out_file.write(line[0]+"\t"+line[1]+"\n")

