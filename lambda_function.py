import sys
from ChartExtractor.extraction.extraction import digitize_sheet
from PIL import Image


def handler(event, context):
    intraop_im = Image.open("data/RC_0001_intraoperative.JPG")
    preop_postop_im = Image.open("data/RC_0001_preoperative_postoperative.JPG")

    return digitize_sheet(intraop_im, preop_postop_im)
