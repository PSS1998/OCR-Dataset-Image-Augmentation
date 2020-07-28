import os
from urllib.parse import quote

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
# from selenium.webdriver.firefox.options import Options
from PIL import Image, ImageFilter
import cv2
import numpy as np
from io import BytesIO
from base64 import b64decode
import Augmentor
import skimage
import arabic_reshaper
from bidi.algorithm import get_display
import unidecode




def noise(noise_typ,image):
    if noise_typ == "gauss":
        row,col,ch= image.shape
        mean = 0
        var = 0.1
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        noisy = image + gauss
        return noisy
    elif noise_typ == "s&p":
        row,col,ch = image.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                for i in image.shape]
        out[tuple(coords)] = 1
        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                for i in image.shape]
        out[tuple(coords)] = 0
        return out
    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_typ =="speckle":
        row,col,ch = image.shape
        gauss = np.random.randn(row,col,ch)
        gauss = gauss.reshape(row,col,ch)        
        noisy = image + image * gauss
        return noisy

def noise_image(image):
    # noise
    pix = np.array(image)
    pix = noise('s&p', pix)
    image = Image.fromarray(np.uint8(pix))
    return image	

def blur_image(image):
    # blur
    image = image.filter(ImageFilter.BoxBlur(0.25))
    return image

def black_and_white(image):
    # convert to black and white
    image = image.convert('1')
    return image

def black_and_white_adaptive_threshold(image):
    # convert to black and white
    image = np.array(image) 
    image = image[:, :, ::-1].copy() 
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    background = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
    background = Image.fromarray(background)
    return background

def Augmentor_filter():
    # perspective , black and white and distortion using Augmentor library
    p = Augmentor.Pipeline('resources/sample-images/')
    p.random_distortion(probability=1, grid_width=16, grid_height=16, magnitude=1)
    p.black_and_white(probability=1)
    p.skew_corner(probability=1, magnitude=0.5)
    p.process()

def convert_to_display_unicode_characters(input_words):
    for i in range(len(input_words)):
        reshaped_text = get_display(arabic_reshaper.reshape(input_words[i]))
        input_words[i] = [input_words[i], reshaped_text]
    return input_words




svg_height = 150
svg_width = 6500

# SVG path builder
# https://mavo.io/demos/svgpath/
# https://codepen.io/anthonydugois/pen/mewdyZ
# defaults
# path='M 50 100 q 250 -100 500 0'
# path_start = 0
# range
# rotationX_degree=0-25, rotationY_degree=0-25, skew_degree=0-25, perspective_distance=200-500

svg_template = '''
<div style="perspective: {8}px;"><svg style="transform: skew({7}deg) rotateY({6}deg) rotateX({5}deg);" version="1.1" baseProfile="full" height="100%" width="100%"
                    xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
                        <defs> 
                            <path id="curve" fill="none" d="{9}" transform="rotate({10} 0 0)" />
                        </defs> 
                        <g style="font-size:50;
                            font-family: '{0}', '{1}';
                            font-weight: {2};
                            font-style : {3};
                            font-feature-settings: 'cswh';
                            fill       : 'black'">
                            <text x="10" y="90" xml:space="preserve" transform="rotate({4})" text-decoration="{11}">
                                <textPath xlink:href="#curve">{12}</textPath>
                            </text>
                        </g>
</svg></div>
'''




def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def create_text_image(driver, text, primary_font_family, secondary_font_family, font_weight='normal',
                      font_style='normal', rotation_degree=0., rotationX_degree=0., rotationY_degree=0., skew_degree=0., perspective_distance=0., text_decoration='none', path='M 50 100 q 250 -100 500 0', path_start=0):
    svg = svg_template.format(primary_font_family, secondary_font_family,
                              font_weight, font_style, rotation_degree, rotationX_degree, rotationY_degree, skew_degree, perspective_distance, path, path_start, text_decoration, text)

    driver.get("data:text/html;charset=utf-8, "+ quote(svg))

    image = BytesIO(b64decode(driver.get_screenshot_as_base64()))
    image = Image.open(image)
    # noise
    image = noise_image(image)
    # blur
    image = blur_image(image)
    background = Image.new('L', image.size, 255)
    background.paste(image, image)
    # convert to black and white
    background = black_and_white(background)
    # background = black_and_white_adaptive_threshold(image)
    return background


if __name__ == '__main__':
    create_directory('resources')
    create_directory('resources/sample-images')
    # chrome
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--window-size=%dx%d' % (svg_width - 6000, svg_height))
    chrome_driver = webdriver.Chrome(chrome_options=chrome_options, executable_path='resources/chromedriver')
    # firefox
    # chrome_driver = webdriver.Firefox(firefox_options=chrome_options, executable_path='resources/geckodriver')
    # chrome_driver.set_window_size(svg_width - 6000,svg_height+200)

    input_words = ['خانه من سرای من', 'خانهٔ', 'سَلامت', 'دانِسته', '   abolfazl     mahdizade   ', 'aaa', 'آمریکا']

    input_words = convert_to_display_unicode_characters(input_words)

    try:
        for word in input_words:
            font = 'times new roman'
            word_image = create_text_image(chrome_driver, word[1], primary_font_family=font,
                                           secondary_font_family='bbcnassim', font_weight='normal', font_style='normal',
                                           rotation_degree=0., rotationX_degree=0, rotationY_degree=0, skew_degree=0, perspective_distance=200, text_decoration='none', path='M 50 100 q 250 -100 500 0', path_start=0)
            word_image.save('resources/sample-images/' + word[0] + '.jpeg')
            word_image = create_text_image(chrome_driver, word[1], primary_font_family=font,
                                           secondary_font_family='bbcnassim', font_weight='bold', font_style='normal',
                                           rotation_degree=.0, rotationX_degree=0., rotationY_degree=0., skew_degree=0., perspective_distance=0., text_decoration='none', path='M 50 100 q 250 -100 500 0', path_start=0)
            word_image.save('resources/sample-images/' + word[0] + '_bold.jpeg')
            word_image = create_text_image(chrome_driver, word[1], primary_font_family=font,
                                           secondary_font_family='bbcnassim', font_weight='normal', font_style='italic',
                                           rotation_degree=.0, rotationX_degree=0., rotationY_degree=0., skew_degree=0., perspective_distance=0., text_decoration='none', path='M 50 100 q 250 -100 500 0', path_start=0)
            word_image.save('resources/sample-images/' + word[0] + '_italic.jpeg')

        # # perspective , black and white and distortion using Augmentor library
        # Augmentor_filter()
        
    finally:
        chrome_driver.quit()

