from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import config

import ast
import os
import sys
import random
import numpy as np
import tensorflow as tf
import scipy
from tqdm import tqdm
from PIL import Image, ImageTk
import cv2
import tkinter as tk
import imageio
import smtplib

from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart

from magenta.models.image_stylization import image_utils
from magenta.models.image_stylization import model

root = tk.Tk()
root.title('Stylizer')
root.config(background='white')

stream_frame = tk.Frame(root, width=300, height=533)
stream_frame.grid(row=0, column=0, columnspan=2)
# The place where images should be placed
image_main = tk.Label(stream_frame)
image_main.grid(row=0, column=0)

capture_button = tk.Button(root, text='Take picture')
capture_button.config(width=50, height=20)
capture_button.grid(row=1, column=0)

send_email_button = tk.Button(root, text='Send Email with Pics', state='disabled')
send_email_button.config(width=50, height=20)
send_email_button.grid(row=1, column=1)

email_entry = tk.Entry(root)
email_entry.grid(row=2, column=1)

images = {}

def initialize_image():
    input_image = './images/me.jpg'
    image = np.expand_dims(image_utils.load_np_image(
            os.path.expanduser(input_image)), 0)

    print(image.shape)
    print(type(image[0,0,0,0]))
    images['latest'] = image

initialize_image()

# checkpoint = './multistyle-pastiche-generator-monet.ckpt'
checkpoint = './multistyle-pastiche-generator-varied.ckpt'

if checkpoint == './multistyle-pastiche-generator-monet.ckpt':
    num_styles = 10
else:
    num_styles = 32

k = 32
styles = list(range(num_styles))
# random.shuffle(styles)
# which_styles = styles[0:4]
which_styles = random.sample(styles, k=k)
num_rendered = len(which_styles)
    
cap = cv2.VideoCapture(0)
def show_frame():
    _, frame = cap.read()
    # Flip so that movements are mirrored
    frame = cv2.flip(frame, 1)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    frame = scipy.misc.imresize(frame, (300, 533), interp='nearest', mode=None)
    new_img = np.delete(np.expand_dims(frame, 0), 3, axis=3)
    # print(new_img.shape)
    images['latest'] = new_img
    img = Image.fromarray(frame)
    # print('frame shape', frame.shape)
    imgtk = ImageTk.PhotoImage(image=img)

    image_main.imgtk = imgtk
    image_main.configure(image=imgtk)
    image_main.after(10, show_frame) 

def generate_images():
    image = images['latest']
    image = image.astype('float32')
    print(type(image[0,0,0,0]))
    with tf.Graph().as_default(), tf.Session() as sess:
        print('Started transforming')
        stylized_images = model.transform(
            tf.concat([image for _ in range(num_rendered)], 0),
            normalizer_params={
                'labels': tf.constant(which_styles),
                'num_categories': num_styles,
                'center': True,
                'scale': True})
        print('Finished transforming')
        model_saver = tf.train.Saver(tf.global_variables())
        print('Saver created')
        model_saver.restore(sess, checkpoint)
        for var in tqdm(tf.global_variables()):
            w = var.eval()
            w = np.nan_to_num(w)
            var.assign(w).eval()
        print('Model restored')
        stylized_images = stylized_images.eval()
        print('Images evaluated')

    for count, s_im in enumerate(stylized_images):
        # Got this warning:
        # Lossy conversion from float32 to uint8. Range [0, 1]. Convert image to uint8 prior to saving to suppress this warning.
        # So we resize the image
        converted_image = scipy.misc.imresize(s_im, (300, 533), interp='nearest', mode=None)
        display_image = scipy.misc.imresize(s_im, (133, 200), interp='nearest', mode=None)
        print('display iamge shape', display_image.shape)
        img = Image.fromarray(display_image)
        display_image = ImageTk.PhotoImage(image=img)
        print(converted_image.shape)
        images[count] = converted_image
        images[count + 10] = display_image
        name = f'./stylized_images/image-test7-{count}.jpg'
        imageio.imwrite(name, converted_image)

def update_images():
    generate_images()
    # style_1 = tk.Label(root)
    print('art image shape', images[11].width(), images[11].height())
    img_1 = images[10]
    style_1.imgtk = img_1
    style_1.config(image=img_1)

    img_2 = images[11]
    style_2.imgtk = img_2
    style_2.config(image=img_2)
    
    img_3 = images[12]
    style_3.imgtk = img_3
    style_3.config(image=img_3)

    img_4 = images[13]
    style_4.imgtk = img_4
    style_4.config(image=img_4)

    img_5 = images[14]
    style_5.imgtk = img_5
    style_5.config(image=img_5)

    img_6 = images[15]
    style_6.imgtk = img_6
    style_6.config(image=img_6)

    send_email_button.config(state='normal')

style_1 = tk.Label(root)
style_1.grid(row=0, column=2)
style_2 = tk.Label(root)
style_2.grid(row=0, column=3)
style_3 = tk.Label(root)
style_3.grid(row=1, column=2)
style_4 = tk.Label(root)
style_4.grid(row=1, column=3)
style_5 = tk.Label(root)
style_5.grid(row=0, column=4)
style_6 = tk.Label(root)
style_6.grid(row=1, column=4)

checkbutton_states = [0] * 6

one = tk.IntVar()
two = tk.IntVar()
three = tk.IntVar()
four = tk.IntVar()
five = tk.IntVar()
six = tk.IntVar()

text = 1
c1 = tk.Checkbutton(root, text=text, variable=one)
c1.grid(row=2, column=text+1)

text += 1
c2 = tk.Checkbutton(root, text=text, variable=two)
c2.grid(row=2, column=text+1)

text += 1
c3 = tk.Checkbutton(root, text=text, variable=three)
c3.grid(row=2, column=text+1)

text += 1
c4 = tk.Checkbutton(root, text=text, variable=four)
c4.grid(row=2, column=text+1)

text += 1
c5 = tk.Checkbutton(root, text=text, variable=five)
c5.grid(row=2, column=text+1)

text += 1
c6 = tk.Checkbutton(root, text=text, variable=six)
c6.grid(row=2, column=text+1)

def send_email():
    email = email_entry.get()

    msg = MIMEMultipart()
    msg['Subject'] = 'Your artistic renderings'
    msg['From'] = 'yashloke@terpmail.umd.edu'
    msg['To'] = email

    

    if one.get() == 1:
        img_data = open('./stylized_images/image-test7-0.jpg', 'rb').read()
        image = MIMEImage(img_data)
        msg.attach(image)
    
    if two.get() == 1:
        img_data = open('./stylized_images/image-test7-1.jpg', 'rb').read()
        image = MIMEImage(img_data)
        msg.attach(image)

    if three.get() == 1:
        img_data = open('./stylized_images/image-test7-2.jpg', 'rb').read()
        image = MIMEImage(img_data)
        msg.attach(image)

    if four.get() == 1:
        img_data = open('./stylized_images/image-test7-3.jpg', 'rb').read()
        image = MIMEImage(img_data)
        msg.attach(image)

    if five.get() == 1:
        img_data = open('./stylized_images/image-test7-4.jpg', 'rb').read()
        image = MIMEImage(img_data)
        msg.attach(image)

    if six.get() == 1:
        img_data = open('./stylized_images/image-test7-5.jpg', 'rb').read()
        image = MIMEImage(img_data)
        msg.attach(image)

    smtp_mailer = smtplib.SMTP('smtp.gmail.com', 587)
    smtp_mailer.ehlo()
    smtp_mailer.starttls()
    smtp_mailer.ehlo()
    smtp_mailer.login(config.FROM_EMAIL, config.FROM_PASS)
    smtp_mailer.sendmail(config.FROM_EMAIL, email, msg.as_string())
    smtp_mailer.quit()
    print('Mail sent')

send_email_button.config(command=send_email)

capture_button.config(command=update_images)

show_frame()
root.mainloop()