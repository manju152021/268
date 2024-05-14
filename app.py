import os
import cv2
import numpy as np
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)

@app.route('/')

#upload_form() will load the upload.html page
def upload_form():
    return render_template('upload.html')


@app.route('/', methods=['POST'])
#This line decorates a function with the route '/' and specifies that it should only accept POST requests.
#@app.route('/') is a decorator that binds a URL to a function 
#so when a POST request is sent to the root URL ('/'), this function will be called.

def upload_image():
    operation_selection = request.form['image_type_selection']
    #request.form is a dictionary-like object that holds the data submitted with the POST request. 
    #it assigns the value of the 'image_type_selection' field to the variable operation_selection.

    image_file = request.files['file']
    #request.files is a dictionary-like object containing uploaded files. This line assigns the uploaded file to the variable image_file.

    filename = secure_filename(image_file.filename)
    #it extracts the filename from the uploaded file and sanitizes it to prevent any potentially malicious filenames. 
    #The secure_filename() function is provided by Flask to perform this sanitization.

    reading_file_data = image_file.read()
    #it reads the content of the uploaded file into a variable named reading_file_data.

    image_array = np.fromstring(reading_file_data, dtype='uint8')
    #it converts the raw file data (bytes) into a NumPy array of unsigned 8-bit integers (uint8).
    # The np.fromstring() function is used to create a NumPy array from the binary data.

    decode_array_to_img = cv2.imdecode(image_array, cv2.IMREAD_UNCHANGED)
    #The cv2.imdecode() function is used to decode an image from a NumPy array. 
    #The flag cv2.IMREAD_UNCHANGED specifies that the image should be loaded as is, including the alpha channel if it exists.



    if operation_selection == 'gray': #C266
        file_data = make_grayscale(decode_array_to_img)
       
    elif operation_selection == 'sketch': #C266
        file_data = image_sketch(decode_array_to_img)
        
    elif operation_selection == 'oil':  #C267
        file_data = oil_effect(decode_array_to_img)

    elif operation_selection == 'rgb': #C267
        file_data = rgb_effect(decode_array_to_img) 

    elif operation_selection == 'water':
        file_data = water_color_effect(decode_array_to_img) #C268

    elif operation_selection == 'invert':
        file_data = invert(decode_array_to_img) #C268

    elif operation_selection == 'hdr':
        file_data = HDR(decode_array_to_img) #268

    else:
        print('No image selected')


    with open(os.path.join('static/', filename),'wb') as f:
        #Opens a file in binary write mode ('wb'). The file is located in the 'static/' directory and is named filename. 
        #The with statement is used here to ensure that the file is properly closed after writing.
        f.write(file_data)
        # This line writes the contents of file_data (which contains the processed image data) to the file opened in the previous line.

    return render_template('upload.html', filename=filename)


#C266 STARTS
def make_grayscale(decode_array_to_img):

    converted_gray_img = cv2.cvtColor(decode_array_to_img, cv2.COLOR_RGB2GRAY)
    #converts the color image represented by decode_array_to_img to grayscale using OpenCV's cv2.cvtColor() function. 
    #It takes two arguments-the image array(decode_array_to_img) and the conversion code cv2.COLOR_RGB2GRAY
    status, output_image = cv2.imencode('.PNG', converted_gray_img)
    #encodes the grayscale image(converted_gray_img) into a PNG format using OpenCV's cv2.imencode() function.
    #It returns two values: status, which indicates whether the encoding was successful
    #and output_image, which contains the encoded image data

    return output_image



def image_sketch(decode_array_to_img):

    converted_gray_img = cv2.cvtColor(decode_array_to_img, cv2.COLOR_BGR2GRAY)
    sharping_gray_img = cv2.bitwise_not(converted_gray_img)
    #This line applies bitwise NOT operation to invert the grayscale image (converted_gray_img).
    #This can be used to enhance edges in the image, creating a sharpening effect.
    blur_img = cv2.GaussianBlur(sharping_gray_img, (111, 111), 30)
    #cv2.GaussianBlur() This function of opencv is used to apply blurriness on images.
    #(111, 111) - is the defined height and width value till where we need to apply a blur filter. 
    #Here we are defining this value as the height and width of the output image to display.
    #0 - is the angle value which means we need to apply the blur filter.
    #But here we need to apply the blur filter on the whole image equally that's why we had given the value as 0.
    sharping_blur_img = cv2.bitwise_not(blur_img)
    #cv2.bitwise_not() This function of opencv is used for sharpening the image.
    #This function takes 1 parameter which the image on which sharpening effects needs to be added.
    sketch_img = cv2.divide(converted_gray_img, sharping_blur_img, scale=256.0)
    #apply the cv2.divide() function of the opencv library and pass the converted_gray_img variable and
    #sharping_blur_img variable which will divide the image pixels and on the scale as 256 i.e range to
    #get the sketch image and store the outcome in the sketch_img variable.
    status, output_img = cv2.imencode('.PNG', sketch_img)
    # encodes the sketch image (sketch_img) into a PNG format using OpenCV's cv2.imencode() function. 
    #It returns two values: status, which indicates whether the encoding was successful
    # and output_img, which contains the encoded image data.
    return output_img

#C266 ENDS


#C267 STARTS
def oil_effect(decode_array_to_img):
    # Convert RGBA to RGB
    rgb_image = cv2.cvtColor(decode_array_to_img, cv2.COLOR_RGBA2RGB)

    oil_effect_img = cv2.xphoto.oilPainting(rgb_image, 7, 1)
    #cv2.xphoto.oilPainting-function of opencv library which is used to convert the uploaded image as an oil paint image. 
    #7- is the size of the pixel on the image we are applying into the Oil Paint effect
    #1- is the white balancing which keeps the color balance with the original by balancing the white tone in the image.
   
    status, output_img = cv2.imencode('.PNG', oil_effect_img)
    return output_img


def rgb_effect(decode_array_to_img):
    rgb_effect_img = cv2.cvtColor(decode_array_to_img, cv2.COLOR_BGR2RGB)
    status, output_img = cv2.imencode('.PNG', rgb_effect_img)

    return output_img
#C267 ENDS

#C268 STARTS
def water_color_effect(decode_array_to_img):
	water_effect = cv2.stylization(decode_array_to_img, sigma_s=60, sigma_r=0.6)
	status, output_img = cv2.imencode('.PNG', water_effect)
	return output_img

def HDR(decode_array_to_img):
    hdr_effect = cv2.detailEnhance(decode_array_to_img, sigma_s=12, sigma_r=0.15)
    status, output_img = cv2.imencode('.PNG', hdr_effect)
    return output_img

def invert(decode_array_to_img):
	invert_effect = cv2.bitwise_not(decode_array_to_img)
	status, output_img = cv2.imencode('.PNG', invert_effect)
	return output_img


#C268 ENDS


@app.route('/display/<filename>')
def display_image(filename):

    return redirect(url_for('static', filename=filename))



if __name__ == "__main__":
    app.run()















