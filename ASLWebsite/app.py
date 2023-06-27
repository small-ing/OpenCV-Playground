from flask import Flask, render_template, request, redirect, url_for, flash, Response
import json
import os, sys
import cv2
sys.path.append("../")
from hand_tracking_module import handTracker
from data_translations import CNN

def get_base_url(port:int) -> str:
    '''
    Returns the base URL to the webserver if available.
    
    i.e. if the webserver is running on coding.ai-camp.org port 12345, then the base url is '/<your project id>/port/12345/'
    
    Inputs: port (int) - the port number of the webserver
    Outputs: base_url (str) - the base url to the webserver
    '''
    
    try:
        info = json.load(open(os.path.join(os.environ['HOME'], '.smc', 'info.json'), 'r'))
        project_id = info['project_id']
        base_url = f'/{project_id}/port/{port}/'
    except Exception as e:
        print(f'Server is probably running in production, so a base url does not apply: \n{e}')
        base_url = '/'
    return base_url

# to run
# cd ASLWebsite
# flask --app app run

# to exit flask app
# ctrl + c

port = 5000
base_url = get_base_url(port)

# Flask App
app = Flask(__name__)
# OpenCV Webcam
cap = cv2.VideoCapture(0)

# Home Page
@app.route(f"{base_url}")
def index():
    print("Loading Home Page...")
    return render_template("main.html")

<<<<<<< Updated upstream
@app.route(f"{base_url}/bio")
def bio():
    return render_template("bio.html")

@app.route(f"{base_url}/demo")
=======
# Introduction Video + How to Use
@app.route(f"{base_url}/intro/")
def intro():
    return render_template("intro.html")

# About Us
@app.route(f"{base_url}/bio/")
def bio():
    return render_template("bio.html")

# Demo Project
@app.route(f"{base_url}/demo/")
>>>>>>> Stashed changes
def demo():
    return render_template("demo.html")

@app.route(f"{base_url}/video_feed/")
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def gen_frames():
    '''
    Generates frames from camera
    '''
    tracker = handTracker(asl=True)
    while True:
        try:
            success, image = cap.read()
            if success:
                image = tracker.hands_finder(image)
                letter=tracker.estimate_letter()
                tracker.letter_display(image,letter=letter)
        
                try:
                    ret, buffer = cv2.imencode('.jpg', cv2.flip(image,1))
                    frame = buffer.tobytes()
                    yield (b'--frame\r\n'
                            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                except Exception as e:
                    pass
        except:
            print("Camera not found")
            break



    

if __name__ == "__main__":
    app.run(debug=True)