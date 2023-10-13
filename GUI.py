import tkinter as tk
from tkinter import *
from tkinter import ttk
from tkinter import messagebox
import numpy as np
import cv2 as cv2
import numpy
from PIL import Image
from PIL import ImageTk
import os
from tkinter import filedialog
from tkinter.font import Font
import tensorflow as tf
import tensorflow_hub as hub


class Program:
    def __init__(self, app):

        self.text_ = 'Welcome to our Brain Tumor Segmentation Tool. We are dedicated to transforming brain tumor ' \
                     'diagnosis. Our cutting-edge technology and deep learning algorithms provide accurate and rapid ' \
                     'segmentation of brain tumors from MRI scans. Join us in the quest for early detection and ' \
                     'effective treatment. SCAN YOUR MRI REPORTS NOW.'

        self.filepath = None
        self.last_captured_frame = np.array([])

        self.app = app
        self.app.geometry("800x500")
        self.app.title("BRAIN TUMOR SEGMENTATION")
        self.app.config(bg="#FFFFFF")
        self.app.resizable(False, False)

        self.custom_font = Font(family="Helvetica", size=34, weight="bold")
        self.custom_font_2 = Font(family="Helvetica", size=22, weight="bold")
        self.custom_font3 = Font(family="Helvetica", size=8, weight="bold")

        # FRAMES -------------------------------------------------------
        self.window = Frame(self.app)
        self.window.grid(row=0, column=0)

        self.window_upload = Frame(self.app)
        # self.window_upload.grid(row=0, column=0)

        self.window_scan = Frame(self.app)
        # self.window_scan.grid(row=0, column=0)

#       ####################################################################------------------------------------------
#       ##########          -----PAGE 1----------     ######################------------------------------------------
#       ####################################################################------------------------------------------

        # LABEL ---> THE BLACK LABEL -----------------------------------
        self.label = Label(self.window, bg='#141414', width=115, height=15)
        self.label.grid(row=0, column=0, columnspan=2)
        self.label_title = Label(self.label, text='Brain Tumor Segmentation Tool', fg='white', width=100,
                                 height=5, bg='#141414', anchor='w', font=self.custom_font, justify='left')
        self.label_title.place(x=10, y=10)

        # LABEL ---> CONTENT AND BUTTON LABEL -----------------------------------
        self.content_label = Label(self.window, bg='#FFFFFF', width=115, height=18, anchor='center', justify='center')
        self.content_label_1 = Label(self.content_label, text='BRAIN TUMOR',font=self.custom_font_2, fg='#141414',
                                    bg='#FFFFFF')
        self.content_label.grid(row=1,column=0)
        self.content_label_1.place(x=280, y=0)

        self.about_label = Label(self.content_label,  text=self.text_, wraplength=600, width=70,font=('Helvetica', 12),
                                 height=5, justify="left", anchor="center", bg='#FFFFFF')
        self.about_label.place(x=85, y=40)

        self.buttons_label = Label(self.content_label, width=75,height=3, bg='#FFFFFF')
        self.buttons_label.place(x=160, y=155)

        self.UPLOAD_BUTTON = Button(self.buttons_label, text="UPLOAD MRI/ X-RAYS", width=30, height=2, fg='white',
                                    bg='#4ca3d9', borderwidth=0,
                                    highlightbackground=self.buttons_label.cget('background'), font=self.custom_font3,
                                    command=self.show_upload_page)
        self.SCAN_BUTTON = Button(self.buttons_label, text="SCAN MRI/ X-RAYS", width=30, height=2, fg='white',
                                  bg='#4ca3d9', borderwidth=0,
                                  highlightbackground=self.buttons_label.cget('background'), font=self.custom_font3,
                                  command=self.show_scan_page)

        self.UPLOAD_BUTTON.grid(row=0, column=0,  padx=(0, 10))
        self.SCAN_BUTTON.grid(row=0, column=1,  padx=(10, 0))

#       ####################################################################------------------------------------------
#       ##########          -----PAGE upload----------     ##################------------------------------------------
#       ####################################################################------------------------------------------

        self.up_txt = "Please Upload Your MRI/CT scan (x-ray image) copy. The copy will be passed to the ML model for"\
                      " prediction. Early Treatment and Diagnosis is necessary for Tumors"

        self.go_back_upload = Label(self.window_upload, width=115, height=2, bg='#141414')
        self.go_Back_upload_button = Button(self.go_back_upload, text='Go back', fg='white', bg='#141414',
                                            borderwidth=0, highlightbackground=self.buttons_label.cget('background'),
                                            justify='left', width=10, height=2, font=("Helvetica", 8),
                                            command=self.go_back_upload_)
        self.upload_label = Label(self.window_upload, width=115, height=31)
        self.upload_title = Label(self.upload_label, text='UPLOAD YOUR MRI SCAN/ X-RAY IMAGE',
                                  font= Font(family="Helvetica", size=22, weight="bold"))
        self.upload_msg = Label(self.upload_label, text=self.up_txt, wraplength=500, width=120,font=('Helvetica', 8),
                                 height=4, justify="left", anchor="center")
        self.upload_label_button = Button(self.upload_label, text='UPLOAD AND PREDICT', width=30, height=2, fg='white',
                                    bg='#4ca3d9', borderwidth=0,
                                    highlightbackground=self.buttons_label.cget('background'), font=self.custom_font3,
                                    command = self.upload_image)
        self.output_upload = Label(self.upload_label, width=50, height=3)

        self.go_back_upload.pack()
        self.go_Back_upload_button.place(x=0, y= 0)
        self.upload_label.pack()
        self.upload_title.place(x=120, y=100)
        self.upload_msg.place(x=45, y=130)
        self.upload_label_button.place(x=280, y=200)
        self.output_upload.place(x=214, y=260)

#       ####################################################################------------------------------------------
#       ##########          -----PAGE scann----------     ##################------------------------------------------
#       ####################################################################------------------------------------------
        self.go_back_scan = Label(self.window_scan, width=115, height=2, bg='#141414')
        self.go_Back_scan_button = Button(self.go_back_scan, text='Go back', fg='white', bg='#141414',
                                            borderwidth=0, highlightbackground=self.buttons_label.cget('background'),
                                            justify='left', width=10, height=2, font=("Helvetica", 8),
                                            command=self.go_back_scan_)

        self.scan_label = Label(self.window_scan, width=115, height=31, bg='#FFFFFF')
        self.blank_label = Label(self.scan_label, width=115, bg='#FFFFFF')
        self.scan_image_label = LabelFrame(self.scan_label, width=10, height=60, bg='#FFFFFF')
        self.image_scan = Label(self.scan_image_label, width=10,height=60, bg='#FFFFFF')
        self.scan_label_button = Button(self.scan_label, text='CLICK AND PREDICT', width=30, height=2, fg='white',
                                          bg='#4ca3d9', borderwidth=0,
                                          highlightbackground=self.buttons_label.cget('background'),
                                          font=self.custom_font3,
                                          command=self.scan_predict_image)
        self.scan_output = Label(self.scan_label, width=50, height=3)

        self.go_back_scan.pack()
        self.go_Back_scan_button.place(x=0, y=0)
        self.scan_label.pack(fill="both", expand=True)
        #self.blank_label.pack()
        self.scan_image_label.pack(fill="both", expand=True)
        self.image_scan.pack(fill="both", expand=True)
        self.scan_label_button.pack()
        self.scan_output.pack()


        self.camera_started = False
        self.start_webcam_feed()

    def start_webcam_feed(self):
        if self.camera_started:
            return

        cap = cv2.VideoCapture(1)

        if not cap.isOpened():
            self.image_scan.config(text='Connect Camera', font=('Helvetica', 16))
            return

        self.camera_started = True
        while self.camera_started:  # Continuously update the label with webcam frames
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, (400, 260))
                # Rotate the frame 90 degrees counter-clockwise
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame)
                # Rotate the image by 180 degrees
                image = image.rotate(180)
                # image = image.rotate(90)

                photo = ImageTk.PhotoImage(image=image)

                self.image_scan.config(image=photo, text="", width=10, height=300)  # Clear the text
                self.image_scan.image = photo
                self.window.update()
            self.image_scan.pack(fill="both", expand=True)

    def scan_predict_image(self):
        self.cap = cv2.VideoCapture(1)
        print("CHECK 1")
        if self.camera_started:
            self.camera_started = False
        if self.cap is not None:
            ret, frame = self.cap.read()
            if ret:
                self.last_captured_frame = frame
                self.scan_image_label.config(text="IMAGE CAPTURED")

                if os.path.exists("file.jpg"):
                    os.remove("file.jpg")


                cv2.imwrite("file.jpg", frame)

                img = tf.io.read_file("file.jpg")
                img = tf.image.decode_image(img)
                img = tf.image.resize(img, size=[224, 224])
                img = img / 255.
                img = tf.expand_dims(img, axis=0)

                # PREDICT
                model = tf.keras.models.load_model(r"C:\Users\Anukriti\PycharmProjects\GUI\efficient_net_model.h5",
                                                   custom_objects={'KerasLayer': hub.KerasLayer})
                classes = ["GLIOMA", "MENINGIOMA", "NO-TUMOR", "PITUITARY"]
                preprocessed_image = img
                predictions = model.predict(preprocessed_image)
                pred_class = classes[np.argmax(predictions) - 1]
                self.scan_output.config(text=f"{pred_class}")

            else:
                print("FAILED TO CAPTURE")
                messagebox.showerror(title="ERROR", message="FAILED TO CAPTURE")
            cv2.destroyAllWindows()

    def upload_image(self):
        try:
            self.output_upload.config(text="Predicting...")
            self.filepath = filedialog.askopenfilename(filetypes=[("Image Files", "*.png *.jpg *.jpeg *.gif *.bmp")])
            img = tf.io.read_file(self.filepath)
            img = tf.image.decode_image(img)
            img = tf.image.resize(img, size=[224, 224])
            img = img/ 255.
            img = tf.expand_dims(img, axis=0)

            # PREDICT
            model = tf.keras.models.load_model(r"C:\Users\Anukriti\PycharmProjects\GUI\efficient_net_model.h5",
                                               custom_objects={'KerasLayer': hub.KerasLayer})
            classes = ["GLIOMA", "MENINGIOMA", "NO-TUMOR", "PITUITARY"]
            preprocessed_image = img
            predictions = model.predict(preprocessed_image)
            pred_class = classes[np.argmax(predictions) - 1]

            self.output_upload.config(text=f"{pred_class}")

        except Exception as e:
            # messagebox.showerror(title="ERROR", message="SOME ERROR OCCURED")
            print(e)

    def show_upload_page(self):
        self.window.grid_forget()
        self.window_upload.grid(row=0, column=0)

    def go_back_upload_(self):
        self.window_upload.grid_forget()
        self.window.grid(row=0, column=0)

    def show_scan_page(self):
        self.window.grid_forget()
        self.window_scan.grid(row=0, column=0)

    def go_back_scan_(self):
        self.window_scan.grid_forget()
        self.window.grid(row=0, column=0)

def main():
    window = Tk()
    app = Program(window)
    window.mainloop()


if __name__ == "__main__":
    main()


