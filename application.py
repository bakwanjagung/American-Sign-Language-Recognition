# -*- coding: utf-8 -*-
"""
Aplication

@author: Mega Pertiwi
"""

import tkinter as tk
import cv2
from PIL import Image, ImageTk
import statistics
from tkinter.filedialog import askopenfilename
from processing import load_model, preprocessing, recognition
import time

root = tk.Tk()
root.title("Aplikasi penerjemah alfabet ASL")
root.iconbitmap('logo.ico')

def main():
    
    def load_process(filepath):
        loaded_model = load_model()
        cap = cv2.VideoCapture(filepath)
        getChar = lambda x: "ABCDEFGHIKLMNOPQRSTUVWXY"[x]
        mulai = time.time()
        while True:
            frame, img = cap.read()
            result_t = []
            while(frame):
                
                img_pre = preprocessing(img)
                
                predhur = recognition(img_pre, loaded_model, getChar)
    
                cv2.putText(img, predhur, (10,300), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2,(0,255,255),2)
                cv2.imshow("grayscaling", img_pre)
                cv2.imshow("prediction", img)
                
                #print("framevideo")
                #print(predhur)
                result_t += [predhur] 
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                frame, img = cap.read()
             
            break
        
        cap.release()
        #print("video selesai")
        #print(statistics.mode(result_t))
        text2_2.configure(text=statistics.mode(result_t)) 
        cv2.destroyAllWindows()
        akhir = time.time()
        print ("Total Waktu Proses ",akhir-mulai," Detik.")
        show_frame2()
        
    def show_frame2():
        frame1.pack_forget()
        frame2.pack()
        
        canvas = tk.Canvas(frame2, width=400, height=250, bg="white")
        canvas.config(highlightthickness=0)
        canvas.grid(columnspan=3)
        
        canvas_2 = tk.Canvas(frame_2, width=400, height=50, bg="white")
        canvas_2.config(highlightthickness=0)
        
        logo = Image.open('logo.png').resize((200, 200), Image.ANTIALIAS)
        logo = ImageTk.PhotoImage(logo)
        logo_label = tk.Label(frame2, image=logo, bg="white")
        logo_label.image = logo
        logo_label.grid(column=1, row=0)
        
        text_2 = tk.Label(frame_2, text="Dikenali isyarat huruf", 
                        bg="white", wraplength=250)
        text_2.config(font=("Monaco", 12))
        
        canvas3_2 = tk.Canvas(frame2, width=400, height=69, bg="white")
        canvas3_2.config(highlightthickness=0)
        
        back_btn = tk.Button(frame2, text="Kembali", command=lambda:back())
        back_btn.config(font=("Monaco", 12), fg="#354e8c", 
                        highlightbackground="#354e8c",
                        width=15, height=2)
        
        canvas4_2 = tk.Canvas(frame2, width=400, height=170, bg="white")
        canvas4_2.config(highlightthickness=0)
    
        frame_2.grid(columnspan=3)
        canvas_2.grid(columnspan=3, row=0)
        text_2.grid(columnspan=3, column=0, row=1)
        text2_2.grid(columnspan=3, column=0, row=2)
        canvas3_2.grid(columnspan=3)
        back_btn.grid(column=1, row=5)
        canvas4_2.grid(columnspan=3)
    
        def back():
            frame2.pack_forget()
            main()
        
    def open_file():
        
        filepath = askopenfilename(
            filetypes=[("Video Files", "*.avi")]
        )
        if filepath:
            browse_btn['text']='Memuat...'
            load_process(filepath)
            
        if not filepath:
            return 
        browse_btn['text']='Pilih Video'

    
    frame1 = tk.Frame(root, width=400, height=667, bg="white")
    frame2 = tk.Frame(root, width=400, height=667, bg="white")
    
    canvas = tk.Canvas(frame1, width=400, height=250, bg="white")
    canvas.config(highlightthickness=0)
    canvas.grid(columnspan=3)
    
    logo = Image.open('logo.png').resize((200, 200), Image.ANTIALIAS)
    logo = ImageTk.PhotoImage(logo)
    logo_label = tk.Label(frame1, image=logo, bg="white")
    logo_label.image = logo
    logo_label.grid(column=1, row=0)
    
    frame = tk.Frame(frame1, bg="white")
    frame.grid(columnspan=3)
    
    text = tk.Label(frame, text="Isyaratnya?", bg="white")
    text.config(font=("Baskerville", 30))
    text.grid(columnspan=3, column=0, row=1)
    
    canvas = tk.Canvas(frame, width=400, height=50, bg="white")
    canvas.config(highlightthickness=0)
    canvas.grid(columnspan=3, column=0, row=2)
    
    text = tk.Label(frame, text="Tekan tombol dibawah untuk memilih\
                    file video gerakan tangan \
                    yang ingin diketahui arti isyaratnya", 
                    bg="white", wraplength=250)
    text.config(font=("Monaco", 12))
    text.grid(columnspan=3, column=0, row=3)
    
    canvas = tk.Canvas(frame, width=400, height=20, bg="white")
    canvas.config(highlightthickness=0)
    canvas.grid(columnspan=3, column=0, row=4)
   
    browse_btn = tk.Button(frame, text='Pilih Video', 
                           command=lambda:open_file(), fg="#354e8c", 
                           highlightbackground="#354e8c",
                           width=15, height=2)
    
    browse_btn.config(font=("Monaco", 12))
    browse_btn.grid(column=1, row=5)
    
    canvas = tk.Canvas(frame1, width=400, height=172, bg="white")
    canvas.config(highlightthickness=0)
    canvas.grid(columnspan=3)
    
    frame1.pack()
    
    frame_2 = tk.Frame(frame2, bg="white")
    
    text2_2 = tk.Label(frame_2, text="_", bg="white")
    text2_2.config(font=("Baskerville", 30))
     
main()
root.eval('tk::PlaceWindow . center')
root.mainloop()

