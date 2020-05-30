import statistics
import time
import timeit
from tkinter import *
from tkinter import filedialog
import cv2
import numpy as np

class Detect_line_sliding_window:
    root = Tk()
    c = Canvas(root, bg="white", height=300, width=300)
    c.pack()


    def __init__(self):
        self.ThreshBinValue = 200  # sem je potrebné zadať hodnotu z tabulky.

        self.pole_right = []
        self.aktualna_vzdialenost = 150
        self.ak_vz = []
        self.pole_left = []

        self.stabil_r = 290; self.stabil_l = 150
        self.count = 0; self.count2 = 0; self.count3 = 0
        self.zmena_pruhu = False
        self.leftx_c=None; self.rightx_c=None
        self.win_x = None

        self.gui()



    def gui(self):
        self.c.create_text(150, 100, fill="black", font="Times 12 bold",text="Video načítate po stlačení tlačidla: načítaj")
        self.c.create_text(150, 130, fill="black", font="Times 12 bold",text="\nVideo vypnete stlačením klávesy: q")
        self.c.create_text(150, 170, fill="black", font="Times 12 bold", text="\nVideo zastavíte/spustíte stlačením klávesy: p")
        b = Button(self.root, text="načítaj video", command=self.open_file)
        b.pack()
        mainloop()

    def open_file(self):  # otvori file explorer
        self.root.fileName = filedialog.askopenfilename(filetypes=(("howcode files", ".mp4"), ("All files", "*.*")))
        self.cap = cv2.VideoCapture(self.root.fileName)
        try:
            self.play_video()

        except:
            print("END")

    def play_video(self): #prehra video a vola funkcie na jeho upravu a detekciu ciar
        self.out = cv2.VideoWriter('outpy.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (640, 480))
        self.ccc = 0
        self.poles = 0
        if (self.cap.isOpened() == False):
            print("Error opening video stream or file")
        while (self.cap.isOpened() ):
            ret, self.frame = self.cap.read()
            img1 = self.frame

            self.gray_video()
            self.frame = cv2.GaussianBlur(self.frame,(5,5),0)

            Minv = self.bird_eye_view()

            thresh, self.frame = cv2.threshold(self.frame, self.ThreshBinValue, 255, cv2.THRESH_BINARY)
            self.frame = cv2.Canny(self.frame,145,255)
            vertices = [np.array([[100,480], [50,0], [360,0], [360,480]], dtype=np.int32)]
            self.frame = self.region_of_interest(self.frame,vertices)

            self.frame,self.r,self.l = self.sliding_windows(self.frame, (0, 255,0))

            self.result_frame(Minv,img1)
            self.change_lane()

            if ret == True:
                self.out.write(self.frame)

                cv2.imshow('Frame', self.frame)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
                if cv2.waitKey(25) == ord('p'):
                    cv2.waitKey(-1)
            else:
                break

        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()


    def bird_eye_view(self):    #pohlad z vtacej perspektivy
        img_size = (self.frame.shape[1], self.frame.shape[0])

        src = np.float32(
            [[(img_size[0] / 2) - 45, img_size[1] / 2 + 35],  # - 100
             [((img_size[0] / 6 - 70)), img_size[1]],
             [(img_size[0] * 5 / 6) + 240, img_size[1]],
             [(img_size[0] / 2 + 240), img_size[1] / 2 + 35]])  # + 240
        dst = np.float32(
            [[(img_size[0] / 4), 0],
             [(img_size[0] / 4), img_size[1]],
             [(img_size[0] * 3 / 4 - 100), img_size[1]],
             [(img_size[0] * 3 / 4) + 200, 0]])

        M = cv2.getPerspectiveTransform(src, dst)  # The transformation matrix
        Minv = cv2.getPerspectiveTransform(dst, src)  # Inverse transformation
        self.frame = cv2.warpPerspective(self.frame, M, img_size, cv2.INTER_LINEAR)
        return Minv

    def region_of_interest(self,img, vertices):  #priestor o ktory mame zaujem
        mask = np.zeros_like(img)

        if len(img.shape) > 2:
            channel_count = img.shape[2]
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255

        cv2.fillPoly(mask, vertices, ignore_mask_color)

        masked_image = cv2.bitwise_and(img, mask)
        return masked_image



    def result_frame(self,Minv,img1): #vysledny frame na zobrazenie/inverzna transformacia + addweight
        img_size = (self.frame.shape[1], self.frame.shape[0])
        self.frame = cv2.warpPerspective(self.frame, Minv, img_size, cv2.INTER_LINEAR)
        self.frame = cv2.addWeighted(img1, 1, self.frame, 0.3, 0)

    def change_lane(self): #signalizacia preradovania
        if (self.zmena_pruhu == True):
            cv2.putText(self.frame, '<---->', (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 2, cv2.LINE_AA)
            self.pole_right = []

    def gray_video(self):
        self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

    def get_hist(self,img):
             hist = np.sum(img[img.shape[0] // 2:, :], axis=0)
             return hist

    def sliding_windows(self, img, color, nwindows=30, margin=15, minpix=1):
        self.pts = []
        self.count = 1
        self.count2 = 1

        if (len(self.pole_right) > 100):
            self.pole_right = self.pole_right[50:]
        out_img = np.dstack((img, img, img)) * 255
        histogram = self.get_hist(img)

        # vrcholy ľavej a pravej polky
        midpoint = int(histogram.shape[0] / 2)
        self.leftx_base = np.argmax(histogram[:midpoint])
        self.rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        self.ak_vz.append(self.rightx_base - self.leftx_base)
        self.check_base()
        n0y,n0x = self.nonzero_img(img)

        window_height = np.int((img.shape[0]/2 +200) / nwindows) # nastavenie vysky okna pri detekcii

        self.leftx_c = self.leftx_base
        self.rightx_c = self.rightx_base

        # listy na pixely lavej a pravej strany
        self.left_lane_inds = []
        self.right_lane_inds = []

        for win in range(nwindows):
            win_y_low,win_y_high,win_xleft_low,win_xleft_high,win_xright_low,win_xright_high = self.set_parametre(img,win,window_height,self.leftx_c,self.rightx_c,margin)
            if(self.aktualna_vzdialenost>120):
                cv2.rectangle(out_img, (int(win_xleft_low), int(win_y_low)), (int(win_xleft_high), int(win_y_high)),color, -1)#vykreslovanie
                cv2.rectangle(out_img, (int(win_xright_low), int(win_y_low)), (int(win_xright_high), int(win_y_high)),color, -1)#vykreslovanie
            good_left_inds, good_right_inds = self.add_nonzeros_to_list(n0y,win_y_low,win_y_high, n0x,win_xleft_low,win_xleft_high,win_xright_low,win_xright_high)
            self.check_if_cond(good_left_inds,minpix,n0x,good_right_inds, nwindows,histogram)
        return out_img, self.rightx_c, self.leftx_c

    def check_base(self):
        if (self.rightx_base < 321 or self.rightx_base > 370):
                self.rightx_base=self.stabil_r
        else:
            self.stabil_r = self.rightx_base
        if (self.leftx_base < 130 or self.leftx_base > 170):
            self.leftx_base = self.stabil_l
        else:
            self.stabil_l = self.leftx_base
        if(self.ak_vz!=[]):
            if(self.rightx_base-self.leftx_base>statistics.median(self.ak_vz)+15):
                self.rightx_base=self.leftx_base+statistics.median(self.ak_vz)+5
            if(self.rightx_base-self.leftx_base<statistics.median(self.ak_vz)-15):
                self.rightx_base=self.leftx_base+statistics.median(self.ak_vz)-5
        self.pole_right.append(self.rightx_base)
        self.pole_left.append(self.leftx_base)
       # self.pole_right_base.append(self.rightx_base)


    def nonzero_img(self,img): # Identify the x and y positions of all nonzero pixels in the image
        nonzero = img.nonzero()
        n0y = np.array(nonzero[0])
        n0x = np.array(nonzero[1])
        return n0y,n0x

    def set_parametre(self,img,win,window_height,leftx_c,rightx_c,margin):
        win_y_low = img.shape[0] - (win + 1) * window_height
        win_y_high = img.shape[0] - win * window_height
        win_xleft_low = leftx_c - margin
        win_xleft_high = leftx_c + margin
        win_xright_low = rightx_c - margin
        win_xright_high = rightx_c + margin
        self.pts.append([leftx_c, win_y_high])
        return win_y_low,win_y_high,win_xleft_low,win_xleft_high,win_xright_low,win_xright_high

    def add_nonzeros_to_list(self,n0y,win_y_low,win_y_high, n0x,win_xleft_low,win_xleft_high,win_xright_low,win_xright_high):
        good_left_inds = ((n0y >= win_y_low) & (n0y < win_y_high) & (n0x >= win_xleft_low) & (n0x < win_xleft_high)).nonzero()[0]
        good_right_inds = ((n0y >= win_y_low) & (n0y < win_y_high) & (n0x >= win_xright_low) & (n0x < win_xright_high)).nonzero()[0]
        self.left_lane_inds.append(good_left_inds)
        self.right_lane_inds.append(good_right_inds)
        return good_left_inds, good_right_inds
        # Identify the nonzero pixels in x and y within the window

    def check_if_cond(self,good_left_inds,minpix,n0x,good_right_inds,nwindows,histogram):
        pom = self.leftx_c
        pom2 = self.rightx_c

        print(len(n0x[good_left_inds]))

        if len(good_left_inds) > minpix:
            self.leftx_c = np.int(np.mean(n0x[good_left_inds]))
        if len(good_right_inds) > minpix:
            self.rightx_c = np.int(np.mean(n0x[good_right_inds]))


        elif(len(good_left_inds) > minpix and len(good_right_inds) <= minpix):
            self.rightx_c = self.leftx_c+(pom2-pom)
        elif(len(good_left_inds) <= minpix and len(good_right_inds) > minpix):
            self.leftx_c = self.rightx_c-(pom2-pom)
        elif(len(good_left_inds) <= minpix and len(good_right_inds) <= minpix):
            self.leftx_c = pom
            self.rightx_c = pom2


        #siignalizacia
        if (self.leftx_c == pom):
            self.count = self.count + 1
        if (self.rightx_c == pom2):
            self.count2 = self.count2 + 1
        if (self.count >= nwindows - 2 and self.count2 >= nwindows - 2):
            a = self.leftx_c + ((self.rightx_c - self.leftx_c) // 2)
            if (np.argmax(histogram[int(a) - 25:int(a) + 25]) > 0):
                self.zmena_pruhu = True
        else:
            self.zmena_pruhu = False

        if(self.rightx_c-self.leftx_c >self.aktualna_vzdialenost+20 or self.rightx_c-self.leftx_c <self.aktualna_vzdialenost-20):
            if(self.ak_vz!=[] and len(self.ak_vz) > 5):
                self.rightx_c=self.leftx_c+statistics.median(self.ak_vz)
            else:
                self.rightx_c = self.leftx_c + self.aktualna_vzdialenost

        if(self.rightx_c-self.leftx_c<180):
            self.aktualna_vzdialenost=self.rightx_c-self.leftx_c
            self.ak_vz.append(self.aktualna_vzdialenost)

        if(self.count3<20):
            self.count3=self.count3+1
            self.pole_right.append(self.rightx_c)
            self.pole_left.append(self.leftx_c)
        else:
                if(abs(statistics.median(self.pole_left)-self.leftx_c) > abs(statistics.median(self.pole_right)-self.rightx_c)):
                    if(len(good_left_inds) <= minpix):
                        #print("pof")
                        self.leftx_c=self.rightx_c-self.aktualna_vzdialenost-1
                        self.aktualna_vzdialenost=self.aktualna_vzdialenost-1

                    self.pole_left.append(self.leftx_c)
                    self.pole_right.append(self.rightx_c)
        if(len(self.pole_left)>100):
            self.pole_left=self.pole_left[50:]
        if(len(self.pole_right)>100):
            self.pole_right=self.pole_right[50:]
        if(len(self.ak_vz)>100):
            self.ak_vz=self.ak_vz[50:]



p=Detect_line_sliding_window()