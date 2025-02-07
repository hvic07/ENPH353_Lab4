#!/usr/bin/env python3

from PyQt5 import QtCore, QtGui, QtWidgets
from python_qt_binding import loadUi
import cv2
import numpy as np
import sys

class My_App(QtWidgets.QMainWindow):
    def __init__(self):
        super(My_App, self).__init__()
        loadUi("./SIFT_app.ui", self)

        self._cam_id = 0
        self._cam_fps = 10
        self._is_cam_enabled = False
        self._is_template_loaded = False
        self.template_image = None
        self.template_keypoints = None
        self.template_descriptors = None
        self.sift = cv2.SIFT_create()
        self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

        self.browse_button.clicked.connect(self.SLOT_browse_button)
        self.toggle_cam_button.clicked.connect(self.SLOT_toggle_camera)

        self._camera_device = cv2.VideoCapture(self._cam_id)
        self._camera_device.set(3, 320)
        self._camera_device.set(4, 240)

        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self.SLOT_query_camera)
        self._timer.setInterval(int(1000 / self._cam_fps))

    def SLOT_browse_button(self):
        dlg = QtWidgets.QFileDialog()
        dlg.setFileMode(QtWidgets.QFileDialog.ExistingFile)

        if dlg.exec_():
            self.template_path = dlg.selectedFiles()[0]
            self.template_image = cv2.imread(self.template_path, cv2.IMREAD_GRAYSCALE)
            self.template_keypoints, self.template_descriptors = self.sift.detectAndCompute(self.template_image, None)
            
            pixmap = QtGui.QPixmap(self.template_path)
            self.template_label.setPixmap(pixmap)
            self._is_template_loaded = True

            print("Loaded template image file: " + self.template_path)

    def convert_cv_to_pixmap(self, cv_img):
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        height, width, channel = cv_img.shape
        bytesPerLine = channel * width
        q_img = QtGui.QImage(cv_img.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
        return QtGui.QPixmap.fromImage(q_img)

    def SLOT_query_camera(self):
        ret, frame = self._camera_device.read()
        if not ret:
            return
        
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.sift.detectAndCompute(gray_frame, None)

        if self._is_template_loaded and descriptors is not None:
            matches = self.matcher.match(self.template_descriptors, descriptors)
            matches = sorted(matches, key=lambda x: x.distance)[:50]
            
            if len(matches) > 10:
                src_pts = np.float32([self.template_keypoints[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([keypoints[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                
                H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                
                if H is not None:
                    h, w = self.template_image.shape
                    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
                    projected_corners = cv2.perspectiveTransform(corners, H)
                    
                    frame = cv2.polylines(frame, [np.int32(projected_corners)], True, (0, 255, 0), 3)
            
        pixmap = self.convert_cv_to_pixmap(frame)
        self.live_image_label.setPixmap(pixmap)

    def SLOT_toggle_camera(self):
        if self._is_cam_enabled:
            self._timer.stop()
            self._is_cam_enabled = False
            self.toggle_cam_button.setText("&Enable camera")
        else:
            self._timer.start()
            self._is_cam_enabled = True
            self.toggle_cam_button.setText("&Disable camera")

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    myApp = My_App()
    myApp.show()
    sys.exit(app.exec_())
