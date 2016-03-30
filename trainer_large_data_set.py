import cv2
import numpy as np
import sys
import os

class preProcessing():
    def __init__(self,image_file):
        self.npaFlattenedImages=np.empty((0,784))
        self.image_file=cv2.imread(image_file)
        #self.preProcessImage()

    def preProcessImage(self):

        MIN_CONTOUR_AREA = 0
        RESIZED_IMAGE_WIDTH = 28
        RESIZED_IMAGE_HEIGHT = 28

        self.imgGray = cv2.cvtColor(self.image_file, cv2.COLOR_BGR2GRAY)
        self.imgBlurred = cv2.GaussianBlur(self.imgGray, (5,5), 0)
        self.imgThresh = cv2.adaptiveThreshold(self.imgBlurred,255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)

        #cv2.imshow('img',self.image_file)
        #cv2.imshow('imgGray',self.imgGray)
        #cv2.imshow('imgBlurred',self.imgBlurred)
        #cv2.imshow('imgThresh',self.imgThresh)

        self.imgThreshCopy = self.imgThresh.copy() 

        self.npaContours, self.npaHierarchy = cv2.findContours(self.imgThreshCopy,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        cv2.drawContours(self.imgThreshCopy,self.npaContours,-1,(255,0,0),1)
        

        check=1
        for self.npaContour in self.npaContours:
            #print cv2.contourArea(self.npaContour)
            if cv2.contourArea(self.npaContour) > MIN_CONTOUR_AREA:
                check=check+1
                [intX, intY, intW, intH] = cv2.boundingRect(self.npaContour)

                cv2.rectangle(self.imgThreshCopy, (intX, intY),(intX+intW,intY+intH),(255, 0, 255),1)
                self.imgROI = self.imgThresh[intY:intY+intH, intX:intX+intW]
                self.imgROIResized = cv2.resize(self.imgROI, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))

                #cv2.imshow("ROI",self.imgROI)
                #cv2.imshow("ROIResized",self.imgROIResized)              

                npaFlattenedImage = self.imgROIResized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))
                self.npaFlattenedImages = np.append(self.npaFlattenedImages, npaFlattenedImage, 0)
                #if check>2:
                    #print '#########################################################################',check
                    #total =1
                    #sys.exit(0)
        #cv2.imshow('imgThreshCopy',self.imgThreshCopy)
        #print self.npaFlattenedImages

        #print "training complete"
        return self.npaFlattenedImages
for i in range(1,5001):
    #if i==500:
    #print "training complete"
    
    #text =obj=preProcessing('samples/sample(0)/node'+str(i)+'.jpg')
    obj=preProcessing('new dataset/(5)/node ('+str(i)+').jpg')
    text = obj.preProcessImage()
    #if check >2:
    	#os.remove('new dataset/(9)/node'+str(i)+'.jpg')
    np.savetxt('new_flat_text/text(5)/flattened_image'+str(i)+'.txt', text)
    print i
cv2.waitKey(0)
cv2.destroyAllWindows()
        
