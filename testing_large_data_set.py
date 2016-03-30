# train_and_test.py

import cv2
import numpy as np
import operator
import os
#import generate_data

svm_params = dict( kernel_type = cv2.SVM_LINEAR,svm_type = cv2.SVM_C_SVC,C=2.67, gamma=5.383 )

# module level variables ##########################################################################
MIN_CONTOUR_AREA = 50
RESIZED_IMAGE_WIDTH = 28
RESIZED_IMAGE_HEIGHT = 28


class ContourWithData():

    npaContour = None           # contour
    boundingRect = None         # bounding rect for contour
    intRectX = 0                # bounding rect top left corner x location
    intRectY = 0                # bounding rect top left corner y location
    intRectWidth = 0            # bounding rect width
    intRectHeight = 0           # bounding rect height
    fltArea = 0.0               # area of contour

    def calculateRectTopLeftPointAndWidthAndHeight(self):               # calculate bounding rect info
        [intX, intY, intWidth, intHeight] = self.boundingRect
        self.intRectX = intX
        self.intRectY = intY
        self.intRectWidth = intWidth
        self.intRectHeight = intHeight

    def checkIfContourIsValid(self):                            # this is oversimplified, for a production grade program
        if self.fltArea < MIN_CONTOUR_AREA: return False        # much better validity checking would be necessary
        return True


def main():
    allContoursWithData = []
    validContoursWithData = []
##################################################################################################
##################################################################################
##
##    #npaClassifications = np.array([9], np.float32)
#####for training for one sample
##    '''
##    npaClassifications= np.loadtxt("classifications.txt", np.float32)
##    npaClassifications.sort()
##    npaClassifications = npaClassifications.reshape((npaClassifications.size, 1))
##    '''
##
###fopr training for multiple samples
    # npaClassifications = []
    # for i in range(0,10):
    #     for j in range(1,5001):
    #         npaClassifications.append([i])
    # npaClassifications = np.array(npaClassifications,np.float32)
    # npaClassifications = npaClassifications.reshape((npaClassifications.size, 1))
   
   
    # print npaClassifications, len(npaClassifications)
    # npaFlattenedImages = np.empty((0,784))
##
#####for training with only one sample
##    '''
##    for i in range(0,10):
##        npaFlattenedImage= np.loadtxt("flat_text/text("+str(i)+")/flattened_images"+str(i+1)+".txt",np.float32)
##
##        npaFlattenedImages = np.append(npaFlattenedImages, [npaFlattenedImage],0)
##        npaFlattenedImages = np.array(npaFlattenedImages,np.float32)
##    '''
###for training with only multiple samples
    # for i in range(0,10):
    #     for j in range(1,5001):
    #         npaFlattenedImage= np.loadtxt("new_flat_text/text("+str(i)+")/flattened_image"+str(j)+".txt",np.float32)
    #         npaFlattenedImages = np.append(npaFlattenedImages, [npaFlattenedImage],0)
    #         print i,j
    # npaFlattenedImages = np.array(npaFlattenedImages,np.float32)
    # print len(npaFlattenedImages),#npaFlattenedImages
    # np.savetxt("saved_data/classifications.txt", npaClassifications)           # write flattened images to file
    # np.savetxt("saved_data/flattened_images.txt", npaFlattenedImages)          #
#####################################################################################
#####################################################################################

###################################################################
    
    npaFlattenedImages= np.loadtxt("saved_data/flattened_images_Large_data_set.txt",np.float32) 
    npaClassifications= np.loadtxt("saved_data/classifications_Large_data_set.txt", np.float32)
    #kNearest = cv2.KNearest()
    #kNearest.train(npaFlattenedImages, npaClassifications)
    svm = cv2.SVM()
    svm.train(npaFlattenedImages, npaClassifications,params = svm_params)
    #print z

    #imgTestingNumbers = cv2.imread('samples/sample(9)/node'+str(6)+'.jpg')
    #imgTestingNumbers = cv2.imread('self_samples/5/sample1.png')
    imgTestingNumbers = cv2.imread("test11.jpg")
    imgTestingNumbers = cv2.resize(imgTestingNumbers, (800, 600))
    #cv2.imshow('imgTestingNumber1',imgTestingNumbers)
    
    imgGray = cv2.cvtColor(imgTestingNumbers, cv2.COLOR_BGR2GRAY)
    imgBlurred = cv2.GaussianBlur(imgGray, (5,5), 0)
    imgThresh = cv2.adaptiveThreshold(imgBlurred,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
    cv2.imshow('imgThresh',imgThresh)
    imgThreshCopy = imgThresh.copy()

    npaContours, npaHierarchy = cv2.findContours(imgThreshCopy,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    #cv2.drawContours(imgTestingNumbers,npaContours,-1,(255,255,0),2)
    for npaContour in npaContours:
        print cv2.contourArea(npaContour)
        contourWithData = ContourWithData()                                             # instantiate a contour with data object
        contourWithData.npaContour = npaContour                                         # assign contour to contour with data
        contourWithData.boundingRect = cv2.boundingRect(contourWithData.npaContour)     # get the bounding rect
        contourWithData.calculateRectTopLeftPointAndWidthAndHeight()                    # get bounding rect info
        contourWithData.fltArea = cv2.contourArea(contourWithData.npaContour)           # calculate the contour area
        allContoursWithData.append(contourWithData)

    for contourWithData in allContoursWithData:
        if contourWithData.checkIfContourIsValid():
            validContoursWithData.append(contourWithData)

    validContoursWithData.sort(key = operator.attrgetter("intRectX"))
    strFinalString = ""
    i=0
    a=0.0
    b=0
    z=0
    for contourWithData in validContoursWithData:
        i+=1
        cv2.rectangle(imgTestingNumbers,(contourWithData.intRectX, contourWithData.intRectY),(contourWithData.intRectX + contourWithData.intRectWidth, contourWithData.intRectY + contourWithData.intRectHeight),(0, 255, 0),2)
        imgROI = imgThresh[contourWithData.intRectY : contourWithData.intRectY + contourWithData.intRectHeight,contourWithData.intRectX : contourWithData.intRectX + contourWithData.intRectWidth]
        imgROIResized = cv2.resize(imgROI, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))
        npaROIResized = imgROIResized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))
        npaROIResized = imgROIResized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))
        npaROIResized = np.float32(npaROIResized)
        #retval, npaResults, neigh_resp, dists = kNearest.find_nearest(npaROIResized, k = 1)
        npaResults = svm.predict_all(npaROIResized)
        strCurrentChar = str(int(npaResults[0][0]))
        strFinalString = strFinalString + strCurrentChar

        cv2.namedWindow('Girl Friend '+str(i),cv2.WINDOW_NORMAL)
        cv2.imshow('Girl Friend '+str(i),imgROI)
        
        print strCurrentChar
        
        b=b+1
        if cv2.waitKey(0) == 121:
             a = a+1
        cv2.destroyAllWindows()
        
        if strCurrentChar == '4':
            a=a+1
    z=(a/b)*100
    print a,b,z
    print 'Accuracy: ',z,' %'

    #print "\n" + strFinalString + "\n"

    cv2.imshow("imgTestingNumbers", imgTestingNumbers)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return

if __name__ == "__main__":
    main()
            
