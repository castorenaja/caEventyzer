import sys
import os.path
from PyQt5 import QtWidgets
from pyqt5_caEventGUI import Ui_caEventGUI
import cv2
import numpy as np
from matplotlib import pyplot as plt
from math import factorial
from PIL import Image
import imutils
import math
from pylab import *



 
class myCaEventGUIProgram(Ui_caEventGUI):
    def __init__(self, dialog):
        Ui_caEventGUI.__init__(self)
        self.setupUi(dialog)
        self.browseButton.clicked.connect(self.openFileNameDialog)
        self.runButton.clicked.connect(self.runAnalysis)
        self.browseButton_2.clicked.connect(self.openFileNameDialogMask)

    def openFileNameDialog(self):   
        global inFile
        options = QtWidgets.QFileDialog.Options()
        inFile, _ = QtWidgets.QFileDialog.getOpenFileName(options=options)
        self.listWidget.addItem(inFile)
        del options

    def openFileNameDialogMask(self):   
        global inMask
        options = QtWidgets.QFileDialog.Options()
        inMask, _ = QtWidgets.QFileDialog.getOpenFileName(options=options)
        self.listWidget_2.addItem(inMask)
        del options


    def runAnalysis(self):  
        global fps
        txtInput = self.myTextInput.text()
        fps = int(txtInput)
                      
        if inFile:
            videoReader(inFile,fps,inMask)



def videoReader(inVideo,fps,inMask):

    imgMask = cv2.imread(inMask) # 1:Color, 0:Grayscale, -1:Unchaged
    myMaskImg = cv2.cvtColor(imgMask, cv2.COLOR_BGR2GRAY)
    retGray, threshGray = cv2.threshold(myMaskImg,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU) # Use THRESH_BINARY_INV if the background is white
    contours, hierarchy = cv2.findContours(threshGray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cntAreas = np.zeros(len(contours),dtype='uint32')   

    for p in range(0,len(contours)):
        cntAreas[p] = cv2.contourArea(contours[p])

    tupleIndexes = np.where(cntAreas>99)
    arrIndexes = list(tupleIndexes[0])

    selectContours = list( contours[i] for i in arrIndexes)
    numCnts = len(selectContours)
    
    videoName = inFile.split("/")[-1]
    directory = inFile[:-len(videoName)]+'/'+videoName[:-4]
    videoName = videoName[:-4]

    if not os.path.exists(directory):
        os.makedirs(directory)
       
#    directory = inFile[:-4].split("/")[-1]

    try:
        cap = cv2.VideoCapture(inVideo)
    except:
        print 'Error Loading Video File'

    numFrames = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))

    timeSpace = np.zeros(numFrames,dtype='float64')
    cellSignal = np.zeros(shape=(numCnts,numFrames),dtype='uint32')
    imName = [None]*numCnts


    for cnt in range(0,numCnts):
    
        try:
            cap = cv2.VideoCapture(inVideo)
        except:
            print 'Error Loading Video File'
    
#        numFrames = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
#        frW  = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
#        frH = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
      
        lineScan = []

        iFrame = 0
    
        while(cap.isOpened()):
            ret, frame = cap.read()
            
            if(ret==True):
                grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Converting to GrayScale
                iFrame = int(iFrame+1)
                
                if cnt == 0:
                    timeSpace[iFrame-1] = float(iFrame-1)/float(fps)

                contourFrame = np.zeros(myMaskImg.shape,np.uint8)
                cv2.drawContours(contourFrame, selectContours,cnt,(255,0,0), -1)
                
                if iFrame == 1:
#                    imName[cnt] = directory+inFile[:-4]+'_Cell_'+str(cnt)+'.tif'
                    imName[cnt] = directory+'/'+videoName[:-4]+'_Cell'+str(cnt)+'.tif'

                    imContour = Image.fromarray(contourFrame)
                    imContour.save(imName[cnt],compression = 'None')                    

                del contourFrame
                
                fullFrame = np.zeros(myMaskImg.shape,np.uint8)
                cv2.drawContours(fullFrame, selectContours,cnt,(255,0,0), -1) # 3rd entry => -1 all cells, any other number specifies a particular cell                
                maskedFullFrame = cv2.bitwise_and(grayFrame,fullFrame)               

                px,py,cwidth,cheight = cv2.boundingRect(selectContours[cnt])
                (rtcx,rtcy),(rtw,rth),cangle = cv2.minAreaRect(selectContours[cnt])

                cropImg = maskedFullFrame[py:py+cheight,px:px+cwidth]
                
                if cangle < 0:
                    rtangle = -(90+cangle)
                    rotImg = imutils.rotate_bound(cropImg, rtangle)
                if cangle > 0:
                    rtangle = 90-cangle  
                    rotImg = imutils.rotate_bound(cropImg, rtangle)
                else:
                    rotImg = cropImg
                
                rotImg = imutils.rotate_bound(rotImg, 90)

                del fullFrame
                
                frW = rotImg.shape[1]
                oneFrameScan = np.zeros(frW,dtype=int)

                for k in range(0,frW):
                    oneFrameScan[k] = np.sum(rotImg[:,k])
                    
                lineScan.append(oneFrameScan)
                
                cellSignal[cnt,iFrame-1] = np.sum(oneFrameScan)
                
                del oneFrameScan           
                    
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
             
            else:
                cap.release()
                cv2.destroyAllWindows
                cv2.waitKey(1)

        lineScanMap = np.transpose(np.asarray(lineScan))
    
        scanH = lineScanMap.shape[0]
        scanW = lineScanMap.shape[1]
       
        diffLineScan = np.zeros(shape=(scanH,scanW),dtype=float)
        normLineScan = np.zeros(shape=(scanH,scanW),dtype=float)
       
        for m in range(0,scanH):
            tempMax = np.amax(lineScanMap[m,:])
            diffLineScan[m,:] = tempMax-lineScanMap[m,:]
                  
        for m in range(0,scanH):
            tempMaxDiff = np.amax(diffLineScan[m,:])
            if tempMaxDiff > 0:
                normLineScan[m,:] = (diffLineScan[m,:]/tempMaxDiff)*255.
                normLineScan[m,:] = np.clip(normLineScan[m,:],0,255)
            else:
                normLineScan[m,:] = 0
                
        uint8lineScan = normLineScan.astype('uint8')
        uint8lineScan = np.flipud(uint8lineScan)      
    
        uint8lineScanInv = cv2.bitwise_not(uint8lineScan) # Inverting Scale - Black=255
     
        imlineScanMap = Image.fromarray(uint8lineScanInv)
#                invLineScanMap = PIL.ImageOps.invert(imlineScanMap)
        outFile = directory+'/'+videoName+'_Cell'+str(cnt)+'_LineByLine_v4.tif'
        imlineScanMap.save(outFile,compression = 'None')
        
        tempCaSTM = cv2.imread(outFile)
        grayCaSTM = cv2.cvtColor(tempCaSTM, cv2.COLOR_BGR2GRAY)
#        thCaSTM = cv2.adaptiveThreshold(grayCaSTM, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
        retTemp, thCaSTM = cv2.threshold(grayCaSTM,150,255,cv2.THRESH_BINARY)
        thCaSTMu8 = thCaSTM.astype('uint8')
        thCaSTMimg = Image.fromarray(thCaSTMu8)        
        outThFile = directory+'/'+videoName+'_Cell'+str(cnt)+'_LineByLine_Thresh_v4.tif'
        thCaSTMimg.save(outThFile,compression = 'None')
    
        del lineScanMap, imlineScanMap, normLineScan, diffLineScan, lineScan


    nRows = math.ceil(numCnts/2)  
    nCols = 4

    figA = plt.figure(figsize=(16,8))
    figA.patch.set_facecolor('white')
    fcnt = 0

    fltSignalArr = np.zeros(shape=(numCnts,numFrames),dtype='float64')
    cellEventsFZero = np.zeros(shape=(numCnts,numFrames),dtype='float64')

    for cnt in range(0,numCnts):  
        fltSignal = savitzky_golay(cellSignal[cnt,:], 721, 5)
        fltSignalArr[cnt,:] = fltSignal
#        subtractedSignal = cellSignal[cnt,:]-fltSignal
        subtractedSignal = cellSignal[cnt,:]
#        subtractedSignal = cellSignal[cnt,:]/fltSignal
#        subtractedSignal = np.clip(subtractedSignal,1,np.amax(subtractedSignal))
#        subtractedSignal = np.clip(cellSignal[cnt,:],fltSignal,np.amax(cellSignal[cnt,:]))
#        subtractedSignal = cellSignal[cnt,:]
        maxSig = np.amax(subtractedSignal)
        minSig = np.amin(subtractedSignal)
#        if minSig < 0:
#            subtractedSignal = subtractedSignal+abs(minSig)
#        cellEvents[cnt,:] = (cellSignal[cnt,:].astype('float64')-minSig)/(maxSig-minSig)
        cellEventsFZero[cnt,:] = subtractedSignal.astype('float64')/abs(minSig)    
        npWrite2File = np.column_stack((timeSpace,cellEventsFZero[cnt,:]))
        np.savetxt(directory+'/'+videoName+'_Cell_'+str(cnt)+'_EventNormF0Signal.txt', npWrite2File, fmt='%10.5f', delimiter="\t")

        fcnt = fcnt+1
        myImage = cv2.imread(imName[cnt],0)
        figA.add_subplot(nRows,nCols,fcnt)
        imshow(myImage,cmap='gray')
        plt.axis('off')
##        plt.title(title)
        fcnt = fcnt+1
        figA.add_subplot(nRows,nCols,fcnt)
##        plt.plot(np.arange(0,numFrames,1), cellEvents[cnt,:],'-')
        plt.plot(timeSpace, cellEventsFZero[cnt,:],'-')
#        plt.plot(timeSpace, cellSignal[cnt,:],'-')
#        plt.plot(timeSpace, fltSignalArr[cnt,:],'-r')
#        plt.ylim((0.99,allMax+0.02))
#        fcnt = fcnt+1
#        figA.add_subplot(nRows,nCols,fcnt)
##        if cnt==10:
##        fDos = plt.figure(figsize=(16,8))       
#        plt.plot(roiFreqDom,roiFreqSTMap2)
##        plt.xlabel('Frequency (Contractions/Minute)')
#        plt.xlim((1,80))
##        plt.ylim((0,1e10))

    plt.savefig(directory+'/'+videoName+'_CellData.pdf',dpi=300)
    plt.show()            
        
        
        
def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError, msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs(y[1:half_window+1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')


 
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    dialog = QtWidgets.QDialog()   
    prog = myCaEventGUIProgram(dialog)
    dialog.show()
    sys.exit(app.exec_())