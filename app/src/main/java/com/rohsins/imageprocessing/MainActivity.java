package com.rohsins.imageprocessing;

import android.app.Activity;
import android.content.Context;
import android.os.Bundle;
import android.provider.ContactsContract;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.WindowManager;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.MatOfPoint3f;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Vector;

public class MainActivity extends Activity implements CvCameraViewListener2 {
    private static final String TAG = "MainActivity";
    private CameraBridgeViewBase cameraBridgeViewBase;

    private Mat inputFrame1;
    private Mat inputFrame2;
    private Mat diffImage;
    private Mat thresholdImage;
    private Mat buffer;
    private List<MatOfPoint> contours;
    private List<MatOfPoint> matchedContours;
    private Mat hierarchy;
    private Mat inputRgbaFrame;
    private Mat inputGrayFrame;

    private File cascadeFile;
    private File cascadeFileEyes;
    private File cascadeFileBody;
    private CascadeClassifier cascadeClassifier;
    private CascadeClassifier cascadeClassifierEyes;
    private CascadeClassifier cascadeClassifierBody;

    private static final Scalar FACE_RECT_COLOR = new Scalar(0, 255, 0 , 255);
    private final Scalar EYES_RECT_COLOR = new Scalar(0, 255, 0, 255);

    private int absoluteFaceSize;
    private int absoluteEyeSizeX;
    private int absoluteEyeSizeY;
    private int absoluteMinBodySizeX;
    private int absoluteMinBodySizeY;
    private int highSpeedObjectDetectionSensitivity = 20;
    private int getHighSpeedObjectDetectionBlur = 20;

    private MenuItem highSpeedObjectDetection;
    private MenuItem faceDetection;
    private MenuItem eyesDetection;
    private MenuItem bodyDetection;
    private MenuItem calibration;
    private MenuItem switchCamera;

    private int modeFlag;
    private int cameraFlag;

    private boolean objectDetected;

    static {
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "OpenCV not Loaded");
        } else {
            Log.d(TAG, "OpenCV Loaded");
        }
    }

    public MainActivity() {
        modeFlag = 0;
        objectDetected = false;
        absoluteFaceSize = 0;
        cameraFlag = 1;
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        cameraBridgeViewBase = (CameraBridgeViewBase) findViewById(R.id.mainactivity_surface_view);
        cameraBridgeViewBase.setCameraIndex(0);
//        cameraBridgeViewBase.enableFpsMeter();
        cameraBridgeViewBase.setVisibility(CameraBridgeViewBase.VISIBLE);
        cameraBridgeViewBase.setCvCameraViewListener(this);
        cameraBridgeViewBase.setCameraIndex(cameraBridgeViewBase.CAMERA_ID_BACK);
//        cameraBridgeViewBase.setCameraIndex(cameraBridgeViewBase.CAMERA_ID_FRONT);
    }

    @Override
    public void onPause() {
        super.onPause();
        if (cameraBridgeViewBase != null) {
            cameraBridgeViewBase.disableView();
        }
    }

    @Override
    public void onResume() {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV Library not found. Using OpenCV Manager for initialization");
        } else {
            Log.d(TAG, "OpenCV Library found inside package.");
            baseLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    @Override
    public void onDestroy() {
        super.onDestroy();
        if (cameraBridgeViewBase != null) {
            cameraBridgeViewBase.disableView();
        }
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        // Inflate the menu; this adds items to the action bar if it is present.
        highSpeedObjectDetection = menu.add("highSpeedObjectDetection");
        faceDetection = menu.add("faceDetection");
        eyesDetection = menu.add("eyesDetection");
        bodyDetection = menu.add("bodyDetection");
        calibration = menu.add("calibration");
        switchCamera = menu.add("switchCamera");
        getMenuInflater().inflate(R.menu.menu_main, menu);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        // Handle action bar item clicks here. The action bar will
        // automatically handle clicks on the Home/Up button, so long
        // as you specify a parent activity in AndroidManifest.xml.
        int id = item.getItemId();
        if (item == highSpeedObjectDetection) {
            modeFlag = 0;
        } else if (item == faceDetection) {
            modeFlag = 1;
            faceDetectionDependenciesInitialization();
        } else if (item == eyesDetection) {
            modeFlag = 2;
            eyesDetectionDependenciesInitialization();
        } else if (item == bodyDetection) {
            modeFlag = 3;
            bodyDetectionDependenciesInitialization();
        }
        else if (item == calibration) {
            modeFlag = 4;
        } else if (item == switchCamera) {
            if (cameraFlag == 0) {
                cameraBridgeViewBase.disableView();
                cameraBridgeViewBase.setCameraIndex(cameraBridgeViewBase.CAMERA_ID_BACK);
                cameraFlag = 1;
                cameraBridgeViewBase.enableView();
            }
            else if (cameraFlag == 1) {
                cameraBridgeViewBase.disableView();
                cameraBridgeViewBase.setCameraIndex(cameraBridgeViewBase.CAMERA_ID_FRONT);
                cameraFlag = 0;
                cameraBridgeViewBase.enableView();
            }
        }
        //noinspection SimplifiableIfStatement
        if (id == R.id.action_settings) {
            return true;
        }

        return super.onOptionsItemSelected(item);
    }

    private void faceDetectionDependenciesInitialization() {
        try {
            InputStream inputStream = getResources().openRawResource(R.raw.lbpcascade_frontalface);
            File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
            cascadeFile = new File(cascadeDir, "lbpcascade_frontalface.xml");
            FileOutputStream outputStream = new FileOutputStream(cascadeFile);

            byte[] byteBuffer = new byte[4096];
            int bytesRead;
            while ((bytesRead = inputStream.read(byteBuffer)) != -1) {
                outputStream.write(byteBuffer, 0, bytesRead);
            }
            inputStream.close();
            outputStream.close();

            cascadeClassifier = new CascadeClassifier(cascadeFile.getAbsolutePath());
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void eyesDetectionDependenciesInitialization() {
        try {
            InputStream inputStream = getResources().openRawResource(R.raw.haarcascade_eye);
            File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
            cascadeFileEyes = new File(cascadeDir, "haarcascade_eye.xml");
            FileOutputStream outputStream = new FileOutputStream(cascadeFileEyes);

            byte[] byteBuffer = new byte[4096];
            int bytesRead;
            while ((bytesRead = inputStream.read(byteBuffer)) != -1) {
                outputStream.write(byteBuffer, 0, bytesRead);
            }
            inputStream.close();
            outputStream.close();

            cascadeClassifierEyes = new CascadeClassifier(cascadeFileEyes.getAbsolutePath());
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void bodyDetectionDependenciesInitialization() {
        try {
            InputStream inputStream = getResources().openRawResource(R.raw.haarcascade_fullbody);
            File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
            cascadeFileBody = new File(cascadeDir, "haarcascade_fullbody.xml");
            FileOutputStream outputStream = new FileOutputStream(cascadeFileBody);

            byte[] byteBuffer = new byte[4096];
            int bytesRead;
            while ((bytesRead = inputStream.read(byteBuffer)) != -1) {
                outputStream.write(byteBuffer, 0, bytesRead);
            }
            inputStream.close();
            outputStream.close();

            cascadeClassifierBody = new CascadeClassifier(cascadeFileBody.getAbsolutePath());
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private BaseLoaderCallback baseLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case BaseLoaderCallback.SUCCESS: {
                    if (modeFlag == 1) {
                        faceDetectionDependenciesInitialization();
                    } else if (modeFlag == 2) {
                        modeFlag = 2;
                        eyesDetectionDependenciesInitialization();
                    } else if (modeFlag == 3) {
                        modeFlag = 3;
                        bodyDetectionDependenciesInitialization();
                    }
                    cameraBridgeViewBase.enableView();
                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };

    @Override
    public void onCameraViewStarted(int width, int height) {
        inputFrame1 = new Mat(height, width, CvType.CV_8UC1);
        inputFrame2 = new Mat(height, width, CvType.CV_8UC1);
        diffImage = new Mat(height, width, CvType.CV_8UC1);
        thresholdImage = new Mat(height, width, CvType.CV_8UC1);
        buffer = new Mat(height, width, CvType.CV_8UC4);
        hierarchy = new Mat(height, width, CvType.CV_8UC4);
        contours = new ArrayList<>();
        matchedContours = new ArrayList<>();
        inputGrayFrame = new Mat();
        inputRgbaFrame = new Mat();

        absoluteFaceSize = (int) (height * 0.2);
        absoluteEyeSizeX = (int) (height * 0.2);
        absoluteEyeSizeY = (int) (height * 0.1);
        absoluteMinBodySizeX = (int) (3);
        absoluteMinBodySizeY = (int) (7);
    }

    @Override
    public void onCameraViewStopped() {
        inputFrame1.release();
        inputFrame2.release();
        diffImage.release();
        thresholdImage.release();
        hierarchy.release();
        matchedContours.clear();
        contours.clear();
    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        switch (modeFlag) {
            case 0: {
                inputFrame1 = inputFrame.gray();
                Core.absdiff(inputFrame1, inputFrame2, diffImage);
                inputFrame2 = inputFrame.gray();
                buffer = inputFrame.rgba();
                Imgproc.threshold(diffImage, thresholdImage, highSpeedObjectDetectionSensitivity, 255, Imgproc.THRESH_BINARY);
                Imgproc.blur(thresholdImage, thresholdImage, new Size(10, 10));
                Imgproc.threshold(thresholdImage, thresholdImage, getHighSpeedObjectDetectionBlur, 255, Imgproc.THRESH_BINARY);
                Imgproc.findContours(thresholdImage, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
                if (contours.size() > 0) objectDetected = true;
                else objectDetected = false;
                if (objectDetected) {
                    for (MatOfPoint contour : contours) {
                        double actualArea = Imgproc.contourArea(contour);
                        double width = contour.width();
                        double radius = width / 2;
                        double calculatedArea = Math.PI * Math.pow(radius, 2);
                        if ((actualArea - calculatedArea) < 10000) {
                            matchedContours.add(contour);
                        }
                    }
                    for (int i = 0; i < contours.size(); i++) {
                        MatOfPoint2f approxCurve = new MatOfPoint2f();
                        MatOfPoint2f matOfPoint2f = new MatOfPoint2f(contours.get(i).toArray());
                        double approxDistance = Imgproc.arcLength(matOfPoint2f, true) * 0.02;
                        Imgproc.approxPolyDP(matOfPoint2f, approxCurve, approxDistance, true);
                        MatOfPoint points = new MatOfPoint(approxCurve.toArray());
                        Rect rect = Imgproc.boundingRect(points);
                        Imgproc.rectangle(buffer, new Point(rect.x, rect.y), new Point(rect.x + rect.width, rect.y + rect.height), new Scalar(0, 255, 0, 255), 2);
                        Imgproc.putText(buffer,"Tracking Object", new Point(rect.x, rect.y), 1, 1, new Scalar(0, 255, 0),1);
                    }
//            Imgproc.drawContours(buffer, matchedContours, -1, new Scalar(0, 255, 0));
                    matchedContours.clear();
                    contours.clear();
                }
            } break;
            case 1: {
                inputRgbaFrame = inputFrame.rgba();
                inputGrayFrame = inputFrame.gray();
                MatOfRect faces = new MatOfRect();
                if (cascadeClassifier != null) {
                    cascadeClassifier.detectMultiScale(inputGrayFrame, faces, 1.1, 3, 3, new Size(absoluteFaceSize, absoluteFaceSize), new Size());
                }
                Rect[] facesArray = faces.toArray();
                for (int i = 0; i < facesArray.length; i++) {
                    Imgproc.rectangle(inputRgbaFrame, facesArray[i].tl(), facesArray[i].br(), FACE_RECT_COLOR, 2);
                    Imgproc.putText(inputRgbaFrame,"You Look Awesome", new Point(facesArray[i].x, facesArray[i].y - 7), 2, 1, new Scalar(0, 255, 0),2);
                }
                buffer = inputRgbaFrame;
            } break;
            case 2: {
                inputRgbaFrame = inputFrame.rgba();
                inputGrayFrame = inputFrame.gray();
                MatOfRect faces = new MatOfRect();
                if (cascadeClassifierEyes != null) {
                    cascadeClassifierEyes.detectMultiScale(inputGrayFrame, faces, 1.1, 1, 1, new Size(absoluteEyeSizeX , absoluteEyeSizeY), new Size());
                }
                Rect[] facesArray = faces.toArray();
                for (int i = 0; i < facesArray.length; i++) {
                    Imgproc.rectangle(inputRgbaFrame, facesArray[i].tl(), facesArray[i].br(), EYES_RECT_COLOR, 1);
                }
                buffer = inputRgbaFrame;
            } break;
            case 3: {
                inputRgbaFrame = inputFrame.rgba();
                inputGrayFrame = inputFrame.gray();
                MatOfRect Bodies = new MatOfRect();
                if (cascadeClassifierBody != null) {
                    cascadeClassifierBody.detectMultiScale(inputGrayFrame, Bodies, 1.1, 2, 18|9, new Size(absoluteMinBodySizeX, absoluteMinBodySizeY), new Size());
                }
                Rect[] BodyArray = Bodies.toArray();
                for (int i = 0; i < BodyArray.length; i++) {
                    Imgproc.rectangle(inputRgbaFrame, BodyArray[i].tl(), BodyArray[i].br(), FACE_RECT_COLOR, 1);
                }
                buffer = inputRgbaFrame;
            } break;
            case 4: {
                inputFrame1 = inputFrame.gray();
                Core.absdiff(inputFrame1, inputFrame2, diffImage);
                inputFrame2 = inputFrame.gray();
                buffer = inputFrame.rgba();
                Imgproc.threshold(diffImage, thresholdImage, 40, 255, Imgproc.THRESH_BINARY);
                Imgproc.blur(thresholdImage, thresholdImage, new Size(40, 40));
                Imgproc.threshold(thresholdImage, thresholdImage, 120, 255, Imgproc.THRESH_BINARY);
                buffer = thresholdImage;
            } break;
        }
        System.gc();
        return buffer;
    }
}