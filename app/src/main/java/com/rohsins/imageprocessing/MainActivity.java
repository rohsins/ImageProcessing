package com.rohsins.imageprocessing;

import android.app.Activity;
import android.os.Bundle;
import android.text.Html;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.ViewDebug;
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
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

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

    private MenuItem highSpeedObjectDetection;
    private MenuItem faceDetection;

    private int modeFlag = 0;

    private boolean objectDetected = false;

    static {
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "OpenCV not Loaded");
        } else {
            Log.d(TAG, "OpenCV Loaded");
        }
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
        }
        if (item == faceDetection) {
            modeFlag = 1;
        }
        //noinspection SimplifiableIfStatement
        if (id == R.id.action_settings) {
            return true;
        }

        return super.onOptionsItemSelected(item);
    }

    private BaseLoaderCallback baseLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case BaseLoaderCallback.SUCCESS:
                {
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
                Imgproc.threshold(diffImage, thresholdImage, 40, 255, Imgproc.THRESH_BINARY);
                Imgproc.blur(thresholdImage, thresholdImage, new Size(40, 40));
                Imgproc.threshold(thresholdImage, thresholdImage, 120, 255, Imgproc.THRESH_BINARY);
                Imgproc.findContours(thresholdImage, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
                if (contours.size() > 0) objectDetected = true;
                if (objectDetected) {
                    objectDetected = false;
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
                        Imgproc.rectangle(buffer, new Point(rect.x, rect.y), new Point(rect.x + rect.width, rect.y + rect.height), new Scalar(255, 0, 0, 255), 2);
                    }
//            Imgproc.drawContours(buffer, matchedContours, -1, new Scalar(0, 255, 0));
                    matchedContours.clear();
                    contours.clear();
                }
            } break;
            case 1: {
                inputFrame1 = inputFrame.gray();
                Core.absdiff(inputFrame1, inputFrame2, diffImage);
                inputFrame2 = inputFrame.gray();
                buffer = inputFrame.rgba();
                Imgproc.threshold(diffImage, thresholdImage, 100, 255, Imgproc.THRESH_BINARY);
                Imgproc.blur(thresholdImage, thresholdImage, new Size(40, 40));
                Imgproc.threshold(thresholdImage, thresholdImage, 120, 255, Imgproc.THRESH_BINARY);
//                Imgproc.findContours(thresholdImage, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
//                if (contours.size() > 0) objectDetected = true;
//                if (objectDetected) {
//                    objectDetected = false;
//                    for (MatOfPoint contour : contours) {
//                        double actualArea = Imgproc.contourArea(contour);
//                        double width = contour.width();
//                        double radius = width / 2;
//                        double calculatedArea = Math.PI * Math.pow(radius, 2);
//                        if ((actualArea - calculatedArea) < 10000) {
//                            matchedContours.add(contour);
//                        }
//                    }
                buffer = thresholdImage;
            } break;
        }
        System.gc();
//        return inputFrame.rgba();
//        return inputFrame1;
//        return thresholdImage;
        return buffer;
    }
}