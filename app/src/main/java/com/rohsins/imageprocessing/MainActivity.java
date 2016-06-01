package com.rohsins.imageprocessing;

import android.app.Activity;
import android.os.Bundle;
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
import org.opencv.imgproc.Imgproc;


public class MainActivity extends Activity implements CvCameraViewListener2 {
    private static final String TAG = "MainActivity";
    private CameraBridgeViewBase cameraBridgeViewBase;

    private Mat inputFrame1;
    private Mat inputFrame2;
    private Mat diffImage;
    private Mat thresholdImage;

    private int flag = 0;

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
        getMenuInflater().inflate(R.menu.menu_main, menu);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        // Handle action bar item clicks here. The action bar will
        // automatically handle clicks on the Home/Up button, so long
        // as you specify a parent activity in AndroidManifest.xml.
        int id = item.getItemId();

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
    }

    @Override
    public void onCameraViewStopped() {
    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        if (flag == 0 ) {
            inputFrame1 = inputFrame.gray();
            flag = 1;
        } else if (flag == 1) {
            inputFrame2 = inputFrame.gray();
            flag = 0;
        }
        Core.absdiff(inputFrame1, inputFrame2, diffImage);
        Imgproc.threshold(diffImage, thresholdImage, 20, 255, Imgproc.THRESH_BINARY);
        System.gc();
//        return inputFrame.rgba();
//        return inputFrame1;
        return thresholdImage;
    }
}