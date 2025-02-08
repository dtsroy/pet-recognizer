package com.dtsroy.PetRecognizer;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.util.Log;

import org.json.JSONArray;
import org.json.JSONException;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;

public class Utils {
    private static float[][][][] preprocessFromBytes(byte[] inputData) {
        float[][][][] retData = new float[1][3][224][224];

        Bitmap bitmap = BitmapFactory.decodeByteArray(inputData, 0, inputData.length);
        Bitmap resizedBitmap = Bitmap.createScaledBitmap(bitmap, 224, 224, true);

        // 转换成float数组支持转换为tensor
        int pixel;
        for (int y=0; y<224; y++) {
            for (int x=0; x<224; x++) {
                pixel = resizedBitmap.getPixel(x, y);

                retData[0][0][y][x] = ((pixel >> 16) & 0xFF) / 255f;
                retData[0][1][y][x] = ((pixel >> 8) & 0xFF) / 255f;
                retData[0][2][y][x] = (pixel & 0xFF) / 255f;
            }
        }
        return retData;
    }

    private static void normalize(float[][][][] inputData) {
        //float[] mean = new float[] {0.5f, 0.5f, 0.5f};
        //float[] std = new float[] {0.5f, 0.5f, 0.5f};
        float[] mean = new float[] {0.485f, 0.456f, 0.406f};
        float[] std = new float[] {0.229f, 0.224f, 0.225f};

        for (int y=0; y<224; y++) {
            for (int x=0; x<224; x++) {
                for (int k=0; k<3; k++) {
                    inputData[0][k][x][y] -= mean[k];
                    inputData[0][k][x][y] /= std[k];
                }
            }
        }
    }

    public static OnnxTensor convertFromBytes(OrtEnvironment env, byte[] inputData) {
        float [][][][] tensorData = preprocessFromBytes(inputData);
        normalize(tensorData);
        try {
            return OnnxTensor.createTensor(env, tensorData);
        } catch (OrtException e) {
            e.printStackTrace();
            return null;
        }
    }

    public static String[] loadClassNames(Context context, int resourceId) {
        try {
            InputStream inputStream = context.getResources().openRawResource(resourceId);
            BufferedReader reader = new BufferedReader(new InputStreamReader(inputStream));
            StringBuilder jsonString = new StringBuilder();
            String line;
            while ((line = reader.readLine()) != null) {
                jsonString.append(line);
            }
            reader.close();
            JSONArray jsonArray = new JSONArray(jsonString.toString());
            String[] ret = new String[jsonArray.length()];
            for (int i = 0; i < jsonArray.length(); i++) {
                ret[i] = jsonArray.getString(i);
            }
            return ret;
        } catch (IOException | JSONException e) {
            Log.e("JsonLoader", "Error loading JSON file", e);
            return null;
        }
    }

}
