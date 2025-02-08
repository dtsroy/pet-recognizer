package com.dtsroy.PetRecognizer;

import android.content.Context;
import android.util.Log;

import java.io.FileInputStream;
import java.io.IOError;
import java.io.IOException;
import java.io.InputStream;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.HashMap;
import java.util.Map;

import ai.onnxruntime.*;

public class ONNXModel {
    private final OrtEnvironment env;
    private OrtSession session;
    private Context context;

    public ONNXModel(Context c, String fp) throws IOException {
        env = OrtEnvironment.getEnvironment();
        context = c;

        // 加载
        try (InputStream inputStream = context.getAssets().open("PetRecognizerM4.onnx")) {
            byte[] modelBytes = new byte[inputStream.available()];
            inputStream.read(modelBytes);
            session = env.createSession(modelBytes);
        } catch (OrtException e) {
            e.printStackTrace();
        }
    }

    public float[] run(OnnxTensor inputTensor) {
        try {
            Map<String, OnnxTensor> inputTensors = new HashMap<String, OnnxTensor>();
            inputTensors.put("input0", inputTensor);
            OrtSession.Result result = session.run(inputTensors);
            float[][] outputData = (float[][])result.get(0).getValue();

            inputTensor.close();
            result.close();

            return softmax(outputData[0]);

        } catch (OrtException e) {
            e.printStackTrace();
            return null;
        }
    }

    private static float[] softmax(float[] input) {
        int length = input.length;
        float[] expValues = new float[length];
        float sum = 0.0f;

        // 计算每个输入值的指数
        for (int i = 0; i < length; i++) {
            expValues[i] = (float) Math.exp(input[i]);
            sum += expValues[i];
            Log.d("MODEL", "i=" + i + ", origin=" + input[i] + ", sum=" + sum);
        }

        // 计算 softmax 值
        float[] output = new float[length];
        for (int i = 0; i < length; i++) {
            output[i] = expValues[i] / sum;
        }

        return output;
    }

    public void test() {
        try (InputStream inputStream = context.getAssets().open("img0.bin")) {
            byte[] data = new byte[inputStream.available()];
            inputStream.read(data);
            OnnxTensor tensor = Utils.convertFromBytes(env, data);
            assert tensor != null;
            float[] res = run(tensor);
            tensor.close();
            Log.i("MODEL", "run successfully!");

            for (int i=0; i<37; i++) {
                Log.d("MODEL", "id=" + i + "p=" + res[i]);
            }

        } catch (IOException e) {
            Log.e("MODEL", "reading test img file failed");
        }
    }

}
