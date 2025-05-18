package dev.samarthguptha.jtensor.core;

import java.util.Arrays;
//ee
public class Tensor {
    private final double[] data;
    private final int[] shape;
    private final int[] strides;
    private final int offset;

    private Tensor(double[] data, int[] shape, int[] strides, int offset) {
        this.data = data;
        this.shape = shape;
        this.strides = strides;
        this.offset = offset;

        if (data == null) {
            throw new IllegalArgumentException("data is null");
        }
        if (shape == null) {
            throw new IllegalArgumentException("shape is null");
        }
        if (strides == null) {
            throw new IllegalArgumentException("strides is null");
        }
    }
        //gettersv1
    public int[] getShape() {
        return Arrays.copyOf(shape, shape.length);
    }
    public int rank() {
        return this.shape.length;
    }
    public int size() {
        if (this.shape.length == 0) {
            return 1;
        }
        int totalSize = 1;
        for (int dimSize : this.shape) {
            totalSize *= dimSize;
        }
        return totalSize;
    }
    @Override
    public String toString() {
        return "Tensor(shape="+Arrays.toString(shape)+")";
    }
    private static int[] calculateStrides(int[] shape) {
        if (shape == null || shape.length == 0) {
            return new int[0]; // Strides for a scalar
        }
        int[] strides = new int[shape.length];
        strides[shape.length - 1] = 1; // Stride for the last dimension is 1
        for (int i = shape.length - 2; i >= 0; i--) {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        return strides;
    }
    public static Tensor zeros(int... shape) {
        if (shape == null || shape.length == 0) {
            return new Tensor(new double[]{0.0}, new int[0], new int[0], 0);
        }

        int totalSize = 1;
        for (int dimSize : shape) {
            if (dimSize <= 0) {
                throw new IllegalArgumentException("Dimension size must be positive: " + dimSize);
            }
            totalSize *= dimSize;
        }

        double[] data = new double[totalSize];
        int[] strides = calculateStrides(shape);
        return new Tensor(data, Arrays.copyOf(shape, shape.length), strides, 0);
    }
}
