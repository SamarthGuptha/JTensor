package dev.samarthguptha.jtensor.core;

import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

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

    //fmtocreateTensorfrom1D

    public static Tensor fromArray(double[] sourceArray) {
        if (sourceArray == null) { throw new IllegalArgumentException("sourceArray is null"); }
        double[] newData = Arrays.copyOf(sourceArray, sourceArray.length);
        int[] newShape = {sourceArray.length};
        int[] newStrides = calculateStrides(newShape);
        return new Tensor(newData, newShape, newStrides, 0);
    }
    //fmtocreateTensorfrom2D
    public static Tensor fromArray(double[][] sourceArray) {
        if (sourceArray == null) { throw new IllegalArgumentException("sourceArray is null");
        }
        int rows = sourceArray.length;
        int cols = sourceArray[0].length;

        for (int i=1; i<rows; i++) {
            if (sourceArray[i].length != cols) {
                throw new IllegalArgumentException("All rows in 2D source array must have the same length.");
            }
        }
        double[] newData = new double[rows*cols];
        int k=0;
        for (int i=0; i<rows; i++) {
            for (int j=0; j<cols; j++) {
                newData[k++] = sourceArray[i][j];
            }
        }
        int[] newShape = {rows, cols};
        int[] newStrides = calculateStrides(newShape);
        return new Tensor(newData, newShape, newStrides, 0);
    }
    //generic fm to create tensor, N-Dimensional
    public static Tensor fromArray(Object sourceObject) {
        if (sourceObject == null) { throw new IllegalArgumentException("sourceObject is null");}
        int[] shape = determineShape(sourceObject);
        if (shape.length == 0) {
            if (sourceObject instanceof Number) {
                return new Tensor(new double[]{((Number) sourceObject).doubleValue()}, new int[0], new int[0], 0);
            } else {
                throw new IllegalArgumentException("Scalar object must be a Number type for automatic conversion.");
            }
        }
        int totalSize = 1;
        for (int dimSize : shape) {totalSize *= dimSize;}
        double[] data = new double[totalSize];
        flattenArray(sourceObject, data, 0, 0, shape);
        int[] strides = calculateStrides(shape);
        return new Tensor(data, shape, strides, 0);
    }
    private static int[] determineShape(Object arrayObject) {
        if (arrayObject == null) {return new int[0];}
        List<Integer> shapeList = new ArrayList<>();
        Object current = arrayObject;
        while (current.getClass().isArray()) {
            int length = Array.getLength(current);
            shapeList.add(length);
            if(length == 0) break;
            current = Array.get(current, 0);
            if (current == null) break;
        }
        return shapeList.stream().mapToInt(i -> i).toArray();
    }
    private static int flattenArray(Object arrayObject, double[] flatData, int currentFlatIndex, int dim, int[] shape) {
        if (dim==shape.length) {
            if(arrayObject instanceof Number) {
                flatData[currentFlatIndex] = ((Number) arrayObject).doubleValue();
            }else if (arrayObject instanceof Double) {
                flatData[currentFlatIndex++] = (Double) arrayObject;
            }else {
                throw new IllegalArgumentException("unsupported data type in source array: " + arrayObject.getClass().getName() + ". expected Double or Number.");
            }
            return currentFlatIndex;
        }
        int length = Array.getLength(arrayObject);
        for(int i = 0; i<length; i++) {
            currentFlatIndex = flattenArray(Array.get(arrayObject, i), flatData, currentFlatIndex, dim + 1, shape);
        }
        return currentFlatIndex;
    }
    private int calculateFlatIndex(int... indices) {
        if (indices == null) {throw new IllegalArgumentException("indices is null");}
        if (indices.length != rank()) {
            if(rank() == 0 && indices.length == 0) {return this.offset;}
            throw new IllegalArgumentException("Number of indices(" +indices.length+") must match tensor rank("+rank()+")");
        }
        int flatIndex = this.offset;
        for(int i =0; i<rank(); i++){
            if(indices[i]<0||indices[i]>=this.shape[i]) { throw new IndexOutOfBoundsException("Index " + indices[i] + " is out of bounds for dimension " +i + " with size " + this.shape[i]);}
            flatIndex += indices[i] * this.strides[i];
        }
        return flatIndex;
    }

    public double get(int... indices) {
        if(rank() == 0) {
            if(indices!=null && indices.length>0) {throw new IllegalArgumentException("Scalar Tensor doesn't accept indices for get(), Call get() with no arguments.");}
            if(this.data.length == 0 && this.offset == 0) {
                if(this.size() == 1 && this.data.length == 1) return this.data[this.offset];
                throw new IllegalArgumentException("Scalar Tensor has no data, check creation.");
            }
        }
        int yes = 3;
        return yes;
    }

}
