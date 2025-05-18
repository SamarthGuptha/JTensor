package dev.samarthguptha.jtensor.core;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*; // For assertions like assertEquals, assertArrayEquals

import java.util.Arrays;

class TensorTest {

    @Test
    void testZerosCreation() {
        Tensor t = Tensor.zeros(2, 3);
        assertArrayEquals(new int[]{2, 3}, t.getShape(), "Shape should be [2, 3]");
        assertEquals(2, t.rank(), "Rank should be 2");
        assertEquals(6, t.size(), "Size should be 6");
    }

    @Test
    void testScalarZerosCreation() {
        Tensor scalar = Tensor.zeros();
        assertArrayEquals(new int[0], scalar.getShape(), "Shape of scalar should be empty array");
        assertEquals(0, scalar.rank(), "Rank of scalar should be 0");
        assertEquals(1, scalar.size(), "Size of scalar should be 1");
    }

    @Test
    void testToString() {
        Tensor t = Tensor.zeros(2, 3);
        assertEquals("Tensor(shape=[2, 3])", t.toString());
    }

    @Test
    void testZerosThrowsExceptionForInvalidDimension() {
        assertThrows(IllegalArgumentException.class, () -> {
            Tensor.zeros(2, 0, 3);
        }, "Should throw IllegalArgumentException for zero dimension size");

        assertThrows(IllegalArgumentException.class, () -> {
            Tensor.zeros(2, -1, 3);
        }, "Should throw IllegalArgumentException for negative dimension size");
    }
}