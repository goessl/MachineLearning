package neural;

import java.util.Arrays;
import java.util.Random;

/**
 * Matrix class for storage & calculations
 * 
 * @author Sebastian Gössl
 * @version 1.1 22.02.2018
 */
public class Matrix {
  
  /** Matrix dimensions */
  private final int height, width;
  /** Matrix elements */
  private final double[][] matrix;
  
  
  
  /**
   * Constructs a new copy of an existing matrix
   * @param input Matrix to copy
   */
  public Matrix(Matrix input) {
    this(input.getHeight(), input.getWidth());
    
    for(int j=0; j<getHeight(); j++) {
      for(int i=0; i<getWidth(); i++) {
        set(input.get(i, j), i, j);
      }
    }
  }
  
  /**
   * Constructs a new Matrix with the content of a 2-dimensional array
   * @param array Array which contents should be copied into this matrix
   * @throws IllegalArgumentException If the input array is not possible
   *         to convert to a matrix
   */
  public Matrix(double[][] array) {
    this(array.length, array[0].length);
    
    for(int j=0; j<getHeight(); j++) {
      if(array[j].length != getWidth()) {
        throw new IllegalArgumentException("Input array not rectangular!");
      }
      
      for(int i=0; i<getWidth(); i++) {
        set(array[j][i], i, j);
      }
    }
  }
  
  /**
   * Constructs a new matrix with the given dimensions
   * @param height Number of rows
   * @param width Number of columns
   * @throws IllegalArgumentException If the dimensions are
   *         impossible to implement
   */
  public Matrix(int height, int width) {
    if(height < 1 || width < 1) {
      throw new IllegalArgumentException("Dimensions less than 1!");
    }
    
    
    this.height = height;
    this.width = width;
    
    matrix = new double[height][width];
  }
  
  
  
  /**
   * Sets the value of a specific element
   * @param value Value to set the element to
   * @param x X-Coordinate of the element
   * @param y Y-Coordinate of the element
   * @throws IllegalArgumentException If the indices are smaller than 0
   *         or bigger than the width/height -1
   */
  public void set(double value, int x, int y) {
    if(x < 0 || x >= getWidth() || y < 0 || y >= getHeight()) {
      throw new IllegalArgumentException("Indices out of bounds!");
    }
    
    
    matrix[y][x] = value;
  }
  
  /**
   * Returns the value of a specific element
   * @param x X-Coordinate of the position to read from
   * @param y Y-Coordinate of the position to read from
   * @return Value of the position
   * @throws IllegalArgumentException If the indices are smaller than 0
   *         or bigger than the width/height -1
   */
  public double get(int x, int y) {
    if(x < 0 || x >= getWidth() || y < 0 || y >= getHeight()) {
      throw new IllegalArgumentException("Indices out of bounds!");
    }
    
    
    return matrix[y][x];
  }
  
  /**
   * Returns the height (number of rows) of the matrix
   * @return Height of the matrix
   */
  public int getHeight() {
    return height;
  }
  
  /**
   * Returns the width (number of columns) of the matrix
   * @return Width of the matrix
   */
  public int getWidth() {
    return width;
  }
  
  
  /**
   * Adds the given matrix to this matrix.
   * C = A + B
   * C[i, j] = A[i, j] + B[i, j]
   * @param matrix2 Second matrix (summand)
   * @return Sum of the two matricies
   * @throws IllegalArgumentException If the matricies are not
   *         of the same dimensions
   */
  public Matrix add(Matrix matrix2) {
    if(matrix2.getHeight() != getHeight()
            || matrix2.getWidth() != getWidth()) {
      throw new IllegalArgumentException("Dimensions not compatible!");
    }
    
    
    final Matrix result = new Matrix(getHeight(), getWidth());
    
    for(int j=0; j<getHeight(); j++) {
      for(int i=0; i<getWidth(); i++) {
        result.set(get(i, j) + matrix2.get(i, j), i, j);
      }
    }
    
    return result;
  }
  
  /**
   * Subtracts the given matrix from this matrix.
   * C = A - B
   * C[i, j] = A[i, j] - B[i, j]
   * @param matrix2 Second matrix to subtract (subtrahend)
   * @return Difference of the two matricies
   * @throws IllegalArgumentException If the matricies are not
   *         of the same dimensions
   */
  public Matrix subtract(Matrix matrix2) {
    if(matrix2.getHeight() != getHeight()
            || matrix2.getWidth() != getWidth()){
      throw new IllegalArgumentException("Dimensions not compatible!");
    }
    
    
    final Matrix result = new Matrix(getHeight(), getWidth());
    
    for(int j=0; j<getHeight(); j++) {
      for(int i=0; i<getWidth(); i++) {
        result.set(get(i, j) - matrix2.get(i, j), i, j);
      }
    }
    
    return result;
  }
  
  /**
   * Scalar multiplies this matrix with the given factor.
   * C = b * A
   * C[i, j] = b * A[i, j]
   * @param factor Scalar to multiply every element with
   * @return Scaled matrix
   */
  public Matrix multiply(double factor) {
    final Matrix result = new Matrix(getHeight(), getWidth());
    
    for(int j=0; j<getHeight(); j++) {
      for(int i=0; i<getWidth(); i++) {
        result.set(factor * get(i, j), i, j);
      }
    }
    
    return result;
  }
  
  /**
   * Multiplies this matrix with the given matrix.
   * C = A o B
   * C[i, j] = A[i, j] * B[i, j]
   * @param matrix2 Second matrix to multiply elementwise (factor)
   * @return Elementwise/Hadamard product
   * @throws IllegalArgumentException If the matricies are not
   *         of the same dimensions
   */
  public Matrix multiplyElementwise(Matrix matrix2) {
    if(matrix2.getHeight() != getHeight()
            || matrix2.getWidth() != getWidth()) {
      throw new IllegalArgumentException("Dimensions not compatible!");
    }
    
    
    final Matrix result = new Matrix(getHeight(), getWidth());
    
    for(int j=0; j<getHeight(); j++) {
      for(int i=0; i<getWidth(); i++) {
        result.set(get(i, j) * matrix2.get(i, j), i, j);
      }
    }
    
    return result;
  }
  
  /**
   * Multiplies this matrix with the given one.
   * C = AB
   * C[i, j] = A[i, 1]*B[1, j] + ... + A[i, m]*B[m, j]
   * @param matrix2 Second matrix to multiply (factor)
   * @return Matrix product
   * @throws IllegalArgumentException If the second matrix is not as high
   *         as this matrix wide
   */
  public Matrix multiply(Matrix matrix2) {
    if(matrix2.getHeight() != getWidth()) {
      throw new IllegalArgumentException("Matrix dimensions not compatible!");
    }
    
    
    final Matrix result = new Matrix(getHeight(), matrix2.getWidth());
    
    for(int j=0; j<result.getHeight(); j++) {
      for(int i=0; i<result.getWidth(); i++) {
        result.set(0, i, j);
        
        for(int k=0; k<getWidth(); k++) {
          result.set(result.get(i, j) + get(k, j)*matrix2.get(i, k), i, j);
        }
      }
    }
    
    return result;
  }
  
  /**
   * Transposes this matrix
   * @return Transpose
   */
  public Matrix transpose() {
    final Matrix result = new Matrix(getWidth(), getHeight());
    
    for(int j=0; j<result.getHeight(); j++) {
      for(int i=0; i<result.getWidth(); i++) {
        result.set(get(j, i), i, j);
      }
    }
    
    return result;
  }
  
  
  /**
   * Appends a row to the end of this matrix
   * @param row Row to append
   * @return New matrix
   * @throws IllegalArgumentException If the row is not as wide as this matrix
   */
  public Matrix appendRow(double[] row) {
    if(row.length != getWidth()) {
      throw new IllegalArgumentException("Row not compatible!");
    }
    
    
    final Matrix result = new Matrix(getHeight() + 1, getWidth());
    
    for(int j=0; j<getHeight(); j++) {
      for(int i=0; i<getWidth(); i++) {
        result.set(get(i, j), i, j);
      }
    }
    
    for(int i=0; i<getWidth(); i++) {
      result.set(row[i], i, getHeight());
    }
    
    return result;
  }
  
  /**
   * Appends a matrix to the bottom end of this matrix
   * @param row Second matrix to append
   * @return New matrix
   * @throws IllegalArgumentException If the given nmatrix
   *         is not as wide as this matrix
   */
  public Matrix appendRow(Matrix row) {
    if(row.getWidth() != getWidth()) {
      throw new IllegalArgumentException("Row not compatible!");
    }
    
    
    final Matrix result = new Matrix(getHeight() + row.getHeight(), getWidth());
    
    for(int j=0; j<getHeight(); j++) {
      for(int i=0; i<getWidth(); i++) {
        result.set(get(i, j), i, j);
      }
    }
    
    for(int j=0; j<row.getHeight(); j++) {
      for(int i=0; i<getWidth(); i++) {
        result.set(row.get(i, j), i, getHeight()+j);
      }
    }
    
    return result;
  }
  
  /**
   * Removes one row of this matrix
   * @param index Index of the row to remove
   * @return New matrix
   * @throws ArrayIndexOutOfBoundsException If this matrix is to small to
   *         remove a row or the index does not point to an existing row
   */
  public Matrix removeRow(int index) {
    if(getHeight() <= 1) {
      throw new ArrayIndexOutOfBoundsException("Matrix to small!");
    }
    if(index < 0 || index > getHeight() - 1) {
      throw new ArrayIndexOutOfBoundsException("Index out of bounds!");
    }
    
    
    final Matrix result = new Matrix(getHeight() - 1, getWidth());
    
    //First half
    for(int j=0; j<index; j++) {
      for(int i=0; i<getWidth(); i++) {
        result.set(get(i, j), i, j);
      }
    }
    //Second half
    for(int j=index+1; j<getHeight(); j++) {
      for(int i=0; i<getWidth(); i++) {
        result.set(get(i, j), i, j-1);
      }
    }
    
    return result;
  }
  
  /**
   * Appends a column to the right end of this matrix
   * @param column Column to append
   * @return New matrix
   * @throws IllegalArgumentException If the column is not as high as the matrix
   */
  public Matrix appendColumn(double[] column) {
    if(column.length != getHeight()) {
      throw new IllegalArgumentException("Column not compatible!");
    }
    
    
    final Matrix result = new Matrix(getHeight(), getWidth() + 1);
    
    for(int j=0; j<getHeight(); j++) {
      for(int i=0; i<getWidth(); i++) {
        result.set(get(i, j), i, j);
      }
      result.set(column[j], getWidth(), j);
    }
    
    return result;
  }
  
  /**
   * Appends a matrix to the right end of this matrix
   * @param column Second matrix to append
   * @return New matrix
   * @throws IllegalArgumentException If the given nmatrix
   *         is not as high as this matrix
   */
  public Matrix appendColumn(Matrix column) {
    if(column.getHeight() != getHeight()) {
      throw new IllegalArgumentException("Column not compatible!");
    }
    
    
    final Matrix result =
            new Matrix(getHeight(), getWidth() + column.getWidth());
    
    for(int j=0; j<getHeight(); j++) {
      for(int i=0; i<getWidth(); i++) {
        result.set(get(i, j), i, j);
      }
      for(int i=0; i<column.getWidth(); i++) {
        result.set(column.get(i, j), getWidth()+i, j);
      }
    }
    
    return result;
  }
  
  /**
   * Removes one column of this matrix
   * @param index Index of the column to remove
   * @return New matrix
   * @throws ArrayIndexOutOfBoundsException If this matrix is to small to
   *         remove a column or the index does not point to an existing column
   */
  public Matrix removeColumn(int index) {
    if(getWidth() <= 1) {
      throw new ArrayIndexOutOfBoundsException("Matrix to small!");
    }
    if(index < 0 || index > getWidth() - 1) {
      throw new ArrayIndexOutOfBoundsException("Index out of bounds!");
    }
    
    
    final Matrix result = new Matrix(getHeight(), getWidth() - 1);
    
    for(int j=0; j<getHeight(); j++) {
      //First half
      for(int i=0; i<index; i++) {
        result.set(get(i, j), i, j);
      }
      //Second half
      for(int i=index+1; i<getWidth(); i++) {
        result.set(get(i, j), i-1, j);
      }
    }
    
    return result;
  }
  
  
  /**
   * Fills the matrix with random double values
   */
  public void rand() {
    rand(new Random(), -Double.MAX_VALUE, Double.MAX_VALUE);
  }
  
  /**
   * Fills the matrix with random double values, based on a seed
   * @param seed Seed for the random number generator
   */
  public void rand(long seed) {
    rand(new Random(seed), -Double.MAX_VALUE, Double.MAX_VALUE);
  }
  
  /**
   * Fills the matrix with random double values between the two given values
   * @param minimum Minimum value
   * @param maximum Maximum value
   */
  public void rand(double minimum, double maximum) {
    rand(new Random(), minimum, maximum);
  }
  
  /**
   * Fills the matrix with random double values, based on a seed,
   * between the two given values
   * @param seed Seed for the random number generator
   * @param minimum Minimum value
   * @param maximum Maximum value
   */
  public void rand(long seed, double minimum, double maximum) {
    rand(new Random(seed), minimum, maximum);
  }
  
  /**
   * Fills the matrix with random double values, between the two given values,
   * from the given random numbers generator
   * @param rand Random number generator
   * @param minimum Minimum value
   * @param maximum Maximum value
   */
  public void rand(Random rand, double minimum, double maximum) {
    final double range = maximum - minimum;
    
    for(int j=0; j<getHeight(); j++) {
      for(int i=0; i<getWidth(); i++) {
        set(range*rand.nextDouble() + minimum, i, j);
      }
    }
  }
  
  
  
  /**
   * Copies the content of the matrix into a 2-dimensional array
   * @return Array copy
   */
  public double[][] toArray() {
    final double[][] array = new double[getHeight()][getWidth()];
    
    for(int j=0; j<getHeight(); j++) {
      for(int i=0; i<getWidth(); i++) {
        array[j][i] = get(i, j);
      }
    }
    
    return array;
  }
  
  
  @Override
  public String toString() {
    final StringBuilder builder = new StringBuilder("[[").append(get(0, 0));
    
    //First row
    for(int i=1; i<getWidth(); i++) {
      builder.append(", ").append(get(i, 0));
    }
    builder.append("]");
    
    //Rest
    for(int j=1; j<getHeight(); j++) {
      builder.append("\n [").append(get(0, j));
      for(int i=1; i<getWidth(); i++) {
        builder.append(", ").append(get(i, j));
      }
      builder.append("]");
    }
    builder.append("]");
    
    
    return builder.toString();
  }
  
  
  
  
  public static void main(String[] args) {
    double scalar = 2;
    double[] array = new double[]{1, 3, 9};
    Matrix matrix1 = new Matrix(new double[][] {
      {1, 2, 3},
      {4, 5, 6},
      {7, 8, 9}});
    Matrix matrix2 = new Matrix(new double[][] {
      {1, 4, 7},
      {2, 5, 8},
      {3, 6, 9}});
    
    
    System.out.println("Scalar:");
    System.out.println(scalar + "\n");
    System.out.println("Array:");
    System.out.println(Arrays.toString(array) + "\n");
    System.out.println("Matrix1:");
    System.out.println(matrix1 + "\n");
    System.out.println("Matrix2:");
    System.out.println(matrix2 + "\n");
    System.out.println();
    
    
    System.out.println("Addition:");
    System.out.println(matrix1.add(matrix2) + "\n");
    System.out.println("Subtraction:");
    System.out.println(matrix1.add(matrix2) + "\n");
    System.out.println("Scalation:");
    System.out.println(matrix1.multiply(scalar) + "\n");
    System.out.println("Elementwise multiplication:");
    System.out.println(matrix1.multiplyElementwise(matrix2) + "\n");
    System.out.println("Matrix multiplication:");
    System.out.println(matrix1.multiply(matrix2) + "\n");
    System.out.println("Transposition:");
    System.out.println(matrix1.transpose() + "\n");
    
    System.out.println("Appending row:");
    System.out.println(matrix1.appendRow(array) + "\n");
    System.out.println("Appending row:");
    System.out.println(matrix1.appendRow(matrix2) + "\n");
    System.out.println("Removing row:");
    System.out.println(matrix1.removeRow(1) + "\n");
    System.out.println("Appending column:");
    System.out.println(matrix1.appendColumn(array) + "\n");
    System.out.println("Appending column:");
    System.out.println(matrix1.appendColumn(matrix2) + "\n");
    System.out.println("Remove column:");
    System.out.println(matrix1.removeColumn(1) + "\n");
  }
}
