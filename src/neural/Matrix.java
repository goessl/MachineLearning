package neural;

import java.util.Random;
import java.util.function.BiFunction;
import java.util.function.DoubleFunction;

/**
 * Matrix class for storage & calculations
 * 
 * @author Sebastian Gössl
 * @version 1.2 26.03.2018
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
        set(j, i, input.get(j, i));
      }
    }
  }
  
  /**
   * Constructs a new Matrix with the content of a 2-dimensional array
   * @param array Array which contents should be copied into this matrix
   * @throws IllegalArgumentException If the input array is not rectangular
   */
  public Matrix(double[][] array) {
    this(array.length, array[0].length);
    
    for(int j=0; j<getHeight(); j++) {
      if(array[j].length != getWidth()) {
        throw new IllegalArgumentException("Input array not rectangular!");
      }
      
      for(int i=0; i<getWidth(); i++) {
        set(j, i, array[j][i]);
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
      throw new IllegalArgumentException("Dimension(s) less than 1!");
    }
    
    
    this.height = height;
    this.width = width;
    
    matrix = new double[height][width];
  }
  
  
  
  /**
   * Sets the value of a specific element
   * @param row Row index of the element
   * @param column Column index of the element
   * @param value Value to set the element to
   * @throws ArrayIndexOutOfBoundsException If the indices are smaller than 0
   *         or bigger than the width/height -1
   */
  public void set(int row, int column, double value) {
    if(row < 0 || row >= getHeight() || column < 0 || column >= getWidth()) {
      throw new ArrayIndexOutOfBoundsException("Indices out of bounds!");
    }
    
    
    matrix[row][column] = value;
  }
  
  /**
   * Returns the value of a specific element
   * @param row Row index of the element
   * @param column Column index of the element
   * @return The value of the element
   * @throws ArrayIndexOutOfBoundsException If the indices are smaller than 0
   *         or bigger than the width/height -1
   */
  public double get(int row, int column) {
    if(row < 0 || row >= getHeight() || column < 0 || column >= getWidth()) {
      throw new ArrayIndexOutOfBoundsException("Indices out of bounds!");
    }
    
    
    return matrix[row][column];
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
   * Sets every element of the matrix to the given value
   * @param value Value to set every element to
   */
  public void fill(double value) {
    for(int j=0; j<getHeight(); j++) {
      for(int i=0; i<getWidth(); i++) {
        set(j, i, value);
      }
    }
  }
  
  
  /**
   * Adds the given matrix to this matrix.
   * C = A + B
   * C[i, j] = A[i, j] + B[i, j]
   * @param matrix2 Second matrix to add (summand)
   * @return Sum of the two matricies
   * @throws IllegalArgumentException If the matricies are not
   *        of the same dimensions
   */
  public Matrix add(Matrix matrix2) {
    if(matrix2.getHeight() != getHeight()
            || matrix2.getWidth() != getWidth()) {
      throw new IllegalArgumentException("Dimensions not compatible!");
    }
    
    
    final Matrix result = new Matrix(getHeight(), getWidth());
    
    for(int j=0; j<result.getHeight(); j++) {
      for(int i=0; i<result.getWidth(); i++) {
        result.set(j, i, get(j, i) + matrix2.get(j, i));
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
   *        of the same dimensions
   */
  public Matrix subtract(Matrix matrix2) {
    if(matrix2.getHeight() != getHeight()
            || matrix2.getWidth() != getWidth()){
      throw new IllegalArgumentException("Dimensions not compatible!");
    }
    
    
    final Matrix result = new Matrix(getHeight(), getWidth());
    
    for(int j=0; j<result.getHeight(); j++) {
      for(int i=0; i<result.getWidth(); i++) {
        result.set(j, i, get(j, i) - matrix2.get(j, i));
      }
    }
    
    return result;
  }
  
  /**
   * Multiplies this matrix with the given one.
   * C = AB
   * C[i, j] = A[i, 1]*B[1, j] + ... + A[i, m]*B[m, j]
   * @param matrix2 Second matrix to multiply (factor)
   * @return Matrix product of the two matricies
   * @throws IllegalArgumentException If the second matrix' height does not
   *        equal this matrix' width
   */
  public Matrix multiply(Matrix matrix2) {
    if(matrix2.getHeight() != getWidth()) {
      throw new IllegalArgumentException("Matrix dimensions not compatible!");
    }
    
    
    final Matrix result = new Matrix(getHeight(), matrix2.getWidth());
    
    for(int j=0; j<result.getHeight(); j++) {
      for(int i=0; i<result.getWidth(); i++) {
        
        double sum = 0;
        for(int k=0; k<getWidth(); k++) {
          sum += get(j, k) * matrix2.get(k, i);
        }
        
        result.set(j, i, sum);
      }
    }
    
    return result;
  }
  
  /**
   * Multiplies this matrix with the given matrix elementwise.
   * C = A o B
   * C[i, j] = A[i, j] * B[i, j]
   * @param matrix2 Second matrix to multiply elementwise (factor)
   * @return Elementwise/Hadamard product
   * @throws IllegalArgumentException If the matricies are not
   *        of the same dimensions
   */
  public Matrix multiplyElementwise(Matrix matrix2) {
    if(matrix2.getHeight() != getHeight()
            || matrix2.getWidth() != getWidth()) {
      throw new IllegalArgumentException("Dimensions not compatible!");
    }
    
    
    final Matrix result = new Matrix(getHeight(), getWidth());
    
    for(int j=0; j<result.getHeight(); j++) {
      for(int i=0; i<result.getWidth(); i++) {
        result.set(j, i, get(j, i) * matrix2.get(j, i));
      }
    }
    
    return result;
  }
  
  /**
   * Scalar multiplies this matrix with the given factor.
   * C = b * A
   * C[i, j] = b * A[i, j]
   * @param value Scalar to multiply every element with
   * @return Scaled matrix
   */
  public Matrix multiply(double value) {
    final Matrix result = new Matrix(getHeight(), getWidth());
    
    for(int j=0; j<result.getHeight(); j++) {
      for(int i=0; i<result.getWidth(); i++) {
        result.set(j, i, value * get(j, i));
      }
    }
    
    return result;
  }
  
  /**
   * Divides this matrix by the given matrix elementwise.
   * C[i, j] = A[i, j] / B[i, j]
   * @param matrix2 Second matrix to divide by elementwise (divisor)
   * @return Elementwise quotient
   * @throws IllegalArgumentException If the matricies are not
   *        of the same dimensions
   */
  public Matrix divideElementwise(Matrix matrix2) {
    if(matrix2.getHeight() != getHeight()
            || matrix2.getWidth() != getWidth()) {
      throw new IllegalArgumentException("Dimensions not compatible!");
    }
    
    
    final Matrix result = new Matrix(getHeight(), getWidth());
    
    for(int j=0; j<result.getHeight(); j++) {
      for(int i=0; i<result.getWidth(); i++) {
        result.set(j, i, get(j, i) / matrix2.get(j, i));
      }
    }
    
    return result;
  }
  
  
  /**
   * Applies the given function on every element of the matrix.
   * B[i, j] = f(A[i, j])
   * @param function Function that gets applied on every element
   * @return Resulting matrix
   */
  public Matrix apply(DoubleFunction<Double> function) {
    final Matrix result = new Matrix(getHeight(), getWidth());
    
    for(int j=0; j<result.getHeight(); j++) {
      for(int i=0; i<result.getWidth(); i++) {
        result.set(j, i, function.apply(get(j, i)));
      }
    }
    
    return result;
  }
  
  /**
   * Applies the given function,
   * on every element of this and the given matrix,
   * and writes the output into the corresponding element of a new matrix.
   * C[i, j] = f(A[i, j], B[i, j])
   * @param matrix2
   * @param function Function that gets applied on every element
   * @return Resulting matrix
   */
  public Matrix apply(Matrix matrix2,
          BiFunction<Double, Double, Double> function) {
    final Matrix result = new Matrix(getHeight(), getWidth());
    
    for(int j=0; j<result.getHeight(); j++) {
      for(int i=0; i<result.getWidth(); i++) {
        result.set(j, i, function.apply(get(j, i), matrix2.get(j, i)));
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
        result.set(j, i, get(i, j));
      }
    }
    
    return result;
  }
  
  
  /**
   * Extracts a single row as a new Matrix
   * @param index Index of the row that should be extracted
   * @return The single row as a new Matrix
   * @throws ArrayIndexOutOfBoundsException If the index does not point
   *        to an existing row
   */
  public Matrix getRow(int index) {
    if(index < 0 || index >= getWidth()) {
      throw new ArrayIndexOutOfBoundsException("Index out of bounds!");
    }
    
    return getRows(index, index + 1);
  }
  
  /**
   * Extracts multiple rows as a new Matrix
   * @param fromIndex Index of the first row
   *        that should be extracted (inclusive)
   * @param toIndex Index of the last row that should be extracted (exclusive)
   * @return The rows as a new Matrix
   * @throws ArrayIndexOutOfBoundsException If an index does not point
   *        to an existing row
   */
  public Matrix getRows(int fromIndex, int toIndex) {
    if(fromIndex < 0 || fromIndex >= getHeight()
            || toIndex < 0 || toIndex > getHeight()) {
      throw new ArrayIndexOutOfBoundsException("Indices out of bounds!");
    }
    if(fromIndex >= toIndex) {
      throw new IllegalArgumentException("Illegal index direction!");
    }
    
    
    final Matrix result = new Matrix(toIndex - fromIndex, getWidth());
    
    for(int j=0; j<result.getHeight(); j++) {
      for(int i=0; i<result.getWidth(); i++) {
        result.set(j, i, get(fromIndex + j, i));
      }
    }
    
    return result;
  }
  
  /**
   * Appends a matrix to the bottom end of this matrix
   * @param rows Matrix to append
   * @return Merged matrix
   * @throws IllegalArgumentException If the givenn matrix
   *        is not as wide as this matrix
   */
  public Matrix appendRows(Matrix rows) {
    if(rows.getWidth() != getWidth()) {
      throw new IllegalArgumentException("Rows not compatible!");
    }
    
    
    final Matrix result =
            new Matrix(getHeight() + rows.getHeight(), getWidth());
    
    for(int j=0; j<getHeight(); j++) {
      for(int i=0; i<result.getWidth(); i++) {
        result.set(j, i, get(j, i));
      }
    }
    
    for(int j=0; j<rows.getHeight(); j++) {
      for(int i=0; i<result.getWidth(); i++) {
        result.set(getHeight() + j, i, rows.get(j, i));
      }
    }
    
    return result;
  }
  
  /**
   * Removes a single row of this matrix
   * @param index Index of the row that should be removed
   * @return Resulting matrix
   * @throws ArrayIndexOutOfBoundsException If this matrix is to small to
   *        remove a row or the index does not point to an existing row
   */
  public Matrix removeRow(int index) {
    if(getHeight() < 2) {
      throw new ArrayIndexOutOfBoundsException("Matrix to small!");
    }
    if(index < 0 || index >= getHeight()) {
      throw new ArrayIndexOutOfBoundsException("Index out of bounds!");
    }
    
    return removeRows(index, index + 1);
  }
  
  /**
   * Removes multiple rows of this matrix
   * @param fromIndex Index of the first row that should be removed (inclusive)
   * @param toIndex Index of the last row that should be removed (exclusive)
   * @return Resulting matrix
   * @throws ArrayIndexOutOfBoundsException If this matrix is to small to
   *        remove the rows or an index does not point to an existing row
   */
  public Matrix removeRows(int fromIndex, int toIndex) {
    if(getHeight() <= toIndex - fromIndex) {
      throw new ArrayIndexOutOfBoundsException("Matrix to small!");
    }
    if(fromIndex < 0 || fromIndex >= getHeight()
            || toIndex < 0 || toIndex > getHeight()) {
      throw new ArrayIndexOutOfBoundsException("Indices out of bounds!");
    }
    if(fromIndex >= toIndex) {
      throw new IllegalArgumentException("Illegal index direction!");
    }
    
    
    final Matrix result =
            new Matrix(getHeight() - (toIndex - fromIndex), getWidth());
    
    for(int j=0; j<fromIndex; j++) {
      for(int i=0; i<result.getWidth(); i++) {
        result.set(j, i, get(j, i));
      }
    }
    for(int j=fromIndex; j<result.getHeight(); j++) {
      for(int i=0; i<result.getWidth(); i++) {
        result.set(j, i, get((toIndex - fromIndex) + j, i));
      }
    }
    
    return result;
  }
  
  /**
   * Extracts a single column as a new Matrix
   * @param index Index of the column that should be extracted
   * @return The single column as a new Matrix
   * @throws ArrayIndexOutOfBoundsException If the index does not point
   *        to an existing column
   */
  public Matrix getColumn(int index) {
    if(index < 0 || index >= getHeight()) {
      throw new ArrayIndexOutOfBoundsException("Index out of bounds!");
    }
    
    return getColumns(index, index + 1);
  }
  
  /**
   * Extracts multiple columns as a new Matrix
   * @param fromIndex Index of the first column
   *        that should be extracted (inclusive)
   * @param toIndex Index of the last column
   *        that should be extracted (exclusive)
   * @return The columns as a new Matrix
   * @throws ArrayIndexOutOfBoundsException If an index does not point
   *        to an existing column
   */
  public Matrix getColumns(int fromIndex, int toIndex) {
    if(fromIndex < 0 || fromIndex >= getWidth()
            || toIndex < 0 || toIndex > getWidth()) {
      throw new ArrayIndexOutOfBoundsException("Indices out of bounds!");
    }
    if(fromIndex >= toIndex) {
      throw new IllegalArgumentException("Illegal index direction!");
    }
    
    
    final Matrix result = new Matrix(getHeight(), toIndex - fromIndex);
    
    for(int j=0; j<result.getHeight(); j++) {
      for(int i=0; i<result.getWidth(); i++) {
        result.set(j, i, get(j, fromIndex + i));
      }
    }
    
    return result;
  }
  
  /**
   * Appends a matrix to the right end of this matrix
   * @param columns Matrix to append
   * @return Merged matrix
   * @throws IllegalArgumentException If the given matrix
   *        is not as high as this matrix
   */
  public Matrix appendColumns(Matrix columns) {
    if(columns.getHeight() != getHeight()) {
      throw new IllegalArgumentException("Column not compatible!");
    }
    
    
    final Matrix result =
            new Matrix(getHeight(), getWidth() + columns.getWidth());
    
    for(int j=0; j<result.getHeight(); j++) {
      for(int i=0; i<getWidth(); i++) {
        result.set(j, i, get(j, i));
      }
      for(int i=0; i<columns.getWidth(); i++) {
        result.set(j, getWidth() + i, columns.get(j, i));
      }
    }
    
    return result;
  }
  
  /**
   * Removes a single column of this matrix
   * @param index Index of the column that should be remove
   * @return Resulting matrix
   * @throws ArrayIndexOutOfBoundsException If this matrix is to small to
   *        remove a column or the index does not point to an existing column
   */
  public Matrix removeColumn(int index) {
    if(getWidth() < 2) {
      throw new ArrayIndexOutOfBoundsException("Matrix to small!");
    }
    if(index < 0 || index >= getWidth()) {
      throw new ArrayIndexOutOfBoundsException("Index out of bounds!");
    }
    
    return removeColumns(index, index + 1);
  }
  
  /**
   * Removes multiple columns of this matrix
   * @param fromIndex Index of the first column
   *        that should be removed (inclusive)
   * @param toIndex Index of the last column that should be removed (exclusive)
   * @return Resulting matrix
   * @throws ArrayIndexOutOfBoundsException If this matrix is to small to
   *        remove the columns or an index does not point to an existing column
   */
  public Matrix removeColumns(int fromIndex, int toIndex) {
    if(getWidth() <= toIndex - fromIndex) {
      throw new ArrayIndexOutOfBoundsException("Matrix to small!");
    }
    if(fromIndex < 0 || fromIndex >= getWidth()
            || toIndex < 0 || toIndex > getWidth()) {
      throw new ArrayIndexOutOfBoundsException("Indices out of bounds!");
    }
    if(fromIndex >= toIndex) {
      throw new IllegalArgumentException("Illegal index direction!");
    }
    
    
    final Matrix result =
            new Matrix(getHeight(), getWidth() - (toIndex - fromIndex));
    
    for(int j=0; j<result.getHeight(); j++) {
      for(int i=0; i<fromIndex; i++) {
        result.set(j, i, get(j, i));
      }
      for(int i=fromIndex; i<result.getWidth(); i++) {
        result.set(j, i, get(j, (toIndex - fromIndex) + i));
      }
    }
    
    return result;
  }
  
  
  /**
   * Fills the matrix with random values
   */
  public void rand() {
    rand(new Random(), -Double.MAX_VALUE, Double.MAX_VALUE);
  }
  
  /**
   * Fills the matrix with random values,
   * from minimum (inclusive) to maximum (exclusive),
   * given by the random number generator
   * @param rand Random number generator
   * @param minimum Minimum value (inclusive)
   * @param maximum Maximum value (exclusive)
   */
  public void rand(Random rand, double minimum, double maximum) {
    final double range = maximum - minimum;
    
    for(int j=0; j<getHeight(); j++) {
      for(int i=0; i<getWidth(); i++) {
        set(j, i, range*rand.nextDouble() + minimum);
      }
    }
  }
  
  
  
  /**
   * Copies the content of the matrix into a 2-dimensional array
   * @return Array copy
   */
  public double[][] toArray() {
    final double[][] array = new double[getHeight()][getWidth()];
    
    for(int j=0; j<array.length; j++) {
      for(int i=0; i<array[j].length; i++) {
        array[j][i] = get(j, i);
      }
    }
    
    return array;
  }
  
  
  @Override
  public String toString() {
    final StringBuilder builder = new StringBuilder("[[").append(get(0, 0));
    
    for(int i=1; i<getWidth(); i++) {
      builder.append(", ").append(get(0, i));
    }
    builder.append("]");
    
    for(int j=1; j<getHeight(); j++) {
      builder.append("\n [").append(get(j, 0));
      for(int i=1; i<getWidth(); i++) {
        builder.append(", ").append(get(j, i));
      }
      builder.append("]");
    }
    builder.append("]");
    
    
    return builder.toString();
  }
  
  
  
  
  public static void main(String[] args) {
    double scalar = 2;
    Matrix matrix1 = new Matrix(new double[][] {
      {1, 2, 3},
      {4, 42, 6},
      {7, 8, 9}});
    Matrix matrix2 = new Matrix(new double[][] {
      {1, 4, 7},
      {2, 5, 8},
      {3, 6, 9}});
    
    
    
    System.out.println("Scalar:");
    System.out.println(scalar + "\n");
    System.out.println("Matrix1:");
    System.out.println(matrix1 + "\n");
    System.out.println("Matrix2:");
    System.out.println(matrix2 + "\n");
    System.out.println();
    
    
    System.out.println("Set [1, 1] to 5:");
    matrix1.set(1, 1, 5);
    System.out.println(matrix1 + "\n");
    System.out.println("Get [1, 1]:" + matrix1.get(1, 1));
    System.out.println("Height: " + matrix1.getHeight());
    System.out.println("Width: " + matrix1.getWidth() + "\n\n");
    
    
    System.out.println("Addition (Matrix1 & Matrix2):");
    System.out.println(matrix1.add(matrix2) + "\n");
    System.out.println("Subtraction (Matrix1 & Matrix2):");
    System.out.println(matrix1.subtract(matrix2) + "\n");
    System.out.println("Matrix multiplication:");
    System.out.println(matrix1.multiply(matrix2) + "\n");
    System.out.println("Elementwise multiplication:");
    System.out.println(matrix1.multiplyElementwise(matrix2) + "\n");
    System.out.println("Scalar multiplication:");
    System.out.println(matrix1.multiply(scalar) + "\n");
    System.out.println("Elementwise division:");
    System.out.println(matrix1.divideElementwise(matrix2) + "\n");
    
    
    System.out.println("Applying sine:");
    System.out.println(matrix1.apply(x -> Math.sin(x)) + "\n\n");
    
    
    System.out.println("Transpose:");
    System.out.println(matrix1.transpose() + "\n\n");
    
    
    System.out.println("Get row 1:");
    System.out.println(matrix1.getRow(1) + "\n");
    System.out.println("Get rows 1 & 2:");
    System.out.println(matrix1.getRows(1, 3) + "\n");
    System.out.println("Append rows:");
    System.out.println(matrix1.appendRows(matrix2) + "\n");
    System.out.println("Remove row 1:");
    System.out.println(matrix1.removeRow(1) + "\n");
    System.out.println("Remove rows 0 & 1:");
    System.out.println(matrix1.removeRows(0, 2) + "\n\n");
    
    System.out.println("Get column 1:");
    System.out.println(matrix1.getColumn(1) + "\n");
    System.out.println("Get columns 1 & 2:");
    System.out.println(matrix1.getColumns(1, 3) + "\n");
    System.out.println("Append columns:");
    System.out.println(matrix1.appendColumns(matrix2) + "\n");
    System.out.println("Remove column 1:");
    System.out.println(matrix1.removeColumn(1) + "\n");
    System.out.println("Remove columns 0 & 1:");
    System.out.println(matrix1.removeColumns(0, 2) + "\n");
    
    
    System.out.println("Randomize within [-1, 1[:");
    matrix1.rand(new Random(), -1, 1);
    System.out.println(matrix1 + "\n");
  }
}
