package neural;

/**
 * Matrix for neural network
 * 
 * @author Sebastian Gössl
 * @version 1.0 22.07.2017
 */
public class Matrix {
  
  /** Matrix dimensions */
  private final int height, width;
  /** Matrix itself */
  private double[][] matrix;
  
  
  
  /**
   * Creates a new matrix with the given dimensions
   * @param height Matrix height = number of rows
   * @param width Matrix width = number of columns
   */
  public Matrix(int height, int width) {
    this.height = height;
    this.width = width;
    
    matrix = new double[height][width];
  }
  
  /**
   * Creates a new Matrix with the content of a 2-dimensional array
   * @param input Array which contents should be copied into this matrix
   */
  public Matrix(double[][] input) {
    this(input.length, input[0].length);
    
    for(int j=0; j<height; j++) {
      for(int i=0; i<width; i++) {
        matrix[j][i] = input[j][i];
      }
    }
  }
  
  /**
   * Creates a new copy of an existing matrix
   * @param input Matrix to copy
   */
  public Matrix(Matrix input) {
    this(input.getHeight(), input.getWidth());
    
    for(int j=0; j<height; j++) {
      for(int i=0; i<width; i++) {
        matrix[j][i] = input.get(i, j);
      }
    }
  }
  
  
  
  /**
   * Sets a specific position to a specific value
   * @param value Value to write into the matrix
   * @param x X-Coordinate of the position to write to
   * @param y Y-Coordinate of the position to write to
   */
  public void set(double value, int x, int y) {
    matrix[y][x] = value;
  }
  
  /**
   * Returns the value of the given position
   * @param x X-Coordinate of the position to read from
   * @param y Y-Coordinate of the position to read from
   * @return Value of the position
   */
  public double get(int x, int y) {
    return matrix[y][x];
  }
  
  /**
   * Returns the height (number of rows)
   * @return Height
   */
  public int getHeight() {
    return height;
  }
  
  /**
   * Returns the width (number of columns)
   * @return Width
   */
  public int getWidth() {
    return width;
  }
  
  
  
  /**
   * Adds to matricies elementwise.
   * C(x, y) = A(x, y) + B(x, y)
   * @param matrix2 Second matrix to add
   * @return Summ of the 2 matricies
   */
  public Matrix add(Matrix matrix2) {
    Matrix result = new Matrix(height, width);
    
    for(int j=0; j<height; j++) {
      for(int i=0; i<width; i++) {
        result.set(get(i, j) + matrix2.get(i, j),
                i, j);
      }
    }
    
    return result;
  }
  
  /**
   * Subtracts the second matrix from this elementwise.
   * C(x, y) = A(x, y) - B(x, y)
   * @param matrix2 Second matrix to add
   * @return Summ of the 2 matricies
   */
  public Matrix subtract(Matrix matrix2) {
    Matrix result = new Matrix(height, width);
    
    for(int j=0; j<height; j++) {
      for(int i=0; i<width; i++) {
        result.set(get(i, j) - matrix2.get(i, j),
                i, j);
      }
    }
    
    return result;
  }
  
  /**
   * Multiplies/Scales every element with the given factor.
   * C(x, y) = factor * A(x, y)
   * @param factor Scalar to multiply every element with
   * @return Scaled matrix
   */
  public Matrix multiply(double factor) {
    Matrix result = new Matrix(height, width);
    
    for(int j=0; j<height; j++) {
      for(int i=0; i<width; i++) {
        result.set(factor * get(i, j),
                i, j);
      }
    }
    
    return result;
  }
  
  /**
   * Multiplies two matricies elemtwise.
   * C(x, y) = A(x, y) * B(x, y)
   * @param matrix2 Second matrix to multiply with
   * @return Dot product
   */
  public Matrix dot(Matrix matrix2) {
    Matrix result = new Matrix(height, width);
    
    for(int j=0; j<height; j++) {
      for(int i=0; i<width; i++) {
        result.set(get(i, j) * matrix2.get(i, j),
                i, j);
      }
    }
    
    return result;
  }
  
  /**
   * Multiplies two matricies.
   * C = AB
   * @param matrix2 Matrix to multiply with
   * @return Matrix product
   */
  public Matrix multiply(Matrix matrix2) {
    Matrix result = new Matrix(height, matrix2.getWidth());
    
    for(int j=0; j<result.height; j++) {
      for(int i=0; i<result.width; i++) {
        result.set(0, i, j);
        
        for(int k=0; k<width; k++) {
          result.set(result.get(i, j) + get(k, j)*matrix2.get(i, k),
                  i, j);
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
    Matrix result = new Matrix(width, height);
    
    for(int j=0; j<result.height; j++) {
      for(int i=0; i<result.width; i++) {
        result.set(get(j, i), i, j);
      }
    }
    
    return result;
  }
  
  
  
  /**
   * Checks every element if there is a positive infinity within the matrix
   * @return If this matrix contains a positive infinity
   */
  public boolean containsPositiveIninity() {
    for(int j=0; j<height; j++) {
      for(int i=0; i<width; i++) {
        if(get(i, j) == Double.POSITIVE_INFINITY) {
          return true;
        }
      }
    }
    
    return false;
  }
  
  /**
   * Checks every element if there is a negative infinity within the matrix
   * @return If this matrix contains a negative infinity
   */
  public boolean containsNegativeIninity() {
    for(int j=0; j<height; j++) {
      for(int i=0; i<width; i++) {
        if(get(i, j) == Double.NEGATIVE_INFINITY) {
          return true;
        }
      }
    }
    
    return false;
  }
  
  /**
   * Checks every element within the matrix if one is not a number (NaN)
   * @return If this matrix contains a NaN
   */
  public boolean containsNAN()
  {
    for(int j=0; j<height; j++) {
      for(int i=0; i<width; i++) {
        if(Double.isNaN(matrix[j][i])) {
          return true;
        }
      }
    }
    
    return false;
  }
  
  
  /**
   * Appends a row to the end of the matrix
   * @param row Row to append
   * @return New matrix
   */
  public Matrix appendRow(double[] row) {
    if(row.length != width) {
      throw new IllegalArgumentException("Row of wrong size");
    }
    
    Matrix result = new Matrix(height + 1, width);
    
    for(int j=0; j<height; j++) {
      for(int i=0; i<width; i++) {
        result.set(matrix[j][i], i, j);
      }
    }
    
    for(int i=0; i<width; i++) {
      result.set(row[i], i, height);
    }
    
    return result;
  }
  
  
  /**
   * Copies the content of the matrix into a 2-dimensional array
   * @return Array copy
   */
  public double[][] toArray() {
    double[][] array = new double[height][width];
    
    for(int j=0; j<height; j++) {
      for(int i=0; i<width; i++) {
        array[j][i] = get(i, j);
      }
    }
    
    return array;
  }
  
  
  
  @Override
  public String toString() {
    StringBuilder builder = new StringBuilder("[[").append(get(0, 0));
    
    //First row
    for(int i=1; i<width; i++) {
      builder.append(", ").append(get(i, 0));
    }
    builder.append("]\n");
    
    //Rows in between
    for(int j=1; j<height-1; j++) {
      builder.append(" [").append(get(0, j));
      for(int i=1; i<width; i++) {
        builder.append(", ").append(get(i, j));
      }
      builder.append("]\n");
    }
    
    //Last row
    builder.append(" [").append(get(0, height-1));
    for(int i=1; i<width; i++) {
      builder.append(", ").append(get(i, height-1));
    }
    builder.append("]]\n");
    
    
    return builder.toString();
  }
  
  
  
  
  public static void main(String[] args) {
    double scalar = 2;
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
    System.out.println("Matrix1:");
    System.out.println(matrix1 + "\n");
    System.out.println("Matrix2:");
    System.out.println(matrix2 + "\n");
    
    
    System.out.println("Sum:");
    System.out.println(matrix1.add(matrix2) + "\n");
    
    System.out.println("Scaled matrix:");
    System.out.println(matrix1.multiply(scalar) + "\n");
    
    System.out.println("Dot product:");
    System.out.println(matrix1.dot(matrix2) + "\n");
    
    System.out.println("Product:");
    System.out.println(matrix1.multiply(matrix2) + "\n");
    
    System.out.println("Transpose:");
    System.out.println(matrix1.transpose());
  }
}