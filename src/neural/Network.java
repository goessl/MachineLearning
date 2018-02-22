package neural;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.Arrays;
import java.util.Random;

/**
 * Neural network
 * 
 * @author Sebastian Gössl
 * @version 1.1 22.02.2018
 */
public class Network {
  
  /**
   * Activation functions
   */
  public enum ActivationFunction {
    NONE, TANH, SIGMOID, RELU, SOFTPLUS, RELU_LEAKY;
    
    private static final double RELU_LEAKY_LEAKAGE = 0.01;
    
    private final static String[] name = {
      "None", "Hyperbolic tangent", "Sigmoid", "Rectified linear unit",
      "SoftPlus", "Leaky rectified linear unit"
    };
    
    
    /**
     * Applies itself to the input
     * @param input The input for the function
     * @return The output of the function
     */
    public double activate(double input) {
      switch(this) {
        case TANH:
          return Math.tanh(input);
        
        case SIGMOID:
          return 1 / (1 + Math.exp(-input));
        
        case RELU:
          if(input >= 0) {
            return input;
          } else {
            return 0;
          }
        
        case SOFTPLUS:
          return Math.log(1 + Math.exp(input));
        
        case RELU_LEAKY:
          if(input >= 0) {
            return  input;
          } else {
            return RELU_LEAKY_LEAKAGE * input;
          }
        
        
        default:
        case NONE:
          return input;
      }
    }
    
    /**
     * Applies it self's derivative to the input
     * @param input The input for the function
     * @return The output of the function's derivative
     */
    public double activatePrime(double input) {
      switch(this) {
        case TANH:
          return 1 - Math.tanh(input) * Math.tanh(input);
        
        case SIGMOID:
          return Math.exp(-input) / ((1 + Math.exp(-input)) * (1 + Math.exp(-input)));
        
        case RELU:
          if(input >= 0) {
            return 1;
          } else {
            return 0;
          }
        
        case SOFTPLUS:
          return 1 / (1 + Math.exp(-input));
        
        case RELU_LEAKY:
          if(input >= 0) {
            return  1;
          } else {
            return RELU_LEAKY_LEAKAGE;
          }
        
        
        default:
        case NONE:
          return input;
      }
    }
    
    
    
    @Override
    public String toString() {
      return name[ordinal()];
    }
  }
  
  
  
  /** Number of input neurons */
  private final int numberOfInputs;
  /** Number of neurons in each layer */
  private final int[] layerSizes;
  /** Each layers activation function */
  private final ActivationFunction[] activationFunctions;
  
  /** Weights */
  private Matrix[] weights;
  /** Activities, needed for backpropagation */
  private Matrix[] activityA;
  private Matrix[] activityZ;
  
  
  
  /**
   * Constructs a new copy of an existing network
   * @param net  Network to copy
   */
  public Network(Network net) {
    this(net.numberOfInputs, net.getLayerSizes(), net.getActivationFunctions());
    
    weights = net.copyWeights();
  }
  
  /**
   * Constructs a new network
   * @param numberOfInputs Number of input neurons
   * @param layerSizes Numbers of neurons in each hidden layer,
   *                   last one is the number of output neurons
   * @param activationFunction Activation function of every layer
   * @throws IllegalArgumentException If the number of input neurons
   *         or the number Of given layers is less than 1
   */
  public Network(int numberOfInputs, int[] layerSizes,
          ActivationFunction activationFunction) {
    if(numberOfInputs < 1) {
      throw new IllegalArgumentException(
              "Number of input neurons less than 1!");
    }
    if(layerSizes.length < 1) {
      throw new IllegalArgumentException("Number of layers less than 1!");
    }
    for(int layerSize : layerSizes) {
      if(layerSize < 1) {
        throw new IllegalArgumentException(
                "Number of neurons in layer less than 1!");
      }
    }
    
    
    //Dimensions
    this.numberOfInputs = numberOfInputs;
    this.layerSizes = layerSizes;
    
    //Activation functions
    this.activationFunctions = new ActivationFunction[layerSizes.length];
    for(int i=0; i<activationFunctions.length; i++) {
      this.activationFunctions[i] = activationFunction;
    }
    
    //Weights
    weights = new Matrix[layerSizes.length];
    weights[0] = new Matrix(numberOfInputs, layerSizes[0]);
    for(int i=1; i<layerSizes.length; i++) {
      weights[i] = new Matrix(layerSizes[i-1], layerSizes[i]);
    }
    
    //Activities
    activityA = new Matrix[layerSizes.length];
    activityZ = new Matrix[layerSizes.length];
  }
  
  /**
   * Constructs a new network
   * @param numberOfInputs Number of inputs
   * @param layerSizes Numbers of nodes in each hidden layer,
   * last one is number of outputs
   * @param activationFunctions Activation functions for every layer
   */
  public Network(int numberOfInputs, int[] layerSizes,
          ActivationFunction[] activationFunctions) {
    if(numberOfInputs < 1) {
      throw new IllegalArgumentException(
              "Number of input neurons less than 1!");
    }
    if(layerSizes.length < 1) {
      throw new IllegalArgumentException("Number of layers less than 1!");
    }
    if(activationFunctions.length != layerSizes.length) {
      throw new IllegalArgumentException(
              "Not as many activation functions as layers!");
    }
    for(int layerSize : layerSizes) {
      if(layerSize < 1) {
        throw new IllegalArgumentException(
                "Number of neurons in layer less than 1!");
      }
    }
    
    
    //Dimensions
    this.numberOfInputs = numberOfInputs;
    this.layerSizes = layerSizes;
    
    //Activation functions
    this.activationFunctions = activationFunctions;
    
    //Weights
    weights = new Matrix[layerSizes.length];
    weights[0] = new Matrix(numberOfInputs, layerSizes[0]);
    for(int i=1; i<layerSizes.length; i++) {
      weights[i] = new Matrix(layerSizes[i-1], layerSizes[i]);
    }
    
    //Activities
    activityA = new Matrix[layerSizes.length];
    activityZ = new Matrix[layerSizes.length];
  }
  
  /**
   * Constructs a new network based on a saved one
   * @param stream Stream to read from
   * @throws IOException 
   */
  public Network(DataInputStream stream) throws IOException {
    //Dimensions
    numberOfInputs = stream.readInt();
    layerSizes = new int[stream.readInt()];
    for(int i=0; i<layerSizes.length; i++) {
      layerSizes[i] = stream.readInt();
    }
    
    //Activation functions
    activationFunctions = new ActivationFunction[layerSizes.length];
    for(int i=0; i<layerSizes.length; i++) {
      activationFunctions[i] = ActivationFunction.values()[stream.readInt()];
    }
    
    //Weights
    weights = new Matrix[layerSizes.length];
    weights[0] = new Matrix(numberOfInputs, layerSizes[0]);
    for(int j=0; j<weights[0].getHeight(); j++) {
      for(int i=0; i<weights[0].getWidth(); i++) {
        weights[0].set(stream.readDouble(), i, j);
      }
    }
    for(int k=1; k<layerSizes.length; k++) {
      weights[k] = new Matrix(layerSizes[k-1], layerSizes[k]);
      for(int j=0; j<weights[k].getHeight(); j++) {
        for(int i=0; i<weights[k].getWidth(); i++) {
          weights[k].set(stream.readDouble(), i, j);
        }
      }
    }
    
    //Activities
    activityA = new Matrix[layerSizes.length];
    activityZ = new Matrix[layerSizes.length];
  }
  
  
  
  /**
   * Seeds the weights within the given boundaries
   * @param minimum Minimum value
   * @param maximum Maximum value
   */
  public void seedWeights(double minimum, double maximum) {
    for(Matrix layer : weights) {
      layer.rand(minimum, maximum);
    }
  }
  
  /**
   * Seeds the weights within the given boundaries, based on a seed
   * @param seed Seed for the random number generator
   * @param minimum Minimum value
   * @param maximum Maximum value
   */
  public void seedWeights(long seed, double minimum, double maximum) {
    for(Matrix layer : weights) {
      layer.rand(seed, minimum, maximum);
    }
  }
  
  /**
   * Seeds the weights within the given boundaries for each layer
   * @param minimums Minimum values for every layer
   * @param maximums Maximum values for every layer
   * @throws IllegalArgumentException If the number of boundaries
   *         does not equal the number of layers
   */
  public void seedWeights(double[] minimums, double[] maximums) {
    if(minimums.length != layerSizes.length
            || maximums.length != layerSizes.length) {
      throw new IllegalArgumentException("Illegal number of boundaries!");
    }
    
    
    for(int i=0; i<weights.length; i++) {
      weights[i].rand(minimums[i], maximums[i]);
    }
  }
  
  /**
   * Seeds the weights within the given boundaries for each layer,
   * based on a seed
   * @param seed Seed for the random number generator
   * @param minimums Minimum values for every layer
   * @param maximums Maximum values for every layer
   * @throws IllegalArgumentException If the number of boundaries
   *         does not equal the number of layers
   */
  public void seedWeights(long seed, double[] minimums, double[] maximums) {
    if(minimums.length != layerSizes.length
            || maximums.length != layerSizes.length) {
      throw new IllegalArgumentException("Illegal number of boundaries!");
    }
    
    
    final Random rand = new Random(seed);
    
    for(int i=0; i<weights.length; i++) {
      weights[i].rand(rand, minimums[i], maximums[i]);
    }
  }
  
  
  /** Keeps weights real */
  public void keepWeightsInBounds() {
    for(Matrix layer : weights) {
      for(int j=0; j<layer.getHeight(); j++) {
        for(int i=0; i<layer.getWidth(); i++)
        {
          final double element = layer.get(i, j);
          
          if(Double.isNaN(element)) {
            layer.set(0, i, j);
          } else if(element == Double.NEGATIVE_INFINITY) {
            layer.set(-Double.MAX_VALUE, i, j);
          } else if(element == Double.POSITIVE_INFINITY) {
            layer.set(Double.MAX_VALUE, i, j);
          }
        }
      }
    }
  }
  
  /** Keeps weights within boundaries
   * @param minimum Minimum value
   * @param maximum Maximum value
   */
  public void keepWeightsInBounds(double minimum, double maximum) {
    for(Matrix layer : weights) {
      for(int j=0; j<layer.getHeight(); j++) {
        for(int i=0; i<layer.getWidth(); i++)
        {
          if(Double.isNaN(layer.get(i, j))) {
            layer.set(0, i, j);
          }
          
          final double element = layer.get(i, j);
          
          if(element < minimum) {
            layer.set(minimum, i, j);
          } else if(element > maximum) {
            layer.set(maximum, i, j);
          }
        }
      }
    }
  }
  
  
  /**
   * Forward propagates a matrix of data sets.
   * Every data set is arranged in a row
   * @param input Input sets
   * @return Output sets
   * @throws IllegalArgumentException If the number of input values
   *         does not equal the number of input neurons
   */
  public Matrix forward(Matrix input) {
    if(input.getWidth() != getNumberOfInputs()) {
      throw new IllegalArgumentException("Illegal number of inputs!");
    }
    
    
    activityZ[0] = input.multiply(weights[0]);
    activityA[0] = activate(activityZ[0], activationFunctions[0]);
    
    for(int i=1; i<layerSizes.length; i++) {
      activityZ[i] = activityA[i-1].multiply(weights[i]);
      activityA[i] = activate(activityZ[i], activationFunctions[i]);
    }
    
    return activityA[layerSizes.length-1];
  }
  
  /**
   * Applies an activation function to a matrix
   * @param input Input matrix
   * @param activationFunction Activation function
   * @return Resulted matrix of the activation
   */
  private Matrix activate(Matrix input,
          ActivationFunction activationFunction) {
    final Matrix output = new Matrix(input.getHeight(), input.getWidth());
    
    for(int j=0; j<input.getHeight(); j++) {
      for(int i=0; i<input.getWidth(); i++) {
        final double inputValue = input.get(i, j);
        final double outputValue = activationFunction.activate(inputValue);
        
        output.set(outputValue, i, j);
      }
    }
    
    return output;
  }
  
  /**
   * Applies the activation function's derivative to a matrix
   * @param input Input matrix
   * @param activationFunction Activation function
   * @return Resulted matrix of the activation
   */
  private Matrix activatePrime(Matrix input,
          ActivationFunction activationFunction) {
    final Matrix output = new Matrix(input.getHeight(), input.getWidth());
    
    for(int j=0; j<input.getHeight(); j++) {
      for(int i=0; i<input.getWidth(); i++) {
        final double inputValue = input.get(i, j);
        final double outputValue = activationFunction.activatePrime(inputValue);
        
        output.set(outputValue, i, j);
      }
    }
    
    return output;
  }
  
  
  /**
   * Calculates the mean squared error of the prediction to the actual output
   * @param input Input sets
   * @param output Actual output sets
   * @return Mean squared error
   * @throws IllegalArgumentException If the number of inputs or outputs
   *         does not fit the dimensions of this network or the number
   *         of input sets is not equal to the number of output sets
   */
  public double cost(Matrix input, Matrix output) {
    if(output.getWidth() != getNumberOfOutputs()) {
      throw new IllegalArgumentException("Illegal number of outputs!");
    }
    if(input.getHeight() != output.getHeight()) {
      throw new IllegalArgumentException(
              "Unequal number of input and output sets!");
    }
    
    
    double cost = 0;
    final Matrix yHat = forward(input);
    
    for(int j=0; j<yHat.getHeight(); j++) {
      for(int i=0; i<yHat.getWidth(); i++) {
        cost += (output.get(i, j) - yHat.get(i, j))
                * (output.get(i, j) - yHat.get(i, j));
      }
    }
    cost /= 2;
    
    
    return cost/input.getHeight();
  }
  
  /**
   * Backpropagates the error to every weight
   * @param input Input sets
   * @param output Wanted output sets
   * @return Derivative of the error to every weight
   * @throws IllegalArgumentException If the number of inputs or outputs
   *         does not fit the dimensions of this network or the number
   *         of input sets is not equal to the number of output sets
   */
  public Matrix[] costPrime(Matrix input, Matrix output) {
    if(input.getWidth() != getNumberOfInputs()) {
      throw new IllegalArgumentException("Illegal number of inputs!");
    }
    if(output.getWidth() != getNumberOfOutputs()) {
      throw new IllegalArgumentException("Illegal number of outputs!");
    }
    if(input.getHeight() != output.getHeight()) {
      throw new IllegalArgumentException(
              "Unequal number of input and output sets!");
    }
    
    
    Matrix delta;
    final Matrix[] dJdW = new Matrix[layerSizes.length];
    final Matrix yHat = forward(input);
    
    delta = yHat.subtract(output).multiplyElementwise(
                activatePrime(activityZ[layerSizes.length-1],
                              activationFunctions[layerSizes.length-1]));
    if(layerSizes.length > 1) {
      dJdW[layerSizes.length-1] = activityA[layerSizes.length-2].transpose()
              .multiply(delta);
    }
    
    for(int i=layerSizes.length-2; i>0; i--) {
      delta = delta.multiply(weights[i+1].transpose()).multiplyElementwise(
              activatePrime(activityZ[i], activationFunctions[i]));
      dJdW[i] = activityA[i-1].transpose().multiply(delta);
    }
    
    if(layerSizes.length > 1) {
      delta = delta.multiply(weights[1].transpose()).multiplyElementwise(
                  activatePrime(activityZ[0], activationFunctions[0]));
    }
    
    dJdW[0] = input.transpose().multiply(delta);
    
    
    return dJdW;
  }
  
  
  /**
   * Trains the network
   * (override the method "keepTraining" to set the continuation condition)
   * @param learningRate Learning rate
   * @param input Input sets
   * @param output Wanted output sets
   * @param printToConsole Print progress to console
   * @return Last cost
   * @throws IllegalArgumentException If the number of inputs or outputs
   *         does not fit the dimensions of this network or the number
   *         of input sets is not equal to the number of output sets
   */
  public double train(double learningRate, Matrix input, Matrix output,
          boolean printToConsole) {
    int iterations = 0;
    double improvement;
    double lastCost = cost(input, output);
    
    if(printToConsole)
      System.out.println("Initial cost: " + lastCost);
    
    
    //Train until there is no change
    while(keepTraining(iterations, learningRate, lastCost) && learningRate>0)
    {
      //Save weights
      final Matrix[] lastWeights = copyWeights();
      
      
      //Try a single training run
      final double currentCost = trainSingle(learningRate, input, output);
      
      //Save improvement
      improvement = 100 * (lastCost - currentCost) / lastCost;
      
      if(printToConsole)
        System.out.print(String.format("%e: %e -> %e (%e%%) ",
                learningRate, lastCost, currentCost, improvement));
      
      //If it got better
      if(currentCost <= lastCost)
      {
        //Remember the new cost
        lastCost = currentCost;
        
        //Optional: Increase learning rate for faster training
        learningRate *= 1.1;
        
        //Output to user
        if(printToConsole)
          System.out.println("(Keep changes)");
      }
      else //If it got worse ...
      {
        //... redo the weight changes ...
        setWeights(lastWeights);
        //... and decreasse the learning rate
        learningRate /= 2;
        
        if(printToConsole)
          System.out.println("(Undo weightchange)");
      }
      
      iterations++;
    }
    
    return lastCost;
  }
  
  /**
   * Tells the network how long to continue to train
   * @param iterations Number of completed training cycles
   * @param learningRate Current learning rate
   * @param cost Current cost
   * @return If the training process should continue
   */
  public boolean keepTraining(int iterations, double learningRate,
          double cost) {
    throw new UnsupportedOperationException("Override for training!");
  }
  
  /**
   * Backpropagates and applies the gradient with a learning rate once
   * @param learningRate Learning rate
   * @param input Input sets
   * @param output Wanted output sets
   * @return New costs after training iteration
   * @throws IllegalArgumentException If the number of inputs or outputs
   *         does not fit the dimensions of this network or the number
   *         of input sets is not equal to the number of output sets
   */
  public double trainSingle(double learningRate, Matrix input, Matrix output) {
    final Matrix[] dJdW = costPrime(input, output);
    
    for(int i=0; i<weights.length; i++) {
      weights[i] = weights[i].add(dJdW[i].multiply(-learningRate));
    }
    keepWeightsInBounds();
    
    return cost(input, output);
  }
  
  
  
  /** 
   * @return Number of inputs */
  public int getNumberOfInputs() {
    return numberOfInputs;
  }
  
  /** 
   * @return Number of outputs (last layer size) */
  public int getNumberOfOutputs() {
    return layerSizes[layerSizes.length-1];
  }
  
  /** 
   * @return Number of nodes in every layer */
  public int[] getLayerSizes() {
    return layerSizes;
  }
  
  /** 
   * @param index Index of layer
   * @return Number of nodes in specific layer */
  public int getLayerSize(int index) {
    return layerSizes[index];
  }
  
  /** 
   * @return Number of hidden Layers */
  public int getNumberOfHiddenLayers() {
    return layerSizes.length;
  }
  
  
  /** 
   * @return Activation function for every layer */
  public ActivationFunction[] getActivationFunctions() {
    return activationFunctions;
  }
  
  /** 
   * @param index Index of layer
   * @return Activation function for specific layer */
  public ActivationFunction getActivationFunction(int index) {
    return activationFunctions[index];
  }
  
  
  /** 
   * @return Weights */
  public Matrix[] getWeights() {
    return weights;
  }
  
  /**
   * @param index Index of layer
   * @return Weights of specific layer */
  public Matrix getWeights(int index) {
    return weights[index];
  }
  
  /**
   * Makes a copy of the weights
   * @return Copy of the weights
   */
  public Matrix[] copyWeights() {
    final Matrix[] copy = new Matrix[layerSizes.length];
    
    for(int i=0; i<weights.length; i++) {
      copy[i] = new Matrix(weights[i]);
    }
    
    return copy;
  }
  
  /**
   * Overrides weights
   * @param weights New weights
   */
  public void setWeights(Matrix[] weights) {
    if(weights.length != this.weights.length) {
      throw new IllegalArgumentException(
              "Number of layers wrong. Is: " + weights.length
                      + ", should be: " + this.weights.length);
    }
    
    
    for(int i=0; i<this.weights.length; i++) {
      setWeights(weights[i], i);
    }
    
    this.weights = weights;
  }
  
  /**
   * Overrides weights of a single layer
   * @param weights New weights
   * @param index Layer index
   */
  public void setWeights(Matrix weights, int index) {
    if(weights.getHeight() != this.weights[index].getHeight()) {
      throw new IllegalArgumentException(
              "Layer height wrong. Is: " + weights.getHeight()
                      + ", should be: " + this.weights[index].getHeight());
    }
    if(weights.getWidth() != this.weights[index].getWidth()) {
      throw new IllegalArgumentException(
              "Layer width wrong. Is: " + weights.getWidth()
                      + ", should be: " + this.weights[index].getWidth());
    }
    
    this.weights[index] = weights;
  }
  
  
  
  /**
   * Writes the network to a stream
   * @param stream Stream to write to
   * @throws IOException 
   */
  public void writeToStream(DataOutputStream stream) throws IOException {
    //Dimensions
    stream.writeInt(numberOfInputs);
    stream.writeInt(layerSizes.length);
    for(int layerSize : layerSizes) {
      stream.writeInt(layerSize);
    }
    
    //Activation functions
    for(ActivationFunction function : activationFunctions) {
      stream.writeInt(function.ordinal());
    }
    
    //Weights
    for(Matrix layer : weights) {
      for(int j=0; j<layer.getHeight(); j++) {
        for(int i=0; i<layer.getWidth(); i++) {
          stream.writeDouble(layer.get(i, j));
        }
      }
    }
  }
  
  
  @Override
  public String toString() {
    final StringBuilder result = new StringBuilder("Network {");
    result.append(numberOfInputs);
    result.append(Arrays.toString(layerSizes));
    result.append("\n");
    
    for(int i=0; i<layerSizes.length; i++) {
      result.append(activationFunctions[i]).append('\n');
      result.append(weights[i]).append('\n');
    }
    result.append('}');
    
    return result.toString();
  }
  
  
  
  
  public static void main(String[] args) {
    //New network
    Network net = new Network(
            2,                          //2 inputs
            new int[]{3, 8, 1},         //3 layers with 2, 8 and 1 neurons
            new Network.ActivationFunction[]{   //1 activation function
              Network.ActivationFunction.NONE,  //for every layer
              Network.ActivationFunction.RELU,
              Network.ActivationFunction.NONE}) {
                //Override keepTraining to return true for 100 cycles
                @Override
                public boolean keepTraining(int iterations,
                        double learningRate, double cost)
                {
                  return iterations < 100;
                }
              };
    //4 training sets (given inputs with wanted outputs)
    Matrix inputs = new Matrix(
            new double[][]{
              {1, 2},                   //<- first input set
              {3, 4},                   //<- second input set ...
              {5, 6},
              {7, 8}});
    Matrix outputs = new Matrix(
            new double[][]{
              {0.25},                   //<- first output set
              {0.50},                   //<- second output set ...
              {0.75},
              {1.00}});
    
    
    //Seed weights
    net.seedWeights(-1, 1);
    //Show net
    System.out.println(net + "\n");
    
    //Show untrained network results
    System.out.println("Cost before training:");
    System.out.println(net.cost(inputs, outputs));
    System.out.println("Result before training:");
    System.out.println(net.forward(inputs) + "\n");
    
    //Train
    System.out.println("Training ...");
    net.train(0.01, inputs, outputs, true);
    System.out.println("Done!" + "\n");
    
    //Show trained network results
    System.out.println("Cost after training:");
    System.out.println(net.cost(inputs, outputs));
    System.out.println("Result after training:");
    System.out.println(net.forward(inputs) + "\n");
  }
}