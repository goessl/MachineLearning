package neural;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.Arrays;
import java.util.Random;
import java.util.function.DoubleFunction;

/**
 * Neural network
 * 
 * @author Sebastian Gössl
 * @version 1.2 26.03.2018
 */
public class Network {
  
  /**
   * Activation functions
   */
  public enum ActivationFunction {
    NONE, TANH, SIGMOID, RELU, SOFTPLUS, RELU_LEAKY;
    
    private static final double RELU_LEAKY_LEAKAGE = 0.01;
    
    private static final String[] name = {
      "None",
      "Hyperbolic tangent",
      "Sigmoid",
      "Rectified linear unit",
      "SoftPlus",
      "Leaky rectified linear unit"
    };
    
    private static final DoubleFunction[] function = {
      //None
      x -> x,
      //Tanh
      x -> Math.tanh(x),
      //Sigmoid
      x -> 1 / (1 + Math.exp(-x)),
      //ReLU
      x -> {
        if(x >= 0) {
          return x;
        } else {
          return 0.0;
        }},
      //SoftPlus
      x -> Math.log(1 + Math.exp(x)),
      //Leaky ReLU
      x -> {
        if(x >= 0) {
          return x;
        } else {
          return RELU_LEAKY_LEAKAGE * x;
        }}
    };
    
    private static final DoubleFunction[] prime = {
      //None
      x -> 1.0,
      //Tanh
      x -> 1 - Math.tanh(x) * Math.tanh(x),
      //Sigmoid
      x -> Math.exp(-x) / ((1 + Math.exp(-x)) * (1 + Math.exp(-x))),
      //ReLU
      x -> {
        if(x >= 0) {
          return 1.0;
        } else {
          return 0.0;
        }},
      //Softplus
      x -> 1 / (1 + Math.exp(-x)),
      //Leaky ReLU
      x -> {
        if(x >= 0) {
          return  1.0;
        } else {
          return RELU_LEAKY_LEAKAGE;
        }}
    };
    
    
    
    /**
     * Returns this activation function as a function
     * @return Function
     */
    public DoubleFunction function() {
      return function[ordinal()];
    }
    
    /**
     * Returns this activation function's derivative as a function
     * @return Function
     */
    public DoubleFunction prime() {
      return prime[ordinal()];
    }
    
    
    
    @Override
    public String toString() {
      return name[ordinal()];
    }
  }
  
  
  
  /**
   * Inputs
   * 
   * Weights[0]
   * 
   * Layer[0]
   *   ActivationZ[0] (Weighted sum)
   *   ActivationA[0] (Activated sum)
   * 
   * Weights[1]
   * 
   * Layer[1]
   *   ActivationZ[1] (Weighted sum)
   *   ActivationA[1] (Activated sum)
   * 
   * ...
   */
  
  /** Number of input neurons */
  private final int numberOfInputs;
  /** Number of neurons in each layer */
  private final int[] layerSizes;
  /** Each layers activation function */
  private final ActivationFunction[] activationFunctions;
  
  /** Weights */
  private final Matrix[] weights;
  /** Activities, needed for backpropagation */
  private final Matrix[] activityA;
  private final Matrix[] activityZ;
  
  
  
  /**
   * Constructs a new copy of an existing network
   * @param net  Network to copy
   */
  public Network(Network net) {
    this(net.getNumberOfInputs(),
            net.copyLayerSizes(),
            net.copyActivationFunctions());
    
    setWeights(net.copyWeights());
  }
  
  /**
   * Constructs a new network
   * @param numberOfInputs Number of inputs
   * @param layerSizes Numbers of neurons in each hidden layer,
   *        last layer is the output layer (number of outputs)
   * @param activationFunctions Activation functions for every layer
   * @throws IllegalArgumentException If the number of layers or
   *        the number of neurons in a layer is smaller than 1 or
   *        if the number of given activation functions
   *        does not equal the number of layers
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
    this.layerSizes = Arrays.copyOf(layerSizes, layerSizes.length);
    
    //Activation functions
    this.activationFunctions = activationFunctions;
    
    //Weights
    weights = new Matrix[layerSizes.length];
    weights[0] = new Matrix(numberOfInputs, layerSizes[0]);
    for(int i=1; i<layerSizes.length; i++) {
      weights[i] = new Matrix(layerSizes[i-1], layerSizes[i]);
    }
    
    //Activities (needed for backpropagation)
    activityA = new Matrix[weights.length];
    activityZ = new Matrix[weights.length];
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
    for(int i=0; i<activationFunctions.length; i++) {
      activationFunctions[i] = ActivationFunction.values()[stream.readInt()];
    }
    
    //Weights
    weights = new Matrix[layerSizes.length];
    weights[0] = new Matrix(numberOfInputs, layerSizes[0]);
    for(int i=1; i<weights.length; i++) {
      weights[i] = new Matrix(layerSizes[i-1], layerSizes[i]);
    }
    
    for(int k=0; k<weights.length; k++) {
      for(int j=0; j<weights[k].getHeight(); j++) {
        for(int i=0; i<weights[k].getWidth(); i++) {
          weights[k].set(j, i, stream.readDouble());
        }
      }
    }
    
    //Activities
    activityA = new Matrix[weights.length];
    activityZ = new Matrix[weights.length];
  }
  
  
  
  /**
   * Returns the number of inputs
   * @return Number of inputs
   */
  public int getNumberOfInputs() {
    return numberOfInputs;
  }
  
  /**
   * Returns the number of outputs (last layer size)
   * @return Number of outputs
   */
  public int getNumberOfOutputs() {
    return layerSizes[layerSizes.length-1];
  }
  
  
  /**
   * Returns the number of layers
   * @return Number of Layers
   */
  public int getNumberOfLayers() {
    return layerSizes.length;
  }
  
  /**
   * Returns the number of neurons in the specified layer
   * @param index Index of the layer
   * @return Number of neurons in the layer
   * @throws ArrayIndexOutOfBoundsException If the Index does not point
   *        to an existing layer
   */
  public int getLayerSize(int index) {
    if(index < 0 || index >= layerSizes.length) {
      throw new ArrayStoreException("Index out of bounds!");
    }
    
    
    return layerSizes[index];
  }
  
  /**
   * Returns a copy of the numbers of neurons in every layer
   * @return Copy of numbers of neurons in every layer
   */
  public int[] copyLayerSizes() {
    return Arrays.copyOf(layerSizes, layerSizes.length);
  }
  
  
  /**
   * Sets the activation function of the specified layer
   * @param index Index of the layer
   * @param function Activation function
   * @throws ArrayIndexOutOfBoundsException If the index does not point
   *        to an existing layer
   */
  public void setActivationFunction(int index, ActivationFunction function) {
    if(index < 0 || index >= activationFunctions.length) {
      throw new ArrayIndexOutOfBoundsException("Index out of bounds!");
    }
    
    
    activationFunctions[index] = function;
  }
  
  /**
   * Returns the activation function of the specific layer
   * @param index Index of the layer
   * @return Activation function of the layer
   * @throws ArrayIndexOutOfBoundsException If the index does not point
   *        to an existing layer
   */
  public ActivationFunction getActivationFunction(int index) {
    if(index < 0 || index >= activationFunctions.length) {
      throw new ArrayIndexOutOfBoundsException("Index out of bounds!");
    }
    
    
    return activationFunctions[index];
  }
  
  /**
   * Returns the activation functions of every layer
   * @return Activation functions
   */
  public ActivationFunction[] getActivationFunctions() {
    return activationFunctions;
  }
  
  /**
   * Returns a copy of the activation functions of every layer
   * @return Copy of the activation functions of every layer
   */
  public ActivationFunction[] copyActivationFunctions() {
    return Arrays.copyOf(activationFunctions, activationFunctions.length);
  }
  
  
  /**
   * Sets the weights of a single layer
   * @param index Layer index
   * @param layer New weights
   * @throws IllegalArgumentException If the index does not point
   *        to an existing matrix or the given matrix dimensions
   *        do not equal the needed size
   */
  public void setWeights(int index, Matrix layer) {
    if(index < 0 || index >= weights.length) {
      throw new ArrayIndexOutOfBoundsException("Index out of bounds!");
    }
    if(layer.getHeight() != weights[index].getHeight()
            || layer.getWidth() != weights[index].getWidth()) {
      throw new IllegalArgumentException("Incorrect layer dimensions!");
    }
    
    
    weights[index] = layer;
  }
  
  /**
   * Sets the weights for every layer
   * @param weights New weights
   * @throws IllegalArgumentException If the number of matricies
   *        does not equal the number of layers
   *        or the dimensions of a matrix do not equal the needed dimensions
   */
  public void setWeights(Matrix[] weights) {
    if(weights.length != this.weights.length) {
      throw new IllegalArgumentException("Incorrect number of layers!");
    }
    for(int i=0; i<this.weights.length; i++) {
      if(weights[i].getHeight() != this.weights[i].getHeight()
              || weights[i].getWidth() != this.weights[i].getWidth()) {
        throw new IllegalArgumentException("Incorrect layer dimensions!");
      }
    }
    
    
    for(int i=0; i<this.weights.length; i++) {
      this.weights[i] = weights[i];
    }
  }
  
  /**
   * Returns the weight matrix of one specific layer
   * @param index Index of the layer
   * @return Weights of the layer
   */
  public Matrix getWeights(int index) {
    if(index < 0 || index >= weights.length) {
      throw new ArrayIndexOutOfBoundsException("Index out of bounds!");
    }
    
    return weights[index];
  }
  
  /**
   * Returns the weights of every layer
   * @return Weights
   */
  public Matrix[] getWeights() {
    return Arrays.copyOf(weights, weights.length);
  }
  
  /**
   * Returns a copy of all weights
   * @return Copy of all weights
   */
  public Matrix[] copyWeights() {
    final Matrix[] copy = new Matrix[weights.length];
    
    for(int i=0; i<copy.length; i++) {
      copy[i] = new Matrix(weights[i]);
    }
    
    return copy;
  }
  
  
  
  /**
   * Seeds the weights within the given boundaries
   * @param minimum Minimum value
   * @param maximum Maximum value
   */
  public void seedWeights(double minimum, double maximum) {
    final Random rand = new Random();
    
    for(Matrix layer : weights) {
      layer.rand(rand, minimum, maximum);
    }
  }
  
  /**
   * Seeds the weights, based on a seed, between the given boundaries
   * @param seed Seed for the random number generator
   * @param minimum Minimum value
   * @param maximum Maximum value
   */
  public void seedWeights(long seed, double minimum, double maximum) {
    final Random rand = new Random(seed);
    
    for(Matrix layer : weights) {
      layer.rand(rand, minimum, maximum);
    }
  }
  
  /**
   * Seeds the weights within the given boundaries for each layer
   * @param minimums Minimum values for each layer
   * @param maximums Maximum values for each layer
   * @throws ArrayIndexOutOfBoundsException If the number of boundaries
   *        does not equal the number of layers
   */
  public void seedWeights(double[] minimums, double[] maximums) {
    if(minimums.length != weights.length
            || maximums.length != weights.length) {
      throw new ArrayIndexOutOfBoundsException("Illegal number of boundaries!");
    }
    
    
    final Random rand = new Random();
    
    for(int i=0; i<weights.length; i++) {
      weights[i].rand(rand, minimums[i], maximums[i]);
    }
  }
  
  /**
   * Seeds the weights, based on a seed,
   * between the given boundaries for each layer
   * @param seed Seed for the random number generator
   * @param minimums Minimum values for each layer
   * @param maximums Maximum values for each layer
   * @throws ArrayIndexOutOfBoundsException If the number of boundaries
   *        does not equal the number of layers
   */
  public void seedWeights(long seed, double[] minimums, double[] maximums) {
    if(minimums.length != weights.length
            || maximums.length != weights.length) {
      throw new ArrayIndexOutOfBoundsException("Illegal number of boundaries!");
    }
    
    
    final Random rand = new Random(seed);
    
    for(int i=0; i<weights.length; i++) {
      weights[i].rand(rand, minimums[i], maximums[i]);
    }
  }
  
  
  /**
   * Eliminates infinite numbers & NaNs
   */
  public void keepWeightsInBounds() {
    for(int i=0; i<weights.length; i++) {
      weights[i] = weights[i].apply(x -> {
        if(Double.isNaN(x)) {
          return 0.0;
        } else if(x <= Double.NEGATIVE_INFINITY) {
          return -Double.MAX_VALUE;
        } else if(x >= Double.POSITIVE_INFINITY) {
          return Double.MAX_VALUE;
        }
        
        return x;
      });
    }
  }
  
  /**
   * Keeps weights within the given boundaries
   * @param minimum Minimum value
   * @param maximum Maximum value
   */
  public void keepWeightsInBounds(double minimum, double maximum) {
    if(minimum >= maximum) {
      throw new IllegalArgumentException(
              "Minimum greater than or equal to maximum!");
    }
    
    
    for(int i=0; i<weights.length; i++) {
      weights[i] = weights[i].apply(x -> {
        if(Double.isNaN(x)) {
          return (minimum + maximum) / 2;
        } else if(x < minimum) {
          return minimum;
        } else if(x > maximum) {
          return maximum;
        }
        
        return x;
      });
    }
  }
  
  
  
  /**
   * Forward propagates a matrix of data sets.
   * Every single row represents one data set
   * Every column gets feed into one input neuron
   * @param input Input sets
   * @return Output sets
   * @throws IllegalArgumentException If the number of input values (columns)
   *        does not equal the number of input neurons
   */
  public Matrix forward(Matrix input) {
    if(input.getWidth() != numberOfInputs) {
      throw new IllegalArgumentException("Illegal number of inputs!");
    }
    
    
    activityZ[0] = input.multiply(weights[0]);
    activityA[0] = activityZ[0].apply(activationFunctions[0].function());
    
    for(int i=1; i<weights.length; i++) {
      activityZ[i] = activityA[i-1].multiply(weights[i]);
      activityA[i] = activityZ[i].apply(activationFunctions[i].function());
    }
    
    
    return new Matrix(activityA[weights.length-1]);
  }
  
  /**
   * Calculates the mean squared error of the prediction to the given output.
   * Every single row represents one data set
   * Every column gets feed into one input/output neuron
   * @param input Input sets
   * @param output Output sets
   * @return Mean squared error
   * @throws IllegalArgumentException If the number of inputs or outputs
   *        does not fit the dimensions of this network or the number
   *        of input sets is not equal to the number of output sets
   */
  public double cost(Matrix input, Matrix output) {
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
    
    
    final Matrix yHat = forward(input);
    final Matrix difference = output.subtract(yHat);
    final Matrix squaredError = difference.multiplyElementwise(difference);
    
    double cost = 0;
    for(int j=0; j<squaredError.getHeight(); j++) {
      for(int i=0; i<squaredError.getWidth(); i++) {
        cost += squaredError.get(j, i);
      }
    }
    cost /= 2;
    cost /= input.getHeight();
    
    
    return cost;
  }
  
  /**
   * Backpropagates the error to every weight.
   * Every single row represents one data set
   * Every column gets feed into one input/output neuron
   * @param input Input sets
   * @param output Output sets
   * @return Derivative of the error to every weight
   * @throws IllegalArgumentException If the number of inputs or outputs
   *        does not fit the dimensions of this network or the number
   *        of input sets is not equal to the number of output sets
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
    final Matrix[] dJdW = new Matrix[weights.length];
    final Matrix yHat = forward(input);
    
    delta = yHat.subtract(output).multiplyElementwise(
            activityZ[weights.length-1].apply(
                    activationFunctions[weights.length-1].prime()));
    
    for(int i=weights.length-1; i>0; i--) {
      dJdW[i] = activityA[i-1].transpose().multiply(delta);
      delta = delta.multiply(weights[i].transpose()).multiplyElementwise(
              activityZ[i-1].apply(activationFunctions[i-1].prime()));
    }
    
    dJdW[0] = input.transpose().multiply(delta);
    
    
    return dJdW;
  }
  
  
  
  /**
   * Trains the network.
   * (override the method "keepTraining" to set the continuation condition)
   * @param learningRate Initial learning rate
   * @param input Input sets
   * @param output Wanted output sets
   * @param printToConsole Print progress to console
   * @return Last cost
   * @throws IllegalArgumentException If the number of inputs or outputs
   *        does not fit the dimensions of this network or the number
   *        of input sets is not equal to the number of output sets
   */
  public double train(double learningRate,
          Matrix input, Matrix output, boolean printToConsole) {
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
    
    
    double lastCost = cost(input, output);
    
    for(int iterations = 0;
            keepTraining(iterations, learningRate, lastCost)
            && learningRate > 0;
            iterations++)
    {
      final Matrix[] lastWeights = copyWeights();
      
      singleGradientDescent(learningRate, input, output);
      final double currentCost = cost(input, output);
      
      if(printToConsole) {
        System.out.println(String.format("%d: %e", iterations, currentCost));
      }
      
      if(currentCost <= lastCost)
      {
        lastCost = currentCost;
        learningRate *= 1.1;
      }
      else
      {
        setWeights(lastWeights);
        learningRate /= 2;
      }
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
    return iterations < 100;
  }
  
  /**
   * Backpropagates and applies the gradient with the given learning rate once
   * @param learningRate Learning rate
   * @param input Input sets
   * @param output Wanted output sets
   * @throws IllegalArgumentException If the number of inputs or outputs
   *        does not fit the dimensions of this network or the number
   *        of input sets is not equal to the number of output sets
   */
  private void singleGradientDescent(double learningRate,
          Matrix input, Matrix output) {
    final Matrix[] dJdW = costPrime(input, output);
    
    for(int i=0; i<weights.length; i++) {
      final Matrix update = dJdW[i].multiply(-learningRate);
      weights[i] = weights[i].add(update);
    }
    keepWeightsInBounds();
  }
  
  
  
  /**
   * Writes the network to a stream
   * @param stream Stream to write to
   * @throws IOException 
   */
  public void writeToStream(DataOutputStream stream) throws IOException {
    //Dimensions
    stream.writeInt(numberOfInputs);
    stream.writeInt(getNumberOfLayers());
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
          stream.writeDouble(layer.get(j, i));
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
    
    for(int i=0; i<getNumberOfLayers(); i++) {
      result.append(activationFunctions[i]).append('\n');
      result.append(weights[i]).append('\n');
    }
    result.append('}');
    
    return result.toString();
  }
  
  
  
  
  public static void main(String[] args) {
    //New network
    final Network net = new Network(
            2,                                    //2 inputs
            new int[]{3, 1},                      //2 layers with 3 & 1 neurons
            new Network.ActivationFunction[]{
              Network.ActivationFunction.NONE,    //both layers with ...
              Network.ActivationFunction.NONE});  //... no activation function
    //Randomize weights
    net.seedWeights(-1, 1);
    //Show network
    System.out.println(net);
    
    
    //Generate 10 training sets
    //Every row represents one training set (10 rows = 10 sets)
    //Every column gets fed into the same input/comes out of the same output
    //(first column gets into the first input)
    //(2 columns = 2 inputs / 1 column = 1 output)
    final Matrix trainInput = new Matrix(10, 2);
    final Matrix trainOutput = new Matrix(10, 1);
    //Fill the training sets
    //Inputs: two random numbers
    //Outputs: average of these two numbers
    final Random rand = new Random();
    for(int set=0; set<trainInput.getHeight(); set++) {
      trainInput.set(set, 0, rand.nextInt(10));
      trainInput.set(set, 1, rand.nextInt(10));
      
      final double out = (trainInput.get(set, 0) + trainInput.get(set, 1)) / 2;
      trainOutput.set(set, 0, out);
    }
    
    
    //Show untrained network results
    System.out.println("Cost before training:");
    System.out.println(net.cost(trainInput, trainOutput));
    System.out.println("Result before training:");
    System.out.println(net.forward(trainInput) + "\n");
    
    //Train
    System.out.println("Training ...");
    net.train(0.2, trainInput, trainOutput, true);
    System.out.println("Done!" + "\n");
    
    //Show trained network results
    System.out.println("Cost after training:");
    System.out.println(net.cost(trainInput, trainOutput));
    System.out.println("Result after training:");
    System.out.println(net.forward(trainInput) + "\n");
  }
}
