package neural;

import java.util.Arrays;

/**
 * Layer of a neural network
 * 
 * @author Sebastian Gössl
 * @version 0.9 28.06.2017
 */
public class Layer
{
  //Activation functions

  /** None. Just outputs the sum of all inputs */
  public static final int ACTIVATION_NONE = 0;
  /** Hyperbolic tangent. Output between -1 and 1 */
  public static final int ACTIVATION_TANH = 1;
  /** Sigmoid / Logistic function. Output between 0 and 1 */
  public static final int ACTIVATION_SIGMOID = 2;
  /** Step function. Everything smaller than 0 becomes -1, everything else 1 */
  public static final int ACTIVATION_HEAVISIDE_STEP_FUNCTION_N1_1 = 3;
  /** Step function. Everything smaller than 0 becomes 0, everything else 1 */
  public static final int ACTIVATION_HEAVISIDE_STEP_FUNCTION_0_1 = 4;
  /** Rectified Linear Unit (ReLU). Everything below 0 becomes 0 */
  public static final int ACTIVATION_RECTIFIED_LINEAR_UNIT = 5;
  
  //Some numbers to define the size

  // Number of inputs.
  private final int numberOfInputs;
  // Number of outputs.
  private final int numberOfOutputs;
  // Weights. Every input gets an array of weights. One for every output
  private double[][] weights;
  // Activation function. Applied in every neuron on the sum of the weighted inputs
  private final int activationFunction;

  /**
   * Constructs a new Layer
   *
   * @param numberOfInputs Number of inputs
   * @param numberOfOutputs Number of outputs
   * @param activationFunction Activation function to apply in every neuron
   *
   * @throws Exception If number of input or output neurons doesn't fit
   */
  public Layer(int numberOfInputs, int numberOfOutputs,
               int activationFunction) throws Exception
  {
    if(numberOfInputs <= 0)
      throw new Exception("Number of inputs must be 1 or more!");
    if(numberOfOutputs <= 0)
      throw new Exception("Number of outputs must be 1 or more!");
    
    
    this.numberOfInputs = numberOfInputs;
    this.numberOfOutputs = numberOfOutputs;
    this.activationFunction = activationFunction;
    
    //Every input has an array of weights, one for every output
    //(Prepare for some awesome ASCII art!)
    //
    //  I1----W11----Z1
    //    \         /
    //     --W12-\ /
    //            \
    //     --W21-/ \
    //    /         \
    //  I2----W22----Z2
    //
    // Z1 = W11*I1 + W21*I2
    // Z2 = W12*I1 + W22*I2
    //
    //  In conclusion: weights[In][Zn]
    //  In: Input n, Zn: Neuron n
    //
    weights = new double[numberOfInputs][numberOfOutputs];
  }
  
  
  /**
   * Calulates the output based on the given input
   * @param input Input to feed into the layer
   * @return The calculated utput of the layer
   * @throws Exception If the size of the input doesn't fit
   */
  public double[] calculate(double[] input) throws Exception
  {
    if(input.length != numberOfInputs)
      throw new Exception("Number of inputs doesn't fit!");
    
    
    double[] output = new double[numberOfOutputs];
    
    //Sum all weighted inputs for every output neuron
    //For every output ...
    for(int j=0; j<output.length; j++)
      //... sum all the weighted inputs
      for(int i=0; i<input.length; i++)
        output[j] += weights[i][j] * input[i];
    
    
    //Apply the activation function on every output neuron
    for(int i=0; i<output.length; i++)
      switch(activationFunction)
      {
        //None. Just outputs the sum of all inputs
        case ACTIVATION_NONE:
          break;
        
        //Hyperbolic tangent. Output between -1 and 1
        case ACTIVATION_TANH:
          output[i] = Math.tanh(output[i]);
          break;
        
        //Sigmoid / Logistic function. Output between 0 and 1
        case ACTIVATION_SIGMOID:
          output[i] = 1/(1+ Math.exp(-output[i]));
          break;
        
        //Step function. Everything smaller than 0 becomes -1, everything else 1
        case ACTIVATION_HEAVISIDE_STEP_FUNCTION_N1_1:
          if(output[i] >= 0)
            output[i] = 1;
          else
            output[i] = 0;
          break;
        
        //Step function. Everything smaller than 0 becomes 0, everything else 1
        case ACTIVATION_HEAVISIDE_STEP_FUNCTION_0_1:
          if(output[i] >= 0)
            output[i] = 1;
          else
            output[i] = 0;
          break;
        
        //Rectified Linear Unit (ReLU). Everything below 0 becomes 0
        case ACTIVATION_RECTIFIED_LINEAR_UNIT:
          if(output[i] < 0)
            output[i] = 0;
          break;
        
        
        default:
          ;
      }
    
    return output;
  }

  /**
   * Initializes all weights with random values
   * @param minimum Minimal value a weight can have
   * @param maximum Maximal value a weight can have
   */
  public void seedWeights(double minimum, double maximum)
  {
    for(double[] weightArray : weights)
      for(int i=0; i<weightArray.length; i++)
        weightArray[i] = (maximum-minimum)*Math.random() + minimum;
  }
  
  
  /**
   * Sets all weights
   * @param weights Array of weight arrays.
   * Every input gets an array of weights, one for every output
   * @throws Exception If the array size doesn't fit
   */
  public void setWeights(double[][] weights) throws Exception
  {
    //As many arrays as inputs?
    if(weights.length != this.weights.length)
      throw new Exception("Wrong size!");
    //As many weights as outputs, for every input?
    for(int i=0; i<this.weights.length; i++)
      if(weights[i].length != this.weights[i].length)
        throw new Exception("Wrong size!");


    this.weights = weights;
  }
  
  
  /**
   * @return Weights
   */
  public double[][] getWeights()
  {
    return weights;
  }
  
  /**
   * @return Number of inputs
   */
  public int getNumberOfInputs()
  {
    return numberOfInputs;
  }
  
  /**
   * @return Number of outputs
   */
  public int getNumberOfOutputs()
  {
    return numberOfOutputs;
  }
  
  /**
   * @return Activation function
   */
  public int getActivationFunction()
  {
    return activationFunction;
  }
  
  
  
  //Output a more or less readable representation
  @Override
  public String toString()
  {
    //Construct a beautiful string!
    StringBuilder result = new StringBuilder();
    
    //All the weights
    //Every column shows all the weights for one input
    //Every row show all the weights for one output
    for(int j = 0; j< numberOfOutputs; j++)
    {
      for(int i = 0; i< numberOfInputs; i++)
        result.append(String.format("| %.5f ", weights[i][j]));
      result.append("|\n");
    }
    
    //Draw an o for every output with an "/ \" above, if there are more outputs
    //than inputs (layer gets bigger),
    //or an "\ /" if it gets smaller or "|" if the size stays the same
    for(int i = 0; i< numberOfOutputs; i++)
    {
      if(numberOfOutputs < numberOfInputs)
        result.append("    \\ /   ");
      else
        if(numberOfOutputs > numberOfInputs)
          result.append("    / \\   ");
        else
          result.append("     |    ");
    }
    result.append("\n");
    
    for(int i = 0; i< numberOfOutputs; i++)
      result.append("     o    ");
    result.append("\n");
    
    //Return out beautiful string!
    return result.toString();
  }
  
  //Test everything!
  public static void main(String[] args) throws Exception
  {
    //4 inputs, 4 neurons, tanh as activation function
    Layer layer = new Layer(4, 4, Layer.ACTIVATION_TANH);
    
    layer.seedWeights(-1, 1);
    System.out.println(layer);
    
    double[] input = new double[]{1, 0, 0.5, -1};
    System.out.println(Arrays.toString(input));
    System.out.println(Arrays.toString(layer.calculate(input)));
  }
}
