package neural;

import java.util.Arrays;

/**
 * Neural network
 * 
 * @author Sebastian Gössl
 * @version 0.92 22.07.2017
 */
public class Network
{
  //Layers. All the Layers in the Network
  private final Layer[] layers;
  
  
  
  /**
   * Constructs a new neural network
   * @param numberOfInputs The number of inputs
   * @param layerSizes An array with the number of neurons for every layer.
   * The last number of neurons are the output neurons
   * @param activationFunctions An array with the activation functions
   * for every layer
   * @throws Exception 
   */
  public Network(int numberOfInputs, int[] layerSizes,
          int[] activationFunctions) throws Exception
  {
    if(layerSizes.length <= 0)
      throw new Exception("Number of hidden layers must be more than 0!");
    if(activationFunctions.length != layerSizes.length)
      throw new Exception
        ("There must be as many activation functions as hidden layers!");
    
    
    //Declare the layers ...
    layers = new Layer[layerSizes.length];
    
    
    //... and initialize them so
    //that every layer has as many neurons as it should have,
    //and as many inputs as the previous layer has neurons
    
    //First layer
    layers[0] = new Layer(numberOfInputs, layerSizes[0],
            activationFunctions[0]);
    //Hidden layers + output layer
    for(int i=1; i<layerSizes.length; i++) {
      layers[i] = new Layer(layerSizes[i-1], layerSizes[i],
              activationFunctions[i]);
    }
  }
  
  
  /**
   * Calulates the output based on the given input. Forward propagation
   * @param input The input to feed into the network
   * @return The calculated output of the network
   * @throws Exception If an error in a layer occurs
   */
  public double[] calculate(double[] input) throws Exception
  {
    //Take the input ...
    double[] temp  = input;
    
    //... and feed it into a layer and feed  the output into the next layer
    for (Layer layer : layers) {
      temp = layer.calculate(temp);
    }
    
    
    return temp;
  }
  
  /**
   * Trains the network with the given input and output
   * @param loops How often it tests all the data sets
   * @param epsilon The difference of the weight to measure
   *                        the gradient of the error
   * @param learningRate The leaning rate that gets applied to the weights
   * @param weightMinimum The minimum value a weight can have
   * @param weightMaximum The maximum value a weight can have
   * @param input The given training input sets
   * @param output The given training output sets
   * @return Documentation of the costs. Only for graphical/debbug purpose
   * @throws Exception If the input & output sizes don't match
   */
  public double[] train(int loops, double epsilon, double learningRate, 
          double weightMinimum, double weightMaximum,
          double[][] input, double[][] output) throws Exception
  {
    if(input.length != output.length)
      throw new Exception
        ("Number of inputs must equal the number of given outputs!");
    
    
    //Documentation of the costs for debugging and visualisation purposes
    double[] costs = new double[loops];
    
    //Go through the whole network many times
    for(int k=0; k<loops; k++) {
      //For every layer ...
      for(Layer layer : layers) {
        // ... save the weights
        double[][] weights = layer.getWeights();
        
        //Adjust every weight ...
        for(int j=0; j<weights.length; j++) {
          for(int i=0; i<weights[j].length; i++) {
            
            double cost_before_change = cost(input, output);
            
            
            //Calculate the change of the cost after the change of the weight
            weights[j][i] += epsilon;
            layer.setWeights(weights);
            double cost1 = cost(input, output);
            
            weights[j][i] -= 2*epsilon;
            layer.setWeights(weights);
            double cost2 = cost(input, output);
            weights[j][i] += epsilon;
            
            
            //Apply changes to the weight so that the costs decrease
            double gradient = (cost1 - cost2) / (2 * epsilon);
            weights[j][i] += - gradient * learningRate;
            
            //Keep the weight in the bounds
            if(weights[j][i] < weightMinimum) {
              weights[j][i] = weightMinimum;
            } else if(weights[j][i] > weightMaximum) {
                weights[j][i] = weightMaximum;
            }
            
            
            //Update the layer
            layer.setWeights(weights);
            
            
//            //For debugging: Check if cost increased
//            if(cost(input, output) > 2*cost_before_change)
//              throw new Exception("Cost increased more than 200%!");
            
          }
        }//Weights
      }//Layers
      
      costs[k] = cost(input, output);
//      //For debugging: output the progress
//      System.out.println(100.0*k/loops + "%: " + costs[k]);
      
    }//Loops
    
    
    return costs;
  }
  
  
  /**
   * Caluclates the costs with the given test inputs and outputs.
   * Costs are the mean squared error.
   * @param input The given test inputs
   * @param output The given test output
   * @return The costs of the given test set
   * @throws Exception If the input & output sizes don't match
   */
  public double cost(double[][] input, double[][] output) throws Exception
  {
    if(input.length != output.length)
      throw new Exception("Number of inputs must equal the number of given outputs!");
    
    
    double cost = 0;
    
    //Sum the costs for every data set
    for(int j=0; j<input.length; j++) {
      //Calculate the actual result
      double[] result = calculate(input[j]);
      double error = 0;
      
      //And compare every actual output with the wanted output
      for(int i=0; i<result.length; i++) {
        error += (output[j][i] - result[i]) * (output[j][i] - result[i]);
      }
      cost += error;
    }
    
    cost *= 0.5;
    
    return cost;
  }
  
  
  /**
   * Initalizes all weights of all layers with random numbers
   * @param minimum Minimal value a weight can have
   * @param maximum Maximal value a weight can have
   */
  public void seedWeights(double minimum, double maximum)
  {
    //Seed the weights of every layer
    for(Layer layer : layers) {
      layer.seedWeights(minimum, maximum);
    }
  }
  
  /**
   * Initalizes all weights of all layers with random numbers based on a seed
   * @param minimum Minimal value a weight can have
   * @param maximum Maximal value a weight can have
   * @param seed Seed for the random weights
   */
  public void seedWeights(double minimum, double maximum, long seed)
  {
    //Seed the weights of every layer
    for(Layer layer : layers) {
      layer.seedWeights(minimum, maximum, seed++);
    }
  }
  
  
  /**
   * Sets the weights of all layers
   * @param weights The weights. Every layer needs a 2-dimensional weight array
   * @throws Exception If the number of weight sets doesn't fit
   */
  public void setWeights(double[][][] weights) throws Exception
  {
    if(weights.length != layers.length)
      throw new Exception("There must be as many weight arrays as layers!");
    
    for(int i=0; i<layers.length; i++) {
      layers[i].setWeights(weights[i]);
    }
  }
  
  /**
   * Sets the weights of a given layer
   * @param layer Index of the layer of which the weights should be set
   * @param weights The weights for the layer
   * @throws Exception If an error occurs in the layer
   */
  public void setWeights(int layer, double[][] weights) throws Exception
  {
    layers[layer].setWeights(weights);
  }
  
  /**
   * @return All layers
   */
  public Layer[] getLayers()
  {
    return layers;
  }
  
  /**
   * @param index Index of the layer that should be returned
   * @return The wanted layer
   */
  public Layer getLayer(int index)
  {
    return layers[index];
  }
  
  
  
  //Output a more or less readable representation
  @Override
  public String toString()
  {
    StringBuilder result = new StringBuilder();
    
    //Just every layer one after another
    for(int i=0; i<layers.length; i++) {
      result.append(layers[i].toString());
    }
    
    
    return result.toString();
  }
  
  
  
  //Test everything!
  public static void main(String[] args) throws Exception
  {
    /**
     * This example is based on Brandon Rohrer's "How Deep Neural Networks Work"
     * https://www.youtube.com/watch?v=ILsA4nyG7I0
     * or
     * https://brohrer.github.io/how_neural_networks_work.html
     * Check it out. It's a great explanation how neural networks work.
     * 
     * Input:
     * The network takes a 4 pixel image
     *  0 1
     *  3 4  = {0, 1, 2, 3}
     * With values between -1 and 1
     * 
     * Output:
     * It outputs a 1 in the corresponding output neuron if the image is solid,
     * or if there is a vertical, diagonal or horizontal line
     * {solid, vertical, diagonal, horizontal}
     */
    
    //4 inputs, 4 layers with 4, 4, 8, 4 neurons
    Network net = new Network(4, new int[]{4, 4, 8, 4},
            new int[]{Layer.ACTIVATION_TANH, Layer.ACTIVATION_TANH,
              Layer.ACTIVATION_RECTIFIED_LINEAR_UNIT, Layer.ACTIVATION_NONE});
    
    //Randomize all weights
    net.seedWeights(-1, 1);
    System.out.println(net);
    
    
    //Train or set the weights manually
    double[][] train_input = new double[][]{{ 1,  1,  1,  1},
                                            {-1, -1, -1, -1},
                                            {-1,  1,  1, -1},
                                            { 1, -1, -1,  1},
                                            { 1, -1,  1, -1},
                                            {-1,  1, -1,  1},
                                            { 1,  1, -1, -1},
                                            {-1, -1,  1,  1}};
    double[][] train_output = new double[][]{{1, 0, 0, 0},
                                            {1, 0, 0, 0},
                                            {0, 1, 0, 0},
                                            {0, 1, 0, 0},
                                            {0, 0, 1, 0},
                                            {0, 0, 1, 0},
                                            {0, 0, 0, 1},
                                            {0, 0, 0, 1}};
    
//    net.setWeights(0, new double[][]{{ 1,  0,  1,  0},
//                                      { 0,  1,  0,  1},
//                                      { 0,  1,  0, -1},
//                                      { 1,  0, -1,  0}});
//    net.setWeights(1, new double[][]{{ 1, -1,  0,  0},
//                                      { 1,  1,  0,  0},
//                                      { 0,  0,  1,  1},
//                                      { 0,  0, -1, 1}});
//    net.setWeights(2, new double[][]{{ 1, -1,  0,  0,  0,  0,  0,  0},
//                                      { 0,  0,  1, -1,  0,  0,  0,  0},
//                                      { 0,  0,  0,  0,  1, -1,  0,  0},
//                                      { 0,  0,  0,  0,  0,  0,  1, -1}});
//    net.setWeights(3, new double[][]{{ 1,  0,  0,  0},
//                                      { 1,  0,  0,  0},
//                                      { 0,  1,  0,  0},
//                                      { 0,  1,  0,  0},
//                                      { 0,  0,  1,  0},
//                                      { 0,  0,  1,  0},
//                                      { 0,  0,  0,  1},
//                                      { 0,  0,  0,  1}});
    
    
    //Train the network
    net.train(100, 0.00001, 0.5, -1, 1, train_input, train_output);
    
    System.out.println("\n" + net);
    System.out.println("Costs: " + net.cost(train_input, train_output));
    
    //Show the input and the resulting output
    for(double[] input : train_input)
      System.out.println(Arrays.toString(input) + ": " +
              Arrays.toString(net.calculate(input)));
  }
}
