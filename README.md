# Machine Learning

Java collection that provides Java packages for developing a machine learning algorithm and that is
- easy to use -> great for small projects or just to learn how machine learning works
- small and simple -> easy to understand and making changes
- lightweight (mostly because I'm a student who just started to learn how to code Java and can't code more complex :P)

## Code Example

### [Neural Network](src/main/java/neural)

If you have some test data consisting of inputs and outputs.
(Be sure that your data is normalized if needed!)

```
Matrix inputs = new Matrix(
        new double[][]{
          {1, 2},                   //<- first input set
          {3, 4}});                 //<- second input set
Matrix outputs = new Matrix(
        new double[][]{
          {0.5},                    //<- first output set
          {0.75}});                 //<- second output set
```

Initialize a new network with a given architecture (number or inputs, their number of neurons in the hidden layers and each layers activation function)
(If you don't know what to choose, here is a rule of thumb for a quadratic looking network: 
- number of hidden layers = number of inputs
- number of neurons per layer: number of inputs (except the last layer is the output layer = as many neurons as outputs)
- activation functions: hyperbolic tangent)

```
Network net = new Network(
        2,                          //2 inputs
        new int[]{3, 8, 1},         //3 layers with 2, 8 and 1 neurons
        new Network.ActivationFunction[]{   //1 activation function
          Network.ActivationFunction.NONE,  //for every layer
          Network.ActivationFunction.TANH,
          Network.ActivationFunction.NONE});
```

Then you can seed the weights in the network = randomize it.

```
net.seedWeights(-1, 1);
```

Now your network is ready for training!
Just say it how drastic the changes should be, how often it should go through a training cycle, the training data of course and if it should print it's progress to the console.
* learning rate: higher = faster training but to high could miss the optimum, slower = better result (sometimer it goes crazy and the cost just increase, then try decreasing the laerning rate)
* iterations: how often it should cycle through the training process (backpropagation and application)
* inputs: training sets inputs
* outputs: wanted outputs the network should imitate
* printToConsole: show the progress in the console


```
net.train(0.001, 40, inputs, outputs, true);
```

Now the network should be trained so let's have a look at the network itself by simply printing a basic representation and try forwarding the inputs.

```
System.out.println(net);
System.out.println(net.forward(inputs));
```

And if we would like to get the mean squared error we just call the cost function on some test data:

```
System.out.println(net.cost(inputs, outputs));
```

## Installation

I included all packages with the source files in the [Source folder /src](src).
Just add them to your project and you are ready to go!

## Tests

In every Java class should be a main method that shows how to use this specific class.

## Contributors

The two people who inspired me to try making my own machine learning project are Brandon Rohrer and Stephen Welch.
Both make awesome YouTube videos that explain how machine learning works.

Brandon Rohrer:
- YouTube https://www.youtube.com/channel/UCsBKTrp45lTfHa_p49I2AEQ
- Blog: https://brohrer.github.io/blog.html
- GitHub: https://github.com/brohrer

Stephen Welch:
- YouTube: https://www.youtube.com/user/Taylorns34
- Homepage: http://www.welchlabs.com/
- GitHub: https://github.com/stephencwelch

## License (MIT)

MIT License

Copyright (c) 2017 Sebastian Gössl

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
