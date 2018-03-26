# Machine Learning

Java collection that provides Java packages for developing machine learning algorithms and that is
- easy to use -> great for small projects or just to learn how machine learning works
- small and simple -> easy to understand and make changes
- lightweight (mostly because I'm a student who just started to learn how to code Java and can't code more complex :P)

## Getting Started

### Prerequisites

This project is written in pure vanilla Java so there is nothing needed than the standard libraries.

### Installation

Just add all packages with the source files in the [Source folder /src](src) to your project and you are ready to go!
Every class has a main test method. After installation just run any class so you can check if the installation was successful.

## Code Example

### [Neural Network](src/main/java/neural)

Initialize a new network with a given architecture (number or inputs, number of neurons in the hidden layers and each layers activation function)
(If you don't know what to choose, here is a rule of thumb for a average looking network: 
- number of hidden layers = 2
- number of neurons per layer: number of inputs (except the last layer is the output layer = as many neurons as outputs)
- activation functions: none)

```
//New network
final Network net = new Network(
        2,                                    //2 inputs
        new int[]{3, 1},                      //2 layers with 3 & 1 neurons
        new Network.ActivationFunction[]{
          Network.ActivationFunction.NONE,    //both layers with ...
          Network.ActivationFunction.NONE});  //... no activation function
```

Then you can seed the weights in the network (= randomize it).

```
net.seedWeights(-1, 1);
```

Prepare your training data and put it into a [Matrix] (src/main/java/neural/Matrix.java)

```
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
```

Now your network is ready for training!
Just tell it how drastic the changes should be, give it the training data and if it should print it's progress to the console.
* learning rate: higher = faster training but to high could miss the optimum, slower = better result (sometimer it goes crazy and the cost just increase, then try decreasing the laerning rate)
* inputs: training set inputs
* outputs: wanted outputs the network should learn from
* printToConsole: show the progress in the console


```
net.train(0.2, trainInput, trainOutput, true);
```

Now the network should be trained so let's have a look at the network itself by simply printing a basic representation and try forwarding the inputs.

```
System.out.println(net);
System.out.println(net.forward(trainInput));
```

And if we would like to get the mean squared error we just call the cost function on some data:

```
System.out.println(net.cost(trainInput, trainOutput));
```

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
