# Machine Learning

Java collection that provides Java packages for developing a machine learning algorithm and that is
- easy to use -> great for small projects or just to learn how machine learning works
- small and simple -> easy to understand and making changes
- lightweight (mostly because I'm a student who just started to learn how to code Java and can't code more complex :P)

## Code Example

### [Neural Network](src/neural)

If you have some test data consisting of inputs and outputs.
(Be sure that your data is normalized if needed!)

```
double[][] train_input = new double[][] {
        {1, 2},
        {3, 4},
        {5, 6}};
double[][] train_output = new double[][] {
        {0.1},
        {0.2},
        {0.3}};
```

Initialize a new network with a given architecture (number or inputs, number of hidden layers their number of neurons and their activation function, number of outputs)
(If you don't know what to choose: 
- number of hidden layers= number of inputs
- number of neurons per layer: number of inputs (except the last layer is the output layer= as many neurons as outputs)
- activation functions: hyperbolic tangent)

```
Network net = new Network(2,
        new int[]{2, 1},
        new int[]{Layer.ACTIVATION_TANH, Layer.ACTIVATION_TANH});
```

Then you can seed the weights in the network = randomize it.

```
net.seedWeights(-1, 1);
```

Now your network is ready for getting trained!
Just feed it how often it should try adjusting every weight and how much, within which range, and of course it also needs the training data.

```
net.train(1000, 0.002, -1, 1, train_input, train_output);
```

If you would like to see the progress, decomment this line in the train method:

```
//      costs[k] = cost(input, output);
//      //For debugging output progress
//      System.out.println(100.0*k/loops + "%: " + costs[k]);
```

Now the network should be trained so let's have a look at the network itself by simply printing a basic representation and try a new input to make a prediction.

```
System.out.println(net);
System.out.println(Arrays.toString(net.calculate(new double[] {7, 8})));
```

And if we would like to get the mean squared error we just call on some new test data:

```
net.cost(train_input, train_output);
```

## Installation

I included all packages with the source files in the [Source folder /src](src).
Just add them to your project and you are ready to go!

## Tests

In every Java class should be a main method that shows how to use this specific class.

## Contributors

The two people who inspired me to try making my own machine learning project are Stephen Welch and Brandon Rohrer.
Both make awesome YouTube videos that explain how machine learning works.

Stephen Welch:
- YouTube: https://www.youtube.com/user/Taylorns34
- Homepage: http://www.welchlabs.com/
- GitHub: https://github.com/stephencwelch

Brandon Rohrer:
- YouTube https://www.youtube.com/channel/UCsBKTrp45lTfHa_p49I2AEQ
- Blog: https://brohrer.github.io/blog.html
- GitHub: https://github.com/brohrer

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
