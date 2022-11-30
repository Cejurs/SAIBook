using NeuralNetwork;

var neuralNetword = new SimpleNeuralNetwork(new Topology(100,10,0.5,60,60));

var input = new double[100];
neuralNetword.Predict(input);
