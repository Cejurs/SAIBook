using NeuralNetwork;

var neuralNetword = new SimpleNeuralNetwork();
var sigmoid = new Sigmoid();
neuralNetword.CreateInputLayer(3, sigmoid);
neuralNetword.CreateHiddenLayer(3, sigmoid);
neuralNetword.CreateOutputLayer(3, sigmoid);
var input = new double[3];
neuralNetword.Predict(input);
neuralNetword.SaveWeights(@"C:\Users\Cejurs\Desktop\Папка с папками\CИИ\2.txt");
var newNeuralnetwork = new SimpleNeuralNetwork();
var sigmoid2 = new Sigmoid();
newNeuralnetwork.CreateInputLayer(3, sigmoid2);
newNeuralnetwork.CreateHiddenLayer(3, sigmoid2);
newNeuralnetwork.CreateOutputLayer(3, sigmoid2);
newNeuralnetwork.LoadWeights(@"C:\Users\Cejurs\Desktop\Папка с папками\CИИ\2.txt");
Console.ReadLine();
