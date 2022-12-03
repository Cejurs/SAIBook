

using System.Text;
using System.Text.Json;

namespace NeuralNetwork
{
    public class SimpleNeuralNetwork
    {
        private IList<Layer> layers;
        public double LearningRate { get; private set; }

        public SimpleNeuralNetwork()
        {
            layers = new List<Layer>();
        }
        public void SetLearningRate(double learningRate)
        {
            this.LearningRate = learningRate;
        }
        public void CreateInputLayer(int neuronsCount,IActivator activationFunction)
        {
            if (layers.Count != 0) throw new ArgumentException("Входной слои может быть создан только 1");
            if(neuronsCount <= 0 ) throw new ArgumentOutOfRangeException("Количество нейронов на входном слое не может быть 0 или отрицательным");
            var inputNeurons = new List<Neuron>();
            for (int i = 0; i < neuronsCount; i++)
            {
                var neuron = new Neuron(1, activationFunction, NeuronType.Input);
                inputNeurons.Add(neuron);
            }
            var inputLayer = new Layer(inputNeurons, NeuronType.Input);
            layers.Add(inputLayer);
        }

        public void CreateHiddenLayer(int neuronsCount, IActivator activationFunction)
        {
            if (layers.Count == 0) throw new ArgumentException("Отсутствует входной слой");
            if (neuronsCount <= 0 ) throw new ArgumentOutOfRangeException("Количество нейронов на скрытом слое не может быть 0 или отрицательным");
            var hiddenNeurons = new List<Neuron>();
            for (int iter = 0; iter < neuronsCount; iter++)
            {
                var lastLayerNeuronsCount = layers.Last().NeuronsCount;
                var neuron = new Neuron(lastLayerNeuronsCount,activationFunction, NeuronType.Hidden);
                hiddenNeurons.Add(neuron);
            }
            var hiddenLayer = new Layer(hiddenNeurons, NeuronType.Hidden);
            layers.Add(hiddenLayer);
        }

        public void CreateOutputLayer(int neuronsCount, IActivator activationFunction)
        {
            if (layers.Count == 0) throw new ArgumentException("Входной слой должен быть создан");
            var outputNeurons = new List<Neuron>();
            var lastLayerNeuronsCount = layers.Last().NeuronsCount;
            for (int i = 0; i < neuronsCount; i++)
            {
                var neuron = new Neuron(lastLayerNeuronsCount, activationFunction, NeuronType.Output);
                outputNeurons.Add(neuron);
            }
            var outputLayer = new Layer(outputNeurons, NeuronType.Output);
            layers.Add(outputLayer);
        }

        public double[] Predict(IList<double> inputSignals)
        {
            if (layers.First().LayerType != NeuronType.Input) throw new Exception("Отсутсвует входной слой");
            if (layers.Last().LayerType != NeuronType.Output) throw new Exception("Отсутсвует выходной слой");
            if (inputSignals.Count() != layers.First().NeuronsCount)
            {
                throw new ArgumentException("Количество входных сигналов не совпдает с количеством нейронов на входном слое");
            }
            SendInputSignalsToInputLayer(inputSignals);
            FeedForward();
            return layers.Last().GetSignals().ToArray();

        }

        private void FeedForward()
        {
            for (int i = 1; i<layers.Count; i++)
            {
                var signals = layers[i-1].GetSignals().ToArray();
                foreach (var neuron in layers[i].Neurons)
                {
                    neuron.FeedForward(signals);
                }
            }
        }

        //public double Learn(double[] inputSignals, double[] expectedResult,int epochCount)
        //{
        //    if(expectedResult.Length != Topology.OutputCount) 
        //    {
        //        throw new ArgumentException(nameof(expectedResult));
        //    }
        //    if (inputSignals.Length != Topology.InputCount)
        //    {
        //        throw new ArgumentException(nameof(inputSignals));
        //    }
        //    if (epochCount < 0)
        //    {
        //        throw new ArgumentException(nameof(epochCount));
        //    }
        //    for (int i = 0; i < epochCount; i++)
        //    {
        //        var totalError = 0.0;

        //    }
        //}

        private double Backpropagation(double[] inputSignals, double[] expectedResult)
        {
            double squareErrorSum = BackpropagationOnOutputLayer(inputSignals, expectedResult);
            BackpropagationOnHiddenLayers();
            return squareErrorSum;
        }
        private void BackpropagationOnHiddenLayers()
        {
            for(int i = layers.Count -2; i>=1;i--)
            {
                var currentLayer = layers[i];
                var previousLayer = layers[i+1];
                Parallel.For(0, currentLayer.NeuronsCount, (n) =>
                {
                    var neuron = currentLayer.Neurons[n];
                    var error = 0.0;
                    for(int j = 0; j< previousLayer.NeuronsCount; j++)
                    {
                        var previousNeuron = previousLayer.Neurons[j];
                        error += previousNeuron.Delta * previousNeuron.Weights[n];
                    }
                    neuron.Learn(error, LearningRate);
                });
            }
        }
        private double BackpropagationOnOutputLayer(double[] inputSignals, double[] expectedResult)
        {
            var squareErrorSum = 0.0;
            var actualResult = Predict(inputSignals);
            var outputLayer = layers.Last();
            for (int i = 0; i < expectedResult.Length; i++)
            {
                var error = actualResult[i] - expectedResult[i];
                outputLayer.Neurons[i].Learn(error,LearningRate);
                squareErrorSum += error * error;
            }
            return squareErrorSum;
        }
        private void SendInputSignalsToInputLayer(IList<double> inputSignals)
        {
            var inputLayer = layers.First();
            for (int i = 0; i < inputSignals.Count(); i++)
            {
                inputLayer.Neurons[i].FeedForward(new double[] { inputSignals[i] });
            }
        }

        public void SaveWeights(string path)
        {
            if (File.Exists(path) == false)
            {
                File.Create(path).Close();
            }
            using (var stream = new StreamWriter(path, false))
            {
                var sb = new StringBuilder();
                for (int i = 0; i < layers.Count-1; i++)
                {
                    sb.Append($"{layers[i].NeuronsCount}|");
                }
                sb.Append(layers.Last().NeuronsCount);
                stream.WriteLine(sb.ToString());
                for (int i = 0; i < layers.Count(); i++)
                {
                    for (int j = 0; j < layers[i].NeuronsCount; j++)
                    {
                        var jsonString = JsonSerializer.Serialize(layers[i].Neurons[j].Weights);
                        stream.WriteLine(jsonString);
                    }
                }
            }
        }

        public  void LoadWeights(string path)
        {
            if(File.Exists(path)==false)
            {
                throw new FileNotFoundException();
            }
            using(var stream = new StreamReader(path))
            {
                var line=stream.ReadLine();
                if (String.IsNullOrEmpty(line)) return; // Можно выкинуть исключение
                var topology = line.Split('|');
                if (topology.Length != layers.Count) throw new ArgumentException("Топология из файла не совпадает с топологией сети");
                for (int i=0; i<layers.Count; i++)
                {
                    if (layers[i].NeuronsCount != int.Parse(topology[i])) throw new ArgumentException("Количество нейронов не совпадает");
                }
                for (int i = 0; i < topology.Length; i++)
                {
                    for ( int j=0; j < int.Parse(topology[i]); j++)
                    {
                        line = stream.ReadLine();
                        if (String.IsNullOrEmpty(line)) throw new Exception();
                        var weights = JsonSerializer.Deserialize(line, typeof(IEnumerable<double>));
                        layers[i].Neurons[j].SetWeights((IEnumerable<double>)weights);
                    }
                }
            }
        }
    }
}
