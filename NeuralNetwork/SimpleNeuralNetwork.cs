using System.Diagnostics;

namespace NeuralNetwork
{
    [Serializable]
    public class SimpleNeuralNetwork
    {
        private IList<Layer> layers;

        public Topology Topology { get;}
        public SimpleNeuralNetwork(Topology topology)
        {
            Topology= topology;
            Topology = topology;
            layers = new List<Layer>();

            CreateInputLayer();
            CreateHiddenLayers();
            CreateOutputLayer();
        }

        private void CreateInputLayer()
        {
            var inputNeurons = new List<Neuron>();
            for (int i = 0; i < Topology.InputCount; i++)
            {
                var neuron = new Neuron(1, NeuronType.Input);
                inputNeurons.Add(neuron);
            }
            var inputLayer = new Layer(inputNeurons, NeuronType.Input);
            layers.Add(inputLayer);
        }

        private void CreateHiddenLayers()
        {
            for (int iter = 0; iter < Topology.HiddenLayers.Length; iter++)
            {
                var hiddenNeurons = new List<Neuron>();
                var lastLayerNeuronsCount = layers.Last().NeuronsCount;
                for (int i = 0; i < Topology.HiddenLayers[iter]; i++)
                {
                    var neuron = new Neuron(lastLayerNeuronsCount, NeuronType.Hidden);
                    hiddenNeurons.Add(neuron);
                }
                var hiddenLayer = new Layer(hiddenNeurons,NeuronType.Hidden);
                layers.Add(hiddenLayer);
            }
        }

        private void CreateOutputLayer()
        {
            var outputNeurons = new List<Neuron>();
            // Слои добавляются последовательно,поэтому сработает
            var lastLayerNeuronsCount = layers.Last().NeuronsCount;
            for (int i = 0; i < Topology.OutputCount; i++)
            {
                var neuron = new Neuron(lastLayerNeuronsCount, NeuronType.Output);
                outputNeurons.Add(neuron);
            }
            var outputLayer = new Layer(outputNeurons, NeuronType.Output);
            layers.Add(outputLayer);
        }

        public double[] Predict(IList<double> inputSignals)
        {
            if (inputSignals.Count() != Topology.InputCount)
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

        public double Learn(double[] inputSignals, double[] expectedResult,int epochCount)
        {
            if(expectedResult.Length != Topology.OutputCount) 
            {
                throw new ArgumentException(nameof(expectedResult));
            }
            if (inputSignals.Length != Topology.InputCount)
            {
                throw new ArgumentException(nameof(inputSignals));
            }
            if (epochCount < 0)
            {
                throw new ArgumentException(nameof(epochCount));
            }
            for (int i = 0; i < epochCount; i++)
            {
                var totalError = 0.0;

            }
        }

        private double Backpropagation(double[] inputSignals, double[] expectedResult)
        {
            var squareErrorSum = 0.0;
            var actualResult = Predict(inputSignals);
            var outputLayer = layers.Last();
            for (int i = 0; i < expectedResult.Length; i++)
            {
                var error = actualResult[i] - expectedResult[i];
                outputLayer.Neurons[i].Learn(error, Topology.LearningRate);
                squareErrorSum += error * error;
            }
        }
        private void SendInputSignalsToInputLayer(IList<double> inputSignals)
        {
            var inputLayer = layers.First();
            for (int i = 0; i < inputSignals.Count(); i++)
            {
                inputLayer.Neurons[i].FeedForward(new double[] { inputSignals[i] });
            }
        }
    }
}
