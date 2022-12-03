using System.Runtime.Serialization;

namespace NeuralNetwork
{
    public enum NeuronType
    {
        Input,
        Hidden,
        Output
    }
    public class Neuron
    {
        public double[] Weights { get; private set; }
        private double[] inputs;

        private IActivator activator;
        public double Delta { get; private set; }
        public NeuronType Type { get; private set; }

        public double Output { get; private set; }

        public Neuron(int inputCount, IActivator activationFunction, NeuronType type)
        {
            if (inputCount <= 0) throw new ArgumentOutOfRangeException(nameof(inputCount));
            Weights = new double[inputCount];
            inputs = new double[inputCount];
            Type = type;
            activator = activationFunction;
            var random = new Random();
            for (int i=0; i < inputCount; i++)
            {
                Weights[i] = Type == NeuronType.Input ? 1 : random.NextDouble();
            }
        }
        internal void SetWeights(IEnumerable<double> weights)
        {
            if(weights.Count() != this.inputs.Length)
            {
                throw new ArgumentOutOfRangeException();
            }
            this.Weights = weights.ToArray();
        }
        public double FeedForward(IEnumerable<double> inputSignals)
        {
            if (inputSignals.Count() != inputs.Length)
            {
                throw new ArgumentException("Количество входных сигналов не совпадает с количеством входов у нейрона");
            }
            inputs = inputSignals.ToArray();
            var sum = 0.0;
            for(int i=0; i < inputs.Length; i++)
            {
                sum = inputs[i] * Weights[i];
            }
            Output = Type == NeuronType.Input ? sum : activator.Activate(sum);
            return Output;
        }

        public void Learn(double error,double learningRate)
        {
            if(Type == NeuronType.Input) return;
            Delta = error * activator.Dx(Output);
            for (int i =0; i< Weights.Length;i++)
            {
                var input = inputs[i];
                var weight = Weights[i];
                Weights[i] = weight - Delta * input * learningRate;
            }
        }
    }
}
