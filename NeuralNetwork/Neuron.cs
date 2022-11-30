using System.Runtime.Serialization;

namespace NeuralNetwork
{
    public enum NeuronType
    {
        Input,
        Hidden,
        Output
    }
    [Serializable]
    public class Neuron
    {
        private double[] weights;
        [NonSerialized]
        private double[] inputs;
        public double Delta { get; private set; }
        public NeuronType Type { get; }

        public double Output { get; private set; }

        public Neuron(int inputCount, NeuronType type)
        {
            if (inputCount <= 0) throw new ArgumentOutOfRangeException(nameof(inputCount));
            weights = new double[inputCount];
            inputs = new double[inputCount];
            Type = type;
            var random = new Random();
            for (int i=0; i < inputCount; i++)
            {
                weights[i] = Type == NeuronType.Input ? 1 : random.NextDouble();
            }
        }

        public double GetWeight(int index) => weights[index];
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
                sum = inputs[i] * weights[i];
            }
            Output = Type == NeuronType.Input ? sum : Sigmoid(sum);
            return Output;
        }

        public void Learn(double error,double learningRate)
        {
            if(Type == NeuronType.Input) return;
            Delta = error * SigmoidDx(Output);
            for (int i =0; i< weights.Length;i++)
            {
                var input = inputs[i];
                var weight = weights[i];
                weights[i] = weight - Delta * input * learningRate;
            }
        }

        private double Sigmoid(double x)
        {
            return 1.0 / (1.0 + Math.Pow(Math.E, -x));
        }
        private double SigmoidDx(double x)
        {
            var sigmoid = Sigmoid(x);
            return sigmoid*(1-sigmoid);
        }
        [OnDeserialized]
        private void OnDeserialized()
        {
            inputs = new double[weights.Length];
        }
    }
}
