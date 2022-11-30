namespace NeuralNetwork
{
    [Serializable]
    public class Layer
    {
        public Neuron[] Neurons { get; }
        public int NeuronsCount { get { return Neurons.Length;} }
        public NeuronType LayerType;

        public Layer(IEnumerable<Neuron> neurons, NeuronType layerType)
        {
            foreach(Neuron neuron in neurons)
            {
                if (neuron.Type != layerType) throw new ArgumentException("Нейроны на слое должны соответсвовать типу слоя");
            }
            Neurons=neurons.ToArray();
        }
        public IEnumerable<double> GetSignals()
        {
            foreach(var neuron in Neurons)
            {
                yield return neuron.Output;
            }
        }
    }
}
