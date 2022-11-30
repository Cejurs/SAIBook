namespace NeuralNetwork
{
    [Serializable]
    public class Topology
    {
        // Количество нейронов на входном слое
        public int InputCount { get; }
        // Количество нейронов на выходном слое
        public int OutputCount { get; }

        // Количество скрытых слоев и нейроннов на них
        public int[] HiddenLayers { get; }

        public double LearningRate { get; }

        public Topology(int inputCount, int outputCount, double leraningRate, params int[] hiddenlayers)
        {
            if (inputCount <= 0) throw new ArgumentException("Количество нейронов на входном слое не может быть 0 или отрицательным");
            if (outputCount <= 0) throw new ArgumentException("Количество нейронов на выходном слое слое не может быть 0 или отрицательным");
            foreach(var hiddenLayerCount in hiddenlayers)
            {
                if (hiddenLayerCount <= 0) throw new ArgumentException("Количество нейронов на скрытом слое не может быть 0 или отрицательным");
            }
            InputCount = inputCount;
            LearningRate = leraningRate;
            OutputCount = outputCount;
            HiddenLayers = hiddenlayers;
        }
    }
}
