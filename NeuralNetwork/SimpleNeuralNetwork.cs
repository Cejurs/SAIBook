using System.Linq;

namespace NeuralNetwork
{
    public class SimpleNeuralNetwork
    {
        private Neuron[][] neurons;

        public SimpleNeuralNetwork(params int[] neuronsCountInLayers)
        {
            neurons = new Neuron[neuronsCountInLayers.Length][];

            for (int i = 0; i < neurons.Length; i++)
            {
                neurons[i] = new Neuron[neuronsCountInLayers[i]];
                if (i != 0)
                {
                    for (int j = 0; j < neuronsCountInLayers[i]; j++)
                    {
                        neurons[i][j] = new Neuron(neuronsCountInLayers[i - 1]);
                    }
                }
            }
        }
        private double[][] FeedForward(double[] input)
        {
            double[][] output = new double[neurons.Length][];

            //Выход первого слоя
            output[1] = new double[neurons[1].Count()];
            for (int j = 0; j < neurons[1].Count(); j++)
            {
                output[1][j] = neurons[1][j].GetOutput(input);
            }

            //Выходы оставшихся слоев
            for (int i = 2; i < output.Length; i++)
            {
                output[i] = new double[neurons[i].Count()];
                {
                    for (int j = 0; j < neurons[i].Count(); j++)
                    {
                        output[i][j] = neurons[i][j].GetOutput(output[i - 1]);
                    }
                }
            }

            return output;
        }
        public double[] Predict(double[] input)
        {
            double[][] output = FeedForward(input);
            return output[neurons.Count() -1];
        }
        public bool Learn(double[] input, double[] correctOutput)
        {
            double[][] output = FeedForward(input);
            int countlearning = 0;
            if (CompareVectors(correctOutput, output[neurons.Count() - 1]))
            {
                return true;
            }

            while (!CompareVectors(correctOutput, output[neurons.Count() - 1]))
            {
                countlearning++;
                if (countlearning > 1000)
                {
                    break;
                }

                for (int i = 0; i < output[neurons.Count() - 1].Count(); i++)
                {
                    neurons[neurons.Count() - 1][i].UpdateErrorOutputLayer(correctOutput[i], output[neurons.Count() - 1][i]);
                }

                for (int i = output.Count() - 2; i >= 1; i--)
                {
                    for (int j = 0; j < output[i].Count(); j++)
                    {
                        neurons[i][j].UpdateError(neurons[i + 1], output[i][j], j);
                    }
                }

                for (int i = neurons.Count() - 1; i >= 1; i--)
                {
                    for (int j = 0; j < neurons[i].Count(); j++)
                    {
                        if (i - 1 != 0)
                        {
                            neurons[i][j].UpdateWeights(output[i - 1]);
                        }
                        else if (i - 1 == 0)
                        {
                            neurons[i][j].UpdateWeights(input);
                        }
                    }
                }
                output = FeedForward(input);
            }

            return false;
        }

        private bool CompareVectors(double[] a, double[] b)
        {
            if (a.Length != b.Length)
                return false;

            for (int i = 0; i < a.Length; i++)
                if ((a[i] == 0 && b[i] > 0.1) || (a[i] == 1 && b[i] < 0.9))
                    return false;

            return true;
        }
    }
}
